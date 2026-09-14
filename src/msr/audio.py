"""Right-aligned analysis and complete-tail WOLA. Output latency <= n_fft-1."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def framed(waveform, window, hop, frames=None):
    if waveform.ndim != 3 or waveform.shape[1] != 2 or waveform.shape[-1] < 1:
        raise ValueError("expected nonempty [B,2,N] waveform")
    left = window - hop
    if frames is None:
        frames = (waveform.shape[-1] + left + hop - 1) // hop
    right = (frames - 1) * hop + window - (waveform.shape[-1] + left)
    if right < 0:
        raise ValueError("insufficient frame count")
    return F.pad(waveform, (left, right)).unfold(-1, window, hop)


class ReferenceSTFT(nn.Module):
    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        # Nonzero endpoints also support hop=n_fft; no NOLA boundary singularity.
        self.register_buffer("window", torch.hamming_window(n_fft, periodic=False))

    def analysis(self, waveform):
        return torch.fft.rfft(framed(waveform, self.n_fft, self.hop) * self.window, dim=-1)

    def synthesis(self, spectrum, length):
        # spectrum [...,T,F], output [...,N]. No mixture-consistency projection.
        frames = torch.fft.irfft(spectrum, n=self.n_fft, dim=-1) * self.window
        leading, count = frames.shape[:-2], frames.shape[-2]
        total = (count - 1) * self.hop + self.n_fft
        flat = frames.reshape(-1, count, self.n_fft).transpose(1, 2)
        output = F.fold(flat, (1, total), (1, self.n_fft), stride=(1, self.hop)).flatten(1)
        weights = self.window.square()[None, :, None].expand(1, -1, count)
        denominator = F.fold(weights, (1, total), (1, self.n_fft), stride=(1, self.hop)).flatten(1)
        output = output / denominator.clamp_min(1e-8)
        left = self.n_fft - self.hop
        if total - left < length:
            raise ValueError("spectrum cannot cover requested length")
        return output[:, left:left + length].reshape(*leading, length)
