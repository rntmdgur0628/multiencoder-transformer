"""Shared reconstruction loss and clearly labelled local diagnostics, not official MSR metrics."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def reconstruction_loss(prediction, target, lengths, fft_sizes=(256, 512, 1024)):
    losses = []
    for pred, truth, length in zip(prediction, target, lengths):
        pred, truth = pred[..., :length], truth[..., :length]
        waveform = (pred - truth).abs().mean()
        spectral = pred.new_zeros(())
        for size in fft_sizes:
            p, y = pred.flatten(0, 1), truth.flatten(0, 1)
            if length < size:
                p, y = F.pad(p, (0, size - length)), F.pad(y, (0, size - length))
            window = torch.hann_window(size, device=pred.device)
            a = torch.stft(p, size, size // 4, window=window, center=False, return_complex=True).abs()
            b = torch.stft(y, size, size // 4, window=window, center=False, return_complex=True).abs()
            spectral = spectral + ((a - b).abs().mean() + (a.log1p() - b.log1p()).abs().mean()) / len(fft_sizes)
        losses.append(waveform + spectral)
    return torch.stack(losses).mean()


def source_diagnostics(prediction, target):
    """Joint-stereo per-source waveform SNR and SI-SDR; silent targets -> None."""
    result = []
    for pred, truth in zip(prediction.double(), target.double()):
        p, y = pred.flatten(), truth.flatten()
        energy = y.square().sum()
        row = {"mae": (p - y).abs().mean().item(), "output_rms": p.square().mean().sqrt().item()}
        if energy <= 1e-12:
            row.update(snr_db=None, sisdr_db=None)
        else:
            row["snr_db"] = (10 * torch.log10(energy / (p - y).square().sum().clamp_min(1e-12))).item()
            p, y = p - p.mean(), y - y.mean()
            projected = (p @ y) * y / y.square().sum().clamp_min(1e-12)
            row["sisdr_db"] = (10 * torch.log10(projected.square().sum().clamp_min(1e-12) /
                                               (p - projected).square().sum().clamp_min(1e-12))).item()
        result.append(row)
    return result
