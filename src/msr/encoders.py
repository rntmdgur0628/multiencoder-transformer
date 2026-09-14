"""Band-preserving baseline. Not compatible with historical all-frequency projections."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .audio import framed
from .config import FAMILIES
from .position import angular_coordinates


@dataclass
class BranchTokens:
    values: torch.Tensor       # [B,T,K,D]
    coordinates: torch.Tensor  # [T,K,3]
    release: torch.Tensor      # [T] inclusive sample endpoint, common frame clock
    valid: torch.Tensor        # [B,T,K]
    family: str

    def with_values(self, values):
        return replace(self, values=values)


def band_edges(n_fft, bands):
    bins = n_fft // 2 + 1
    return tuple(i * bins // bands for i in range(bands + 1))


class SpectralEncoder(nn.Module):
    def __init__(self, cfg, chirplet=False):
        super().__init__()
        self.cfg, self.family = cfg, "chirplet" if chirplet else "gabor"
        self.edges = band_edges(cfg.n_fft, cfg.bands)
        n = torch.arange(cfg.n_fft, dtype=torch.float32) - (cfg.n_fft - 1) / 2
        g = torch.exp(-0.5 * (n / (cfg.n_fft * cfg.gaussian_sigma_fraction)).square())
        g = g / g.norm()
        rates = cfg.chirp_rates if chirplet else (0.0,)
        windows = [g * torch.exp(-1j * math.pi * q * (n / cfg.n_fft).square()) for q in rates]
        self.register_buffer("windows", torch.stack(windows))
        self.projections = nn.ModuleList(nn.Linear((b - a) * 4 * len(rates), cfg.d_model)
                                         for a, b in zip(self.edges[:-1], self.edges[1:]))
        self.register_buffer("frequency_hz", torch.tensor([(a + b - 1) / 2 * cfg.sample_rate / cfg.n_fft
                                                          for a, b in zip(self.edges[:-1], self.edges[1:])]))

    def forward(self, waveform, count):
        cfg = self.cfg
        frames = framed(waveform, cfg.n_fft, cfg.hop, count)
        spectra = torch.fft.fft(frames[:, None] * self.windows[None, :, None, None], dim=-1)
        spectra = spectra[..., :cfg.n_fft // 2 + 1]  # [B,Q,C,T,F], preserve real/imag
        bands = []
        for projection, a, b in zip(self.projections, self.edges[:-1], self.edges[1:]):
            coefficients = torch.view_as_real(spectra[..., a:b]).permute(0, 3, 1, 2, 4, 5).flatten(2)
            bands.append(projection(coefficients))
        values = torch.stack(bands, dim=2)
        release = torch.arange(count, device=waveform.device) * cfg.hop + cfg.hop - 1
        center = (release.float() - (cfg.n_fft - 1) / 2) / cfg.sample_rate
        p = angular_coordinates(center, self.frequency_hz, torch.ones_like(self.frequency_hz, dtype=torch.bool),
                                cfg.sample_rate, cfg.angular_span, cfg.time_scale)
        return BranchTokens(values, p, release, torch.ones(values.shape[:-1], device=values.device, dtype=torch.bool), self.family)


class TemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(2, cfg.d_model, cfg.temporal_kernel, stride=cfg.hop, bias=False)

    def forward(self, waveform, count):
        cfg = self.cfg
        frames = framed(waveform, cfg.temporal_kernel, cfg.hop, count)
        values = F.gelu(torch.einsum("bctw,dcw->btd", frames, self.conv.weight))[:, :, None]
        release = torch.arange(count, device=waveform.device) * cfg.hop + cfg.hop - 1
        center = (release.float() - (cfg.temporal_kernel - 1) / 2) / cfg.sample_rate
        zero = waveform.new_zeros(1)
        p = angular_coordinates(center, zero, zero.bool(), cfg.sample_rate, cfg.angular_span, cfg.time_scale)
        return BranchTokens(values, p, release, torch.ones(values.shape[:-1], device=values.device, dtype=torch.bool), FAMILIES[0])
