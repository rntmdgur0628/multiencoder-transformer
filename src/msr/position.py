"""Angular input chart and block LieRE; family IDs are NOT group coordinates."""
from __future__ import annotations

import torch
from torch import nn


def angular_coordinates(time_seconds, frequency_hz, frequency_valid, sample_rate,
                        span, time_scale=1.0):
    """[T], [K], [K] -> [T,K,3]. Frequency is position, not energy or signal phase."""
    theta = span * frequency_hz / (sample_rate / 2)
    valid = frequency_valid.to(theta.dtype)
    t = (time_seconds / time_scale)[:, None].expand(-1, theta.numel())
    return torch.stack((t, (valid * theta.cos())[None].expand_as(t),
                        (valid * theta.sin())[None].expand_as(t)), -1)


class LieRE(nn.Module):
    """Head-specific noncommuting SO(block) factors, shared across families.

    General generators do not guarantee exact relative-displacement invariance.
    The sin/cos INPUT chart is periodic, not an assertion of frequency symmetry.
    """
    def __init__(self, heads, head_dim, block=4):
        super().__init__()
        if block < 2 or head_dim % block:
            raise ValueError("invalid LieRE block")
        self.heads, self.head_dim, self.block = heads, head_dim, block
        self.raw = nn.Parameter(torch.randn(heads, head_dim // block, 3, block, block) * 0.02)

    def rotations(self, coordinates):
        # Matrix exponentials remain FP32 under AMP; preserve FP64 for gradcheck.
        dtype = torch.float64 if self.raw.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=self.raw.device.type, enabled=False):
            a = self.raw.to(dtype)
            a = (a - a.transpose(-1, -2)) * 0.5
            skew = torch.einsum("...r,hgrij->...hgij", coordinates.to(dtype), a)
            return torch.matrix_exp(skew)

    def forward(self, x, coordinates):
        # x [B,T,K,H,Dh], coordinates [T,K,3]. Same R is used for Q and K.
        return self.apply_rotations(x, self.rotations(coordinates))

    def apply_rotations(self, x, r):
        blocks = x.reshape(*x.shape[:-1], self.head_dim // self.block, self.block)
        rotated = torch.einsum("tkhgij,btkhgj->btkhgi", r, blocks.to(r.dtype))
        return rotated.flatten(-2).to(x.dtype)


def baseline_position(coordinates, width):
    """Fixed coordinate features used by the COMMON trunk in all three arms."""
    index = torch.arange(width, device=coordinates.device)
    axes = index % 3
    scales = torch.exp(-(index // 6).float() * 0.5)
    phase = coordinates[..., axes] * scales
    return torch.where((index % 2) == 0, phase.sin(), phase.cos())
