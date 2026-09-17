"""Matched TC / multiresolution-STFT adapters for the actual frozen SW parent.

This supersedes (but does not edit) the earlier tiny community API prototype.
No new PE, source heads, output blending, or parent fine-tuning is introduced.
"""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class AdapterConfig:
    bands: tuple
    n_fft: int
    hop: int
    base_dim: int
    short: int = 512
    width: int = 64
    heads: int = 4
    seed: int = 20260917

    def __post_init__(self):
        if min(self.n_fft, self.hop, self.base_dim, self.short, self.width, self.heads) < 1:
            raise ValueError("positive dimensions required")
        if self.n_fft % 4 or self.short % 2 or self.short > self.n_fft or self.hop > self.n_fft:
            raise ValueError("FFT divisible by 4, even short window <= FFT, hop <= FFT required")
        if len(self.bands) < 2 or min(self.bands) < 1 or sum(self.bands) != self.n_fft//2+1:
            raise ValueError("bands must partition the positive FFT frequencies")
        if self.width % self.heads:
            raise ValueError("width must be divisible by attention heads")


class Memory(nn.Module):
    def __init__(self, cfg, arm):
        super().__init__()
        if arm not in ("tc", "stft"):
            raise ValueError("arm must be tc or stft")
        self.cfg, self.arm = cfg, arm
        self.short_projection = nn.Linear(2*cfg.short, cfg.width)
        self.band_projections = nn.ModuleList(nn.Linear(8*b, cfg.width) for b in cfg.bands)
        self.norm = nn.LayerNorm(cfg.width)
        self.type_embedding = nn.Parameter(torch.zeros(2, cfg.width))
        n = torch.arange(cfg.n_fft) - (cfg.n_fft-1)/2
        if arm == "tc":
            gaussian = torch.exp(-0.5*(n/(cfg.n_fft/6))**2)
            windows = torch.stack([gaussian*torch.exp(-1j*math.pi*q*(n/cfg.n_fft)**2) for q in (-4., 4.)])
        else:
            # Two genuinely different STFT windows; not duplicate long spectra.
            half = F.pad(torch.hann_window(cfg.n_fft//2), (cfg.n_fft//4, cfg.n_fft//4))
            windows = torch.stack((half, torch.hann_window(cfg.n_fft))).to(torch.complex64)
        self.register_buffer("windows", windows/windows.norm(dim=-1, keepdim=True))
        self.register_buffer("short_window", F.normalize(torch.hann_window(cfg.short), dim=0)*math.sqrt(cfg.short))

    def forward(self, waveform):
        cfg = self.cfg
        with torch.autocast(waveform.device.type, enabled=False):
            frames = F.pad(waveform.float(), (cfg.n_fft//2, cfg.n_fft//2), mode="reflect")
            frames = frames.unfold(-1, cfg.n_fft, cfg.hop)  # B,C,T,N
            start = (cfg.n_fft-cfg.short)//2
            short = frames[..., start:start+cfg.short]
            if self.arm == "stft":
                fft = torch.fft.rfft(short*self.short_window, norm="ortho")
                # Real rFFT endpoints have identically-zero imaginary parts.
                # Excluding those TWO zero coordinates retains all information
                # and gives exactly N real scalars, matching the waveform arm.
                real = torch.cat((fft.real[..., :1], fft.real[..., 1:-1]*math.sqrt(2), fft.real[..., -1:]), dim=-1)
                short = torch.cat((real, fft.imag[..., 1:-1]*math.sqrt(2)), dim=-1)
            short = short.permute(0, 2, 1, 3).flatten(2)
            spectra = torch.fft.fft(frames.unsqueeze(-2)*self.windows, dim=-1)[..., :cfg.n_fft//2+1]
        temporal = F.gelu(self.short_projection(short))[:, :, None] + self.type_embedding[0]
        bands, offset = [], 0
        for bins, projection in zip(cfg.bands, self.band_projections):
            values = torch.view_as_real(spectra[..., offset:offset+bins]).permute(0, 2, 1, 3, 4, 5).flatten(2)
            bands.append(F.gelu(projection(values)))
            offset += bins
        spectral = torch.stack(bands, dim=2) + self.type_embedding[1]
        return self.norm(torch.cat((temporal, spectral), dim=2))


class Injection(nn.Module):
    def __init__(self, cfg, arm):
        super().__init__()
        self.cfg = cfg
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.query_norm = nn.LayerNorm(cfg.base_dim)
            self.query = nn.Linear(cfg.base_dim, cfg.width)
            self.attention = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True, dropout=0)
            self.output = nn.Linear(cfg.width, cfg.base_dim, bias=False)
            self.gate = nn.Parameter(torch.zeros(()))
            self.memory = Memory(cfg, arm)

    def forward(self, tokens, waveform):
        b, t, f, d = tokens.shape
        if (f, d) != (len(self.cfg.bands), self.cfg.base_dim):
            raise ValueError("parent token geometry mismatch")
        memory = self.memory(waveform)
        if memory.shape[:2] != (b, t):
            raise ValueError("memory/parent clocks differ")
        q = self.query(self.query_norm(tokens)).reshape(b*t, f, self.cfg.width)
        kv = memory.reshape(b*t, memory.shape[2], self.cfg.width)
        delta = self.output(self.attention(q, kv, kv, need_weights=False)[0]).reshape(b, t, f, d)
        return tokens + self.gate*delta


class CheckpointBlock(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(self.inner, x, use_reentrant=False)
        return self.inner(x)


class FrozenSW(nn.Module):
    def __init__(self, backbone, cfg, arm, checkpoint_blocks=True):
        super().__init__()
        if getattr(backbone, "use_torch_checkpoint", False):
            raise ValueError("upstream whole-model checkpoint flag must stay false; band hook cannot be replayed")
        if any(backbone.stft_kwargs[k] != v for k, v in
               {"n_fft": cfg.n_fft, "hop_length": cfg.hop, "win_length": cfg.n_fft}.items()):
            raise ValueError("adapter clock differs from parent STFT")
        if backbone.audio_channels != 2:
            raise ValueError("stereo parent required")
        self.backbone, self.cfg = backbone, cfg
        self.backbone.requires_grad_(False).eval()
        # Checkpoint ONLY downstream blocks, never band_split / its temporary
        # injection hook. Thus backward recomputation cannot omit the adapter.
        if checkpoint_blocks:
            for block in backbone.layers:
                for index, inner in enumerate(block):
                    block[index] = CheckpointBlock(inner)
            for index, inner in enumerate(backbone.mask_estimators):
                backbone.mask_estimators[index] = CheckpointBlock(inner)
        self.adapter = Injection(cfg, arm)
        self._busy = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, waveform, *, gate_off=False):
        if waveform.ndim != 3 or waveform.shape[1] != 2 or waveform.dtype != torch.float32:
            raise ValueError("B,2,L float32 input required")
        if waveform.shape[-1] <= self.cfg.n_fft//2 or not torch.isfinite(waveform).all():
            raise ValueError("nonfinite/too short input")
        if self._busy:
            raise RuntimeError("concurrent forwards unsupported")
        self._busy = True
        handle = None
        try:
            if not gate_off:
                handle = self.backbone.band_split.register_forward_hook(
                    lambda module, inputs, tokens: self.adapter(tokens, waveform))
            return self.backbone(waveform)
        finally:
            if handle:
                handle.remove()
            self._busy = False


def state_fingerprint(module):
    """All persistent tensors, not just gradient flags. Use outside the step loop."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()
