"""One common separator/head. No expert masks, waveform sum, or Q3 pair tokens."""
from __future__ import annotations

from dataclasses import replace
import torch
from torch import nn

from .audio import ReferenceSTFT
from .attention import ResidualInjection, TransformerBlock
from .encoders import SpectralEncoder, TemporalEncoder, band_edges
from .position import baseline_position


class RestorationModel(nn.Module):
    def __init__(self, cfg, variant="A0"):
        super().__init__()
        if variant not in ("A0", "A1", "A2"):
            raise ValueError("variant must be A0, A1, or A2")
        self.cfg, self.variant, self.adapter_only = cfg, variant, False
        # All arms allocate the same modules in the same order. PE is disabled
        # and frozen in A1; the whole injection is disabled/frozen in A0.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.reference = ReferenceSTFT(cfg.n_fft, cfg.hop)
            self.encoders = nn.ModuleList([TemporalEncoder(cfg), SpectralEncoder(cfg), SpectralEncoder(cfg, True)])
            self.family_embedding = nn.Parameter(torch.randn(3, cfg.d_model) * 0.02)
            self.injection = ResidualInjection(cfg)
            self.trunk = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.depth))
            edges = band_edges(cfg.n_fft, cfg.bands)
            self.heads = nn.ModuleList(nn.Linear(cfg.d_model, len(cfg.sources) * 4 * (b - a))
                                       for a, b in zip(edges[:-1], edges[1:]))
        self.configure_training(False)

    @property
    def latency_samples(self):
        return self.cfg.n_fft - 1

    @property
    def preroll_samples(self):
        c = self.cfg
        return max(c.n_fft, c.temporal_kernel) + c.n_fft + (c.depth * (c.context_frames - 1) + c.injection_frames - 1) * c.hop

    def configure_training(self, adapter_only):
        self.adapter_only = bool(adapter_only)
        self.requires_grad_(not adapter_only)
        self.injection.requires_grad_(self.variant != "A0")
        self.injection.position.requires_grad_(self.variant == "A2")
        if self.variant == "A0":
            self.injection.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.adapter_only:
            self.encoders.eval()
            self.trunk.eval()
            self.heads.eval()
        return self

    def encode(self, waveform):
        count = (waveform.shape[-1] + self.cfg.n_fft - self.cfg.hop + self.cfg.hop - 1) // self.cfg.hop
        return [encoder(waveform, count) for encoder in self.encoders]

    def forward(self, waveform, gate_off=False, return_aux=False, intervention=None, memory_waveform=None):
        if waveform.dtype != torch.float32:
            raise ValueError("waveform input must be float32; precision contract is FP32")
        original = self.encode(waveform)
        branches, memories = original, None
        if intervention not in (None, "frequency_shuffle", "kv_joint_permutation", "wrong_track"):
            raise ValueError("unknown intervention")
        if intervention and (gate_off or self.variant == "A0"):
            raise ValueError("injection interventions require an active injection")
        if intervention == "frequency_shuffle":
            if self.variant != "A2":
                raise ValueError("frequency-coordinate intervention requires A2")
            branches = []
            for branch in original:
                p = branch.coordinates.clone()
                # Cyclic permutation WITHIN each frame; time/content/masks stay fixed.
                p[..., 1:] = p[..., 1:].roll(1, dims=1)
                branches.append(replace(branch, coordinates=p))
        elif intervention == "kv_joint_permutation":
            memories = [replace(b, values=b.values.flip(2), coordinates=b.coordinates.flip(1), valid=b.valid.flip(2)) for b in original]
        elif intervention == "wrong_track":
            if memory_waveform is None or memory_waveform.shape != waveform.shape:
                raise ValueError("wrong-track intervention needs explicit shape-matched donor waveform")
            memories = self.encode(memory_waveform)
        updated, aux = self.injection(branches, self.variant == "A2", gate_off or self.variant == "A0", memories)
        # Interventions affect ONLY the added block, never the common trunk PE.
        branches = [b.with_values(z.values) for b, z in zip(original, updated)]
        sequences = [b.values + self.family_embedding[i] + baseline_position(b.coordinates, self.cfg.d_model)
                     for i, b in enumerate(branches)]
        x = torch.cat(sequences, dim=2)
        for layer in self.trunk:
            x = layer(x)
        # Gabor-grid slots after JOINT separation are output locations, not an
        # independent Gabor separator. All three families can influence them.
        grid = x[:, :, 1:1 + self.cfg.bands]
        masks = []
        for i, head in enumerate(self.heads):
            z = head(grid[:, :, i]).reshape(x.shape[0], x.shape[1], len(self.cfg.sources), 2, -1, 2)
            masks.append(torch.view_as_complex(z.contiguous()))
        mask = torch.cat(masks, dim=-1).permute(0, 2, 3, 1, 4)
        spec = self.reference.analysis(waveform)
        output = self.reference.synthesis(mask * spec[:, None], waveform.shape[-1])
        if return_aux:
            return output, aux
        return output
