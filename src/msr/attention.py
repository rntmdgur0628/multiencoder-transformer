"""Local causal cross-branch attention; updates are computed simultaneously."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .position import LieRE
from .config import FAMILIES


def local_attention(q, k, v, valid_q, valid_k, context):
    """[B,T,K,H,Dh] -> same. Past+current frames only; bounded memory, no T² grid."""
    batch, time, queries, heads, dim = q.shape
    memories = k.shape[2]
    context = min(context, time)
    index = torch.arange(time, device=q.device)[:, None] + torch.arange(1 - context, 1, device=q.device)[None]
    in_range = index >= 0
    index = index.clamp_min(0)
    keys = k[:, index].reshape(batch, time, context * memories, heads, dim)
    vals = v[:, index].reshape_as(keys)
    allowed = (valid_k[:, index] & in_range[None, :, :, None]).flatten(2)
    query = q.permute(0, 1, 3, 2, 4)
    keys = keys.permute(0, 1, 3, 2, 4)
    vals = vals.permute(0, 1, 3, 2, 4)
    mask = allowed[:, :, None, None, :] & valid_q[:, :, None, :, None]
    # Fused CUDA SDPA expects four-dimensional Q/K/V, not separate batch/time
    # batch axes. Flatten these axes without changing frame causality.
    result = F.scaled_dot_product_attention(
        query.reshape(batch * time, heads, queries, dim),
        keys.reshape(batch * time, heads, context * memories, dim),
        vals.reshape(batch * time, heads, context * memories, dim),
        attn_mask=mask.reshape(batch * time, 1, queries, context * memories), dropout_p=0.0,
    ).reshape(batch, time, heads, queries, dim)
    return result.permute(0, 1, 3, 2, 4)


class ResidualInjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.d_model) for _ in range(3))
        self.qkv = nn.ModuleList(nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False) for _ in range(3))
        self.output = nn.ModuleList(nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(3))
        self.gates = nn.Parameter(torch.full((3,), cfg.gate_init))
        self.position = LieRE(cfg.heads, cfg.d_model // cfg.heads, cfg.liere_block)

    def forward(self, branches, use_pe, gate_off=False, memories=None):
        if len(branches) != 3:
            raise ValueError("exactly three label families are required")
        if gate_off:
            return branches, {"gates": self.gates.detach() * 0, "residual_rms_ratio": self.gates.detach() * 0}
        memories = branches if memories is None else memories
        if len(memories) != 3:
            raise ValueError("exactly three memory families are required")
        if tuple(b.family for b in branches) != FAMILIES or tuple(b.family for b in memories) != FAMILIES:
            raise ValueError("family labels must be temporal/gabor/chirplet, not positional group elements")
        if any(not torch.equal(b.release, branches[0].release) for b in [*branches, *memories]):
            raise ValueError("this implementation requires a common release clock")
        clock = branches[0].release
        if clock.ndim != 1 or clock.numel() < 1 or not torch.all(clock[1:] - clock[:-1] == self.cfg.hop):
            raise ValueError("release samples must increase by exactly one configured hop")
        for branch in [*branches, *memories]:
            if branch.values.ndim != 4 or branch.values.shape[1] != clock.numel() or branch.values.shape[-1] != self.cfg.d_model:
                raise ValueError("branch values must follow [B,T,K,D] on the release clock")
            if branch.valid.shape != branch.values.shape[:-1] or branch.valid.dtype != torch.bool:
                raise ValueError("branch validity must be boolean [B,T,K]")
            if branch.coordinates.shape != (*branch.values.shape[1:3], 3):
                raise ValueError("coordinates must be [T,K,3]")
        q, keys, vals = [], [], []
        for i, (branch, memory) in enumerate(zip(branches, memories)):
            b, t, k, _ = branch.values.shape
            h, d = self.cfg.heads, self.cfg.d_model // self.cfg.heads
            projected = self.qkv[i](self.norms[i](branch.values)).reshape(b, t, k, 3, h, d)
            query = projected[:, :, :, 0]
            kv = projected if memory is branch else self.qkv[i](self.norms[i](memory.values)).reshape(b, t, memory.values.shape[2], 3, h, d)
            key, value = kv[:, :, :, 1], kv[:, :, :, 2]
            if use_pe:
                rotations = self.position.rotations(branch.coordinates)
                query = self.position.apply_rotations(query, rotations)
                key_rotations = rotations if memory is branch else self.position.rotations(memory.coordinates)
                key = self.position.apply_rotations(key, key_rotations)
            q.append(query)
            keys.append(key)
            vals.append(value)
        updated, ratios = [], []
        for i, branch in enumerate(branches):
            others = [j for j in range(3) if j != i]
            message = local_attention(q[i], torch.cat([keys[j] for j in others], 2),
                                      torch.cat([vals[j] for j in others], 2), branch.valid,
                                      torch.cat([memories[j].valid for j in others], 2), self.cfg.injection_frames)
            delta = self.gates[i] * self.output[i](message.flatten(-2))
            updated.append(branch.with_values(branch.values + delta))
            ratios.append((delta.detach().square().mean().sqrt() /
                           branch.values.detach().square().mean().sqrt().clamp_min(1e-8)))
        return updated, {"gates": self.gates.detach().clone(), "residual_rms_ratio": torch.stack(ratios)}


class TransformerBlock(nn.Module):
    """Joint family/band attention per frame, then causal time attention per slot."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spatial_norm = nn.LayerNorm(cfg.d_model)
        self.spatial = nn.MultiheadAttention(cfg.d_model, cfg.heads, dropout=0, batch_first=True)
        self.time_norm = nn.LayerNorm(cfg.d_model)
        self.time_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.time_output = nn.Linear(cfg.d_model, cfg.d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model * 4),
                                 nn.GELU(), nn.Linear(cfg.d_model * 4, cfg.d_model))

    def forward(self, x):
        b, t, slots, d = x.shape
        norm = self.spatial_norm(x).reshape(b * t, slots, d)
        mixed = self.spatial(norm, norm, norm, need_weights=False)[0].reshape_as(x)
        x = x + mixed
        z = self.time_norm(x).permute(0, 2, 1, 3).reshape(b * slots, t, 1, d)
        qkv = self.time_qkv(z).reshape(b * slots, t, 1, 3, self.cfg.heads, d // self.cfg.heads)
        valid = torch.ones(z.shape[:-1], dtype=torch.bool, device=x.device)
        out = local_attention(qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2], valid, valid, self.cfg.context_frames)
        out = out.flatten(-2).reshape(b, slots, t, d).permute(0, 2, 1, 3)
        x = x + self.time_output(out)
        return x + self.ffn(x)
