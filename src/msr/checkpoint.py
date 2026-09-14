"""Versioned contracts, strict loading, and explicit 6 -> 8 head transfer."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch

from .config import ModelConfig, MOISES_SOURCES, MSR_SOURCES
from .model import RestorationModel

SCHEMA = 1


def contract_hash(cfg):
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()


def save_checkpoint(path, model, *, stage, step, optimizer=None, run_contract=None, best_validation_loss=None):
    payload = {"schema_version": SCHEMA, "config": model.cfg.to_dict(), "config_sha256": contract_hash(model.cfg),
               "variant": model.variant, "stage": stage, "step": step, "model": model.state_dict(),
               "optimizer": optimizer.state_dict() if optimizer else None,
               "run_contract": run_contract or {}, "cpu_rng": torch.get_rng_state()}
    if best_validation_loss is not None:
        payload["best_validation_loss"] = best_validation_loss
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported checkpoint lineage/schema (historical MSS checkpoints cannot be loaded)")
    cfg = ModelConfig.from_dict(payload["config"])
    if contract_hash(cfg) != payload["config_sha256"]:
        raise ValueError("checkpoint configuration hash mismatch")
    model = RestorationModel(cfg, payload["variant"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def make_adapter(parent, variant):
    if parent.variant != "A0" or parent.cfg.sources != MOISES_SOURCES:
        raise ValueError("first adapter screen must start from the same six-stem A0")
    if variant not in ("A1", "A2"):
        raise ValueError("A0 is an evaluation-only frozen anchor during adapter screen")
    model = RestorationModel(parent.cfg, variant)
    model.load_state_dict(parent.state_dict(), strict=True)
    model.configure_training(True)
    return model


def transfer_to_msr(parent, variant):
    """Warm-start exactly corresponding heads; piano -> keyboards is NOT presumed."""
    if parent.cfg.sources != MOISES_SOURCES or parent.variant != "A0":
        raise ValueError("controlled MSR transfer starts from the common six-stem A0 checkpoint")
    cfg = replace(parent.cfg, sources=MSR_SOURCES)
    model = RestorationModel(cfg, variant)
    state = model.state_dict()
    for name, value in parent.state_dict().items():
        if not name.startswith("heads."):
            state[name] = value.clone()
    mapping = {"vocals": "vocals", "bass": "bass", "drums": "drums", "guitars": "guitar"}
    for i, (old, new) in enumerate(zip(parent.heads, model.heads)):
        for field in ("weight", "bias"):
            original = getattr(old, field).detach().reshape(len(parent.cfg.sources), -1)
            transferred = getattr(new, field).detach().clone().reshape(len(cfg.sources), -1)
            for destination, source in mapping.items():
                transferred[cfg.sources.index(destination)] = original[parent.cfg.sources.index(source)]
            state[f"heads.{i}.{field}"] = transferred.reshape_as(getattr(new, field))
    model.load_state_dict(state, strict=True)
    model.configure_training(False)
    return model, mapping
