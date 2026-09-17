"""Pinned MSST inference backend for SW-Fixed; separate from the tiny API probe."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import urllib.request

import torch
import yaml

REVISION = "178d6242a51ac56d9b4369c55e14fbbbb2a01ab6"
REPOSITORY = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
SW_SHA256 = "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
FILES = {
    "models/bs_roformer/bs_roformer.py": "446d1e80b40cc3e378c69781450fd501b0c662ee06d5b0cbeb2f919a1040ef1e",
    "models/bs_roformer/attend.py": "0459d799ade55541df2994b0becf7aec12214491360c5a06e346f6d615eaed15",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_backend(cache):
    """Explicit setup only. Evaluation never downloads or executes mutable HEAD."""
    root = Path(cache).expanduser().resolve() / f"msst-{REVISION}"
    for relative, expected in FILES.items():
        path = root / relative
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f"modified backend; not overwriting: {path}")
            continue
        url = f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"upstream hash mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    # Minimal package scaffolding avoids importing unrelated MelBand models.
    # The two upstream implementation files are used byte-for-byte unchanged.
    for relative in ("models/__init__.py", "models/bs_roformer/__init__.py"):
        path = root / relative
        if not path.exists():
            path.write_text("")
        if path.read_bytes() != b"":
            raise ValueError(f"unexpected package initializer: {path}")
    (root / "ORIGIN.json").write_text(json.dumps({"repository": REPOSITORY,
        "revision": REVISION, "files": FILES, "scaffolding": "empty package initializers"}, indent=2))
    return root


class TupleSafeLoader(yaml.SafeLoader):
    pass


TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                               lambda loader, node: tuple(loader.construct_sequence(node)))


def read_config(path):
    config = yaml.load(Path(path).read_text(), Loader=TupleSafeLoader)
    if not isinstance(config, dict):
        raise ValueError("expected a model YAML mapping")
    names = config["training"]["instruments"]
    if len(names) != 6 or set(names) != {"vocals", "bass", "drums", "other", "piano", "guitar"}:
        raise ValueError("this baseline runner requires six named Moises stems")
    if config["model"]["num_stems"] != 6 or not config["model"]["stereo"]:
        raise ValueError("expected six-stem stereo checkpoint")
    if config["audio"]["sample_rate"] != 44100 or config["audio"]["num_channels"] != 2:
        raise ValueError("SW-Fixed expects 44100 Hz stereo")
    if config["inference"].get("normalize", False) or config["training"].get("target_instrument"):
        raise ValueError("normalization/single-target mode is outside this evaluation contract")
    chunk = config["audio"]["chunk_size"]
    overlap = config["inference"]["num_overlap"]
    if not isinstance(chunk, int) or chunk < 10 or not isinstance(overlap, int) or not 2 <= overlap <= chunk:
        raise ValueError("expected positive chunk and overlap >= 2")
    if chunk % config["model"]["stft_hop_length"]:
        raise ValueError("chunk must align to the model STFT hop")
    return config


def load_backend(config, checkpoint, cache, device, precision="fp16"):
    actual = sha256(checkpoint)
    if actual != SW_SHA256:
        raise ValueError(f"not the verified SW-Fixed checkpoint: SHA256={actual}")
    root = Path(cache).expanduser().resolve() / f"msst-{REVISION}"
    for relative, expected in FILES.items():
        if not (root / relative).is_file() or sha256(root / relative) != expected:
            raise ValueError("backend missing/modified; run scripts/setup_bsr_baseline.sh first")
    for relative in ("models/__init__.py", "models/bs_roformer/__init__.py"):
        if (root / relative).read_bytes() != b"":
            raise ValueError("backend package initializer was modified")
    if "models" in sys.modules:
        existing = Path(sys.modules["models"].__file__).resolve()
        if existing != root / "models/__init__.py":
            raise RuntimeError("another 'models' package is already imported; use a fresh process")
    sys.path.insert(0, str(root))
    module = importlib.import_module("models.bs_roformer.bs_roformer")
    model = module.BSRoformer(**config["model"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError("expected a raw tensor state_dict; no guessed key conversions")
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite checkpoint weights")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    # The pinned upstream chooses Flash-only on compute capability >=8.0.
    # Flash does not accept fp32: explicitly allow math/memory-efficient SDPA
    # for that user-selected diagnostic mode, without changing parameters.
    if device.type == "cuda" and precision == "fp32":
        for layer in model.modules():
            if getattr(layer, "cuda_config", None) is not None:
                layer.cuda_config = layer.cuda_config._replace(enable_flash=False, enable_math=True, enable_mem_efficient=True)
    return model, {"checkpoint_sha256": actual, "backend_revision": REVISION,
                   "backend_files": FILES, "state_entries": len(state),
                   "parameters": sum(p.numel() for p in model.parameters()), "strict_load": True,
                   "sdpa_policy": "fp32 math/memory-efficient" if device.type == "cuda" and precision == "fp32" else "pinned upstream"}
