"""Serializable experiment contracts. Defaults are engineering choices, not results."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

MOISES_SOURCES = ("vocals", "bass", "drums", "other", "piano", "guitar")
MSR_SOURCES = ("vocals", "guitars", "keyboards", "bass", "synthesizers", "drums", "percussion", "orchestral")
FAMILIES = ("temporal", "gabor", "chirplet")


@dataclass(frozen=True)
class ModelConfig:
    sample_rate: int = 48000
    n_fft: int = 1024
    hop: int = 256
    bands: int = 16
    temporal_kernel: int = 512
    d_model: int = 128
    heads: int = 4
    depth: int = 4
    context_frames: int = 32
    injection_frames: int = 8
    liere_block: int = 4
    gate_init: float = 0.001
    angular_span: float = math.pi
    time_scale: float = 1.0
    gaussian_sigma_fraction: float = 1 / 6
    chirp_rates: tuple = (-4.0, 4.0)
    sources: tuple = MOISES_SOURCES
    seed: int = 20260914

    def __post_init__(self):
        positive = ("sample_rate", "n_fft", "hop", "bands", "temporal_kernel", "d_model", "heads", "depth", "context_frames", "injection_frames", "liere_block")
        if any(not isinstance(getattr(self, k), int) or getattr(self, k) < 1 for k in positive):
            raise ValueError("integer dimensions must be positive")
        if self.n_fft % 2 or self.hop > self.n_fft or self.temporal_kernel < self.hop:
            raise ValueError("need even FFT, hop <= FFT, temporal kernel >= hop")
        if self.bands > self.n_fft // 2 + 1:
            raise ValueError("every band must contain at least one bin")
        if self.d_model % self.heads or (self.d_model // self.heads) % self.liere_block:
            raise ValueError("head width must divide into LieRE blocks")
        if self.liere_block < 2:
            raise ValueError("rotation blocks require dimension >= 2")
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError("source labels must be nonempty and unique")
        if any(not isinstance(s, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", s) for s in self.sources):
            raise ValueError("source labels must be safe plain identifiers")
        if not self.chirp_rates or any(not math.isfinite(q) or q == 0 for q in self.chirp_rates):
            raise ValueError("chirplet rates must be finite and nonzero")
        if any(not math.isfinite(v) or v <= 0 for v in (self.time_scale, self.angular_span, self.gaussian_sigma_fraction)):
            raise ValueError("coordinate/window scales must be finite and positive")
        if not math.isfinite(self.gate_init):
            raise ValueError("gate must be finite")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for name in ("sources", "chirp_rates"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)

    @classmethod
    def tiny(cls):
        return cls(sample_rate=8000, n_fft=64, hop=16, bands=4, temporal_kernel=32,
                   d_model=16, heads=2, depth=1, context_frames=3, injection_frames=2)
