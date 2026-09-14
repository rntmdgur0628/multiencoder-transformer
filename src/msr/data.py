"""Explicit paired-audio manifests: missing labels are never silently zero-filled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch

from .config import MOISES_SOURCES, MSR_SOURCES


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AudioManifest:
    """Relative audio paths resolve against the manifest, not the shell cwd.

    Every track needs stable id + underlying-song group + explicit split.
    [] is an explicitly known absent target; omission is an error.
    """
    def __init__(self, path, audit_audio=True):
        self.path = Path(path).expanduser().resolve()
        self.document = json.loads(self.path.read_text())
        doc = self.document
        if doc.get("schema_version") != 1 or doc.get("task") not in ("moises6", "msr8"):
            raise ValueError("manifest needs schema_version=1 and task=moises6|msr8")
        self.task = doc["task"]
        self.sample_rate = doc["sample_rate"]
        self.sources = tuple(doc["sources"])
        expected = MOISES_SOURCES if self.task == "moises6" else MSR_SOURCES
        if self.sources != expected:
            raise ValueError("manifest source order differs from canonical task taxonomy")
        self.tracks = doc["tracks"]
        if not self.tracks:
            raise ValueError("empty track manifest")
        ids, groups, paths = set(), {}, {}
        audio_hashes = {}
        for track in self.tracks:
            name, group, split = track["id"], track["group"], track["split"]
            if not isinstance(name, str) or not name or not isinstance(group, str) or not group:
                raise ValueError("id/group must be stable nonempty strings")
            if name in ids or split not in ("train", "validation", "test"):
                raise ValueError("duplicate track id or invalid split")
            ids.add(name)
            if group in groups and groups[group] != split:
                raise ValueError("underlying song group leaks across splits")
            groups[group] = split
            if track.get("provenance") in ("official_validation", "official_test") and split == "train":
                raise ValueError("official evaluation audio cannot be silently used for training")
            if set(track["targets"]) != set(self.sources):
                raise ValueError("all target labels must be explicit; use [] only for known absence")
            if not isinstance(track.get("length"), int) or track["length"] < 1:
                raise ValueError("track length must be a positive sample count")
            files = []
            for source in self.sources:
                values = track["targets"][source]
                if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                    raise ValueError("each target must be an explicit list of audio paths")
                files.extend(values)
            if not files:
                raise ValueError("at least one target recording is required")
            if len(files) != len(set(files)):
                raise ValueError("an audio recording cannot occur in multiple target slots")
            mixture = track.get("mixture")
            if self.task == "msr8" and not mixture:
                raise ValueError("MSR requires an explicit degraded input, never sum(raw targets)")
            if mixture:
                files.append(mixture)
            for value in files:
                resolved = self.resolve(value)
                if resolved in paths and paths[resolved] != split:
                    raise ValueError("audio path leaks across splits")
                paths[resolved] = split
                if audit_audio:
                    info = sf.info(resolved)
                    if info.samplerate != self.sample_rate or info.channels != 2 or info.frames != track["length"]:
                        raise ValueError(f"audio must be aligned stereo at manifest rate/length: {resolved}")
                    if resolved not in audio_hashes:
                        audio_hashes[resolved] = sha256_file(resolved)
        self.groups = groups
        self.sha256 = sha256_file(self.path)
        self.audio_sha256 = hashlib.sha256(json.dumps(audio_hashes, sort_keys=True).encode()).hexdigest() if audit_audio else None

    def resolve(self, value):
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else self.path.parent / path).resolve())

    def partition(self, split):
        tracks = [t for t in self.tracks if t["split"] == split]
        if not tracks:
            raise ValueError(f"manifest has no {split} tracks")
        return tracks

    def read(self, path, offset, count):
        output = np.zeros((count, 2), dtype=np.float32)
        with sf.SoundFile(self.resolve(path)) as audio:
            start = max(0, offset)
            stop = min(len(audio), offset + count)
            if stop > start:
                audio.seek(start)
                output[start - offset:stop - offset] = audio.read(stop - start, dtype="float32", always_2d=True)
        if not np.isfinite(output).all():
            raise ValueError(f"nonfinite audio: {path}")
        return torch.from_numpy(output.T.copy())

    def segment(self, track, start, core_samples, preroll, lookahead):
        count, offset = preroll + core_samples + lookahead, start - preroll
        targets = []
        for source in self.sources:
            value = torch.zeros(2, count)
            for path in track["targets"][source]:
                value += self.read(path, offset, count)
            targets.append(value)
        target = torch.stack(targets)
        mixture = self.read(track["mixture"], offset, count) if track.get("mixture") else target.sum(0)
        valid = min(core_samples, track["length"] - start)
        return mixture, target[..., preroll:preroll + core_samples], valid


def training_batch(manifest, step, batch_size, seed, core_samples, preroll, lookahead, hop):
    # Local RNG does not depend on model construction, variant, or global RNG.
    rng = random.Random(f"{seed}:{step}")
    tracks = manifest.partition("train")
    inputs, targets, lengths, records = [], [], [], []
    for _ in range(batch_size):
        track = tracks[rng.randrange(len(tracks))]
        maximum = max(0, track["length"] - core_samples) // hop
        start = rng.randrange(maximum + 1) * hop
        x, y, length = manifest.segment(track, start, core_samples, preroll, lookahead)
        inputs.append(x)
        targets.append(y)
        lengths.append(length)
        records.append({"id": track["id"], "start": start, "valid_samples": length})
    return torch.stack(inputs), torch.stack(targets), lengths, records
