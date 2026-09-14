"""Offline MoisesDB export using explicit song splits; no automatic data download."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MOISES_SOURCES
from .data import sha256_file


def export_moises(root, split_path, output_dir, sample_rate=48000):
    """splits JSON: {track_uuid: {split, group}}; group is a canonical song identity.

    Metadata layout follows moises-ai/moises-db MoisesDBTrack._parse_sources.
    Unknown top-level stemName categories go to other, never disappear.
    """
    from scipy.signal import resample_poly

    root, output = Path(root).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    split = json.loads(Path(split_path).read_text())
    if not isinstance(split, dict) or not split:
        raise ValueError("explicit nonempty track->split/group mapping is required")
    if sample_rate < 1:
        raise ValueError("sample rate must be positive")
    if output == root or root in output.parents:
        raise ValueError("prepared output must be separate from the original dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepared output must be empty")
    candidates = {}
    for metadata in root.rglob("data.json"):
        if metadata.parent.name in split:
            if metadata.parent.name in candidates:
                raise ValueError("duplicate track UUID in dataset")
            candidates[metadata.parent.name] = metadata
    if set(candidates) != set(split):
        raise ValueError("some requested track UUIDs have no unique data.json")
    groups = {}
    for info in split.values():
        group, partition = info["group"], info["split"]
        if not group or partition not in ("train", "validation", "test"):
            raise ValueError("each split entry requires a group and train/validation/test role")
        if group in groups and groups[group] != partition:
            raise ValueError("song group crosses split boundaries")
        groups[group] = partition
    output.mkdir(parents=True, exist_ok=True)
    tracks, audit = [], []
    for track_id, metadata in sorted(candidates.items()):
        doc = json.loads(metadata.read_text())
        if not isinstance(doc.get("stems"), list) or not doc["stems"]:
            raise ValueError(f"missing complete stem metadata: {metadata}")
        gathered = {source: [] for source in MOISES_SOURCES}
        details, seen = [], set()
        for stem in doc["stems"]:
            name = stem["stemName"]
            if not isinstance(name, str) or not name or not stem.get("tracks"):
                raise ValueError("stem must have a name and nonempty recording list")
            destination = name if name in MOISES_SOURCES else "other"
            for recording in stem["tracks"]:
                path = (metadata.parent / name / f"{recording['id']}.{recording['extension']}").resolve()
                if metadata.parent not in path.parents or path in seen:
                    raise ValueError("invalid/duplicate metadata audio path")
                seen.add(path)
                audio, sr = sf.read(path, dtype="float32", always_2d=True)
                if len(audio) < 1 or audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
                    raise ValueError(f"unsupported source audio: {path}")
                if audio.shape[1] == 1:
                    audio = np.repeat(audio, 2, axis=1)
                if sr != sample_rate:
                    factor = math.gcd(sr, sample_rate)
                    audio = resample_poly(audio, sample_rate // factor, sr // factor, axis=0).astype(np.float32)
                gathered[destination].append(audio)
                details.append({"path": str(path), "sha256": sha256_file(path), "category": name,
                                "mapped_to": destination, "resampled_length": len(audio),
                                "has_bleed": recording.get("has_bleed")})
        length = max(len(x) for values in gathered.values() for x in values)
        directory = output / track_id
        directory.mkdir()
        targets = {}
        for source, arrays in gathered.items():
            targets[source] = []
            if not arrays:
                continue
            mixed = np.zeros((length, 2), dtype=np.float32)
            for audio in arrays:
                mixed[:len(audio)] += audio
            path = directory / f"{source}.wav"
            sf.write(path, mixed, sample_rate, subtype="FLOAT")
            targets[source] = [str(path.relative_to(output))]
        tracks.append({"id": track_id, **split[track_id], "length": length, "targets": targets,
                       "mixture": None, "provenance": "moisesdb_metadata_export"})
        audit.append({"id": track_id, "metadata_sha256": sha256_file(metadata), "sources": details})
        print(f"prepared {track_id}", flush=True)
    manifest = {"schema_version": 1, "task": "moises6", "sample_rate": sample_rate,
                "sources": list(MOISES_SOURCES), "tracks": tracks}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "preparation.json").write_text(json.dumps({"split_sha256": sha256_file(split_path),
        "resampling": "scipy.signal.resample_poly", "mono_policy": "duplicate to stereo",
        "length_policy": "zero-pad shorter recordings; no trimming", "tracks": audit}, indent=2))
    return output / "manifest.json"
