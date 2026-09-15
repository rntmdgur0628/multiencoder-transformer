"""Deterministic MoisesDB song-group splits derived from metadata, with audit."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import re
import unicodedata


def _label(value):
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", value)


def _partition(group, seed, train, validation):
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < train else "validation" if value < train + validation else "test"


def make_moises_splits(root, output_path, *, seed=20260915, train=0.8, validation=0.1, test=0.1):
    """Write the `{track UUID: {group, split}}` contract consumed by prepare-moises.

    Metadata grouping is conservative: normalized `artist + song` defines a
    same-song group. A missing artist or song falls back to a unique UUID
    group and is reported, rather than silently claiming song-level isolation.
    """
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    fractions = (train, validation, test)
    if any(not math.isfinite(value) or value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError("train/validation/test fractions must be positive and sum exactly to 1")
    root, output = Path(root).expanduser().resolve(), Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("split output already exists; choose a new immutable filename")
    candidates = {}
    for metadata in root.rglob("data.json"):
        track_id = metadata.parent.name
        if track_id in candidates:
            raise ValueError(f"duplicate track UUID: {track_id}")
        candidates[track_id] = metadata
    if not candidates:
        raise ValueError("no data.json files found under root")

    result, audit, groups = {}, [], defaultdict(list)
    fallback = []
    for track_id, metadata in sorted(candidates.items()):
        document = json.loads(metadata.read_text())
        artist, song = _label(document.get("artist")), _label(document.get("song"))
        if artist and song:
            group = f"moises:{artist}|{song}"
            grouping = "normalized_artist_and_song"
        else:
            group = f"moises:uuid:{track_id}"
            grouping = "uuid_fallback_missing_artist_or_song"
            fallback.append(track_id)
        partition = _partition(group, seed, train, validation)
        result[track_id] = {"group": group, "split": partition}
        groups[group].append(track_id)
        audit.append({"id": track_id, "metadata_path": str(metadata), "metadata_sha256": _sha256(metadata),
                      "group": group, "split": partition, "grouping": grouping,
                      "artist_present": bool(artist), "song_present": bool(song)})
    # Group is deterministic, so this should be impossible; keep a fail-closed audit.
    if any(len({result[track_id]["split"] for track_id in members}) != 1 for members in groups.values()):
        raise RuntimeError("a canonical group crossed partitions")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(json.dumps({"schema_version": 1, "root": str(root), "seed": seed,
        "fractions": {"train": train, "validation": validation, "test": test},
        "group_rule": "normalized artist + song; UUID fallback is explicitly listed",
        "tracks": len(result), "groups": len(groups), "uuid_fallback_tracks": fallback,
        "partition_tracks": {name: sum(item["split"] == name for item in result.values()) for name in ("train", "validation", "test")},
        "partition_groups": {name: sum(_partition(group, seed, train, validation) == name for group in groups) for name in ("train", "validation", "test")},
        "entries": audit}, indent=2) + "\n")
    return output, audit_path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
