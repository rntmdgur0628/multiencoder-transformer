"""Paired held-out effects with the underlying song as bootstrap unit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compare_results(baseline_path, candidate_path, metric="sisdr_db", seed=20260914, repeats=2000):
    baseline = json.loads(Path(baseline_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    for key in ("manifest_sha256", "audio_sha256", "core_samples", "split", "metric_scope"):
        if not baseline.get(key) or baseline[key] != candidate.get(key):
            raise ValueError(f"paired evaluation contract mismatch: {key}")
    a, b = {t["id"]: t for t in baseline["tracks"]}, {t["id"]: t for t in candidate["tracks"]}
    if not a or set(a) != set(b):
        raise ValueError("paired comparisons require identical nonempty track sets")
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    grouped = {}
    per_track = []
    for name in sorted(a):
        if a[name]["group"] != b[name]["group"] or set(a[name]["sources"]) != set(b[name]["sources"]):
            raise ValueError("song identity/source taxonomy differs")
        differences = {}
        for source in a[name]["sources"]:
            x, y = a[name]["sources"][source][metric], b[name]["sources"][source][metric]
            if x is not None and y is not None:
                differences[source] = y - x
                grouped.setdefault(source, {}).setdefault(a[name]["group"], []).append(y - x)
        per_track.append({"id": name, "group": a[name]["group"], "candidate_minus_baseline": differences})
        if differences:
            grouped.setdefault("active_source_mean", {}).setdefault(a[name]["group"], []).append(float(np.mean(list(differences.values()))))
    summaries = {}
    rng = np.random.default_rng(seed)
    for source, songs in grouped.items():
        values = np.array([np.mean(v) for _, v in sorted(songs.items())])
        bootstrap = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
        summaries[source] = {"song_macro_effect": float(values.mean()), "songs": len(values),
                             "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist() if len(values) > 1 else None}
    return {"metric": metric, "direction": "candidate minus baseline; higher is better for dB metrics",
            "bootstrap_unit": "canonical underlying song; repeated views first averaged within song", "seed": seed,
            "repeats": repeats, "effects": summaries, "per_track": per_track,
            "claim_limit": "paired performance difference; not proof of LieRE-specific mechanism or human attention"}
