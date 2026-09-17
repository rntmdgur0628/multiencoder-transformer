"""Matched SW TC/STFT pilot: one command, two arms, fixed final-step evaluation.

No baseline retraining/re-evaluation, no validation-best selection, no auto OOM
fallback. Existing B0 contract/data hashes must match before comparing scores.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.nn import functional as F

from msr.data import AudioManifest
from .baseline import aggregate, audit_tracks, load_track, log_event, score, separate, write_json
from .sw_adapter import AdapterConfig, FrozenSW, state_fingerprint
from .sw_backend import load_backend, read_config, sha256


def make_schedule(tracks, updates, seed, rate, input_rate, samples, hop):
    """Shuffled song cycles: all 194 train songs before a second visit."""
    if updates < 1 or samples < 1:
        raise ValueError("positive update/sample count required")
    tracks = sorted(tracks, key=lambda x: x["id"])
    if not tracks or any(t["split"] != "train" for t in tracks):
        raise ValueError("training schedule requires train-only tracks")
    schedule, cycle = [], 0
    while len(schedule) < updates:
        ordered = list(tracks)
        random.Random(f"{seed}:songs:{cycle}").shuffle(ordered)
        for track in ordered:
            step = len(schedule) + 1
            length = math.ceil(track["length"]*rate/input_rate)
            if length < samples:
                raise ValueError(f"train song shorter than crop, no silent omission: {track['id']}")
            start = random.Random(f"{seed}:crop:{step}").randrange((length-samples)//hop+1)*hop
            schedule.append({"step": step, "id": track["id"], "start": start, "samples": samples})
            if len(schedule) == updates:
                break
        cycle += 1
    return schedule


def read_crop(manifest, track, start, count, rate=44100, hashes=None):
    """Bounded IO with resampler halo and globally aligned polyphase origin.

    At 48k->44.1k a read offset must be divisible by160, giving an integral
    147-sample output offset. 32 polyphase periods of context exceed the
    scipy default FIR support; crop tests compare against full-file resampling.
    """
    if track["split"] != "train" or start < 0 or count < 1:
        raise ValueError("invalid train crop")
    factor = math.gcd(rate, manifest.sample_rate)
    up, down = rate//factor, manifest.sample_rate//factor
    margin = 32*max(up, down)
    left = max(0, (math.floor(start*down/up)-margin)//down*down)
    right = min(track["length"], math.ceil(((start+count)*down/up+margin)/down)*down)
    offset = start-left*up//down
    target = np.zeros((len(manifest.sources), 2, count), dtype=np.float32)
    for index, source in enumerate(manifest.sources):
        for value in track["targets"][source]:
            path = manifest.resolve(value)
            if hashes is not None and path not in hashes:
                hashes[path] = sha256(path)
            with sf.SoundFile(path) as audio:
                if (audio.samplerate, audio.channels, len(audio)) != (manifest.sample_rate, 2, track["length"]):
                    raise ValueError(f"audio header mismatch: {path}")
                audio.seek(left)
                segment = audio.read(right-left, dtype="float32", always_2d=True)
            if up != down:
                segment = resample_poly(segment, up, down, axis=0).astype(np.float32)
            segment = segment[offset:offset+count]
            if segment.shape != (count, 2) or not np.isfinite(segment).all():
                raise ValueError("invalid crop/resampler context")
            target[index] += segment.T
    return torch.from_numpy(target.sum(0).copy())[None], torch.from_numpy(target)[None]


def sw_loss(prediction, target, config):
    """Pinned parent's waveform + complex multi-STFT L1, over ALL six stems.

    Explicit float32 spectral loss even during fp16 model autocast. No
    normalization, active-stem masking, added loss, or tuned stem weighting.
    """
    with torch.autocast(prediction.device.type, enabled=False):
        p, y = prediction.float(), target.float()
        waveform = F.l1_loss(p, y)
        spectral = torch.zeros((), device=p.device)
        cfg = config["model"]
        for size in cfg["multi_stft_resolutions_window_sizes"]:
            kw = dict(n_fft=max(size, cfg["stft_n_fft"]), win_length=size,
                      hop_length=cfg["multi_stft_hop_size"], normalized=cfg["multi_stft_normalized"],
                      window=torch.hann_window(size, device=p.device), return_complex=True)
            spectral = spectral + F.l1_loss(torch.stft(p.flatten(0, 2), **kw), torch.stft(y.flatten(0, 2), **kw))
        total = waveform + cfg["multi_stft_resolution_loss_weight"]*spectral
    per_source = (p.detach()-y).abs().mean(dim=(0, 2, 3)).cpu().tolist()
    return total, {"waveform_l1": float(waveform.detach()), "complex_multistft_l1": float(spectral.detach()),
                   "waveform_l1_by_source": dict(zip(config["training"]["instruments"], per_source))}


def amp_context(device, precision):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and precision == "fp16" else nullcontext()


def gradient_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def check_baseline(directory, manifest, model_config, checkpoint, config_path, precision):
    directory = Path(directory)
    contract = json.loads((directory/"contract.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    rows = [json.loads(line) for line in (directory/"tracks.jsonl").read_text().splitlines()]
    selected = sorted(t["id"] for t in manifest.partition("validation"))
    if not summary["complete"] or summary["kind"] != "full_track" or summary["tracks"] != 28:
        raise ValueError("requires the completed 28-full-track B0, not a profile")
    expected = {"manifest_sha256": manifest.sha256, "config_sha256": sha256(config_path),
                "checkpoint_sha256": sha256(checkpoint), "precision": precision,
                "chunk_size": model_config["audio"]["chunk_size"], "overlap": model_config["inference"]["num_overlap"],
                "selected_ids": selected, "source_order_metrics": list(manifest.sources), "max_seconds": 0}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"B0 comparison contract mismatch: {key}")
    if [r["id"] for r in rows] != selected or aggregate(rows, manifest.sources) != summary["sources"]:
        raise ValueError("B0 rows/aggregation mismatch")
    tracks = {t["id"]: t for t in manifest.partition("validation")}
    for row in rows:
        expected_length = math.ceil(tracks[row["id"]]["length"]*44100/manifest.sample_rate)
        if row["start_sample_44100"] != 0 or row["samples_44100"] != expected_length:
            raise ValueError("B0 row is not a full-track evaluation")
    for key, parent in (("code_sha256", Path(__file__).parent), ("shared_code_sha256", Path(__file__).parents[1])):
        # Verify every file recorded by B0, allow NEW adapter files.
        for name, expected_hash in contract[key].items():
            if sha256(parent/name) != expected_hash:
                raise ValueError(f"B0 evaluator code changed: {name}")
    for name, expected_version in contract["versions"].items():
        if importlib.metadata.version(name) != expected_version:
            raise ValueError(f"B0 dependency version changed: {name}")
    if torch.__version__ != contract["torch"]:
        raise ValueError("B0 PyTorch version changed")
    return contract, rows


def save_checkpoint(path, model, optimizer, scaler, step, metadata):
    temporary = path.with_suffix(".tmp")
    torch.save({"schema_version": 1, "adapter": model.adapter.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "metadata": metadata}, temporary)
    temporary.replace(path)


def train_arm(model, config, manifest, schedule, settings, directory, device, metadata):
    directory.mkdir()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=settings["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings["precision"] == "fp16", init_scale=256)
    fingerprint = state_fingerprint(model.backbone)
    train_tracks = {t["id"]: t for t in manifest.partition("train")}
    order = [manifest.sources.index(s) for s in config["training"]["instruments"]]
    hashes, seen_memory_gradient = {}, False
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for item in schedule:
        step_start = time.perf_counter()
        x, y = read_crop(manifest, train_tracks[item["id"]], item["start"], item["samples"], hashes=hashes)
        x, y = x.to(device), y[:, order].to(device)
        model.train()
        if item["step"] == 1:
            # Use the real first crop, no extra training data or optimizer step.
            with torch.no_grad(), amp_context(device, settings["precision"]):
                parent = model(x, gate_off=True)
                initial = model(x)
            error = float((parent-initial).abs().max())
            if error != 0.0:
                raise RuntimeError(f"zero-gate does not exactly recover SW: {error}")
            write_json(directory/"startup_checks.json", {"zero_gate_max_abs_error": error,
                "parent_fingerprint": fingerprint, "first_crop": item, "parent_eval": not model.backbone.training})
            del parent, initial
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, settings["precision"]):
            prediction = model(x)
        loss, loss_parts = sw_loss(prediction, y, config)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = gradient_norm(model.adapter.parameters())
        memory_norm = gradient_norm(model.adapter.memory.parameters())
        if not math.isfinite(norm) or not math.isfinite(memory_norm):
            raise RuntimeError("nonfinite gradients; stop instead of silently skipping one arm's update")
        if any(p.grad is not None or p.requires_grad for p in model.backbone.parameters()) or model.backbone.training:
            raise RuntimeError("parent is not frozen/eval")
        seen_memory_gradient |= memory_norm > 0
        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), settings["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        if any(not torch.isfinite(p).all() for p in model.adapter.parameters()):
            raise RuntimeError("nonfinite adapter parameters")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"event": "train", **item, "loss": float(loss.detach()), **loss_parts,
               "gradient_norm": norm, "memory_gradient_norm": memory_norm,
               "gate": float(model.adapter.gate.detach()), "step_seconds": time.perf_counter()-step_start,
               "elapsed_seconds": time.perf_counter()-started,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        log_event(directory, row)
        if item["step"] == 3:
            log_event(directory, {"event": "startup_training_profile", "memory_gradient_seen": seen_memory_gradient,
                "estimated_arm_training_seconds": (time.perf_counter()-started)/3*len(schedule),
                "note": "first three steps include zero-gate checks and initial IO; not evaluation time",
                "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"]})
        if item["step"] == 3 and not seen_memory_gradient:
            raise RuntimeError("no memory gradient after three updates")
        if item["step"] % 50 == 0 or item["step"] == len(schedule):
            save_checkpoint(directory/"last.pt", model, optimizer, scaler, item["step"], metadata)
        del prediction, loss, x, y
    if not seen_memory_gradient or state_fingerprint(model.backbone) != fingerprint:
        raise RuntimeError("memory never learned or persistent parent tensors changed")
    # Gate bypass still yields the parent by construction; startup tested gate=0.
    result = {"updates": len(schedule), "unique_train_songs": len({x["id"] for x in schedule}),
              "trainable_parameters": sum(p.numel() for p in model.adapter.parameters()),
              "memory_gradient_seen": seen_memory_gradient, "parent_unchanged": True,
              "elapsed_seconds": time.perf_counter()-started, "final_gate": float(model.adapter.gate.detach()),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    write_json(directory/"train_summary.json", result)
    write_json(directory/"training_audio_hashes.json", hashes)
    return result, hashes


def evaluate_arm(model, manifest, config, baseline_rows, directory, device, precision):
    directory.mkdir()
    model.eval()
    rows = []
    previous = {row["id"]: row for row in baseline_rows}
    order = [config["training"]["instruments"].index(s) for s in manifest.sources]
    started = time.perf_counter()
    for track in sorted(manifest.partition("validation"), key=lambda t: t["id"]):
        begin = time.perf_counter()
        x, targets, data = load_track(manifest, track, 44100)
        if data["audio_sha256"] != previous[track["id"]]["audio_sha256"]:
            raise ValueError(f"validation audio differs from B0: {track['id']}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        def progress(done, total):
            if done == 1 or done % 10 == 0 or done == total:
                log_event(directory, {"event": "chunks", "id": track["id"], "done": done, "total": total})
        pred = separate(model, x, config, device, precision, progress)[order]
        row = {"id": track["id"], "group": track["group"], **data,
               "sources": {s: score(pred[i], targets[i]) for i, s in enumerate(manifest.sources)},
               "elapsed_seconds": time.perf_counter()-begin,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        rows.append(row)
        with (directory/"tracks.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        log_event(directory, {"event": "track_complete", "id": track["id"], "elapsed_seconds": row["elapsed_seconds"]})
        # Same first-two-track previews as B0, bounded disk use, no full WAV export.
        if len(rows) <= 2:
            dest = directory/"audio"/track["id"]
            dest.mkdir(parents=True)
            size = min(x.shape[-1], 441000)
            offset = (x.shape[-1]-size)//2
            for i, s in enumerate(manifest.sources):
                sf.write(dest/f"{s}_prediction.wav", pred[i, :, offset:offset+size].T, 44100, subtype="FLOAT")
            write_json(dest/"excerpt.json", {"start_sample_44100": offset, "samples": size})
    result = {"complete": True, "kind": "full_track", "tracks": len(rows),
              "sources": aggregate(rows, manifest.sources), "elapsed_seconds": time.perf_counter()-started}
    write_json(directory/"summary.json", result)
    return rows


def paired_comparison(reference, candidate, sources):
    if len({r["id"] for r in candidate}) != len(candidate) or {r["id"] for r in reference} != {r["id"] for r in candidate}:
        raise ValueError("paired comparison needs identical unique track IDs")
    by_id = {r["id"]: r for r in candidate}
    result = {}
    for source in sources:
        metrics = {}
        for metric in ("si_sdr_db", "snr_db"):
            deltas, invalid = [], []
            for row in reference:
                a, b = row["sources"][source], by_id[row["id"]]["sources"][source]
                if a["status"] == "silent_target":
                    if b["status"] != "silent_target":
                        raise ValueError("target activity changed")
                    continue
                if a[metric] is None or b[metric] is None:
                    invalid.append(row["id"])
                else:
                    deltas.append({"id": row["id"], "delta_db": b[metric]-a[metric]})
            values = [d["delta_db"] for d in deltas]
            metrics[metric] = {"n": len(values), "invalid_active_ids": invalid,
                "mean_delta_db": float(np.mean(values)) if values else None,
                "median_delta_db": float(np.median(values)) if values else None,
                "improved": sum(v > 0 for v in values), "worsened": sum(v < 0 for v in values), "tracks": deltas}
        metrics["absent_output_rms"] = [{"id": row["id"], "reference": row["sources"][source]["prediction_rms"],
            "candidate": by_id[row["id"]]["sources"][source]["prediction_rms"]}
            for row in reference if row["sources"][source]["status"] == "silent_target"]
        result[source] = metrics
    return result


def run(args):
    started = time.perf_counter()
    settings = json.loads(Path(args.config).read_text())
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("expected available CUDA or explicit CPU")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    config = read_config(settings["model_config"])
    manifest = AudioManifest(settings["manifest"], audit_audio=False)
    if manifest.task != "moises6" or len(manifest.partition("train")) != 194:
        raise ValueError("requires existing Moises6 194-train/28-validation split")
    if settings["updates"] < 3 or settings["learning_rate"] <= 0 or settings["gradient_clip"] <= 0:
        raise ValueError("at least three updates and positive learning rate/clip required")
    precision = settings["precision"] if device.type == "cuda" else "fp32"
    if settings["precision"] not in ("fp16", "fp32"):
        raise ValueError("precision must be fp16 or fp32")
    base_contract, base_rows = check_baseline(settings["baseline_run"], manifest, config,
        settings["checkpoint"], settings["model_config"], precision)
    train = manifest.partition("train")
    samples = config["audio"]["chunk_size"]
    schedule = make_schedule(train, settings["updates"], settings["seed"], 44100, manifest.sample_rate,
                             samples, config["model"]["stft_hop_length"])
    cfg = AdapterConfig(tuple(config["model"]["freqs_per_bands"]), config["model"]["stft_n_fft"],
        config["model"]["stft_hop_length"], config["model"]["dim"], seed=settings["seed"])
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("pilot output is nonempty; choose a NEW output directory")
    try:
        write_json(output/"settings.json", settings)
        write_json(output/"schedule.json", schedule)
        audit_tracks(manifest, train)
        contract = {"schema_version": 1, "settings": settings, "adapter_config": asdict(cfg),
            "baseline_contract": base_contract, "baseline_tracks_sha256": sha256(Path(settings["baseline_run"])/"tracks.jsonl"),
            "schedule_sha256": sha256(output/"schedule.json"),
            "new_code_sha256": {name: sha256(Path(__file__).parent/name) for name in ("pilot.py", "sw_adapter.py")},
            "loss": "SW waveform L1 + sum(complex multi-STFT L1), all six sources; float32 loss",
            "optimizer": "AdamW, weight_decay=0, fixed learning rate, grad clipping",
            "selection": "fixed final update, no validation-best selection", "checkpointing": "downstream blocks/head only",
            "training_overlap": "SW pretraining songs unknown; exploratory"}
        write_json(output/"contract.json", contract)
        evaluations, summaries, train_hashes = {}, {}, {}
        for arm in ("tc", "stft"):
            log_event(output, {"event": "arm_start", "arm": arm})
            backbone, backend = load_backend(config, settings["checkpoint"], settings["backend_cache"], device, precision)
            for key in ("checkpoint_sha256", "backend_revision", "backend_files"):
                if backend[key] != base_contract[key]:
                    raise ValueError(f"parent changed from B0: {key}")
            model = FrozenSW(backbone, cfg, arm).to(device)
            metadata = {"arm": arm, "contract_sha256": sha256(output/"contract.json"), "adapter_config": asdict(cfg)}
            summaries[arm], train_hashes[arm] = train_arm(model, config, manifest, schedule, settings, output/arm, device, metadata)
            if arm == "stft":
                if train_hashes["tc"] != train_hashes["stft"]:
                    raise ValueError("training files differ between arms")
                if summaries["tc"]["trainable_parameters"] != summaries["stft"]["trainable_parameters"]:
                    raise ValueError("adapter parameter counts do not match")
            evaluations[arm] = evaluate_arm(model, manifest, config, base_rows, output/arm/"evaluation", device, precision)
            write_json(output/arm/"vs_b0.json", paired_comparison(base_rows, evaluations[arm], manifest.sources))
            del model, backbone
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        comparisons = {"tc_minus_b0": paired_comparison(base_rows, evaluations["tc"], manifest.sources),
            "stft_minus_b0": paired_comparison(base_rows, evaluations["stft"], manifest.sources),
            "tc_minus_stft": paired_comparison(evaluations["stft"], evaluations["tc"], manifest.sources)}
        write_json(output/"comparison.json", comparisons)
        write_json(output/"summary.json", {"complete": True, "arms": summaries,
            "elapsed_seconds": time.perf_counter()-started, "comparison": "comparison.json"})
        log_event(output, {"event": "complete", "output": str(output)})
    except BaseException as error:
        write_json(output/"failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                                          "elapsed_seconds": time.perf_counter()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pilot JSON settings, not the model YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
