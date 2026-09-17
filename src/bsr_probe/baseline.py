"""Offline B0 evaluation. No training, adapter, mixture consistency, or TTA.

Scores are local full-track, stereo-joint SI-SDR / waveform SNR, not museval
BSS Eval or official MSR metrics. Silent targets never receive a fabricated SDR.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.nn import functional as F

from msr.data import AudioManifest
from .sw_backend import load_backend, read_config, sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def log_event(run, value):
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    with (run / "progress.jsonl").open("a") as stream:
        stream.write(line + "\n")


def chunk_count(length, chunk, overlap):
    step = chunk // overlap
    border = chunk - step
    padded = length + (2 * border if length > 2 * border else 0)
    return math.ceil(padded / step)


@torch.inference_mode()
def separate(model, mixture, config, device, precision="fp16", progress=None):
    """MSST generic batch-1 fade/reflect protocol; reject NaNs instead of hiding.

    Buffers live on CPU. GPU holds one chunk only. Boundary handling matches the
    pinned upstream generic demix for batch_size=1, num_overlap>=2.
    """
    x = torch.as_tensor(mixture, dtype=torch.float32, device="cpu")
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[-1] < 1 or not torch.isfinite(x).all():
        raise ValueError("expected finite nonempty stereo mixture")
    size = config["audio"]["chunk_size"]
    overlap = config["inference"]["num_overlap"]
    step, fade = size // overlap, size // 10
    border, length = size - step, x.shape[-1]
    padded = length > 2 * border
    if padded:
        x = F.pad(x, (border, border), mode="reflect")
    output = torch.zeros(6, 2, x.shape[-1])
    weight = torch.zeros(x.shape[-1])
    window = torch.ones(size)
    window[:fade] = torch.linspace(0, 1, fade)
    window[-fade:] = torch.linspace(1, 0, fade)
    total = math.ceil(x.shape[-1] / step)
    for index, start in enumerate(range(0, x.shape[-1], step)):
        part = x[:, start:start + size].to(device)
        valid = part.shape[-1]
        part = F.pad(part, (0, size - valid), mode="reflect" if valid > size // 2 else "constant")
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and precision == "fp16" else nullcontext()
        with context:
            predicted = model(part[None])
        if predicted.shape != (1, 6, 2, size) or not torch.isfinite(predicted).all():
            raise RuntimeError("nonfinite/wrong-shaped prediction; no silent zero replacement")
        w = window.clone()
        if start == 0:
            w[:fade] = 1
        elif start + step >= x.shape[-1]:
            w[-fade:] = 1
        output[..., start:start + valid] += predicted[0, ..., :valid].float().cpu() * w[:valid]
        weight[start:start + valid] += w[:valid]
        if progress:
            progress(index + 1, total)
    if torch.any(weight <= 0):
        raise RuntimeError("overlap-add has uncovered samples")
    output /= weight
    if padded:
        output = output[..., border:-border]
    if output.shape[-1] != length:
        raise RuntimeError("inference changed track length")
    return output.numpy()


def score(prediction, target, activity_rms=1e-6):
    """Joint stereo centering, no optimal time shift, gain fit only for SI-SDR.

    Accumulate in blocks to avoid whole-track float64 copies. dB ratios saturate
    at +/-120 dB. A zero/constant prediction has undefined SI-SDR (null), with
    status/counts reported; its ordinary SNR remains meaningful.
    """
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("metric inputs must have the same nonempty shape")
    totals = np.zeros(6, dtype=np.float64)
    for start in range(0, target.shape[-1], 262144):
        p = prediction[..., start:start + 262144].astype(np.float64)
        y = target[..., start:start + 262144].astype(np.float64)
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite metric input")
        totals += [np.sum(p), np.sum(y), np.sum(p*p), np.sum(y*y), np.sum(p*y), np.sum((p-y)**2)]
    sp, sy, pp, yy, py, error = totals
    n = target.size
    target_rms, predicted_rms = math.sqrt(yy/n), math.sqrt(pp/n)
    record = {"target_rms": target_rms, "prediction_rms": predicted_rms,
              "snr_db": None, "si_sdr_db": None, "status": "silent_target"}
    def db(a, b):
        if a <= 0:
            return -120.0
        return float(np.clip(10 * math.log10(a / max(b, a*1e-12)), -120, 120))
    if target_rms <= activity_rms:
        return record
    record["snr_db"] = db(yy, error)
    yc, pc, cross = max(0.0, yy-sy*sy/n), max(0.0, pp-sp*sp/n), py-sp*sy/n
    record["status"] = "degenerate_prediction"
    if yc/n <= activity_rms**2:
        record["status"] = "constant_target"
    elif pc > 1e-24 * n:
        projection = min(pc, max(0.0, cross*cross/yc))
        record["si_sdr_db"] = db(projection, max(0.0, pc-projection))
        record["status"] = "ok"
    return record


def audit_tracks(manifest, tracks):
    records = []
    for track in tracks:
        if track["id"] in (".", "..") or Path(track["id"]).name != track["id"]:
            raise ValueError("track id must be a safe single directory component")
        for source, values in track["targets"].items():
            for path in values:
                resolved = manifest.resolve(path)
                info = sf.info(resolved)
                if (info.samplerate, info.channels, info.frames) != (manifest.sample_rate, 2, track["length"]):
                    raise ValueError(f"manifest/header mismatch: {resolved}")
                records.append({"id": track["id"], "source": source, "path": resolved,
                                "frames": info.frames, "sample_rate": info.samplerate})
        if track.get("mixture"):
            raise ValueError("B0 Moises contract uses sum of prepared six targets; explicit mixtures unsupported")
    return records


def load_track(manifest, track, rate, max_seconds=0):
    # Resample the full recording BEFORE selecting a profile excerpt, so the
    # excerpt boundary does not change the resampling filter context.
    expected = math.ceil(track["length"] * rate / manifest.sample_rate)
    count = min(expected, round(max_seconds * rate)) if max_seconds else expected
    start = (expected - count) // 2 if max_seconds else 0
    target = np.zeros((len(manifest.sources), 2, count), dtype=np.float32)
    audio_hashes = {}
    factor = math.gcd(rate, manifest.sample_rate)
    for index, source in enumerate(manifest.sources):
        for value in track["targets"][source]:
            path = manifest.resolve(value)
            audio_hashes[path] = sha256(path)
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            if (sr, audio.shape) != (manifest.sample_rate, (track["length"], 2)) or not np.isfinite(audio).all():
                raise ValueError(f"audio changed or contains nonfinite values: {path}")
            if sr != rate:
                audio = resample_poly(audio, rate // factor, sr // factor, axis=0).astype(np.float32)
            if len(audio) != expected:
                raise ValueError("unexpected resampling length")
            target[index] += audio[start:start + count].T
    if not np.isfinite(target).all():
        raise ValueError("target sum overflow")
    return target.sum(axis=0), target, {"start_sample_44100": start,
        "samples_44100": count, "full_samples_44100": expected, "audio_sha256": audio_hashes}


def aggregate(rows, sources):
    result = {}
    for source in sources:
        entries = [row["sources"][source] for row in rows]
        scores = {}
        for name in ("snr_db", "si_sdr_db"):
            values = [x[name] for x in entries if x[name] is not None]
            scores[name] = {"count": len(values), "mean": float(np.mean(values)) if values else None,
                            "median": float(np.median(values)) if values else None}
        scores["status_counts"] = {status: sum(x["status"] == status for x in entries)
                                  for status in sorted({x["status"] for x in entries})}
        scores["silent_target_prediction_rms"] = [x["prediction_rms"] for x in entries if x["status"] == "silent_target"]
        result[source] = scores
    return result


def run(args):
    start_time = time.perf_counter()
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("supported devices: cuda:N or cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CONTAINER GPU unavailable; inspect mounts/device exposure before changing models")
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    config = read_config(args.config)
    manifest = AudioManifest(args.manifest, audit_audio=False)
    if manifest.task != "moises6":
        raise ValueError("this experiment is MSS/Moises6, not MSR8")
    tracks = sorted(manifest.partition("validation"), key=lambda t: t["id"])
    if len(tracks) != args.expected_tracks:
        raise ValueError(f"expected {args.expected_tracks} validation tracks, got {len(tracks)}")
    all_tracks = tracks
    if args.limit_tracks:
        tracks = tracks[:args.limit_tracks]
    run_dir = Path(args.output).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise ValueError("output directory is nonempty; choose a NEW --output (do not delete old runs)")
    try:
        write_json(run_dir / "arguments.json", vars(args))
        log_event(run_dir, {"event": "preflight", "tracks": len(tracks), "device": str(device)})
        inventory = audit_tracks(manifest, tracks)
        write_json(run_dir / "input_headers.json", inventory)
        model, backend = load_backend(config, args.checkpoint, args.backend_cache, device, args.precision)
        rate = config["audio"]["sample_rate"]
        contract = {**backend, "config_sha256": sha256(args.config), "config": config,
            "manifest_sha256": manifest.sha256, "dataset_rate": manifest.sample_rate,
            "evaluation_rate": rate, "source_order_model": config["training"]["instruments"],
            "source_order_metrics": list(manifest.sources), "selected_ids": [t["id"] for t in tracks],
            "split": "validation", "training_overlap": "unknown; exploratory evaluation",
            "metrics": "joint-stereo full-excerpt centered SI-SDR and uncentered waveform SNR; not BSS Eval",
            "resampling": "full-stem scipy.signal.resample_poly; mixture=sum(resampled targets)",
            "precision": args.precision if device.type == "cuda" else "fp32",
            "chunk_size": config["audio"]["chunk_size"], "overlap": config["inference"]["num_overlap"],
            "batch_size": 1, "max_seconds": args.max_seconds,
            "python": platform.python_version(), "torch": torch.__version__,
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile", "PyYAML", "einops", "rotary-embedding-torch", "beartype")},
            "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "shared_code_sha256": {str(p.relative_to(Path(__file__).parents[1])): sha256(p)
                for p in (Path(__file__).parents[1] / "msr/data.py", Path(__file__).parents[1] / "msr/config.py")},
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        write_json(run_dir / "contract.json", contract)
        log_event(run_dir, {"event": "strict_load_passed", **backend})
        rows = []
        order = [config["training"]["instruments"].index(s) for s in manifest.sources]
        for track_index, track in enumerate(tracks):
            begin = time.perf_counter()
            log_event(run_dir, {"event": "loading_track", "id": track["id"], "index": track_index + 1})
            mixture, targets, data = load_track(manifest, track, rate, args.max_seconds)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            infer_start = time.perf_counter()
            def progress(done, total):
                if done == 1 or done % 10 == 0 or done == total:
                    log_event(run_dir, {"event": "chunks", "id": track["id"], "done": done, "total": total,
                                        "elapsed_seconds": time.perf_counter() - infer_start})
            predictions = separate(model, mixture, config, device, args.precision, progress)[order]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - infer_start
            values = {source: score(predictions[i], targets[i]) for i, source in enumerate(manifest.sources)}
            if track_index < args.save_audio_tracks:
                destination = run_dir / "audio" / track["id"]
                destination.mkdir(parents=True)
                size = min(mixture.shape[-1], round(args.preview_seconds * rate))
                offset = (mixture.shape[-1] - size) // 2
                section = slice(offset, offset + size)
                sf.write(destination / "mixture.wav", mixture[:, section].T, rate, subtype="FLOAT")
                for i, source in enumerate(manifest.sources):
                    sf.write(destination / f"{source}_prediction.wav", predictions[i, :, section].T, rate, subtype="FLOAT")
                    sf.write(destination / f"{source}_target.wav", targets[i, :, section].T, rate, subtype="FLOAT")
                write_json(destination / "excerpt.json", {"start_sample_44100": data["start_sample_44100"] + offset, "samples": size})
            duration = mixture.shape[-1] / rate
            row = {"id": track["id"], "group": track["group"], **data, "sources": values,
                   "audio_seconds": duration, "inference_seconds": inference_seconds,
                   "real_time_factor": inference_seconds / duration,
                   "chunks": chunk_count(mixture.shape[-1], config["audio"]["chunk_size"], config["inference"]["num_overlap"]),
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                   "elapsed_seconds": time.perf_counter() - begin}
            rows.append(row)
            with (run_dir / "tracks.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            log_event(run_dir, {"event": "track_complete", "id": track["id"], "inference_seconds": inference_seconds,
                                "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"]})
        full_chunks = sum(chunk_count(math.ceil(t["length"] * rate / manifest.sample_rate),
                          config["audio"]["chunk_size"], config["inference"]["num_overlap"]) for t in all_tracks)
        summary = {"complete": True, "kind": "profile_excerpt" if args.max_seconds else "full_track",
                   "tracks": len(rows), "sources": aggregate(rows, manifest.sources),
                   "elapsed_seconds": time.perf_counter() - start_time,
                   "estimated_full_validation_inference_seconds": sum(r["inference_seconds"] for r in rows) / sum(r["chunks"] for r in rows) * full_chunks,
                   "estimate_note": "same GPU/precision/chunk settings; excludes IO, resampling, metrics; first-pass warmup included",
                   "training_overlap": "unknown"}
        write_json(run_dir / "summary.json", summary)
        log_event(run_dir, {"event": "complete", "summary": str(run_dir / "summary.json")})
    except Exception as error:
        write_json(run_dir / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                                              "elapsed_seconds": time.perf_counter() - start_time})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "manifest", "backend-cache", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--expected-tracks", type=int, default=28)
    parser.add_argument("--limit-tracks", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--save-audio-tracks", type=int, default=2)
    parser.add_argument("--preview-seconds", type=float, default=10)
    args = parser.parse_args()
    if (not math.isfinite(args.max_seconds) or not math.isfinite(args.preview_seconds)
            or args.cpu_threads < 1 or args.expected_tracks < 1
            or min(args.limit_tracks, args.max_seconds, args.save_audio_tracks) < 0 or args.preview_seconds <= 0):
        parser.error("invalid limits/threads/preview duration")
    if args.max_seconds and round(args.max_seconds * 44100) < 1:
        parser.error("max-seconds is shorter than one sample")
    run(args)


if __name__ == "__main__":
    main()
