"""Deterministic local training, track-level evaluation, and reproducible resume."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch

from .checkpoint import save_checkpoint
from .data import training_batch
from .objectives import reconstruction_loss


class TrackMetrics:
    """Sufficient statistics: do not average crop SI-SDRs and call that track SI-SDR."""
    def __init__(self, sources):
        self.sums = torch.zeros(sources, 7, dtype=torch.float64)

    def update(self, prediction, target):
        p, y = prediction.detach().cpu().double().flatten(1), target.cpu().double().flatten(1)
        self.sums += torch.stack((p.sum(-1), y.sum(-1), p.square().sum(-1), y.square().sum(-1),
                                 (p * y).sum(-1), (p - y).abs().sum(-1), torch.full((p.shape[0],), p.shape[-1])), -1)

    def compute(self):
        rows = []
        for sum_p, sum_y, p2, y2, py, absolute, count in self.sums.tolist():
            row = {"mae": absolute / count, "output_rms": math.sqrt(max(p2 / count, 0))}
            if y2 <= 1e-12:
                row.update(snr_db=None, sisdr_db=None)
            else:
                row["snr_db"] = 10 * math.log10(y2 / max(p2 + y2 - 2 * py, 1e-12))
                yc = max(y2 - sum_y * sum_y / count, 1e-12)
                pc = max(p2 - sum_p * sum_p / count, 0)
                cross = py - sum_y * sum_p / count
                projected = cross * cross / yc
                row["sisdr_db"] = 10 * math.log10(max(projected, 1e-12) / max(pc - projected, 1e-12))
            rows.append(row)
        return rows


def crop_contract(model, core_samples):
    if core_samples <= 0 or core_samples % model.cfg.hop:
        raise ValueError("core sample count must be a positive multiple of hop")
    hop = model.cfg.hop
    pre = math.ceil(model.preroll_samples / hop) * hop
    look = math.ceil(model.latency_samples / hop) * hop
    return pre, look


@torch.no_grad()
def evaluate(model, manifest, split, core_samples, device, fft_sizes, gate_off=False, intervention=None):
    was_training = model.training
    model.eval()
    pre, look = crop_contract(model, core_samples)
    tracks = []
    partition = manifest.partition(split)
    for track in partition:
        donors = [item for item in partition if item["group"] != track["group"]]
        if intervention == "wrong_track" and not donors:
            raise ValueError("wrong-track diagnostic requires another held-out song group")
        metrics = TrackMetrics(len(model.cfg.sources))
        total_loss, total_samples = 0.0, 0
        for start in range(0, track["length"], core_samples):
            x, y, length = manifest.segment(track, start, core_samples, pre, look)
            donor = None
            if intervention == "wrong_track":
                donor_start = round(start / max(1, track["length"] - core_samples) * max(0, donors[0]["length"] - core_samples))
                donor_start = (donor_start // model.cfg.hop) * model.cfg.hop
                donor, _, _ = manifest.segment(donors[0], donor_start, core_samples, pre, look)
                donor = donor[None].to(device)
            pred = model(x[None].to(device), gate_off=gate_off, intervention=intervention,
                         memory_waveform=donor)[..., pre:pre + core_samples]
            truth = y[None].to(device)
            loss = reconstruction_loss(pred, truth, [length], fft_sizes)
            if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                raise RuntimeError("nonfinite evaluation output/loss")
            metrics.update(pred[0, ..., :length], y[..., :length])
            total_loss += loss.item() * length
            total_samples += length
        tracks.append({"id": track["id"], "group": track["group"], "loss": total_loss / total_samples,
                       "sources": dict(zip(model.cfg.sources, metrics.compute())),
                       "donor_id": donors[0]["id"] if intervention == "wrong_track" else None})
    model.train(was_training)
    source_summary = {}
    for source in model.cfg.sources:
        source_summary[source] = {}
        for metric in ("mae", "snr_db", "sisdr_db", "output_rms"):
            values = [t["sources"][source][metric] for t in tracks if t["sources"][source][metric] is not None]
            source_summary[source][metric] = sum(values) / len(values) if values else None
            source_summary[source][metric + "_tracks"] = len(values)
    return {"split": split, "tracks": tracks, "track_macro_loss": sum(t["loss"] for t in tracks) / len(tracks),
            "source_track_macro": source_summary, "metric_scope": "local waveform diagnostics; NOT official Multi-Mel-SNR/Zimtohrli/FAD", "gate_off": gate_off, "intervention": intervention}


def train(model, manifest, run_dir, *, stage, steps, batch_size, learning_rate, data_seed,
          core_samples, validation_every, device, fft_sizes, run_contract, resume=None):
    pre, look = crop_contract(model, core_samples)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters: frozen A0 is evaluation-only")
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)
    start_step = 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["cpu_rng"])
        start_step = resume["step"]
    if steps <= start_step:
        raise ValueError("requested steps must exceed completed checkpoint step")
    run = Path(run_dir)
    best_path = run / "checkpoint_best.pt"
    best = float("inf")
    if resume and best_path.exists():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
        if best_payload["run_contract"] != run_contract:
            raise ValueError("best checkpoint belongs to another run")
        best = best_payload.get("best_validation_loss", float("inf"))
    # A fresh journal per resume prevents partial logs from silently duplicating steps.
    journal = run / f"training_{start_step}_{time.time_ns()}.jsonl"
    started = time.monotonic()

    def checkpoint(path, step, score=None):
        save_checkpoint(path, model, stage=stage, step=step, optimizer=optimizer,
                        run_contract=run_contract, best_validation_loss=score)

    def validation(step, stream):
        nonlocal best
        scores = evaluate(model, manifest, "validation", core_samples, device, fft_sizes)
        (run / f"validation_{step}.json").write_text(json.dumps(scores, indent=2, allow_nan=False))
        value = scores["track_macro_loss"]
        if value < best:
            best = value
            checkpoint(best_path, step, best)
        stream.write(json.dumps({"event": "validation", "step": step, "track_macro_loss": value, "best_validation_loss": best}) + "\n")
        stream.flush()

    with journal.open("x") as stream:
        if not resume:
            validation(0, stream)
            checkpoint(run / "checkpoint_last.pt", 0)
        model.train()
        for step in range(start_step, steps):
            x, y, lengths, records = training_batch(manifest, step, batch_size, data_seed, core_samples, pre, look, model.cfg.hop)
            optimizer.zero_grad(set_to_none=True)
            prediction, aux = model(x.to(device), return_aux=True)
            loss = reconstruction_loss(prediction[..., pre:pre + core_samples], y.to(device), lengths, fft_sizes)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
            pe_grad = model.injection.position.raw.grad
            optimizer.step()
            row = {"event": "train", "step": step + 1, "loss": loss.item(), "gradient_norm": grad_norm.item(),
                   "pe_gradient_norm": pe_grad.norm().item() if pe_grad is not None else None,
                   "gates": aux["gates"].cpu().tolist(), "residual_rms_ratio": aux["residual_rms_ratio"].cpu().tolist(),
                   "samples": records, "elapsed_seconds": time.monotonic() - started}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({k: row[k] for k in ("step", "loss", "gates", "pe_gradient_norm")}), flush=True)
            if (step + 1) % validation_every == 0 or step + 1 == steps:
                validation(step + 1, stream)
                checkpoint(run / "checkpoint_last.pt", step + 1)
    return {"completed_steps": steps, "best_validation_loss": best, "elapsed_seconds": time.monotonic() - started}
