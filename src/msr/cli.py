"""Entry points intentionally do not download data or start remote jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import soundfile as sf
import torch

from . import ModelConfig, RestorationModel
from .checkpoint import load_checkpoint, make_adapter, transfer_to_msr
from .data import AudioManifest, sha256_file
from .engine import crop_contract, evaluate, train
from .objectives import reconstruction_loss


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_manifest(model, manifest):
    if model.cfg.sample_rate != manifest.sample_rate or model.cfg.sources != manifest.sources:
        raise ValueError("model and data sample rate/source taxonomy must match exactly")


def lineage_contract(manifest, inherited=None):
    inherited = inherited or {}
    prior_train = set(inherited.get("lineage_training_groups", []))
    prior_selection = set(inherited.get("lineage_selection_groups", []))
    prior_train.update(g for g, split in inherited.get("groups", {}).items() if split == "train")
    prior_selection.update(g for g, split in inherited.get("groups", {}).items() if split == "validation")
    for track in manifest.tracks:
        if track["split"] != "train" and track["group"] in prior_train:
            raise ValueError("held-out group was used for parent checkpoint training")
        if track["split"] == "test" and track["group"] in prior_selection:
            raise ValueError("test group was used for parent checkpoint selection")
    return {"lineage_training_groups": sorted(prior_train | {g for g, s in manifest.groups.items() if s == "train"}),
            "lineage_selection_groups": sorted(prior_selection | {g for g, s in manifest.groups.items() if s == "validation"})}


def train_command(args):
    if args.steps < 1 or args.batch_size < 1 or args.validation_every < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("steps, batch, validation interval and learning rate must be positive")
    manifest = AudioManifest(args.manifest)
    manifest.partition("train")
    manifest.partition("validation")
    resume = None
    inherited = {}
    if args.resume:
        if args.parent or args.config:
            raise ValueError("resume restores the checkpoint config; do not combine with parent/config")
        model, resume = load_checkpoint(args.resume)
        inherited = resume["run_contract"]
        if resume["stage"] != args.stage or model.variant != args.variant:
            raise ValueError("resume stage/variant mismatch")
    elif args.stage == "pretrain":
        if args.parent or args.variant != "A0" or manifest.task != "moises6":
            raise ValueError("pretrain creates the common six-stem A0; no parent is allowed")
        cfg = ModelConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else ModelConfig()
        model = RestorationModel(cfg, "A0")
    else:
        if not args.parent or args.config:
            raise ValueError("adapt/finetune require a parent checkpoint and preserve its backbone config")
        parent, payload = load_checkpoint(args.parent)
        inherited = payload["run_contract"]
        if payload["stage"] != "pretrain":
            raise ValueError("first screen must use a pretrain baseline checkpoint")
        if args.stage == "adapt":
            model = make_adapter(parent, args.variant)
        else:
            model, _ = transfer_to_msr(parent, args.variant)
    model.configure_training(args.stage == "adapt")
    check_manifest(model, manifest)
    core = args.core_samples or model.cfg.sample_rate * 2
    pre, look = crop_contract(model, core)
    fft_sizes = tuple(args.loss_ffts)
    if any(n < 4 or n % 2 for n in fft_sizes):
        raise ValueError("loss FFT sizes must be even and >=4")
    run_contract = {"manifest_sha256": manifest.sha256, "source_sha256": source_hash(), "stage": args.stage,
                    "variant": args.variant, "batch_size": args.batch_size, "lr": args.lr,
                    "data_seed": args.data_seed, "core_samples": core, "preroll_samples": pre,
                    "lookahead_samples": look, "loss_ffts": list(fft_sizes), "precision": "float32",
                    "groups": manifest.groups, "audio_sha256": manifest.audio_sha256,
                    "device": args.device, "torch_version": str(torch.__version__),
                    "parent_sha256": sha256_file(args.parent) if args.parent else None,
                    **lineage_contract(manifest, inherited)}
    run = Path(args.run_dir).expanduser().resolve()
    if resume:
        run_contract["parent_sha256"] = resume["run_contract"].get("parent_sha256")
        if resume["run_contract"] != run_contract:
            raise ValueError("resume contract mismatch: data/code/optimizer/crop settings changed")
        if Path(args.resume).resolve().parent != run:
            raise ValueError("resume checkpoint must belong to the requested run directory")
    elif run.exists() and any(run.iterdir()):
        raise ValueError("run directory is not empty; choose a new directory or explicit --resume")
    run.mkdir(parents=True, exist_ok=True)
    if not resume:
        (run / "run.json").write_text(json.dumps({"contract": run_contract, "config": model.cfg.to_dict(),
                                                   "torch": torch.__version__, "python": platform.python_version(),
                                                   "device": args.device}, indent=2))
        # Keep paths executable when the snapshot moves away from its source manifest.
        snapshot = json.loads(json.dumps(manifest.document))
        for track in snapshot["tracks"]:
            track["targets"] = {s: [manifest.resolve(p) for p in paths] for s, paths in track["targets"].items()}
            if track.get("mixture"):
                track["mixture"] = manifest.resolve(track["mixture"])
        (run / "manifest.json").write_text(json.dumps(snapshot, indent=2))
    model.to(args.device)
    print(json.dumps({"allocated_parameters": sum(p.numel() for p in model.parameters()),
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "core_samples": core, "preroll_samples": pre, "lookahead_samples": look}), flush=True)
    summary = train(model, manifest, run, stage=args.stage, steps=args.steps, batch_size=args.batch_size,
                    learning_rate=args.lr, data_seed=args.data_seed, core_samples=core,
                    validation_every=args.validation_every, device=args.device, fft_sizes=fft_sizes,
                    run_contract=run_contract, resume=resume)
    (run / "summary.json").write_text(json.dumps(summary, indent=2))


def evaluate_command(args):
    model, payload = load_checkpoint(args.checkpoint, args.device)
    manifest = AudioManifest(args.manifest)
    check_manifest(model, manifest)
    # The benchmark's held-out groups must not occur in the checkpoint's training set.
    train_groups = {g for g, split in payload["run_contract"].get("groups", {}).items() if split == "train"}
    train_groups.update(payload["run_contract"].get("lineage_training_groups", []))
    if args.split == "test":
        train_groups.update(payload["run_contract"].get("lineage_selection_groups", []))
    if any(t["group"] in train_groups for t in manifest.partition(args.split)):
        raise ValueError("evaluation groups overlap checkpoint training groups")
    contract = payload["run_contract"]
    core = args.core_samples or contract.get("core_samples", model.cfg.sample_rate * 2)
    scores = evaluate(model, manifest, args.split, core, args.device, tuple(contract.get("loss_ffts", (256, 512, 1024))), args.gate_off, args.intervention)
    scores.update(checkpoint_sha256=sha256_file(args.checkpoint), manifest_sha256=manifest.sha256,
                  audio_sha256=manifest.audio_sha256, core_samples=core, variant=model.variant,
                  model_config=model.cfg.to_dict(), training_contract=contract)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(scores, stream, indent=2, allow_nan=False)
    print(json.dumps({"track_macro_loss": scores["track_macro_loss"], "output": str(output.resolve())}))


def infer_command(args):
    model, payload = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    info = sf.info(args.input)
    if info.samplerate != model.cfg.sample_rate or info.channels != 2:
        raise ValueError("input must already be stereo at checkpoint sample rate")
    core = args.core_samples or payload["run_contract"].get("core_samples", model.cfg.sample_rate * 2)
    pre, look = crop_contract(model, core)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("inference output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        handles = [sf.SoundFile(output / f"{name}.wav", mode="w", samplerate=model.cfg.sample_rate,
                                channels=2, subtype="FLOAT") for name in model.cfg.sources]
        with sf.SoundFile(args.input) as audio, torch.no_grad():
            for start in range(0, info.frames, core):
                x = torch.zeros(2, pre + core + look)
                offset = start - pre
                a, b = max(0, offset), min(info.frames, start + core + look)
                audio.seek(a)
                x[:, a - offset:b - offset] = torch.from_numpy(audio.read(b - a, dtype="float32", always_2d=True).T.copy())
                pred = model(x[None].to(args.device), gate_off=args.gate_off)[0, ..., pre:pre + min(core, info.frames - start)]
                if not torch.isfinite(pred).all():
                    raise RuntimeError("nonfinite inference output")
                for handle, stem in zip(handles, pred.cpu()):
                    handle.write(stem.T.numpy())
    finally:
        for handle in handles:
            handle.close()
    (output / "provenance.json").write_text(json.dumps({"checkpoint_sha256": sha256_file(args.checkpoint),
                                                        "input_sha256": sha256_file(args.input), "core_samples": core,
                                                        "gate_off": args.gate_off, "latency_bound_samples": model.latency_samples}, indent=2))


def smoke_command(args):
    torch.set_num_threads(args.threads)
    cfg = ModelConfig.tiny() if not args.config else ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    torch.manual_seed(cfg.seed)
    x = torch.randn(1, 2, args.samples or cfg.hop * 8, device=args.device) * 0.1
    rows = []
    baseline = None
    for variant in ("A0", "A1", "A2"):
        model = RestorationModel(cfg, variant).to(args.device)
        start = time.monotonic()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        y, aux = model(x, return_aux=True)
        target = x[:, None].expand(-1, len(cfg.sources), -1, -1) / len(cfg.sources)
        loss = reconstruction_loss(y, target, [x.shape[-1]], (cfg.n_fft,))
        loss.backward()
        if not torch.isfinite(y).all():
            raise RuntimeError("nonfinite smoke output")
        if args.device == "cuda":
            torch.cuda.synchronize()
        row = {"variant": variant, "shape": list(y.shape), "loss": loss.item(),
               "seconds": time.monotonic() - start, "parameters": sum(p.numel() for p in model.parameters()),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
        with torch.no_grad():
            zero = model(x, gate_off=True)
            if baseline is None:
                baseline = zero
            row["gate_off_baseline_max_error"] = (baseline - zero).abs().max().item()
        rows.append(row)
    print(json.dumps({"scope": "synthetic implementation smoke, not restoration quality", "results": rows}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--stage", choices=("pretrain", "adapt", "finetune"), required=True)
    training.add_argument("--variant", choices=("A0", "A1", "A2"), default="A0")
    training.add_argument("--manifest", required=True)
    training.add_argument("--run-dir", required=True)
    training.add_argument("--config")
    training.add_argument("--parent")
    training.add_argument("--resume")
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--lr", type=float, default=0.0001)
    training.add_argument("--data-seed", type=int, default=20260914)
    training.add_argument("--validation-every", type=int, default=100)
    training.add_argument("--loss-ffts", type=int, nargs="+", default=[256, 512, 1024])
    training.set_defaults(func=train_command)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--gate-off", action="store_true")
    evaluation.add_argument("--intervention", choices=("frequency_shuffle", "kv_joint_permutation", "wrong_track"))
    evaluation.set_defaults(func=evaluate_command)
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--output-dir", required=True)
    inference.add_argument("--gate-off", action="store_true")
    inference.set_defaults(func=infer_command)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config")
    smoke.add_argument("--samples", type=int)
    smoke.add_argument("--threads", type=int, default=2)
    smoke.set_defaults(func=smoke_command)
    prepare = commands.add_parser("prepare-moises")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--splits", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--sample-rate", type=int, default=48000)
    def prepare_command(args):
        from .prepare import export_moises
        print(export_moises(args.root, args.splits, args.output_dir, args.sample_rate))
    prepare.set_defaults(func=prepare_command)
    split = commands.add_parser("make-moises-splits")
    split.add_argument("--root", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=20260915)
    split.add_argument("--train", type=float, default=0.8)
    split.add_argument("--validation", type=float, default=0.1)
    split.add_argument("--test", type=float, default=0.1)
    def split_command(args):
        from .splits import make_moises_splits
        split_path, audit_path = make_moises_splits(args.root, args.output, seed=args.seed,
            train=args.train, validation=args.validation, test=args.test)
        print(json.dumps({"splits": str(split_path), "audit": str(audit_path)}))
    split.set_defaults(func=split_command)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--metric", choices=("sisdr_db", "snr_db"), default="sisdr_db")
    compare.add_argument("--output", required=True)
    def compare_command(args):
        from .compare import compare_results
        result = compare_results(args.baseline, args.candidate, args.metric)
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result["effects"], indent=2))
    compare.set_defaults(func=compare_command)
    for command in (training, evaluation, inference, smoke):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for command in (training, evaluation, inference):
        command.add_argument("--core-samples", type=int)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
