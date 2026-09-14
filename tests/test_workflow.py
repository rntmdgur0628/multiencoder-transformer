import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from msr import ModelConfig, MOISES_SOURCES, MSR_SOURCES, RestorationModel
from msr.checkpoint import load_checkpoint, make_adapter, save_checkpoint, transfer_to_msr
from msr.cli import main, lineage_contract
from msr.data import AudioManifest, training_batch
from msr.engine import TrackMetrics
from msr.objectives import source_diagnostics


def fixture(directory, task="moises6"):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig.tiny()
    sources = MOISES_SOURCES if task == "moises6" else MSR_SOURCES
    rng = np.random.default_rng(42)
    tracks = []
    for i, split in enumerate(("train", "validation", "test")):
        y = rng.standard_normal((160, 2)).astype(np.float32) * 0.1
        sf.write(root / f"{i}_target.wav", y, cfg.sample_rate, subtype="FLOAT")
        targets = {s: [] for s in sources}
        targets[sources[0]] = [f"{i}_target.wav"]
        mixture = None
        if task == "msr8":
            mixture = f"{i}_mixture.wav"
            sf.write(root / mixture, y * 0.7, cfg.sample_rate, subtype="FLOAT")
        tracks.append({"id": f"{task}-{i}", "group": f"{task}-{i}", "split": split,
                       "mixture": mixture, "targets": targets, "length": len(y), "provenance": "synthetic_test_only"})
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "task": task, "sample_rate": cfg.sample_rate,
                                "sources": list(sources), "tracks": tracks}))
    (root / "config.json").write_text(json.dumps(cfg.to_dict()))
    return path


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = fixture(self.root / "moises")

    def tearDown(self):
        self.temp.cleanup()

    def call(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            main(args)

    def train_args(self, run, steps=2):
        return ["train", "--stage", "pretrain", "--manifest", str(self.path), "--run-dir", str(run),
                "--steps", str(steps), "--core-samples", "64", "--validation-every", "1", "--loss-ffts", "32", "64"]

    def test_manifest_absence_and_msr_input(self):
        manifest = AudioManifest(self.path)
        x, y, valid = manifest.segment(manifest.tracks[0], 0, 64, 32, 32)
        torch.testing.assert_close(x[..., 32:96], y.sum(0))
        self.assertEqual(y[1:].count_nonzero(), 0)
        doc = json.loads(self.path.read_text())
        del doc["tracks"][0]["targets"]["piano"]
        self.path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "all target labels"):
            AudioManifest(self.path)
        path = fixture(self.root / "msr", "msr8")
        doc = json.loads(path.read_text())
        doc["tracks"][0]["mixture"] = None
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "degraded input"):
            AudioManifest(path)

    def test_group_leakage_and_official_validation_rejected(self):
        doc = json.loads(self.path.read_text())
        doc["tracks"][1]["group"] = doc["tracks"][0]["group"]
        self.path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "group leaks"):
            AudioManifest(self.path)
        self.path = fixture(self.root / "moises")
        doc = json.loads(self.path.read_text())
        doc["tracks"][0]["provenance"] = "official_validation"
        self.path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "official evaluation"):
            AudioManifest(self.path)

    def test_sampler_independent_of_model_rng(self):
        manifest = AudioManifest(self.path)
        a = training_batch(manifest, 3, 2, 123, 64, 32, 32, 16)
        torch.manual_seed(999)
        RestorationModel(ModelConfig.tiny(), "A2")
        b = training_batch(manifest, 3, 2, 123, 64, 32, 32, 16)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        self.assertEqual(a[3], b[3])

    def test_parent_training_and_selection_leaks_rejected(self):
        manifest = AudioManifest(self.path)
        with self.assertRaisesRegex(ValueError, "parent checkpoint training"):
            lineage_contract(manifest, {"lineage_training_groups": ["moises6-2"]})
        with self.assertRaisesRegex(ValueError, "parent checkpoint selection"):
            lineage_contract(manifest, {"lineage_selection_groups": ["moises6-2"]})

    def test_audio_content_mutation_changes_fingerprint(self):
        a = AudioManifest(self.path)
        audio = self.path.parent / "0_target.wav"
        data, sr = sf.read(audio, dtype="float32")
        sf.write(audio, data * 0.5, sr, subtype="FLOAT")
        b = AudioManifest(self.path)
        self.assertEqual(a.sha256, b.sha256)
        self.assertNotEqual(a.audio_sha256, b.audio_sha256)

    def test_taxonomy_transfer_and_shared_new_head_initialization(self):
        parent = RestorationModel(ModelConfig.tiny(), "A0")
        a, mapping = transfer_to_msr(parent, "A1")
        b, _ = transfer_to_msr(parent, "A2")
        for old, new_a, new_b in zip(parent.heads, a.heads, b.heads):
            torch.testing.assert_close(new_a.weight, new_b.weight, atol=0, rtol=0)
            original = old.weight.reshape(6, -1)
            transferred = new_a.weight.reshape(8, -1)
            for target, source in mapping.items():
                torch.testing.assert_close(transferred[MSR_SOURCES.index(target)], original[MOISES_SOURCES.index(source)])
        self.assertNotIn("keyboards", mapping)

    def test_track_statistics_match_whole_waveform(self):
        y = torch.randn(6, 2, 150)
        p = y * 0.8 + torch.randn_like(y) * 0.1
        acc = TrackMetrics(6)
        acc.update(p[..., :64], y[..., :64])
        acc.update(p[..., 64:], y[..., 64:])
        for a, b in zip(acc.compute(), source_diagnostics(p, y)):
            for metric in ("snr_db", "sisdr_db", "mae"):
                self.assertAlmostEqual(a[metric], b[metric], places=7)

    def test_resume_is_bitwise_same_as_uninterrupted(self):
        full, partial = self.root / "full", self.root / "partial"
        config = str(self.path.parent / "config.json")
        self.call(self.train_args(full) + ["--config", config])
        self.call(self.train_args(partial, 1) + ["--config", config])
        self.call(self.train_args(partial) + ["--resume", str(partial / "checkpoint_last.pt")])
        a, _ = load_checkpoint(full / "checkpoint_last.pt")
        b, _ = load_checkpoint(partial / "checkpoint_last.pt")
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)

    def test_end_to_end_adapter_transfer_evaluate_infer(self):
        run = self.root / "pretrain"
        self.call(self.train_args(run, 1) + ["--config", str(self.path.parent / "config.json")])
        parent = run / "checkpoint_last.pt"
        for variant in ("A1", "A2"):
            adapted = self.root / variant
            self.call(["train", "--stage", "adapt", "--variant", variant, "--parent", str(parent),
                       "--manifest", str(self.path), "--run-dir", str(adapted), "--steps", "1", "--core-samples", "64", "--loss-ffts", "32", "64"])
            self.call(["evaluate", "--checkpoint", str(adapted / "checkpoint_last.pt"), "--manifest", str(self.path),
                       "--output", str(adapted / "test.json")])
        msr_path = fixture(self.root / "msr", "msr8")
        msr_run = self.root / "finetune"
        self.call(["train", "--stage", "finetune", "--variant", "A2", "--parent", str(parent), "--manifest", str(msr_path),
                   "--run-dir", str(msr_run), "--steps", "1", "--core-samples", "64", "--loss-ffts", "32", "64"])
        self.call(["infer", "--checkpoint", str(msr_run / "checkpoint_last.pt"), "--input", str(msr_path.parent / "2_mixture.wav"),
                   "--output-dir", str(self.root / "inference")])
        for source in MSR_SOURCES:
            info = sf.info(self.root / "inference" / f"{source}.wav")
            self.assertEqual((info.frames, info.channels, info.samplerate), (160, 2, 8000))


if __name__ == "__main__":
    unittest.main()
