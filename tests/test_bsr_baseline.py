"""Offline baseline protocol tests, no weights/network/GPU required."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import yaml

from bsr_probe.baseline import aggregate, audit_tracks, chunk_count, load_track, run, score, separate
from bsr_probe.sw_backend import read_config, TupleSafeLoader
from msr.config import MOISES_SOURCES
from msr.data import AudioManifest


NAMES = ["bass", "drums", "other", "vocals", "guitar", "piano"]


def config():
    return {"audio": {"sample_rate": 44100, "num_channels": 2, "chunk_size": 64},
            "model": {"num_stems": 6, "stereo": True, "stft_hop_length": 16},
            "training": {"instruments": NAMES}, "inference": {"num_overlap": 2}}


class Echo(torch.nn.Module):
    def forward(self, x):
        return x[:, None].repeat(1, 6, 1, 1)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_overlap_identity_all_boundaries(self):
        rng = np.random.default_rng(31)
        for overlap in (2, 3, 4):
            c = config()
            c["inference"]["num_overlap"] = overlap
            for length in (1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 333):
                x = rng.normal(size=(2, length)).astype(np.float32)
                progress = []
                y = separate(Echo(), x, c, torch.device("cpu"), progress=lambda d, n: progress.append((d, n)))
                self.assertEqual(y.shape, (6, 2, length))
                np.testing.assert_allclose(y, np.broadcast_to(x, y.shape), atol=5e-7)
                self.assertEqual(progress[-1], (chunk_count(length, 64, overlap),)*2)

    def test_nonfinite_predictions_fail(self):
        class Bad(Echo):
            def forward(self, x):
                return super().forward(x) * float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            separate(Bad(), np.ones((2, 30)), config(), torch.device("cpu"))

    def test_metric_scale_silence_and_zero(self):
        x = np.random.default_rng(2).normal(size=(2, 99)).astype(np.float32)
        perfect = score(x, x)
        self.assertAlmostEqual(perfect["snr_db"], 120)
        self.assertGreater(perfect["si_sdr_db"], 100)
        gain = score(x * 2, x)
        self.assertAlmostEqual(gain["snr_db"], 0)
        self.assertGreater(gain["si_sdr_db"], 100)
        zero = score(np.zeros_like(x), x)
        self.assertAlmostEqual(zero["snr_db"], 0)
        self.assertIsNone(zero["si_sdr_db"])
        self.assertEqual(zero["status"], "degenerate_prediction")
        absent = score(x, np.zeros_like(x))
        self.assertEqual(absent["status"], "silent_target")
        self.assertGreater(absent["prediction_rms"], 0)
        self.assertIsNone(absent["snr_db"])

    def test_metric_matches_direct_equation(self):
        rng = np.random.default_rng(44)
        y = rng.normal(size=(2, 270000)).astype(np.float32)
        p = 0.7*y + rng.normal(size=y.shape).astype(np.float32)*0.3
        a, b = p.astype(np.float64).ravel(), y.astype(np.float64).ravel()
        a -= a.mean()
        b -= b.mean()
        projected = np.dot(a, b) / np.dot(b, b) * b
        reference = 10*np.log10(np.sum(projected**2)/np.sum((a-projected)**2))
        self.assertAlmostEqual(score(p, y)["si_sdr_db"], reference, places=9)

    def test_aggregation_counts_missing_stems(self):
        x = np.ones((2, 10))
        rows = [{"sources": {"piano": score(x, np.zeros_like(x))}}]
        result = aggregate(rows, ["piano"])["piano"]
        self.assertEqual(result["si_sdr_db"]["count"], 0)
        self.assertIsNone(result["si_sdr_db"]["mean"])
        self.assertEqual(result["status_counts"], {"silent_target": 1})
        json.dumps(result, allow_nan=False)

    def test_yaml_allows_only_tuple_extension(self):
        self.assertEqual(yaml.load("x: !!python/tuple [1, 2]", Loader=TupleSafeLoader), {"x": (1, 2)})
        with self.assertRaises(yaml.constructor.ConstructorError):
            yaml.load("!!python/object/apply:os.system ['false']", Loader=TupleSafeLoader)

    def test_bad_source_taxonomy_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            c = config()
            c["training"]["instruments"] = ["vocals"] * 6
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, "six named"):
                read_config(path)

    def fixture(self, root):
        length = 321
        x = np.random.default_rng(5).normal(size=(length, 2)).astype(np.float32)*0.1
        sf.write(root / "vocals.wav", x, 48000, subtype="FLOAT")
        targets = {s: ["vocals.wav"] if s == "vocals" else [] for s in MOISES_SOURCES}
        track = {"id": "song", "group": "song", "split": "validation", "length": length, "targets": targets}
        (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "task": "moises6",
            "sample_rate": 48000, "sources": list(MOISES_SOURCES), "tracks": [track]}))
        return AudioManifest(root / "manifest.json", audit_audio=False), track

    def test_resampling_crop_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, track = self.fixture(root)
            audit_tracks(manifest, [track])
            x, y, data = load_track(manifest, track, 44100)
            self.assertEqual(x.shape[-1], 295)
            np.testing.assert_array_equal(x, y[0])
            self.assertTrue(np.all(y[1:] == 0))
            xc, yc, crop = load_track(manifest, track, 44100, 0.002)
            start, count = crop["start_sample_44100"], crop["samples_44100"]
            np.testing.assert_array_equal(xc, x[:, start:start+count])
            np.testing.assert_array_equal(yc, y[..., start:start+count])
            self.assertEqual(len(data["audio_sha256"]), 1)
            track["targets"]["vocals"] = ["missing.wav"]
            with self.assertRaises(sf.LibsndfileError):
                audit_tracks(manifest, [track])

    def test_runner_writes_outputs_reorders_and_rejects_overwrite(self):
        class VocalOnly(Echo):
            def forward(self, x):
                y = torch.zeros(x.shape[0], 6, 2, x.shape[-1])
                y[:, NAMES.index("vocals")] = x
                return y
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "model.yaml").write_text(yaml.safe_dump(config()))
            args = SimpleNamespace(config=str(root / "model.yaml"), manifest=str(root / "manifest.json"),
                checkpoint="fake-not-loaded", backend_cache="fake-not-loaded", output=str(root / "out"),
                device="cpu", precision="fp32", expected_tracks=1, limit_tracks=0, max_seconds=0,
                cpu_threads=1, save_audio_tracks=1, preview_seconds=0.001)
            with patch("bsr_probe.baseline.load_backend", return_value=(VocalOnly(), {"strict_load": True})):
                run(args)
                summary = json.loads((root / "out/summary.json").read_text())
                self.assertTrue(summary["complete"])
                self.assertGreater(summary["sources"]["vocals"]["si_sdr_db"]["mean"], 100)
                self.assertEqual(summary["sources"]["bass"]["status_counts"], {"silent_target": 1})
                self.assertEqual(len(list((root / "out/audio/song").glob("*.wav"))), 13)
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    run(args)
                args.output = str(root / "failed")
                with patch("bsr_probe.baseline.load_backend", side_effect=RuntimeError("strict-load failure")):
                    with self.assertRaisesRegex(RuntimeError, "strict-load"):
                        run(args)
                self.assertTrue((root / "failed/failure.json").is_file())
                self.assertFalse((root / "failed/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
