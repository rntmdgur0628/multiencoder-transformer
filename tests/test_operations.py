import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from msr import ModelConfig, RestorationModel
from msr.checkpoint import load_checkpoint, save_checkpoint
from msr.compare import compare_results
from msr.data import AudioManifest
from msr.prepare import export_moises


class OperationsTests(unittest.TestCase):
    def test_moises_export_reads_actual_metadata_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "raw" / "provider" / "uuid"
            (native / "vocals").mkdir(parents=True)
            (native / "strings").mkdir()
            sf.write(native / "vocals" / "a.wav", np.ones((101, 1), dtype=np.float32) * 0.1, 4000, subtype="FLOAT")
            sf.write(native / "strings" / "b.wav", np.ones((150, 2), dtype=np.float32) * 0.1, 8000, subtype="FLOAT")
            metadata = {"stems": [{"stemName": name, "tracks": [{"id": recording, "extension": "wav", "trackType": "test"}]}
                                   for name, recording in (("vocals", "a"), ("strings", "b"))]}
            (native / "data.json").write_text(json.dumps(metadata))
            split = root / "splits.json"
            split.write_text(json.dumps({"uuid": {"split": "train", "group": "song-one"}}))
            with contextlib.redirect_stdout(io.StringIO()):
                path = export_moises(root / "raw", split, root / "prepared", 8000)
            manifest = AudioManifest(path)
            track = manifest.tracks[0]
            self.assertEqual(track["length"], 202)
            self.assertTrue(track["targets"]["other"])
            self.assertEqual(track["targets"]["guitar"], [])
            self.assertEqual(sf.info(path.parent / track["targets"]["vocals"][0]).channels, 2)

    def test_corrupt_checkpoint_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            save_checkpoint(path, RestorationModel(ModelConfig.tiny()), stage="pretrain", step=0)
            payload = torch.load(path, weights_only=True)
            payload["config"]["angular_span"] = 1.0
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "configuration hash"):
                load_checkpoint(path)

    def test_paired_comparison_clusters_repeated_song_views(self):
        base = {"manifest_sha256": "a", "audio_sha256": "b", "core_samples": 64, "split": "test", "metric_scope": "local", "tracks": []}
        for i, group in enumerate(("song1", "song1", "song2")):
            base["tracks"].append({"id": str(i), "group": group, "sources": {"vocals": {"sisdr_db": 1.0}}})
        candidate = json.loads(json.dumps(base))
        for t in candidate["tracks"]:
            t["sources"]["vocals"]["sisdr_db"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.json", Path(directory) / "b.json"
            a.write_text(json.dumps(base))
            b.write_text(json.dumps(candidate))
            result = compare_results(a, b)
            self.assertEqual(result["effects"]["vocals"]["songs"], 2)
            self.assertEqual(result["effects"]["vocals"]["song_macro_effect"], 1.0)
            candidate["core_samples"] = 128
            b.write_text(json.dumps(candidate))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                compare_results(a, b)


if __name__ == "__main__":
    unittest.main()
