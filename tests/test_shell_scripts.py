from pathlib import Path
import subprocess
import unittest


class ShellScriptTests(unittest.TestCase):
    def test_docker_files_and_scripts_parse(self):
        root = Path(__file__).parents[1]
        self.assertIn("FROM ${BASE_IMAGE}", (root / "Dockerfile").read_text())
        for script in ("build_local_image.sh", "run_local_container.sh"):
            path = root / "scripts" / script
            self.assertTrue(path.is_file())
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_runner_declares_explicit_mounts_and_preflight(self):
        text = (Path(__file__).parents[1] / "scripts" / "run_local_container.sh").read_text()
        for required in ("MOISES_DIR", "CACHE_DIR", "RUN_DIR", "/workspace/data/moises:ro",
                         "/workspace/cache", "/workspace/runs", "torch.cuda.is_available"):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
