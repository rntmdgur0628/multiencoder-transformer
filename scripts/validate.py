"""Generate local implementation-verification artifacts, never quality results."""
import contextlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import torch

from msr.cli import main, source_hash


def validate():
    root = Path(__file__).resolve().parents[1]
    output = root / "validation_artifacts"
    output.mkdir(exist_ok=True)
    test_stream = io.StringIO()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(stream=test_stream, verbosity=2).run(suite)
    (output / "tests.log").write_text(test_stream.getvalue())
    print(test_stream.getvalue())
    summary = {"scope": "synthetic implementation verification only", "python": platform.python_version(),
               "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu",
               "source_sha256": source_hash(), "tests": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "test_seconds": time.monotonic() - started, "smoke": []}
    if result.wasSuccessful():
        for arguments in (["smoke"], ["smoke", "--config", str(root / "configs" / "moises6.json"), "--samples", "4096"]):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                main(arguments)
            summary["smoke"].append(json.loads(stream.getvalue()))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(validate())
