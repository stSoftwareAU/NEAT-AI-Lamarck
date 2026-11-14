#!/usr/bin/env python3
"""Offline integration checks using real AMM and the real Rust server.

Run in the pinned scratch environment, outside routine CI:
  env PYTHONPATH=... python3 scripts/test-amm-reference.py --bench BIN
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "amm_reference", Path(__file__).with_name("amm-reference.py")
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ReferenceProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="amm-integration-")
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.root = Path(cls.scratch.name)
        subprocess.run([cls.bench, str(cls.root / "smoke"), "--smoke"],
                       check=True, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, timeout=60)
        cls.manifest = str(cls.root / "smoke" / "manifest.json")
        cls.request = ("tanh", 20, 1001, 10)

    def test_real_reference_completes_within_budget(self):
        row = driver.run_one((self.bench, self.manifest, self.request))
        self.assertIsNone(row["error"])
        self.assertLessEqual(row["evaluationsUsed"], 200)
        self.assertLessEqual(row["normalisedScreeningLoss"], 1.0)
        self.assertEqual(row["confirmationEvaluations"], 5)

    def test_missing_executable_becomes_a_failed_row(self):
        row = driver.run_one((str(self.root / "absent"), self.manifest, self.request))
        self.assertIn("FileNotFoundError", row["error"])
        self.assertEqual(row["evaluationsUsed"], 0)
        self.assertIsNone(row["normalisedScreeningLoss"])

    def test_server_failure_before_header_becomes_a_failed_row(self):
        wrapper = self.root / "fail-before-header"
        wrapper.write_text("#!/usr/bin/env python3\nimport sys\n"
                           "print('deliberate header failure', file=sys.stderr)\n"
                           "print('wrapper: request could not finish', file=sys.stderr)\n"
                           "sys.exit(9)\n")
        wrapper.chmod(0o700)
        row = driver.run_one((str(wrapper), self.manifest, self.request))
        self.assertIn("no header from server: deliberate header failure", row["error"])
        self.assertEqual(row["evaluationsUsed"], 0)

    def test_verbose_server_does_not_block_objective_replies(self):
        wrapper = self.root / "verbose-server"
        wrapper.write_text(
            "#!/usr/bin/env python3\nimport subprocess, sys\n"
            "sys.stderr.write('diagnostic padding ' * 65536)\nsys.stderr.flush()\n"
            f"subprocess.run([{json.dumps(self.bench)}, *sys.argv[1:]], check=True)\n"
        )
        wrapper.chmod(0o700)
        row = driver.run_one((str(wrapper), self.manifest, self.request))
        self.assertIsNone(row["error"])
        self.assertEqual(row["confirmationEvaluations"], 5)
        self.assertLessEqual(row["normalisedScreeningLoss"], 1.0)

    def test_nonzero_exit_after_real_row_keeps_costs_and_marks_failure(self):
        wrapper = self.root / "fail-after-row"
        wrapper.write_text(
            "#!/usr/bin/env python3\nimport subprocess, sys\n"
            f"subprocess.run([{json.dumps(self.bench)}, *sys.argv[1:]], check=True)\n"
            "print('deliberate post-row process failure', file=sys.stderr)\n"
            "sys.exit(7)\n"
        )
        wrapper.chmod(0o700)
        control = driver.run_one((self.bench, self.manifest, self.request))
        self.assertIsNone(control["error"])
        row = driver.run_one((str(wrapper), self.manifest, self.request))
        self.assertIn("server exit 7: deliberate post-row process failure", row["error"] or "")
        self.assertEqual(row["evaluationsUsed"], control["evaluationsUsed"])
        self.assertEqual(row["confirmationEvaluations"], 5)
        self.assertEqual(row["normalisedScreeningLoss"], control["normalisedScreeningLoss"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    args = parser.parse_args()
    ReferenceProcessTests.bench = str(Path(args.bench).resolve(strict=True))
    unittest.main(argv=[__file__])
