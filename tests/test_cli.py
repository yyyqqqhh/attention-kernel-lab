"""Run public commands with site-packages disabled to verify the no-extras path."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def invoke(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-S", "-m", "attention_kernel_lab", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=25,
        check=False,
    )


class CliTests(unittest.TestCase):
    def test_help_without_frameworks(self):
        process = invoke("--help")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("doctor", process.stdout)
        self.assertIn("bench", process.stdout)

    def test_demo_rectangular_causal_and_json_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "nested" / "demo.json"
            process = invoke("demo", "--nq", "9", "--nk", "5", "--dim", "3", "--block-size", "2", "--causal", "--output", str(report))
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result, json.loads(report.read_text(encoding="utf-8")))
            self.assertEqual(result["status"], "passed")
            self.assertLess(result["max_abs_error"], 1e-10)

    def test_doctor_is_read_only_inventory(self):
        process = invoke("doctor", "--json")
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertIn("python", result)
        self.assertIn("packages", result)
        self.assertIn("gpu_preflight_ready", result)
        self.assertIn("not evidence", result["validation_note"])

    def test_missing_optional_framework_has_clear_failure(self):
        for command in ("check", "bench"):
            with self.subTest(command=command):
                process = invoke(command)
                self.assertEqual(process.returncode, 2)
                self.assertIn("PyTorch is unavailable", process.stderr)
                self.assertEqual(process.stdout, "")

    def test_bad_arguments_fail_before_work(self):
        process = invoke("demo", "--nq", "0")
        self.assertEqual(process.returncode, 2)
        self.assertIn("positive", process.stderr)
        process = invoke("demo", "--scale", "nan")
        self.assertEqual(process.returncode, 2)
        self.assertIn("finite", process.stderr)


if __name__ == "__main__":
    unittest.main()
