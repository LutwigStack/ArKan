"""Offline script fixtures only: no ArKan build, application, or benchmark."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("check_command.py")


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.write_text("fixed input")

    def run_case(self, code, timeout=2, changed=False):
        self.assertTrue(SCRIPT.is_file(), "guarded command runner is missing")
        contract = dict(schema=1, cwd=str(self.root), argv=[sys.executable, "-c", code],
                        timeout_seconds=timeout, inputs=[dict(path="input", sha256=hashlib.sha256(b"fixed input").hexdigest())])
        manifest = self.root / "check.json"
        manifest.write_text(json.dumps(contract))
        if changed:
            self.source.write_text("changed")
        result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest),
                                 "--output", str(self.root / "evidence")], capture_output=True, timeout=5)
        receipt = json.loads((self.root / "evidence" / "receipt.json").read_text())
        return result, receipt

    def test_nonzero_child_keeps_partial_output_and_exit_status(self):
        result, receipt = self.run_case("import sys; print('partial', flush=True); print('failure', file=sys.stderr); sys.exit(7)")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["status"], "FAIL")
        self.assertEqual(receipt["child_exit"], 7)
        self.assertEqual((self.root / "evidence" / "stdout.log").read_text(), "partial\n")
        self.assertEqual((self.root / "evidence" / "stderr.log").read_text(), "failure\n")

    def test_deadline_terminates_child_and_keeps_partial_output(self):
        result, receipt = self.run_case("import time; print('before timeout', flush=True); time.sleep(20)", .2)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["status"], "TIMEOUT")
        self.assertIsInstance(receipt["child_exit"], int)
        self.assertEqual((self.root / "evidence" / "stdout.log").read_text(), "before timeout\n")
        self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists(), "timed out child was not reaped")

    def test_deadline_kills_child_that_ignores_termination(self):
        result, receipt = self.run_case("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(20)", .2)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["status"], "TIMEOUT")
        self.assertEqual(receipt["child_exit"], -9)
        self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists())

    def test_launch_failure_keeps_refusal_receipt(self):
        self.assertTrue(SCRIPT.is_file())
        manifest = self.root / "check.json"
        manifest.write_text(json.dumps(dict(schema=1, cwd=str(self.root), argv=[str(self.root / "missing")],
                                            timeout_seconds=1, inputs=[dict(path="input", sha256=hashlib.sha256(b"fixed input").hexdigest())])))
        result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest),
                                 "--output", str(self.root / "evidence")], capture_output=True, timeout=5)
        receipt = json.loads((self.root / "evidence" / "receipt.json").read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["status"], "REFUSED")
        self.assertIsNone(receipt["pid"])

    def test_duplicate_manifest_key_refuses_before_launch(self):
        self.assertTrue(SCRIPT.is_file())
        manifest = self.root / "check.json"
        manifest.write_text('{"schema":1,"schema":1}')
        result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest),
                                 "--output", str(self.root / "evidence")], capture_output=True, timeout=5)
        receipt = json.loads((self.root / "evidence" / "receipt.json").read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["status"], "REFUSED")
        self.assertIsNone(receipt["pid"])

    def test_input_mismatch_refuses_before_child_side_effect(self):
        result, receipt = self.run_case("from pathlib import Path; Path('launched').touch()", changed=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["status"], "REFUSED")
        self.assertIsNone(receipt["pid"])
        self.assertFalse((self.root / "launched").exists())

    def test_mutation_after_launch_invalidates_success(self):
        result, receipt = self.run_case("from pathlib import Path; Path('input').write_text('changed')")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["child_exit"], 0)
        self.assertEqual(receipt["status"], "INPUT_CHANGED")

    def test_success_receipt_cannot_overwrite_or_repeat_existing_evidence(self):
        result, receipt = self.run_case("print('complete')")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(receipt["status"], "PASS")
        before = (self.root / "evidence" / "receipt.json").read_bytes()
        rerun = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(self.root / "check.json"),
                                "--output", str(self.root / "evidence")], capture_output=True, timeout=5)
        self.assertNotEqual(rerun.returncode, 0)
        self.assertEqual((self.root / "evidence" / "receipt.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
