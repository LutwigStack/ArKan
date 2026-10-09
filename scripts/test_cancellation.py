"""Real Linux Python fixtures for process groups and handled launch interruption.

The test process temporarily adopts its fixture orphans so they can be reaped;
the production utility remains POSIX and never claims to reap orphan descendants.
"""
import ctypes
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).with_name("check_command.py")
EVIDENCE_ROOT = None


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.libc = ctypes.CDLL(None, use_errno=True)
        old = ctypes.c_int()
        self.assertEqual(self.libc.prctl(37, ctypes.byref(old), 0, 0, 0), 0)
        self.assertEqual(self.libc.prctl(36, 1, 0, 0, 0), 0)
        self.addCleanup(lambda: self.libc.prctl(36, old.value, 0, 0, 0))
        self.children = set()
        self.wrapper = None
        self.addCleanup(self.cleanup_children)

    def adopt_and_reap(self):
        for pid in list(self.children):
            try:
                waited, _ = os.waitpid(pid, os.WNOHANG)
                if waited:
                    self.children.remove(pid)
            except ChildProcessError:
                pass

    def cleanup_children(self):
        # RED must be safe too: cleanup only recorded fixture-owned identities.
        for name in ("leader.pid", "descendant.pid"):
            path = self.root / name
            if path.exists():
                self.children.add(int(path.read_text()))
        for pid in self.children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.wrapper is not None and self.wrapper.poll() is None:
            self.wrapper.kill()
        if self.wrapper is not None:
            self.wrapper.wait(timeout=3)
        deadline = time.monotonic()+3
        while self.children and time.monotonic() < deadline:
            self.adopt_and_reap()
            for pid in list(self.children):
                if not Path(f"/proc/{pid}").exists():
                    self.children.remove(pid)
            time.sleep(.01)
        self.assertFalse(self.children, "test fixture cleanup left owned children")
        if EVIDENCE_ROOT is not None:
            with (EVIDENCE_ROOT / self._testMethodName / "fixture-cleanup.json").open("x") as stream:
                json.dump(dict(owned_fixture_children_remaining=sorted(self.children)), stream)

    def write_manifest(self, code, timeout):
        source = self.root / "input"
        source.write_text("fixed")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(schema=1, cwd=str(self.root),
            argv=[sys.executable, "-c", code], timeout_seconds=timeout,
            inputs=[dict(path="input", sha256=hashlib.sha256(b"fixed").hexdigest())])))
        return manifest

    def run_fixture(self, mode):
        descendant = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('descendant.pid').write_text(str(os.getpid())); print('descendant partial',flush=True); time.sleep(30)"
        leader = "import os,subprocess,sys,time; from pathlib import Path; print('leader partial',flush=True); Path('leader.pid').write_text(str(os.getpid())); "
        if mode in ("timeout", "waiting"):
            leader += "p=subprocess.Popen([sys.executable,'-c',"+repr(descendant)+"]); p.wait()"
        else:
            leader += "time.sleep(30)"
        manifest = self.write_manifest(leader, .3 if mode == "timeout" else 20)
        command = [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--output", str(self.root / "evidence")]
        if mode in ("popen", "registered"):
            # Inject an actual handled signal at a deterministic real-child boundary.
            injection = "import importlib.util,os,signal,time; from pathlib import Path; spec=importlib.util.spec_from_file_location('utility',"+repr(str(SCRIPT))+"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            injection += "def ready():\n end=time.monotonic()+2\n while not Path("+repr(str(self.root / 'leader.pid'))+").exists():\n  assert time.monotonic()<end\n  time.sleep(.005)\n"
            if mode == "popen":
                injection += "real=m.subprocess.Popen\ndef launch(*a,**k):\n p=real(*a,**k)\n ready()\n os.kill(os.getpid(),signal.SIGTERM)\n return p\nm.subprocess.Popen=launch\n"
            else:
                injection += "class Receipt(dict):\n def update(self,*a,**k):\n  super().update(*a,**k)\n  if 'pid' in k:\n   ready()\n   os.kill(os.getpid(),signal.SIGTERM)\nm.dict=Receipt\n"
            injection += "raise SystemExit(m.run("+repr(str(manifest))+","+repr(str(self.root / 'evidence'))+"))\n"
            command = [sys.executable, "-c", injection]
        self.wrapper = subprocess.Popen(command, cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic()+9
        interrupted = False
        while self.wrapper.poll() is None and time.monotonic() < deadline:
            for name in ("leader.pid", "descendant.pid"):
                path = self.root / name
                if path.exists():
                    self.children.add(int(path.read_text()))
            if mode == "waiting" and (self.root / "descendant.pid").exists() and not interrupted:
                os.kill(self.wrapper.pid, signal.SIGTERM)
                interrupted = True
            self.adopt_and_reap()
            time.sleep(.01)
        self.assertIsNotNone(self.wrapper.poll(), "utility exceeded bounded fixture deadline")
        receipt = json.loads((self.root / "evidence" / "receipt.json").read_text())
        if EVIDENCE_ROOT is not None:
            saved = EVIDENCE_ROOT / self._testMethodName
            saved.mkdir(parents=True, exist_ok=False)
            shutil.copytree(self.root / "evidence", saved / "utility-evidence")
            states = {}
            for pid in self.children:
                path = Path(f"/proc/{pid}/stat")
                states[str(pid)] = path.read_text().rsplit(")", 1)[1].split()[0] if path.exists() else "absent"
            with (saved / "observation.json").open("x") as stream:
                json.dump(dict(argv=command, wrapper_exit=self.wrapper.returncode, mode=mode,
                               fixture_states_after_receipt=states), stream, indent=2)
        self.assertNotEqual(self.wrapper.returncode, 0)
        self.assertEqual(receipt["status"], "TIMEOUT" if mode == "timeout" else "INTERRUPTED")
        self.assertIn("leader partial", (self.root / "evidence" / "stdout.log").read_text())
        for pid in self.children:
            # Zombies are not executing, but the test still reaps them in teardown.
            status = Path(f"/proc/{pid}/stat")
            if status.exists():
                state = status.read_text().rsplit(")", 1)[1].split()[0]
                self.assertEqual(state, "Z", "owned fixture remained running after utility receipt")
        self.assertTrue(receipt["group_cleanup"]["direct_child_reaped"])
        self.assertFalse(receipt["group_cleanup"]["group_present_after_cleanup"])
        self.assertLess(receipt["group_cleanup"]["elapsed_seconds"], 6.5)
        if mode in ("timeout", "waiting"):
            self.assertTrue(receipt["group_cleanup"]["kill_sent"])
        self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists(), "direct child was not reaped")

    def test_timeout_kills_descendant_after_leader_exits_on_term(self):
        self.run_fixture("timeout")

    def test_waiting_interruption_kills_same_group_descendant(self):
        self.run_fixture("waiting")

    def test_handled_signal_before_popen_returns_is_deferred_until_owned(self):
        self.run_fixture("popen")

    def test_handled_signal_after_handle_registration_reaches_cleanup(self):
        self.run_fixture("registered")

    def run_finalization_fixture(self, mode):
        manifest = self.write_manifest(
            "import os; from pathlib import Path; "
            "Path('leader.pid').write_text(str(os.getpid())); print('completed child', flush=True)", 2)
        # Real child/hash/publication boundaries; only inject the external signal/error.
        injection = f"""
import importlib.util, json, os, signal
from pathlib import Path
spec = importlib.util.spec_from_file_location('utility', {str(SCRIPT)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original_handlers = {{s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}}
def restored():
    return {{str(s): signal.getsignal(s) == h for s, h in original_handlers.items()}}
real_cleanup = m.cleanup_group
def cleanup(process):
    result = real_cleanup(process)
    Path('cleanup-observation.json').write_text(json.dumps(result))
    return result
m.cleanup_group = cleanup
def boundary(phase):
    Path('boundary.json').write_text(json.dumps(dict(phase=phase, restored_handlers=restored())))
if {mode!r} == 'publication':
    real_dump = m.json.dump
    def publish(*args, **kwargs):
        boundary('receipt-publication')
        os.kill(os.getpid(), signal.SIGTERM)
        return real_dump(*args, **kwargs)
    m.json.dump = publish
else:
    real_digest = m.digest
    def digest(path):
        if Path(path).name == 'stdout.log':
            boundary('post-cleanup-stdout-hash')
            if {mode!r} == 'hash-error':
                raise OSError('final stdout hash failure')
            os.kill(os.getpid(), signal.SIGTERM)
        return real_digest(path)
    m.digest = digest
try:
    result = m.run({str(manifest)!r}, {str(self.root / 'evidence')!r})
except OSError as error:
    Path('error.json').write_text(json.dumps(dict(error=str(error), restored_handlers=restored())))
    raise SystemExit(1)
raise SystemExit(result)
"""
        command = [sys.executable, "-c", injection]
        with (self.root / "wrapper.stdout").open("wb") as stdout, (self.root / "wrapper.stderr").open("wb") as stderr:
            self.wrapper = subprocess.Popen(command, cwd=self.root, stdout=stdout, stderr=stderr)
            deadline = time.monotonic()+9
            while self.wrapper.poll() is None and time.monotonic() < deadline:
                path = self.root / "leader.pid"
                if path.exists():
                    self.children.add(int(path.read_text()))
                self.adopt_and_reap()
                time.sleep(.01)
        receipt_path = self.root / "evidence" / "receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            receipt = None
        if EVIDENCE_ROOT is not None:
            saved = EVIDENCE_ROOT / self._testMethodName
            saved.mkdir(parents=True, exist_ok=False)
            if (self.root / "evidence").exists():
                shutil.copytree(self.root / "evidence", saved / "utility-evidence")
            for name in ("boundary.json", "error.json", "cleanup-observation.json", "wrapper.stdout", "wrapper.stderr"):
                if (self.root / name).exists():
                    shutil.copyfile(self.root / name, saved / name)
            (saved / "observation.json").write_text(json.dumps(dict(
                argv=command, mode=mode, wrapper_exit=self.wrapper.returncode,
                complete_receipt=receipt is not None, receipt_status=receipt.get("status") if receipt else None)))
        self.assertIsNotNone(self.wrapper.poll(), "finalization fixture exceeded deadline")
        self.assertNotEqual(self.wrapper.returncode, 0, "interrupted/error finalization reported process success")
        cleanup = json.loads((self.root / "cleanup-observation.json").read_text())
        self.assertTrue(cleanup["direct_child_reaped"])
        self.assertFalse(cleanup["group_present_after_cleanup"])
        self.assertIn("completed child", (self.root / "evidence" / "stdout.log").read_text())
        if mode == "hash-signal":
            self.assertIsNotNone(receipt)
            self.assertEqual(receipt["status"], "INTERRUPTED")
            self.assertEqual(receipt["interruption_signal"], signal.SIGTERM)
        elif mode == "publication":
            self.assertEqual(self.wrapper.returncode, -signal.SIGTERM)
            self.assertIsNone(receipt, "post-handoff termination must leave incomplete evidence")
            boundary = json.loads((self.root / "boundary.json").read_text())
            self.assertTrue(all(boundary["restored_handlers"].values()))
        else:
            error = json.loads((self.root / "error.json").read_text())
            self.assertEqual(error["error"], "final stdout hash failure")
            self.assertTrue(all(error["restored_handlers"].values()), "hash failure leaked deferred handlers")
            self.assertIsNone(receipt)

    def test_sigterm_during_final_stdout_hash_is_interrupted(self):
        self.run_finalization_fixture("hash-signal")

    def test_sigterm_at_receipt_publication_uses_original_handlers(self):
        self.run_finalization_fixture("publication")

    def test_final_stdout_hash_error_restores_both_handlers(self):
        self.run_finalization_fixture("hash-error")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, default=SCRIPT)
    parser.add_argument("--evidence-dir", type=Path)
    args, test_args = parser.parse_known_args()
    SCRIPT = args.script.resolve()
    EVIDENCE_ROOT = args.evidence_dir
    unittest.main(argv=[sys.argv[0]]+test_args)
