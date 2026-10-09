"""Record one guarded foreground command on POSIX; never a scientific acceptance claim.

Children must join their descendants and must not detach process groups. Full
campaign measurement uses its separately reviewed driver and supervisor.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value
    def constant(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def guard(contract, base):
    require(contract["schema"] == 1, "unsupported schema")
    require(isinstance(contract["argv"], list) and contract["argv"] and
            all(isinstance(v, str) and v for v in contract["argv"]), "argv must be a nonempty string vector")
    require(Path(contract["argv"][0]).is_absolute(), "executable path must be absolute")
    limit = contract["timeout_seconds"]
    require(type(limit) in (int, float) and math.isfinite(limit) and limit > 0, "invalid deadline")
    require(contract["inputs"], "immutable input manifest required")
    paths = []
    for item in contract["inputs"]:
        path = (base / item["path"]).resolve()
        require(path not in paths, "duplicate input path")
        paths.append(path)
        require(digest(path) == item["sha256"], "immutable input changed: " + str(path))


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup_group(process):
    """Bound termination separately from direct reaping; zombies can keep a group present."""
    started = time.monotonic()
    deadline = started+6
    result = dict(term_sent=False, kill_sent=False, direct_child_reaped=False)
    for signum, field in ((signal.SIGTERM, "term_sent"), (signal.SIGKILL, "kill_sent")):
        try:
            os.killpg(process.pid, signum)
            result[field] = True
        except ProcessLookupError:
            pass
        if signum == signal.SIGTERM and result[field]:
            time.sleep(1)
    try:
        process.wait(timeout=max(.001, deadline-time.monotonic()))
        result["direct_child_reaped"] = True
    except subprocess.TimeoutExpired:
        pass
    while group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(min(.02, max(0, deadline-time.monotonic())))
    result.update(group_present_after_cleanup=group_exists(process.pid),
                  elapsed_seconds=time.monotonic()-started)
    return result


def run(manifest, output):
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=1, status="REFUSED", child_exit=None, pid=None,
                   started_wall_ns=time.time_ns(), manifest_path=str(manifest))
    started = time.monotonic()
    process = None
    previous = {}
    pending_signal = None
    def interrupted(signum, frame):
        # Never interrupt Popen or cleanup before the owned handle is registered.
        nonlocal pending_signal
        pending_signal = signum
    try:
        data = manifest.read_bytes()
        receipt["manifest_sha256"] = hashlib.sha256(data).hexdigest()
        (output / "manifest.json").write_bytes(data)
        contract = strict_json(data)
        guard(contract, manifest.parent)
        cwd = (manifest.parent / contract["cwd"]).resolve()
        require(cwd.is_dir(), "working directory missing")
        receipt.update(argv=contract["argv"], cwd=str(cwd), timeout_seconds=contract["timeout_seconds"])
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            try:
                process = subprocess.Popen(contract["argv"], cwd=cwd, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                receipt.update(pid=process.pid, status="RUNNING")
                deadline = time.monotonic()+contract["timeout_seconds"]
                while True:
                    if pending_signal is not None:
                        receipt.update(status="INTERRUPTED", interruption_signal=pending_signal)
                        break
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        receipt["status"] = "TIMEOUT"
                        break
                    try:
                        receipt["child_exit"] = process.wait(timeout=min(.1, remaining))
                        receipt["status"] = "PASS" if process.returncode == 0 else "FAIL"
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process is not None:
                    if receipt["status"] == "PASS" and group_exists(process.pid):
                        receipt.update(status="ERROR", error="foreground descendants outlived direct child")
                    receipt["group_cleanup"] = cleanup_group(process)
                    receipt["child_exit"] = process.returncode
                    if (not receipt["group_cleanup"]["direct_child_reaped"] or
                            receipt["group_cleanup"]["group_present_after_cleanup"]):
                        receipt.update(status="ERROR", error="bounded group cleanup not confirmed")
        try:
            require(digest(manifest) == receipt["manifest_sha256"], "command manifest changed")
            guard(contract, manifest.parent)
        except (ValueError, OSError) as error:
            receipt.update(status="INPUT_CHANGED", error=str(error))
    except (ValueError, OSError, KeyError, TypeError) as error:
        receipt.update(status="REFUSED" if process is None else "ERROR", error=str(error))
    finally:
        if pending_signal is not None and receipt["status"] == "PASS":
            receipt.update(status="INTERRUPTED", interruption_signal=pending_signal)
        receipt.update(elapsed_seconds=time.monotonic()-started, ended_wall_ns=time.time_ns())
        receipt["evidence"] = [dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
                               for p in sorted(output.iterdir()) if p.is_file()]
        with (output / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.output))
