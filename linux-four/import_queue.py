"""Durable private song requests. Only the worker holding a job's lease may prepare it."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def save(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        try:
            json.dump(value, tmp, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(tmp.name, path)
            sync_dir(path.parent)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path):
    return json.loads(path.read_text())


def job_root(imports):
    library = imports.resolve()
    # Unscoped historical folders carry no reliable destination. Only adopt
    # them when the operator has identified their library.
    legacy = library.parent / "jobs"
    marker = legacy / ".library.json"
    try:
        if marker.is_file() and read(marker) == str(library):
            return legacy
    except (OSError, ValueError):
        pass
    return library.parent / ".songs.requests" if library.name == "songs" else library / ".requests"


def enqueue(imports, source, notes, model, bars, sections):
    root = job_root(imports)
    root.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="song-", dir=root))
    save(job / "request.json", dict(source=source, notes=notes, model=model, bars=bars,
                                    sections=sections, created=time.time()))
    sync_dir(root)
    return job


def status(job):
    path = job / "status.json"
    return read(path) if path.is_file() else {"state": "queued", "attempt": 0}


def requests(imports):
    records = {}
    for path in job_root(imports).glob("*/request.json"):
        job = path.parent
        try:
            item = read(path)
            if (not isinstance(item, dict)
                    or not all(isinstance(item.get(k), str) for k in ("source", "notes", "model"))
                    or not isinstance(item.get("created"), (float, int))
                    or item.get("bars") not in (4, 8, 16, 32) or type(item.get("sections")) is not bool):
                raise ValueError("Incomplete song request")
        except (OSError, ValueError):
            records[job] = dict(source=job.name, notes="", created=0, state="invalid")
            continue
        try:
            state = status(job)
            if (state["state"] not in ("running", "queued", "failed", "cancelled", "ready")
                    or type(state["attempt"]) is not int):
                raise ValueError("Invalid import status")
            item["state"] = state["state"]
            item["attempt"] = state["attempt"]
        except (ValueError, TypeError, KeyError, OSError):
            item["state"] = "failed"
        if (imports / job.name / "song.json").is_file():
            item["state"] = "ready"
        elif item["state"] == "ready":
            item["state"] = "failed"
        elif (job / "cancel").exists():
            with lease(job) as lock:
                item["state"] = "cancelling" if lock is None else "cancelled"
        records[job] = item
    return dict(sorted(records.items(), key=lambda pair: pair[1]["created"]))


@contextmanager
def lease(job, name=".worker.lock", *, wait=False):
    with (job / name).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield lock


def retry(job):
    with lease(job) as lock:
        if lock is None:
            return False
        save(job / "status.json", {"state": "queued", "attempt": 0})
        if (job / "cancel").exists():
            (job / "cancel").unlink()
            sync_dir(job)
        return True


def cancel(job):
    save(job / "cancel", True)


def launch(job):
    cmd = [sys.executable, str(Path(__file__).resolve()), str(job)]
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    with (job / "worker.log").open("a") as log:
        if runtime and (Path(runtime) / "systemd/private").exists():
            if not shutil.which("systemd-run"):
                raise OSError("The background worker could not start")
            unit = "karaoke-import-" + hashlib.sha256(os.fsencode(job.resolve())).hexdigest()[:24]
            # Type=exec acknowledges only after the independent service executes.
            # Forward variable names, never secret values on a command line.
            skip = {"INVOCATION_ID", "JOURNAL_STREAM", "SYSTEMD_EXEC_PID", "NOTIFY_SOCKET",
                    "WATCHDOG_PID", "WATCHDOG_USEC", "LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES"}
            env = ["--setenv=" + k for k in os.environ if k not in skip and k.isidentifier()]
            start = ["systemd-run", "--user", "--collect", "--quiet", "--service-type=exec",
                     "--unit=" + unit, "--working-directory=" + os.getcwd(),
                     "--property=StandardOutput=append:" + str(job / "worker.log"),
                     "--property=StandardError=inherit", *env, "--", *cmd]
            try:
                if subprocess.run(start, stdout=log, stderr=log, timeout=3).returncode == 0:
                    return None
            except subprocess.TimeoutExpired:
                pass
            # Concurrent desktops can race to start the same named service.
            try:
                active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit],
                                        stdout=log, stderr=log, timeout=3)
                if active.returncode == 0:
                    return None
            except subprocess.TimeoutExpired:
                pass
            raise OSError("The background worker could not start")
        return subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)


def supervise(job, *, wait=False):
    from library import IMPORTS

    if job.parent.resolve() != job_root(IMPORTS):
        raise ValueError("Request belongs to a different library")
    with lease(job, wait=wait) as lock:
        if lock is None:
            return
        previous = status(job)
        if (job / "cancel").exists() or previous["state"] not in ("queued", "running"):
            return
        if (IMPORTS / job.name / "song.json").is_file():
            save(job / "status.json", dict(state="ready", attempt=previous["attempt"]))
            return
        # Every acknowledged addition has a detached worker, including jobs
        # waiting their turn when the desk closes. Serialize CPU-heavy preparation.
        while not (job / "cancel").exists():
            with lease(job.parent, ".preparation.lock") as slot:
                if slot:
                    prepare_with_lease(job, previous["attempt"], (lock.fileno(), slot.fileno()))
                    return
            time.sleep(.3)


def prepare_with_lease(job, previous_attempt, lock_fds):
    from library import IMPORTS

    attempt = previous_attempt + 1

    def update(state, message):
        save(job / "status.json", dict(state=state, message=message, attempt=attempt))

    if attempt > 3:
        update("failed", "Preparation was interrupted three times. Retry to continue.")
        return
    update("running", "")
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("import_job.py")), "--job", str(job)]
    try:
        # A supervisor crash must not allow a second writer while preparation continues.
        with subprocess.Popen(cmd, start_new_session=True, pass_fds=lock_fds,
                              env={**os.environ, "KARAOKE_IMPORT_PARENT": str(os.getpid()),
                                   "KARAOKE_IMPORT_LEASE": " ".join(map(str, lock_fds))}) as proc:
            # Keep the leader's PID reserved until its helpers have been signalled.
            # A killed preparer can otherwise strand a stalled Codex holding the leases.
            while not (job / "cancel").exists():
                if os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                    break
                time.sleep(.3)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if (IMPORTS / job.name / "song.json").is_file():
            update("ready", "")
        elif (job / "cancel").exists():
            update("cancelled", "Cancelled")
        elif proc.returncode < 0 or proc.returncode == 75:
            update("running", "Interrupted")
        else:
            error = job / "error.json"
            update("failed", read(error) if error.is_file() else f"Worker exited {proc.returncode}. See worker.log.")
    except OSError as exc:
        update("failed", str(exc))

if __name__ == "__main__":
    supervise(Path(sys.argv[1]))
