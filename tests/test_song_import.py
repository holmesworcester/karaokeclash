"""Exercise Add song, the detached worker, Codex handoff and recovery through the GTK list."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
import import_queue
import library
from backing_loops import backing_loops
from import_job import resolve_recording, run
from routing import SongLibrary
from song_import import SongImport
from test_playback import write_tones


# The remote model is the only stand-in. Its executable receives the desk's
# actual argv, skill and prompt, and returns files consumed by the real publisher.
CODEX = """#!/usr/bin/python3
import json, os, shutil, sys, time
from pathlib import Path
job = Path(sys.argv[sys.argv.index("-C") + 1])
prompt = sys.stdin.read()
with (job / "calls").open("a") as out:
    print(json.dumps([sys.argv[1:], prompt]), file=out)
fixture = Path(os.environ["IMPORT_FIXTURE"])
if "Song requested:" in prompt:
    shutil.copyfile(fixture / "backing.wav", job / "recording.wav")
else:
    (job / "enter.tmp").write_text(str(os.getpgrp()))
    (job / "enter.tmp").replace(job / "entered")
    if (fixture / "fail").exists():
        sys.exit(23)
    while not (fixture / "release").exists():
        time.sleep(.03)
    for filename in ("guide.wav", "drums.wav", "backing.wav", "analysis.json"):
        shutil.copyfile(fixture / filename, job / filename)
"""


class TestDesk:
    def __init__(self):
        self.library = SongLibrary()
        self.jobs = SongImport(self)
        self.reload_songs()

    def reload_songs(self):
        self.songs = library.catalog()
        self.library.show_catalog(self.songs, backing_loops(self.songs))

    def close(self):
        self.jobs.close()
        self.jobs.destroy()
        self.library.destroy()


class SongRequests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.imports = self.root / "songs"
        self.fake = self.root / "codex"
        self.fake.write_text(CODEX)
        self.fake.chmod(0o700)
        skill = self.root / "SKILL.md"
        skill.write_text("Karaoke skill fixture: verify each stem.")
        paths = [str(Path(__file__).resolve().parents[1] / "linux-four"),
                 str(Path(library.__file__).parent), os.environ.get("PYTHONPATH", "")]
        self.user_runtime = os.environ.get("XDG_RUNTIME_DIR", "")
        self.enterContext(patch.dict(os.environ, KARAOKE_IMPORTS=str(self.imports),
            KARAOKE_CODEX=str(self.fake), KARAOKE_SKILL=str(skill), IMPORT_FIXTURE=str(self.root),
            PYTHONPATH=os.pathsep.join(paths), XDG_RUNTIME_DIR=str(self.root)))
        self.enterContext(patch("song_import.IMPORTS", self.imports))
        self.enterContext(patch.object(library, "IMPORTS", self.imports))
        for name, hz in [("backing", [220, 110]), ("drums", [110]), ("guide", [660])]:
            write_tones(self.root / (name + ".wav"), 8, hz)
        self.meta = dict(title="Recovered edition", artist="Singer", key="C major", meter=4,
                    bpm=120, review="Fixture", downbeats=[0, 2, 4, 6, 8], fullStartMs=350,
                    rows=[[["Found"], [[500, 900]]]],
                    teaser=dict(label="Teaser phrase", start_bar=0, bars=4, voice_ms=[400, 1000]),
                    hooks=[dict(label="Whole chorus", start_bar=0, bars=4, voice_ms=[0, 8000])])
        (self.root / "analysis.json").write_text(json.dumps(self.meta))
        self.desks = []
        self.procs = set()
        self.addCleanup(self.shutdown)

    def open(self):
        desk = TestDesk()
        self.desks.append(desk)
        return desk

    def shutdown(self):
        (self.root / "release").touch()
        for desk in self.desks:
            if desk.jobs.timer:
                desk.close()
            self.procs.update(desk.jobs.workers.values())
        for proc in self.procs:
            proc.wait(timeout=45)

        # A desk killed by systemd leaves no Popen handle in this interpreter.
        for job in import_queue.job_root(self.imports).glob("song-*"):
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                with import_queue.lease(job) as lock:
                    if lock:
                        break
                time.sleep(.05)
            else:
                self.fail(f"Worker still holds {job.name} after test cleanup")

    def until(self, desk, predicate):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            desk.jobs.poll()
            self.procs.update(desk.jobs.workers.values())
            if predicate():
                return
            time.sleep(.05)
        logs = [p.read_text()[-1200:] for p in import_queue.job_root(self.imports).glob("*/worker.log")]
        self.fail(f"Import timed out: {desk.jobs.progress.get_text()} {logs}")

    def submit(self, desk):
        desk.jobs.source.set_text("miss new booty")
        desk.jobs.notes.set_text("Teaser with a break. Full without intro.")
        desk.jobs.model.set_text("chosen-model")
        desk.jobs.bars.set_active_id("4")
        desk.jobs.action.clicked()
        job, = desk.library.request_rows
        row = desk.library.request_rows[job]
        self.assertFalse(row.get_selectable())
        self.assertIn("In progress", row.get_accessible().get_name())
        self.assertIn("Teaser with a break.", row.get_accessible().get_name())
        details = row.get_child().get_children()[1]
        prompt, = [w for w in details.get_children() if isinstance(w, Gtk.Expander)]
        self.assertFalse(prompt.get_expanded())
        prompt.set_expanded(True)
        self.assertEqual(prompt.get_child().get_text(), "Teaser with a break. Full without intro.")
        return job

    def test_reopen_and_worker_crash_resume_and_publish_one_assignable_song(self):
        desk = self.open()
        job = self.submit(desk)
        self.until(desk, lambda: (job / "entered").exists())
        desk.close()
        replacement = self.open()
        replacement.jobs.poll()
        self.assertIn(job, replacement.library.request_rows)
        self.assertFalse(replacement.jobs.workers)  # Reattached to the original request.
        # A prior worker can leave a partial render before the atomic rename.
        staging = self.imports / (".publish-" + job.name) / "song"
        staging.mkdir(parents=True)
        (staging / "song.json").write_text("partial")
        self.assertNotIn(job.name + "/full", library.catalog())
        os.killpg(int((job / "entered").read_text()), signal.SIGKILL)
        self.until(replacement, lambda: len((job / "calls").read_text().splitlines()) >= 3)
        (self.root / "release").touch()
        key = job.name + "/full"
        self.until(replacement, lambda: key in replacement.songs)
        self.assertFalse(replacement.library.request_rows)
        self.assertEqual([p.name for p in self.imports.iterdir()], [job.name])
        calls = [json.loads(line) for line in (job / "calls").read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum("Song requested:" in prompt for _, prompt in calls), 1)
        for args, prompt in calls:
            self.assertEqual(args[0], "exec")
            self.assertEqual(args[args.index("-m") + 1], "chosen-model")
            self.assertIn("Karaoke skill fixture", prompt)
            self.assertIn("Teaser with a break. Full without intro.", prompt)
        song = replacement.songs[key]
        self.assertEqual(song["fullStartMs"], 350)
        self.assertEqual(song["importRequest"]["source"], "miss new booty")
        replacement.library.search.set_text("without intro")
        self.assertTrue(replacement.library.items[key].get_child_visible())
        self.assertIn(song["importRequest"]["notes"], replacement.library.items[key].get_tooltip_text())
        # Lose the final status write after the atomic catalog rename.
        import_queue.save(job / "status.json", {"state":"running", "attempt":2})
        self.assertEqual(import_queue.requests(self.imports)[job]["state"], "ready")
        self.assertEqual(len(list(self.imports.glob("*/song.json"))), 1)

    def test_isolated_destinations_and_damaged_neighbor_records(self):
        foreign = self.root / "jobs" / "unbound"
        foreign.mkdir(parents=True)
        import_queue.save(foreign / "request.json", dict(source="legacy song", notes="", model="", bars=4,
                                                         sections=False, created=0))
        for destination in (self.imports, self.root/"second"):
            import_queue.enqueue(destination, destination.name, "private prompt", "", 4, False)
        broken = import_queue.enqueue(self.imports, "Broken", "", "", 4, False)
        (broken / "request.json").write_text("{")
        bad_state = import_queue.enqueue(self.imports, "Recoverable", "notes", "", 4, False)
        (bad_state / "status.json").write_text("[]")
        seen = import_queue.requests(self.imports)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[broken]["state"], "invalid")
        self.assertEqual(seen[bad_state]["state"], "failed")
        import_queue.save(bad_state/"status.json", dict(state="failed", attempt=1, created=[]))
        self.assertIsInstance(import_queue.requests(self.imports)[bad_state]["created"], float)
        self.assertTrue(import_queue.retry(bad_state))
        self.assertEqual(import_queue.status(bad_state)["state"], "queued")
        self.assertEqual([r["source"] for r in import_queue.requests(self.root/"second").values()], ["second"])

    def test_unavailable_systemd_service_leaves_a_retryable_request(self):
        socket = self.root / "systemd/private"
        socket.parent.mkdir()
        socket.touch()
        desk = self.open()
        with patch("import_queue.shutil.which", return_value="/usr/bin/systemd-run"), patch(
            "import_queue.subprocess.run", return_value=subprocess.CompletedProcess([], 1)
        ), patch("import_queue.subprocess.Popen") as local:
            job = self.submit(desk)
            desk.jobs.poll()
            local.assert_not_called()
        self.assertEqual(desk.library.request_rows[job].record["state"], "failed")
        self.assertTrue((job / "request.json").is_file())

    def test_cli_waits_for_a_desk_that_already_claimed_its_job(self):
        def claim(*args):
            job = import_queue.enqueue(*args)
            self.procs.add(import_queue.launch(job))
            deadline = time.monotonic() + 15
            while not (job / "entered").exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.05)
            (self.root / "release").touch()
            return job
        with patch("import_job.IMPORTS", self.imports), patch("import_job.enqueue", side_effect=claim):
            target = run("miss new booty", "", 4, "CLI request")
        self.assertTrue((target / "song.json").is_file())

    def test_resume_rejects_external_or_symlink_recording(self):
        job = import_queue.enqueue(self.imports, "song", "", "", 4, False)
        link = job/"recording.wav"
        link.symlink_to(self.root/"backing.wav")
        for name in (str(self.root/"backing.wav"), "../../backing.wav", link.name):
            with self.subTest(name=name):
                import_queue.save(job/"recording.json", name)
                with self.assertRaisesRegex(ValueError, "job-local"):
                    resolve_recording(job, "song", self.fail, self.fail)

    def test_cancel_after_supervisor_crash_releases_queue(self):
        desk = self.open()
        job = self.submit(desk)
        self.until(desk, lambda: (job / "entered").exists())
        desk.jobs.workers[job].kill()
        import_queue.cancel(job)
        self.until(desk, lambda: import_queue.requests(self.imports)[job]["state"] == "cancelled")
        self.assertTrue(import_queue.retry(job))
        (self.root / "release").touch()
        self.until(desk, lambda: job.name + "/full" in desk.songs)

    def test_killing_only_preparer_releases_stalled_codex(self):
        desk = self.open()
        job = self.submit(desk)
        self.until(desk, lambda: (job / "entered").exists())
        os.kill(int((job / "entered").read_text()), signal.SIGKILL)
        self.until(desk, lambda: len((job / "calls").read_text().splitlines()) == 3)
        (self.root / "release").touch()
        self.until(desk, lambda: job.name + "/full" in desk.songs)

    def test_recover_killed_codex_and_replace_incomplete_publication(self):
        desk = self.open()
        job = self.submit(desk)
        self.until(desk, lambda: (job / "entered").exists())
        pipeline = int((job / "entered").read_text())
        live_codex, = Path(f"/proc/{pipeline}/task/{pipeline}/children").read_text().split()
        os.kill(int(live_codex), signal.SIGKILL)
        self.until(desk, lambda: len((job / "calls").read_text().splitlines()) == 3)
        (self.root / "release").touch()
        self.until(desk, lambda: (self.imports / job.name / "song.json").is_file() and not desk.jobs.workers)
        (self.imports / job.name / "song.json").unlink()
        self.assertEqual(import_queue.requests(self.imports)[job]["state"], "failed")
        self.assertTrue(import_queue.retry(job))
        self.until(desk, lambda: (self.imports / job.name / "song.json").is_file())
        self.assertEqual(len((job / "calls").read_text().splitlines()), 3)

    @unittest.skipUnless(os.environ.get("KARAOKE_TEST_IMPORT_SCOPE"), "requires a systemd user manager")
    def test_desk_service_stop_leaves_codex_in_its_own_scope(self):
        os.environ["XDG_RUNTIME_DIR"] = self.user_runtime
        unit = "karaoke-test-" + self.root.name
        launch = """from test_song_import import TestDesk, Gtk
desk = TestDesk()
desk.jobs.source.set_text("miss new booty")
desk.jobs.notes.set_text("Survive service stop")
desk.jobs.bars.set_active_id("4")
desk.jobs.action.clicked()
from pathlib import Path
import os
Path(os.environ["IMPORT_FIXTURE"], "clicked").touch()
Gtk.main()
"""
        keys = ("DISPLAY", "XAUTHORITY", "PYTHONPATH", "KARAOKE_IMPORTS", "KARAOKE_CODEX",
                "KARAOKE_SKILL", "IMPORT_FIXTURE", "OPENBLAS_NUM_THREADS", "TMPDIR")
        env = ["--setenv=" + key for key in keys if key in os.environ]
        subprocess.run(["systemd-run", "--user", "--collect", "--unit=" + unit,
                        *env, sys.executable, "-c", launch], check=True)
        try:
            deadline = time.monotonic() + 30
            while not (self.root / "clicked").exists():
                if time.monotonic() > deadline:
                    log = subprocess.check_output(["journalctl", "--user", "-u", unit, "-n", "20"], text=True)
                    self.fail("Desk did not acknowledge Add song: " + log)
                time.sleep(.02)
        finally:
            subprocess.run(["systemctl", "--user", "stop", unit], check=True)
        while not (entries := list(import_queue.job_root(self.imports).glob("*/entered"))):
            self.assertLess(time.monotonic(), deadline, "Worker did not survive the immediate stop")
            time.sleep(.05)
        entered, = entries
        pid = int(entered.read_text())
        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        self.assertNotIn(unit, cgroup)
        self.assertIn("/karaoke-import-", cgroup)
        job = entered.parent
        self.assertTrue(Path(f"/proc/{pid}").is_dir())
        desk = self.open()
        desk.jobs.poll()
        self.assertFalse(desk.jobs.workers)
        self.assertIn("Survive service stop", desk.library.request_rows[job].get_accessible().get_name())
        (self.root / "release").touch()
        self.until(desk, lambda: job.name + "/full" in desk.songs)
        self.assertEqual(len((job / "calls").read_text().splitlines()), 2)
        self.assertEqual(import_queue.status(job)["attempt"], 1)

    def test_queued_requests_finish_after_desk_closes(self):
        desk = self.open()
        first = self.submit(desk)
        self.until(desk, lambda: (first / "entered").exists())
        desk.jobs.source.set_text("Another recording")
        desk.jobs.notes.set_text("Second request")
        desk.jobs.bars.set_active_id("4")
        desk.jobs.action.clicked()
        second, = set(desk.library.request_rows) - {first}
        self.assertEqual(desk.library.request_rows[second].record["state"], "queued")
        desk.close()
        (self.root / "release").touch()
        for proc in desk.jobs.workers.values():
            self.assertEqual(proc.wait(timeout=45), 0)
        replacement = self.open()
        replacement.jobs.poll()
        self.assertFalse(replacement.library.request_rows)
        for job in (first, second):
            self.assertIn(job.name + "/full", replacement.library.items)
            self.assertEqual(len((job / "calls").read_text().splitlines()), 2)

    def test_failure_cancel_and_retry_keep_prompt_and_settings(self):
        (self.root / "fail").touch()
        desk = self.open()
        job = self.submit(desk)
        self.until(desk, lambda: desk.library.request_rows[job].record["state"] == "failed"
                   and not desk.jobs.workers)
        self.assertIn("Codex exited with status 23", import_queue.read(job / "error.json"))
        self.assertNotIn("workflow.log", desk.library.request_rows[job].get_accessible().get_name())
        request = import_queue.read(job / "request.json")
        (self.root / "fail").unlink()

        def click(label):
            row = desk.library.request_rows[job]
            details = row.get_child().get_children()[1]
            button, = [w for w in details.get_children() if isinstance(w, Gtk.Button)]
            self.assertEqual(button.get_accessible().get_name(), label)
            button.clicked()

        click("Retry")
        self.until(desk, lambda: len((job / "calls").read_text().splitlines()) == 3)
        click("Cancel")
        self.until(desk, lambda: not desk.jobs.workers)
        self.assertEqual(desk.library.request_rows[job].record["state"], "cancelled")
        self.assertFalse(self.imports.exists())
        click("Retry")
        (self.root / "release").touch()
        self.until(desk, lambda: job.name + "/full" in desk.songs)
        self.assertEqual(import_queue.read(job / "request.json"), request)
