from gi.repository import Gtk, GLib, Pango

from library import IMPORTS
import import_queue


class SongImport(Gtk.Box):
    def __init__(self, desk):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.desk = desk
        self.source = Gtk.Entry(
            placeholder_text="Song, link or file path", width_chars=10
        )
        self.source.get_accessible().set_name("Song to add")
        self.notes = Gtk.Entry(placeholder_text="Prompt (optional)", width_chars=10)
        self.notes.get_accessible().set_name("Song preparation prompt")
        self.model = Gtk.Entry(placeholder_text="Default Codex model", width_chars=10)
        self.model.get_accessible().set_name("Cloud model")
        self.bars = Gtk.ComboBoxText()
        for n in (4, 8, 16, 32):
            self.bars.append(str(n), f"{n} bars")
        self.bars.set_active_id("32")
        self.bars.get_accessible().set_name("Loop length at 126 BPM")
        self.sections = Gtk.CheckButton(label="Split whole song")
        options = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        for w in (self.model, self.bars, self.sections):
            options.add(w)
        self.options = Gtk.Expander(label="Import options")
        self.options.add(options)
        self.action = Gtk.Button(label="Add song")
        self.action.connect("clicked", self.activate)
        for field in (self.source, self.notes):
            field.connect("activate", lambda *_: self.action.clicked())
        self.progress = Gtk.Label(
            xalign=0, wrap=True, selectable=True, max_width_chars=24
        )
        self.progress.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
        for w in (self.source, self.notes, self.options, self.action, self.progress):
            self.pack_start(w, False, False, 0)
        self.workers = {}
        self.timer = GLib.timeout_add(800, self.poll)
        self.seen_ready = set()

    def activate(self, _):
        source = self.source.get_text().strip()
        if not source:
            self.progress.set_text("Enter a song, link or recording path.")
            return
        try:
            import_queue.enqueue(IMPORTS, source, self.notes.get_text(), self.model.get_text(),
                                 int(self.bars.get_active_id()), self.sections.get_active())
        except OSError as exc:
            self.progress.set_text(f"Could not save request: {exc}")
            return
        self.source.set_text("")
        self.notes.set_text("")
        self.progress.set_text("")
        self.desk.library.search.set_text("")
        self.poll()
        self.desk.library.scroll.get_vadjustment().set_value(0)

    def change_request(self, job, state):
        if state in ("failed", "cancelled"):
            if not import_queue.retry(job):
                self.progress.set_text("The worker is stopping. Retry in a moment.")
                return
        else:
            import_queue.cancel(job)
        self.progress.set_text("")
        self.poll()

    def poll(self):
        try:
            self.refresh()
        except (ValueError, OSError) as exc:
            self.progress.set_text(f"Could not update imports: {exc}")
        return True

    def refresh(self):
        records = import_queue.requests(IMPORTS)
        ready = {job for job, r in records.items() if r["state"] == "ready"}
        new = [job.name + "/full" for job in ready - self.seen_ready
               if job.name + "/full" not in self.desk.songs]
        if new:
            self.desk.reload_songs()
            titles = [self.desk.songs[k]["title"] for k in new if k in self.desk.songs]
            if titles:
                self.progress.set_text("Added " + ", ".join(titles))
        self.seen_ready = ready
        pending = {job: r for job, r in records.items() if job not in ready}
        self.desk.library.show_requests(pending, self.change_request)

        for job, proc in list(self.workers.items()):
            result = proc.poll()
            if result is None:
                continue
            del self.workers[job]
            if result > 0 and job in pending:
                with import_queue.lease(job) as lock:
                    if lock and import_queue.status(job)["state"] in ("running", "queued"):
                        import_queue.save(job / "status.json", dict(state="failed", attempt=0,
                                          message=f"Worker exited {result}. See worker.log."))
                        return
        for job, record in pending.items():
            if job in self.workers or record["state"] not in ("queued", "running"):
                continue
            with import_queue.lease(job) as lock:
                if lock is None:
                    continue
            # The worker claims its own lease. Concurrent desks may both launch,
            # but only one worker can open the recording or contact Codex.
            try:
                proc = import_queue.launch(job)
                if proc is not None:
                    self.workers[job] = proc
            except OSError as exc:
                with import_queue.lease(job) as lock:
                    if lock:
                        import_queue.save(job / "status.json", dict(state="failed", attempt=0, message=str(exc)))

    def close(self):
        if self.timer:
            GLib.source_remove(self.timer)
            self.timer = None
