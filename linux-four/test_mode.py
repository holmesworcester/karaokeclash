"""Route dry rehearsal sources through the live microphone effects."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from gi.repository import Gtk

from audience import MIC_NAMES
from test_singing import prepare_vocal


VOCAL_SOURCES = {"auto": "Auto", "extracted": "Extracted", "computer": "Computer"}


class TestMode:
    def __init__(self, desk):
        self.desk = desk
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.jobs = {}
        self.dry_sources = {}
        self.keys = [None] * 4
        self.pending = [False] * 4
        self.messages = [""] * 4
        self.active = [False] * 4
        self.start_when_ready = False
        self.buttons = []
        self.sources = []
        for lane in range(4):
            button = Gtk.ToggleButton(label="Test vocal")
            button.set_tooltip_text("Play this pane’s assigned vocal through its mic effects")
            button.set_sensitive(False)
            button.connect("toggled", self.toggle, lane)
            self.buttons.append(button)
            source = Gtk.ComboBoxText()
            for key, label in VOCAL_SOURCES.items():
                source.append(key, label)
            source.set_active_id(desk.strips[lane]["test_source"])
            source.get_accessible().set_name(f"{MIC_NAMES[lane]} test vocal source")
            source.set_tooltip_text("Choose the extracted recording or computer-sung lyrics. Auto prefers the recording.")
            source.connect("changed", self.change_source, lane)
            self.sources.append(source)

    def change_source(self, widget, lane):
        self.desk.strips[lane]["test_source"] = widget.get_active_id()
        if self.active[lane]:
            self.desk.configure()
            self.refresh()
        self.desk.save()

    def song_key(self, lane):
        d = self.desk
        backing, _, _ = d.pane_transport(lane)
        # A Hear-only mic auditions the selected cut on the same clock as a singer.
        song = d.stage.panes[lane].song_for(backing, d.bar_pairs.get(lane), d.strips[lane]["anchor"])
        if song and song.get("id") in d.songs:
            return song["id"]
        return None

    def toggle(self, button, lane):
        d = self.desk
        enabled = button.get_active()
        if enabled == self.active[lane]:
            return
        if enabled and (not self.song_key(lane) or not d.open_audio()):
            button.set_active(False)
            return
        self.active[lane] = enabled
        if enabled:
            d.audio.call("hear", lane, -1)
            self.start_when_ready = not d.audio.call("playing")
        elif not any(self.active):
            self.start_when_ready = False
        d.audio.call("test_lanes", sum(1 << i for i, active in enumerate(self.active) if active))
        d.configure()
        self.refresh()

    def refresh(self):
        d = self.desk
        wanted = []
        worker_requests = []
        for lane, strip in enumerate(d.strips):
            name = self.song_key(lane)
            self.buttons[lane].set_sensitive(bool(name) or self.active[lane])
            if not self.active[lane]:
                name = None
            voice = strip["voice"]
            wanted.append((name, strip["test_source"]) if name else None)
            # Both inputs use this strip's live converter. Invalidate outstanding
            # packets when the operator switches between a test source and a mic.
            enabled = bool(name) and d.ready if self.active[lane] else d.live[lane]
            worker_requests.append((voice, d.session.get("gpu_hops", 4), self.active[lane])
                                   if enabled and voice >= 3 else None)
        d.sync_character_workers(worker_requests)
        if any(self.active) and d.ready:
            songs = {key[0] for key in wanted if key}
            for key in list(self.dry_sources):
                if key[0] not in songs:
                    del self.dry_sources[key]
        for key in self.jobs.keys() - set(wanted):
            job, cancel = self.jobs.pop(key)
            cancel.set()
            job.cancel()
        for lane, key in enumerate(wanted):
            binding = (key, id(d.backing)) if key else None
            previous = self.keys[lane]
            if binding != previous:
                if previous and previous[0] != key:
                    d.audio.test_source(lane)
                self.keys[lane] = binding
                self.pending[lane] = bool(key)
                self.messages[lane] = "Preparing test vocal" if key else ""
            if not self.pending[lane]:
                continue
            song = d.songs[key[0]]
            if key not in self.jobs and key not in self.dry_sources:
                cancel = Event()
                self.jobs[key] = self.pool.submit(prepare_vocal, song, cancel, source=key[1]), cancel
            if d.ready and (key in self.dry_sources or self.jobs[key][0].done()):
                self.pending[lane] = False
                try:
                    if key in self.dry_sources:
                        pcm = self.dry_sources[key]
                    else:
                        pcm = self.jobs[key][0].result()
                        self.dry_sources[key] = pcm
                    d.audio.test_source(lane, pcm, song, d.backing)
                    self.messages[lane] = "Test vocal" if len(pcm) else "Instrumental"
                except Exception as error:
                    self.messages[lane] = f"Test vocal unavailable · {error}"
        warming = any(active and status.startswith("Warming GPU")
                      for active, status in zip(self.active, d.worker_status))
        if self.start_when_ready and d.ready and not any(self.pending) and not warming:
            self.start_when_ready = False
            if not d.audio.call("playing"):
                d.play_pause()

    def close(self):
        for job, cancel in self.jobs.values():
            cancel.set()
            job.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
