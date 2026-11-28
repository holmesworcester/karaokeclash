#!/usr/bin/python3
import concurrent.futures
import ctypes as C
import json
import os
from pathlib import Path
import re
import subprocess
import wave
from functools import lru_cache

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gdk, GLib, Gtk
import numpy as np

from clip_clock import ClipClock, launch_origin
from desk_audio import DeskAudio as Audio
from backing_loops import backing_loops, calibrated_catalog, family_key
from character_monitor import character_worker
from audio import IDS, PARAMS, VOICES, decode
from audience import Audience, MIC_NAMES, grid_position, mic_panel, transport_bar
from playback import catalog, playback_files
from routing import SongLibrary, SongTarget
from song_import import SongImport
from test_mode import TestMode, VOCAL_SOURCES
from dial import Dial
from tal_controls import TalControls, TITLES
from midi_versions import VERSIONS
from pane_presets import PanePresets
from library import lyric_position, resolve, scale_mask

STATE = Path(
    os.environ.get(
        "KARAOKE_STATE", Path.home() / ".config/karaoke-playground-four/session.json"
    )
)


def button(text, action):
    b = Gtk.Button(label=text)
    b.connect("clicked", lambda _: action())
    return b


def picker(label, items, select, disabled=lambda _: False):
    b = Gtk.MenuButton(label=label)
    pop = Gtk.Popover.new(b)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=7)
    search = Gtk.SearchEntry(placeholder_text="Filter choices")
    box.pack_start(search, False, False, 0)
    scroll = Gtk.ScrolledWindow(min_content_width=260, min_content_height=235)
    scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    rows = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
    scroll.add(rows)
    box.add(scroll)
    pop.add(box)
    b.set_popover(pop)

    def build(*_):
        for row in rows.get_children():
            rows.remove(row)
        for key, title in items():
            row = Gtk.ListBoxRow()
            row.key = key
            row.title = title.casefold()
            row.add(Gtk.Label(label=title, xalign=0, margin=5))
            row.set_sensitive(not disabled(key))
            rows.add(row)
        rows.show_all()
        search.set_text("")
        search.grab_focus()

    rows.set_filter_func(lambda row: search.get_text().casefold() in row.title)
    search.connect("search-changed", lambda _: rows.invalidate_filter())

    def choose(_, row):
        pop.popdown()
        select(row.key)

    rows.connect("row-activated", choose)
    pop.connect("show", build)
    box.show_all()
    return b


VOCODER_PARAMS = (
    ("Dry / Wet", "%", 0, 100, 100),
    ("Consonants", "%", 0, 100, 50),
    ("Vocal attack", " ms", 1, 80, 5),
    ("Release", " ms", 5, 300, 40),
    ("Formant shift", " st", -12, 12, 0),
    ("Carrier harmonics", "%", 0, 100, 55),
    ("Synth register", "", -2, 1, 0),
    ("Glide", " ms", 0, 240, 100),
)
VOCODER_HINTS = (
    "At 100% only the speech-shaped carrier is heard while a carrier note plays.",
    "Resynthesize S, F and T sounds with noise in the upper speech bands.",
    "How quickly a syllable opens the frequency bands.",
    "How long the bands ring after a syllable. Short times separate rap syllables.",
    "Move the vowel resonances up or down while keeping the carrier notes.",
    "Blend a rounded saw with a brighter saw and pulse for clearer speech.",
    "Transpose the synthesized chord together. Speech range lowers high notes.",
    "Bend connected MIDI melody notes. Zero plays the original scored attacks.",
)
VOCODER_PRESETS = {
    "Tight speech": [100, 75, 2, 20, 0, 80, 1, 100],
    "Vowel stack": [100, 38, 5, 64, 0, 88, 0, 100],
    "Smooth vowels": [100, 30, 12, 95, -2, 35, 0, 100],
}



def strip(lyrics):
    return dict(
        lyrics=lyrics,
        anchor="natural",
        playback_mode="full",
        midi_version="all",
        test_source="auto",
        muted=False,
        voice=0,
        pulse=False,
        delay=False,
        compressor=True,
        reverb=0,
        lead=70,
        stereo_width=100,
        tal_params={},
        vocoder=[default for _, _, _, _, default in VOCODER_PARAMS],
        values=[p[3] for p in PARAMS],
    )


# Negative IDs leave saved character IDs and rehearsal sample slots intact.
VOCODER_HOLD = -1
TUNELET = -2  # Keep saved Tunelet selections on their original engine.
CHORD_TUNE = -4
TAL_VOCODER = -3
VOICE_OPTIONS = dict(enumerate(VOICES))
VOICE_OPTIONS.update({VOCODER_HOLD: "Vocoder Hold", CHORD_TUNE: "Chord Tune (Chorale VST3)", TUNELET: "Chord Tune (Tunelet)", TAL_VOCODER: "TAL Vocoder"})
SCORE_VOICES = (1, VOCODER_HOLD, TAL_VOCODER, CHORD_TUNE, TUNELET)


def carrier_id(source, audio):
    anchor = source["anchor"]
    if anchor in ("natural", "autotune"):
        return source["lyrics"] or ""
    if anchor in ("Lights Burn Dimmer", "My Heart Will Go On") and source["lyrics"]:
        duet = anchor + " / " + source["lyrics"]
        if audio.call("has_carrier", duet.encode()):
            return duet
    return anchor


def settings():
    result = dict(
        strips=[strip(None) for _ in range(4)],
        presets={},
        ceiling=-6,
        routes=[""] * 6,
    )
    try:
        saved = json.loads(STATE.read_text())
        if len(saved["strips"]) != 4:
            raise ValueError()
        for i, s in enumerate(saved["strips"]):
            if s["voice"] not in VOICE_OPTIONS:
                raise ValueError()
            s["lyrics"] = None
            s["anchor"] = "natural"
            if len(s["values"]) != len(PARAMS):
                raise ValueError()
            s["values"] = [
                max(lo, min(hi, float(v)))
                for (_, lo, hi, _), v in zip(PARAMS, s["values"])
            ]
            if s.get("playback_mode") not in ("full", "stripped", "carrier"):
                s["playback_mode"] = "stripped" if saved.get("playback_mode") == "stripped" else "full"
            if s.get("test_source") not in tuple(VOCAL_SOURCES):
                s.pop("test_source", None)
            if s.get("midi_version") not in VERSIONS:
                s["midi_version"] = "all"
            knobs = s.get("vocoder")
            defaults = [spec[-1] for spec in VOCODER_PARAMS]
            if isinstance(knobs, list) and len(knobs) == 4:
                knobs = defaults if knobs == [55, 50, 2, 16] else knobs + defaults[4:]
            if isinstance(knobs, list) and len(knobs) == 7:
                knobs = knobs + defaults[7:]
            if isinstance(knobs, list) and len(knobs) == len(VOCODER_PARAMS):
                s["vocoder"] = [max(lo, min(hi, val)) if isinstance(val, (int, float)) else default
                                for val, (_, _, lo, hi, default) in zip(knobs, VOCODER_PARAMS)]
            else:
                s.pop("vocoder", None)
            params = s.get("tal_params")
            s["tal_params"] = {k: v for k, v in params.items() if isinstance(v, (int, float))
                               and 0 <= v <= 1} if isinstance(params, dict) else {}
            width = s.get("stereo_width", 100)
            s["stereo_width"] = max(0, min(100, width)) if isinstance(width, (int, float)) else 100
            lead = s.get("lead", 70)
            s["lead"] = max(0, min(100, lead)) if isinstance(lead, (int, float)) else 70
            saved["strips"][i] = {**strip(s["lyrics"]), **s}
        saved.pop("backing", None)
        saved.pop("playback_mode", None)
        if len(saved["routes"]) != 6:
            raise ValueError()
        if not isinstance(saved.get("presets"), dict):
            saved["presets"] = {}
        result.update(saved)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return result


def default_output_ports(ports):
    try:
        sink = subprocess.run(
            ["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"], capture_output=True,
            text=True, timeout=2, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        sink = ""
    names = set(re.findall(r'(?:node\.(?:name|description|nick)) = "([^"]+)"', sink))
    devices = {}
    playback = {}
    for port in ports:
        device, _, channel = port.rpartition(":")
        if channel.lower().startswith("playback"):
            playback.setdefault(device, []).append(port)
        match = re.fullmatch(r"playback_(FL|FR|1|2)", channel, re.I)
        if match:
            side = 0 if match[1].upper() in ("FL", "1") else 1
            devices.setdefault(device, ["", ""])[side] = port
    for device, channels in playback.items():
        devices.setdefault(device, channels[:2])
    if not devices:
        return []

    def priority(device):
        return (device in names, len(devices[device]) == 2 and all(devices[device]), "dummy" not in device.lower())

    return devices[max(devices, key=priority)]


def update_label(widget, text):
    if widget.get_label() != text:
        widget.set_label(text)


class Desk:
    def __init__(self):
        self.songs = calibrated_catalog(catalog())
        self.session = settings()
        self.strips = self.session["strips"]
        self.audio = Audio()
        self.audio.resources()
        self.audio.import_scores(self.songs)
        self.audio.call("buffer", self.session.get("gpu_hops", 4))
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.live = [False] * 4
        self.workers = [None] * 4
        self.worker_requests = [None] * 4
        self.epochs = [0] * 4
        self.worker_status = [""] * 4
        self.routes = [""] * 6
        self.ports_known = False
        self.loading = None
        self.ready = False
        self.backing = None
        self.selected_mix = None
        self.request = None
        self.pending = None
        self.source_seconds = 0
        self.loop_choices = {}
        self.slot_clocks = {}
        self.native_panes = {}
        self.actions = {}
        self.nudges = {}
        self.master_clock = None
        self.playing_slot = None
        self.loop_controls = {}
        self.shown_source = (0, None)
        self.backing_slot = None
        self.resume_at = 0
        self.resume_playing = False
        self.bar_pairs = {}
        self.hear_modes = []
        self.pair_controls = []
        self.source_buttons = {}
        self.mix_controls = {}
        self.midi_controls = []
        self.preset_controls = []
        self.closed = False
        self.dirty = 0
        self.muted = False
        self.duration = 0
        self.updating = False
        self.cells = []
        self.vocoder_panels = []
        self.effect_menus = []
        self.window = Gtk.Window(title="Karaoke Playground 4")
        display = self.window.get_display()
        work = (display.get_primary_monitor() or display.get_monitor(0)).get_workarea()
        self.window.set_default_size(min(1750, work.width - 24), min(1160, work.height - 64))
        self.window.connect("destroy", self.close)
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=8)
        # Preserve the full mic racks at smaller desktop sizes.
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.add(outer)
        self.window.add(scroll)
        self.stage = Audience(self)
        header = Gtk.Box(spacing=6)
        self.play = button("Play", self.play_pause)
        self.play.set_tooltip_text("Play or pause (Space)")
        self.library = SongLibrary()
        self.test = TestMode(self)
        self.window.connect("key-press-event", self.transport_key)
        self.stage.connect("key-press-event", self.transport_key)
        self.mute = Gtk.ToggleButton(label="Mute all")
        self.mute.set_active(self.session.get("muted", False))
        self.mute.connect("toggled", self.master)
        self.ceiling = Gtk.SpinButton.new_with_range(-24, -1, 1)
        self.ceiling.set_value(self.session["ceiling"])
        self.ceiling.connect("value-changed", self.master)
        self.loop_style = Gtk.CssProvider()
        self.loop_style.load_from_data(b"""
            button.backing-note:checked {background-image:none;background-color:#294a64;color:white;
                border-color:#172c40;box-shadow:inset 0 3px 3px rgba(0,0,0,0.6);}
            button.clip-control {padding: 2px 5px; min-width: 16px;}
            button.playing {background-image:none;background-color:#c5dfbf;color:#18371d;}
            button.queued {background-image:none;background-color:#ffde98;color:#382808;}
        """)
        for w in [
            self.play,
            self.mute,
            Gtk.Label(label="Ceiling dBFS"),
            self.ceiling,
            button("Audio I/O", self.devices),
            button("Audience ↗", self.show_stage),
        ]:
            header.pack_start(w, False, False, 0)
        outer.pack_start(header, False, False, 0)
        bar = Gtk.Box(spacing=8)
        self.time = Gtk.Label(label="0:00")
        self.seek = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1, 0.01)
        self.seek.set_draw_value(False)
        self.seek.set_hexpand(True)
        self.seek.connect("value-changed", self.seek_change)
        self.tempo = Gtk.SpinButton.new_with_range(20, 999, .1)
        self.tempo.set_digits(1)
        self.tempo.get_accessible().set_name("Global BPM")
        self.tempo.set_tooltip_text("Tempo for all songs and panes")
        saved_bpm = self.session.get("bpm", 126)
        self.tempo.set_value(saved_bpm if isinstance(saved_bpm, (int, float)) else 126)
        self.audio.call("tempo", self.tempo.get_value())
        self.tempo.connect("value-changed", self.tempo_changed)
        self.quant = Gtk.ComboBoxText()
        self.quant.get_accessible().set_name("Launch quantization")
        for ident, title in [("cycle", "Next loop"), ("8", "8 bars"), ("4", "4 bars"), ("2", "2 bars"),
                             ("1", "1 bar"), (".5", "½ bar"), (".25", "¼ bar"),
                             (".125", "⅛ bar"), (".0625", "¹⁄₁₆ bar")]:
            self.quant.append(ident, title)
        if not self.quant.set_active_id(self.session.get("quantization", "cycle")):
            self.quant.set_active_id("cycle")
        self.quant.connect("changed", self.quant_changed)
        bar.add(self.time)
        bar.add(self.seek)
        bar.add(Gtk.Label(label="BPM"))
        bar.add(self.tempo)
        bar.add(Gtk.Label(label="Launch on"))
        bar.add(self.quant)
        outer.pack_start(bar, False, False, 0)
        grid = Gtk.Grid(
            column_homogeneous=True,
            row_homogeneous=False,
            column_spacing=8,
            row_spacing=8,
        )
        body = Gtk.Box(spacing=8)
        outer.pack_start(body, True, True, 0)
        library = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=8, width_request=215
        )
        library.pack_start(self.library, True, True, 0)
        library.add(Gtk.Separator())
        self.jobs = SongImport(self)
        library.pack_start(self.jobs, False, False, 0)
        body.pack_start(library, False, False, 0)
        body.pack_start(grid, True, True, 0)
        for i in range(4):
            frame = Gtk.Frame()
            grid.attach(frame, i % 2, i // 2, 1, 1)
            frame.add(mic_panel(i, self.cell(i)))
        self.status = Gtk.Label(xalign=0)
        self.status.set_ellipsize(3)
        outer.pack_start(self.status, False, False, 0)
        self.info = ""
        self.refresh_lists()
        self.refresh_loop_controls()
        self.window.show_all()
        self.configure()
        self.master()
        self.apply_mutes()
        self.seek.set_sensitive(False)
        GLib.timeout_add(35, self.tick)

    def save(self):
        if self.dirty:
            GLib.source_remove(self.dirty)
        self.dirty = GLib.timeout_add(250, self.flush)

    def flush(self):
        self.dirty = 0
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_name(STATE.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.session, ensure_ascii=False))
        tmp.replace(STATE)
        return False

    def refresh_lists(self):
        self.loops = backing_loops(self.songs)
        self.library.show_catalog(self.songs, self.loops)
        groups = {family_key(self.songs[group]): group for group in self.loops}
        self.song_families = {
            key: groups[family_key(song)] for key, song in self.songs.items()
        }

    def source_button(self, slot):
        first = next(iter(self.source_buttons.values()), None)
        control = Gtk.RadioButton.new_with_label_from_widget(first, "♫")
        control.set_mode(False)
        control.set_valign(Gtk.Align.START)
        control.set_tooltip_text("Use this slot for the backing track")
        style = control.get_style_context()
        style.add_class("backing-note")
        style.add_provider(self.loop_style, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        control.get_accessible().set_name("Audio from " + MIC_NAMES[slot - 1])
        control.connect("toggled", lambda w: self.select_source() if w.get_active() else None)
        self.source_buttons[slot] = control
        return control

    def source_key(self, slot):
        mic = self.strips[slot - 1]
        # Following another mic borrows its vocal score. Backing ownership
        # stays with this slot's directly assigned song or its lyric edition.
        anchor = mic["anchor"]
        return anchor if anchor in self.songs else mic["lyrics"]

    def assignment_changed(self, slot):
        active = next(i for i, w in self.source_buttons.items() if w.get_active())
        if not self.source_key(active) and self.source_key(slot):
            self.source_buttons[slot].set_active(True)
        self.select_source()

    def change_mix(self, widget, slot):
        mode = widget.get_active_id()
        strip = self.strips[slot - 1]
        strip["muted"] = mode == "mute"
        if not strip["muted"]:
            strip["playback_mode"] = mode
        self.apply_mutes()
        if self.source_buttons[slot].get_active():
            self.select_source()
        self.save()

    def apply_mutes(self):
        self.audio.call("mute_lanes", sum(1 << i for i, s in enumerate(self.strips) if s["muted"]))

    def tempo_changed(self, widget):
        bpm = widget.get_value()
        self.audio.call("tempo", bpm)
        self.session["bpm"] = bpm
        self.save()

    def scheduling_time(self):
        period = max(1, self.audio.call("status", 0)) / 48000
        bar = self.conductor_bar() + period * self.tempo.get_value() / 240
        return self.master_clock.seconds(bar)

    def quant_changed(self, widget):
        self.session["quantization"] = widget.get_active_id()
        if self.ready:
            self.sync_launch()
            self.apply_slot_actions()
            future = self.scheduling_time()
            edge = self.master_clock.bar(self.master_clock.next_edge(future, widget.get_active_id()))
            for slot, (key, choice, clock, old_edge) in list(self.actions.items()):
                current = self.slot_clocks.get(slot)
                if current and current[:2] != (key, choice):
                    clock = ClipClock(clock.region, clock.origin + launch_origin(choice, edge) - launch_origin(choice, old_edge))
                self.actions[slot] = (key, choice, clock, edge)
            if self.pending:
                self.cancel_launch()
                self.select_source()
        self.save()

    def loop_control(self, slot):
        box = Gtk.Box(spacing=0)
        box.get_style_context().add_class("linked")
        name = MIC_NAMES[slot - 1]
        back = button("◀", lambda: self.skip_bar(slot, -1))
        modes = {choice: button(initial, lambda c=choice: self.choose_loop(slot, c))
                 for choice, initial in (("teaser", "T"), ("main", "M"), ("full", "F"))}
        forward = button("▶", lambda: self.skip_bar(slot, 1))
        extra = picker("▾", lambda: self.extra_loops(slot),
                       lambda choice: self.choose_loop(slot, choice))
        controls = [(back, "Back one bar on the launch grid"),
                    *((w, c.title() + " loop") for c, w in modes.items()),
                    (forward, "Forward one bar on the launch grid"), (extra, "Other loops")]
        for widget, hint in controls:
            widget.get_accessible().set_name(f"{name} · {hint}")
            widget.set_tooltip_text(hint)
            style = widget.get_style_context()
            style.add_provider(self.loop_style, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            style.add_class("clip-control")
            box.add(widget)
        extra.set_no_show_all(True)
        self.loop_controls[slot] = (modes, back, forward, extra)
        return box

    def extra_loops(self, slot):
        family = self.song_families.get(self.source_key(slot))
        return [(k, item.label) for k, item in self.loops.get(family, {}).items()
                if k not in ("teaser", "main", "full")]

    def loop_choice(self, slot, key):
        saved_key, choice = self.loop_choices.get(slot, (None, None))
        family = self.song_families[key]
        if saved_key == key and choice in self.loops[family]:
            return choice
        return key if key != family else "full"

    @staticmethod
    @lru_cache(maxsize=100)
    def wave_duration(path):
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()

    def slot_region(self, slot, choice):
        key = self.source_key(slot)
        loop = self.loops[self.song_families[key]][choice]
        song = self.songs[loop.source]
        files = playback_files(song)
        path = files[self.strips[slot - 1]["playback_mode"] == "stripped"]
        seconds = self.source_seconds if self.ready and path == self.active_path else self.wave_duration(path)
        return loop.region(song, seconds)

    def conductor_bar(self):
        return self.master_clock.bar(self.elapsed()) if self.ready else 0.0

    def apply_slot_actions(self):
        now = self.conductor_bar()
        for slot, (key, choice, clock, edge) in list(self.actions.items()):
            if key != self.source_key(slot):
                del self.actions[slot]
            elif edge <= now + 1e-8 and slot != self.shown_source[0]:
                self.slot_clocks[slot] = (key, choice, clock)
                del self.actions[slot]

    def choose_loop(self, slot, choice):
        self.queue_move(slot, choice, 0)

    def partner_full_clock(self, slot, region, edge):
        """Join a different singer's Full part at its position on the same recording."""
        for other, current in self.slot_clocks.items():
            if other == slot:
                continue
            move = self.actions.get(other)
            if move and move[3] <= edge + 1e-8:
                current = move[:3]
            key, choice, clock = current
            if (choice == "full" and key == self.source_key(other) and clock.region["id"] != region["id"]
                    and clock.region["track"] == region["track"]
                    and clock.region["start"] == region["start"] and clock.region["end"] == region["end"]):
                return clock
        return None

    def skip_bar(self, slot, bars):
        key = self.source_key(slot)
        if key:
            self.queue_move(slot, self.loop_choice(slot, key), bars)

    def queue_move(self, slot, choice, bars):
        key = self.source_key(slot)
        if not key:
            return
        self.sync_launch()
        self.apply_slot_actions()
        try:
            region = self.slot_region(slot, choice)
        except (OSError, wave.Error) as error:
            self.info = str(error)
            return
        selected = slot == self.shown_source[0]
        now = self.conductor_bar()
        running = self.ready and self.audio.call("playing")
        edge = now
        if running:
            elapsed = self.scheduling_time()
            edge = self.master_clock.bar(self.master_clock.next_edge(elapsed, self.quant.get_active_id()))
        old = self.actions.get(slot)
        current = self.slot_clocks.get(slot)
        if selected and not bars and self.selected_mix and self.selected_mix[:2] == (self.song_families[key], choice):
            current = (key, choice, self.master_clock)
        if not current or current[0] != key:
            default = self.loop_choice(slot, key)
            initial = self.slot_region(slot, default)
            origin = grid_position(initial, initial["start"]) if not self.ready else 0
            if default == "full" and (partner := self.partner_full_clock(slot, initial, now)):
                origin = partner.origin
            current = (key, default, ClipClock(initial, origin))
        if old and old[0] == key and old[3] > now:
            edge = old[3]
        self.slot_clocks[slot] = current
        base = old if old and old[:2] == (key, choice) else current
        origin = base[2].origin if base[:2] == (key, choice) else launch_origin(choice, edge)
        if choice == "full" and not bars and (partner := self.partner_full_clock(slot, region, edge)):
            origin = partner.origin
        clock = ClipClock(region, origin - bars)
        self.loop_choices[slot] = (key, choice)
        if bars and running:
            self.nudges[slot] = (self.nudges.get(slot, 0) if old else 0) + bars
        else:
            self.nudges.pop(slot, None)
        if selected:
            self.cancel_launch()
        if current[:2] == (key, choice) and abs(current[2].origin - clock.origin) < 1e-8:
            self.actions.pop(slot, None)
        else:
            self.actions[slot] = (key, choice, clock, edge)
        if selected:
            self.select_source()
        elif not running:
            self.slot_clocks[slot] = (key, choice, clock)
            self.actions.pop(slot, None)
        self.refresh_loop_controls()

    def pane_transport(self, index):
        """Use the pane's own loop while another source supplies the audio."""
        slot = index + 1
        if slot == self.playing_slot or not self.ready:
            return self.backing, self.elapsed(), self.pending
        key = self.source_key(slot)
        current = self.slot_clocks.get(slot)
        move = self.actions.get(slot)
        if not current or current[0] != key:
            return self.backing, self.elapsed(), self.pending
        clock = current[2]
        now = self.conductor_bar()
        pending = None
        if move and move[0] == key:
            pending = dict(region=move[2].region, boundary=clock.seconds(move[3]),
                           offset=move[2].seconds(move[3]))
        return clock.region, clock.seconds(now), pending

    def update_pane_audio(self):
        for lane in range(4):
            slot = lane + 1
            current = self.slot_clocks.get(slot)
            move = self.actions.get(slot)
            key = self.source_key(slot)
            part = self.songs.get(self.test.song_key(lane), {})
            now = current[2] if current and current[0] == key else None
            then = move[2] if move and move[0] == key else None
            edge = move[3] if then else float("inf")
            signature = (now, then, edge, part.get("track"))
            if self.native_panes.get(slot) != signature:
                self.audio.call("pane_clock", lane, edge, *self.audio.clock_args(now, part.get("track")),
                                *self.audio.clock_args(then, part.get("track")))
                self.native_panes[slot] = signature

    def refresh_loop_controls(self):
        self.update_pane_audio()
        queued = self.pending["target"] if self.pending else self.request
        for slot, (modes, back, forward, extra) in self.loop_controls.items():
            key = self.source_key(slot)
            family = self.song_families.get(key)
            choice = self.loop_choice(slot, key) if family else None
            active = self.playing_slot == slot
            current = self.slot_clocks.get(slot)
            audible = self.selected_mix[1] if active and self.selected_mix else None
            current_choice = audible if active else current[1] if current and current[0] == key else choice
            waiting = slot in self.actions or (slot == self.shown_source[0] and queued)
            for mode, widget in modes.items():
                style = widget.get_style_context()
                for cls, enabled in (("playing", mode == current_choice),
                                     ("queued", waiting and mode == choice)):
                    if enabled:
                        style.add_class(cls)
                    else:
                        style.remove_class(cls)
                widget.set_sensitive(bool(family))
            for arrow, direction in ((back, -1), (forward, 1)):
                queued_arrow = slot in self.actions and self.nudges.get(slot, 0) * direction > 0
                ctx = arrow.get_style_context()
                if queued_arrow:
                    ctx.add_class("queued")
                else:
                    ctx.remove_class("queued")
            back.set_sensitive(bool(key and self.ready))
            forward.set_sensitive(bool(key and self.ready))
            extra.set_visible(bool(family and len(self.loops[family]) > 3))

    def cancel_launch(self):
        if self.loading:
            self.loading.cancel()
            self.loading = None
        if self.pending:
            self.audio.call("cancel")
            self.sync_launch()
        self.pending = None
        self.request = None
        self.info = ""

    def select_source(self):
        slot = next(i for i, w in self.source_buttons.items() if w.get_active())
        key = self.source_key(slot)
        backing_slot = slot if key else None
        route_changed = backing_slot != self.backing_slot
        self.backing_slot = backing_slot
        if not key:
            self.audio.call("backing_enabled", False)
            # An empty backing selector silences accompaniment. Keep an assigned
            # song as the timing reference for the singer panes and live voices.
            candidates = [self.playing_slot, *self.source_buttons]
            slot = next((i for i in candidates if i and self.source_key(i)), slot)
            key = self.source_key(slot)
        family = self.song_families.get(key)
        same_selection = not route_changed and self.shown_source == (slot, key)
        self.shown_source = (slot, key)
        if not key:
            self.cancel_launch()
            self.info = "No backing selected"
            return
        choice = self.loop_choice(slot, key)
        target = (family, choice, self.strips[slot - 1]["playback_mode"])
        if same_selection and (self.request == target or (self.pending and self.pending["target"] == target)):
            return
        self.cancel_launch()
        self.info = ""
        previous = self.slot_clocks.get(slot)
        aligned = (not previous or previous[:2] != (key, choice) or not self.master_clock
                   or abs(previous[2].origin - self.master_clock.origin) < 1e-8)
        if target == self.selected_mix and aligned and slot not in self.actions:
            self.audio.call("active_lane", slot - 1)
            self.audio.call("backing_enabled", self.backing_slot is not None)
            self.playing_slot = slot
            self.slot_clocks[slot] = (key, choice, self.master_clock)
            return
        source = self.songs[self.loops[family][choice].source]
        full, stripped = playback_files(source)
        path = stripped if target[2] == "stripped" else full
        if not path.is_file():
            self.backing_unavailable(f"Missing {target[2]} backing for {source['label']} · {path}")
            return
        self.request = target
        self.info = "Preparing " + source["label"]
        if self.ready and self.active_path == path:
            self.install()
        else:
            self.loading = self.pool.submit(decode, path, 2)

    def reload_songs(self):
        self.songs = calibrated_catalog(catalog())
        self.audio.import_scores(self.songs)
        self.refresh_lists()
        self.configure()

    def pair_panes(self, first, second):
        for index in (first, second):
            pair = self.bar_pairs.get(index)
            if pair:
                for member in (pair["first"], pair["second"]):
                    del self.bar_pairs[member]
        if second is not None:
            key = self.strips[first]["lyrics"] or self.strips[second]["lyrics"]
            pair = dict(first=first, second=second, lyrics=key)
            self.bar_pairs[first] = self.bar_pairs[second] = pair
        self.configure()

    def cell(self, i):
        s = self.strips[i]
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7, margin=7)

        def change(key, v):
            if key in ("lyrics", "anchor"):
                self.set_song_assignment(i, key, v)
            else:
                s[key] = v
            self.configure()
            if key in ("lyrics", "anchor"):
                self.assignment_changed(i + 1)
            self.save()

        def assign(field, key):
            if key not in self.songs:
                return False
            if field == "lyrics" and i in self.bar_pairs:
                self.bar_pairs[i]["lyrics"] = key
            change(field, key)
            return True

        presets = PanePresets(self.session["presets"], self.save, MIC_NAMES[i])
        self.preset_controls.append(presets)
        presets.bind("voice", "Voice", None, lambda: s["voice"], lambda v: self.voice(i, v) if v != s["voice"] else None)
        midi = Gtk.ComboBoxText()
        midi.get_accessible().set_name(MIC_NAMES[i] + " MIDI arrangement")
        midi.set_tooltip_text("Choose the arrangement for the MIDI monitor and scored voices on this pane")
        midi.append("all", VERSIONS["all"])
        midi.set_active_id("all")
        midi.options = {"all": VERSIONS["all"]}
        self.midi_controls.append(midi)
        def choose_version(widget):
            value = widget.get_active_id()
            if value:
                change("midi_version", value)
        midi.handler = midi.connect("changed", choose_version)
        presets.bind("midi_version", "MIDI arrangement", None, lambda: s["midi_version"],
                     lambda v: change("midi_version", v))

        targets = Gtk.Box(spacing=4)
        anchor = SongTarget("Hear", self.library, lambda key: assign("anchor", key))
        lyrics = SongTarget("See", self.library, lambda key: assign("lyrics", key))
        targets.pack_start(anchor, True, True, 0)
        targets.pack_start(lyrics, True, True, 0)
        targets.add(self.loop_control(i + 1))
        targets.pack_end(self.source_button(i + 1), False, False, 0)
        box.pack_start(targets, False, False, 0)
        pairs = [(None, "Unbind panes")] + [
            (j, f"Alternate with {MIC_NAMES[j]}") for j in range(4) if j != i
        ]
        pairing = picker("Alternate bars", lambda: pairs, lambda j: self.pair_panes(i, j))
        self.pair_controls.append(pairing)
        modes = [("natural", "Natural voice"), ("autotune", "Autotune")]
        modes.extend((f"mic:{j}", f"Follow {MIC_NAMES[j]}") for j in range(4) if j != i)
        mode = picker("Hear mode", lambda: modes, lambda v: change("anchor", v),
                      lambda v: v.startswith("mic:") and resolve(self.strips, int(v[4:]), i) is None)

        self.hear_modes.append(mode)

        controls = Gtk.Box(spacing=5)
        effects = picker(
            "", lambda: [(v, VOICE_OPTIONS[v]) for v in (0, *SCORE_VOICES, *range(2, len(VOICES)))], lambda v: self.voice(i, v)
        )
        live = button("Live", lambda: self.monitor(i))
        hear = button("Hear", lambda: self.audition(i))
        meter = Gtk.LevelBar(min_value=0, max_value=1)
        meter.set_mode(Gtk.LevelBarMode.DISCRETE)
        meter.set_max_value(10)
        meter_style = Gtk.CssProvider()
        meter_style.load_from_data(
            b"levelbar trough {padding: 1px; background: #d2d6d1;} "
            b"levelbar block {min-width: 3px; min-height: 5px; margin: 0 1px; border: none;} "
            b"levelbar block.empty {background: #b1b8b0;} "
            b"levelbar block.filled {background: #4f8564;}"
        )
        meter.get_style_context().add_provider(
            meter_style, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        meter.set_valign(Gtk.Align.CENTER)
        meter.set_tooltip_text(f"{MIC_NAMES[i]} input level")
        meter.get_accessible().set_name(f"{MIC_NAMES[i]} input level")
        meter.set_hexpand(True)
        rack = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        rack.set_size_request(210, -1)
        rack.set_hexpand(False)
        box.pack_start(rack, False, False, 0)
        mix = Gtk.ComboBoxText()
        for key, label in (("full", "Full"), ("stripped", "Stripped"),
                           ("carrier", "MIDI carrier"), ("mute", "Mute backing")):
            mix.append(key, label)
        mix.set_active_id("mute" if s["muted"] else s["playback_mode"])
        mix.get_accessible().set_name(MIC_NAMES[i] + " mix")
        mix.set_tooltip_text("Choose the accompaniment, drums, or the pane’s vocoder synth. MIDI carrier plays without a microphone.")
        mix.connect("changed", self.change_mix, i + 1)
        self.mix_controls[i + 1] = mix
        presets.bind("mix", "Backing mix", None, mix.get_active_id, mix.set_active_id)
        for w in [mode, effects, hear, live, meter, pairing]:
            controls.add(w)
        rack.pack_start(controls, False, False, 0)
        knobs = Gtk.Grid(column_homogeneous=True, column_spacing=3, row_spacing=1)
        rack.pack_start(knobs, False, False, 0)
        names = (
            ("pulse", "Pulse", (0, 1, 2, 3, 7)),
            ("delay", "Delay", (4, 5, 6)),
            ("compressor", "Comp", (8, 9, 10, 11, 12, 13)),
        )
        units = (
            "%",
            "%",
            "%",
            "%",
            "beat",
            "%",
            "%",
            "%",
            " dB",
            ":1",
            " ms",
            " ms",
            " dB",
            " dB",
        )
        for group, (effect, tab, parameters) in enumerate(names):
            toggle = Gtk.CheckButton(label=tab)
            toggle.set_active(s[effect])
            toggle.connect("toggled", lambda b, k=effect: change(k, b.get_active()))
            presets.bind(effect + ".enabled", "Enabled", effect, toggle.get_active, toggle.set_active,
                         group_label=tab)
            knobs.attach(toggle, 0, group * 2, 1, 2)
            for pos, j in enumerate(parameters):
                name, lo, hi, default = PARAMS[j]
                value = Dial(
                    f"{MIC_NAMES[i]} {name}", lo, hi, s["values"][j], default, units[j]
                )

                def adjusted(w, n=j):
                    s["values"][n] = w.get_value()
                    self.configure()
                    self.save()

                value.connect("value-changed", adjusted)
                presets.bind(f"values.{j}", name, effect, value.get_value, value.set_value, group_label=tab)
                presets.attach_control(f"values.{j}", value)
                caption = Gtk.EventBox()
                caption.add(Gtk.Label(label=name))
                caption.connect(
                    "button-press-event",
                    lambda _, e, w=value, d=default: w.set_value(d)
                    if e.type == Gdk.EventType.DOUBLE_BUTTON_PRESS
                    else False,
                )
                knobs.attach(caption, pos + 1, group * 2, 1, 1)
                knobs.attach(value, pos + 1, group * 2 + 1, 1, 1)
        room = Dial(f"{MIC_NAMES[i]} room reverb", 0, 65, s["reverb"], 0, "%")
        room.connect("value-changed", lambda w: change("reverb", round(w.get_value())))
        knobs.attach(Gtk.Label(label="Reverb"), 6, 0, 1, 1)
        knobs.attach(room, 6, 1, 1, 1)
        lead = Dial(f"{MIC_NAMES[i]} vocal lead", 0, 100, s["lead"], 70, "%")
        lead.set_tooltip_text("Level of the scored vocal melody in the vocoder carrier, when available")
        lead.connect("value-changed", lambda w: change("lead", round(w.get_value())))
        knobs.attach(Gtk.Label(label="Lead"), 6, 2, 1, 1)
        knobs.attach(lead, 6, 3, 1, 1)
        presets.bind("reverb", "Reverb", None, room.get_value, room.set_value)
        presets.bind("lead", "Vocal lead", "vocoder", lead.get_value, lead.set_value)
        presets.attach_control("reverb", room)
        presets.attach_control("lead", lead)
        status = Gtk.Label(xalign=0)
        status.set_ellipsize(3)
        playback = Gtk.Box(spacing=5)
        playback.add(self.test.buttons[i])
        playback.add(self.test.sources[i])
        playback.add(mix)
        tal = TalControls(self.audio, i, self.strips[i]["tal_params"], self.save)
        def tal_binding(index, widget=None):
            param = tal.params[index]
            title = param.name.decode()
            key = f"tal.{param.id}"
            presets.bind(key, TITLES.get(title, title), "tal", lambda: tal.value(index),
                         lambda v: tal.apply_parameter(index, v), group_label="TAL Vocoder", in_pane=False)
            if widget:
                presets.attach_control(key, widget)
        tal.bind_control = tal_binding
        tal.set_no_show_all(True)
        playback.add(self.vocoder_controls(i, tal))
        playback.pack_start(status, True, True, 0)
        rack.pack_start(playback, False, False, 0)
        arrangement = Gtk.Box(spacing=6)
        arrangement.add(Gtk.Label(label="MIDI"))
        arrangement.add(midi)
        arrangement.pack_end(presets, False, False, 0)
        rack.pack_start(arrangement, False, False, 0)
        rack.pack_start(tal, False, False, 0)
        presets.bind("tal_params", "All TAL parameters", "tal", lambda: tal.saved,
                     tal.restore, group_label="TAL Vocoder")
        for index, param in enumerate(tal.params):
            name = param.name.decode()
            if name.startswith("MIDI CC "):
                continue
            tal_binding(index)
        self.cells.append((lyrics, anchor, effects, live, hear, meter, status))
        return box

    def vocoder_controls(self, lane, tal):
        menu = Gtk.MenuButton(label="Vocoder…")
        menu.set_tooltip_text("Adjust the vocoder and its scored carrier")
        menu.get_accessible().set_name(f"{MIC_NAMES[lane]} vocoder settings")
        pop = Gtk.Popover.new(menu)
        menu.set_popover(pop)
        grid = Gtk.Grid(column_spacing=8, row_spacing=6, margin=10)
        pop.add(grid)
        values = self.strips[lane]["vocoder"]
        sliders = {}
        preset = Gtk.ComboBoxText()
        preset.get_accessible().set_name(f"{MIC_NAMES[lane]} vocoder preset")
        defaults = [spec[-1] for spec in VOCODER_PARAMS]
        presets = {"Balanced": defaults, **VOCODER_PRESETS}
        for name in (*presets, "Custom"):
            preset.append(name, name)
        grid.attach(Gtk.Label(label="Preset", xalign=0), 0, 0, 1, 1)
        grid.attach(preset, 1, 0, 1, 1)
        setting_preset = False

        def apply():
            self.audio.call("vocoder", lane, *values[:4])
            self.audio.call("vocoder_shape", lane, *values[4:7])
            self.audio.call("glide", lane, values[7])
            name = next((key for key, params in presets.items() if params == values), "Custom")
            if preset.get_active_id() != name:
                preset.set_active_id(name)

        def changed(widget, index):
            values[index] = int(widget.get_active_id()) if index == 6 else round(widget.get_value())
            if not setting_preset:
                apply()
                self.save()

        def use_preset(widget):
            nonlocal setting_preset
            selected = presets.get(widget.get_active_id())
            if selected is None:
                apply()
                return
            if selected == values:
                return
            setting_preset = True
            for index, value in enumerate(selected):
                if index == 6:
                    sliders[index].set_active_id(str(value))
                else:
                    sliders[index].set_value(value)
            setting_preset = False
            apply()
            self.save()

        for row, i in enumerate((0, 6, 7, 4, 5, 1, 2, 3), 1):
            name, unit, lo, hi, _ = VOCODER_PARAMS[i]
            label = Gtk.Label(label=name, xalign=0)
            grid.attach(label, 0, row, 1, 1)
            if i == 0:
                mix_label = label
            if i == 6:
                slider = Gtk.ComboBoxText()
                for mode, text in [(0, "Score pitch"), (1, "Speech range"), (-1, "−1 octave"), (-2, "−2 octaves")]:
                    slider.append(str(mode), text)
                slider.set_active_id(str(int(values[i])))
                slider.connect("changed", changed, i)
            else:
                slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, lo, hi, 1)
                slider.set_size_request(190, -1)
                slider.set_digits(0)
                slider.set_value_pos(Gtk.PositionType.RIGHT)
                slider.connect("format-value", lambda _, val, u=unit: f"{val:.0f}{u}")
                slider.set_value(values[i])
                slider.connect("value-changed", changed, i)
            slider.get_accessible().set_name(f"{MIC_NAMES[lane]} vocoder {name.lower()}")
            slider.set_tooltip_text(VOCODER_HINTS[i])
            sliders[i] = slider
            setter = (lambda v, w=slider: w.set_active_id(str(int(v)))) if i == 6 else slider.set_value
            self.preset_controls[lane].bind(f"vocoder.{i}", name, "vocoder",
                                            lambda j=i: values[j], setter)
            self.preset_controls[lane].attach_control(f"vocoder.{i}", slider)
            grid.attach(slider, 1, row, 1, 1)
        width = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        width.set_value(self.strips[lane]["stereo_width"])
        width.set_digits(0)
        width.set_tooltip_text("Use each instrument’s recorded stereo placement in the MIDI monitor and vocoder.")
        width.get_accessible().set_name(f"{MIC_NAMES[lane]} vocoder stereo width")
        def set_width(widget):
            value = round(widget.get_value())
            self.strips[lane]["stereo_width"] = value
            self.audio.call("stereo_width", lane, value)
            self.save()
        width.connect("value-changed", set_width)
        self.preset_controls[lane].bind("stereo_width", "Stereo width", "vocoder",
                                        width.get_value, width.set_value)
        self.preset_controls[lane].attach_control("stereo_width", width)
        self.audio.call("stereo_width", lane, width.get_value())
        grid.attach(Gtk.Label(label="Stereo width (%)", xalign=0), 0, len(sliders)+1, 1, 1)
        grid.attach(width, 1, len(sliders)+1, 1, 1)
        restore = Gtk.Button(label="Reset to defaults")
        restore.connect("clicked", lambda _: preset.set_active_id("Balanced"))
        restore.connect("clicked", lambda _: width.set_value(100))
        grid.attach(restore, 1, len(sliders)+2, 1, 1)
        preset.connect("changed", use_preset)
        self.effect_menus.append((menu, grid, preset, restore, tal))
        self.vocoder_panels.append((mix_label, sliders))
        grid.show_all()
        apply()
        return menu

    def set_song_assignment(self, lane, field, value):
        self.strips[lane][field] = value

    def pitch_mask(self, lane):
        source = resolve(self.strips, lane)
        key = self.songs.get(source["lyrics"], {}).get("key")
        return scale_mask(key) if source["anchor"] == "autotune" else 0

    def show_midi_choices(self, lane, carrier, leader):
        if self.ready and carrier in self.songs:
            clip, _, _ = self.pane_transport(leader)
            if clip and clip.get("id") in self.songs and family_key(clip) == family_key(self.songs[carrier]):
                carrier = clip["id"]
        options = self.audio.version_labels.get(carrier) or {"all": VERSIONS["all"]}
        midi = self.midi_controls[lane]
        preferred = self.strips[lane]["midi_version"]
        selected = preferred if preferred in options else "all"
        if midi.options != options or midi.get_active_id() != selected:
            with midi.handler_block(midi.handler):
                if midi.options != options:
                    midi.remove_all()
                    for key, label in options.items():
                        midi.append(key, label)
                    midi.options = options
                midi.set_active_id(selected)
        midi.set_sensitive(len(options) > 1)

    def configure(self):
        self.carrier_names = []
        for i, s in enumerate(self.strips):
            source = resolve(self.strips, i)
            carrier = carrier_id(source, self.audio)
            carrier_song = self.songs.get(carrier) or self.songs.get(source["anchor"]) or self.songs.get(source["lyrics"], {})
            score_id = self.songs.get(carrier, {}).get("scoreId", carrier)
            loaded_id = self.audio.backing_ids.get(carrier, score_id)
            self.audio.call("midi_version", i, list(VERSIONS).index(s["midi_version"]))
            has_carrier = self.audio.call("has_carrier", loaded_id.encode())
            display_key = carrier if carrier in self.audio.backing_ids else score_id
            self.carrier_names.append(display_key if has_carrier else None)
            # Imported full-length scores carry recording timestamps. Repeating
            # chant scores start on beat one and follow the room's beat count.
            leader = next(index for index, strip in enumerate(self.strips) if strip is source)
            self.show_midi_choices(i, carrier, leader)
            self.audio.call("carrier_clock", i, leader, carrier_song.get("bpm", 0))
            mask = self.pitch_mask(i)
            flags = int(s["pulse"]) | int(s["delay"]) << 1 | int(s["compressor"]) << 2
            self.audio.call("lead_level", i, s["lead"])
            self.audio.call("reverb", i, s["reverb"] / 100)
            self.audio.call(
                "config",
                i,
                s["voice"],
                loaded_id.encode(),
                mask,
                flags,
                (C.c_float * 14)(*s["values"]),
            )
            lyrics, anc, fx, *_ = self.cells[i]
            pair = self.bar_pairs.get(i)
            song = self.songs.get(pair["lyrics"]) if pair else self.songs.get(s["lyrics"])
            lyrics.show_assignment(song["label"] if song else None)
            partner = (pair["second"] if i == pair["first"] else pair["first"]) if pair else None
            self.pair_controls[i].set_label(f"Bars ↔ {MIC_NAMES[partner]}" if pair else "Alternate bars")
            selected = s["anchor"]
            if selected.startswith("mic:"):
                title = f"Follow {MIC_NAMES[int(selected[4:])]}"
            elif selected == "autotune":
                title = "Autotune · " + (self.songs.get(source["lyrics"], {}).get("key") or "Chromatic")
            elif selected == "natural":
                title = None
            else:
                title = self.songs[selected]["label"]
            anc.show_assignment(title)
            label = title if selected.startswith("mic:") else {
                "natural": "Natural", "autotune": "Autotune"
            }.get(selected, "Song")
            mode = self.hear_modes[i]
            mode.set_label(label)
            resolved = self.songs.get(source["anchor"], {}).get("label") or {
                "natural": "Natural voice", "autotune": "Autotune"
            }.get(source["anchor"], source["anchor"])
            routing = f"{MIC_NAMES[i]} hear mode · {label} · {resolved}"
            mode.set_tooltip_text(routing)
            mode.get_accessible().set_name(routing)
            fx.set_label(VOICE_OPTIONS[s["voice"]])
            hint = "Select a voice effect"
            if s["voice"] in SCORE_VOICES:
                hint = (f"Vocoder notes from {carrier}" if has_carrier
                        else "No vocoder score for this assignment. Passing the voice through.")
            vst_harmony = s["voice"] == CHORD_TUNE
            harmony = vst_harmony or s["voice"] == TUNELET
            tal = s["voice"] == TAL_VOCODER
            mix_label, sliders = self.vocoder_panels[i]
            menu, grid, preset, restore, tal_menu = self.effect_menus[i]
            tal_menu.set_visible(tal)
            group = "Harmony" if harmony else "Carrier" if tal else "Vocoder"
            menu.set_label(group + "…")
            menu.set_tooltip_text("Adjust the scored singing harmony" if harmony else "Adjust the vocoder and its scored carrier")
            menu.get_accessible().set_name(f"{MIC_NAMES[i]} {group.lower()} settings")
            mix_name = "Harmony level" if harmony else "Wet level" if tal else "Dry / Wet"
            mix_label.set_text(mix_name)
            sliders[0].get_accessible().set_name(f"{MIC_NAMES[i]} {group.lower()} {mix_name.lower()}")
            sliders[0].set_tooltip_text("Level of the tuned voices." if harmony else
                                        "Level of the fully vocoded voice. Zero mutes it." if tal else VOCODER_HINTS[0])
            grid.get_child_at(0, 2).set_text("Vocal register" if harmony else "Synth register")
            for widget in (grid.get_child_at(0, 0), preset, restore,
                           *(grid.get_child_at(col, row) for row in (4, 5, 6, 7, 8) for col in (0, 1))):
                hide = harmony or (tal and widget not in (grid.get_child_at(0, 5), sliders[5]))
                widget.set_no_show_all(hide)
                widget.set_visible(not hide)
            for index in (1, 2, 3, 4):
                sliders[index].set_sensitive(not (harmony or tal))
            # The Chorale MIDI mode follows exact note targets with its own transitions.
            for widget in (sliders[7], grid.get_child_at(0, 3)):
                widget.set_no_show_all(vst_harmony)
                widget.set_visible(not vst_harmony)
            if harmony:
                engine = "Chorale VST3" if vst_harmony else "Tunelet"
                hint = (f"{engine} · singer harmonized to notes from {carrier}" if has_carrier
                        else "Chord Tune is silent until a score is assigned.")
                if vst_harmony and not self.audio.call("chorale_ready", i):
                    hint = "Chorale unavailable. Run scripts/install-chorale-vst3 and reopen the desk."
            if tal:
                hint = (f"TAL-Vocoder-2 · fully wet · notes from {carrier}" if has_carrier
                        else "TAL Vocoder is silent until a score is assigned.")
                if not self.audio.call("tal_ready", i):
                    hint = "TAL Vocoder is silent. Run scripts/install-tal-vst3 and reopen the desk."
            fx.set_tooltip_text(hint)

    def master(self, *_):
        self.muted = self.mute.get_active()
        self.session["muted"] = self.muted
        self.session["ceiling"] = self.ceiling.get_value()
        self.audio.call("master", self.session["ceiling"], self.muted)
        self.save()

    def elapsed(self):
        return self.audio.call("clock") / 48000 if self.ready else 0.0

    def install(self):
        if self.request is None:
            return
        target, self.request = self.request, None
        task, self.loading = self.loading, None
        family, choice, mode = target
        item = self.loops[family][choice]
        song = self.songs[item.source]
        try:
            bed = task.result() if task else None
            seconds = len(bed) / 96000 if task else self.source_seconds
            region = item.region(song, seconds)
            if not self.audio.prepare_track(bed, None, region):
                raise ValueError("Backing contains no audio in this cut")
            playing = bool(self.audio.call("playing")) if self.ready else self.resume_playing
            pos = self.elapsed()
            boundary = -1
            same_loop = self.selected_mix and target[:2] == self.selected_mix[:2]
            slot, key = self.shown_source
            move = self.actions.get(slot)
            now = self.conductor_bar()
            edge = now if self.ready else grid_position(region, region["start"])
            if playing and self.ready and (not same_loop or move or slot != self.playing_slot):
                future = self.scheduling_time()
                at = self.master_clock.next_edge(future, self.quant.get_active_id())
                if move and self.master_clock.seconds(move[3]) >= future:
                    at = self.master_clock.seconds(move[3])
                boundary = round(at * 48000)
                edge = self.master_clock.bar(at)
            previous = self.slot_clocks.get(slot)
            if move:
                pos = move[2].seconds(edge)
            elif previous and previous[:2] == (key, choice):
                pos = previous[2].seconds(edge)
            elif choice == "full" and (partner := self.partner_full_clock(slot, region, edge)):
                pos = partner.seconds(edge)
            elif choice in ("teaser", "main"):
                pos = ClipClock(region, launch_origin(choice, edge)).seconds(edge)
            elif boundary >= 0:
                pos = 0
            elif self.ready and self.active_assignment != key:
                pos = max(0, lyric_position(song, self.backing, pos) - region["start"])
            elif not same_loop:
                pos = 0
            length = region["end"] - region["start"]
            pos = (round(pos * 48000) % round(length * 48000)) / 48000
            if self.ready:
                outgoing = boundary / 48000 if boundary >= 0 else self.elapsed()
                region["turn_offset"] = transport_bar(self.backing, outgoing) - transport_bar(region, pos)
            clock = ClipClock(region, edge - ClipClock(region).bar(pos))
            self.audio.call("stage_lane", slot - 1)
            self.audio.call("stage_backing_enabled", self.backing_slot is not None)
            self.audio.call("stage_carrier_backing", mode == "carrier")
            self.audio.call("stage_clock", *self.audio.clock_args(clock))
            token = self.audio.call("launch_at", boundary, round(pos * 48000))
            if not token:
                self.select_source()
                return
            if boundary < 0:
                self.audio.call("transport", 0, playing)
            self.pending = dict(target=target, region=region, token=token, clock=clock,
                                seconds=seconds, assignment=key, slot=slot,
                                boundary=boundary / 48000, offset=pos)
            self.sync_launch()
        except Exception as exc:
            self.backing_unavailable("Backing unavailable · " + str(exc))

    def backing_unavailable(self, message):
        self.audio.call("backing_enabled", False)
        self.backing_slot = None
        self.info = message

    def sync_launch(self):
        p = self.pending
        if not p or self.audio.call("active") != p["token"]:
            return
        self.pending = None
        self.master_clock = p["clock"]
        self.playing_slot = p["slot"]
        self.slot_clocks[p["slot"]] = (p["assignment"], p["target"][1], self.master_clock)
        self.actions.pop(p["slot"], None)
        self.backing = p["region"]
        self.selected_mix = p["target"]
        self.active_path = playback_files(self.backing)[self.selected_mix[2] == "stripped"]
        self.source_seconds = p["seconds"]
        self.active_assignment = p["assignment"]
        self.duration = self.backing["end"] - self.backing["start"]
        self.updating = True
        self.seek.set_range(0, self.duration)
        self.updating = False
        self.ready = True
        self.resume_playing = False
        self.info = ""

    def open_audio(self):
        try:
            self.audio.open()
            if not self.ports_known:
                self.autoroute()
            return True
        except RuntimeError as error:
            self.info = str(error)
            return False

    def autoroute(self):
        self.ports_known = True
        for cap in (True, False):
            ports = self.audio.ports(cap)
            motu = [
                p
                for p in ports
                if re.search(r"motu|\bM4\b", p, re.I)
                and "monitor" not in p.lower().split(":")[-1]
            ]
            choices = motu if cap else (default_output_ports(ports) or motu)
            for i in range(4 if cap else 2):
                idx = i if cap else i + 4
                preferred = self.session["routes"][idx]
                fallback = choices[i] if i < len(choices) else ""
                name = preferred if preferred in ports else fallback
                if preferred is not None and name:
                    self.route(idx, name, remember=False)

    def route(self, idx, name, remember=True):
        if (
            self.audio.call(
                "route", idx < 4, idx if idx < 4 else idx - 4, name.encode()
            )
            == 0
        ):
            self.routes[idx] = name
            if remember:
                self.session["routes"][idx] = name or None
                self.save()
        else:
            self.info = "Could not connect " + name

    def choose_route(self, idx, choice):
        if choice == "auto":
            self.session["routes"][idx] = ""
            self.route(idx, "", remember=False)
            self.autoroute()
            self.save()
        elif choice is not None and choice != "missing":
            self.route(idx, choice)

    def devices(self):
        if not self.open_audio():
            return
        d = Gtk.Dialog(title="Audio I/O · 48 kHz", transient_for=self.window)
        d.add_button("Close", Gtk.ResponseType.CLOSE)
        area = d.get_content_area()
        area.set_border_width(10)
        for i in range(6):
            row = Gtk.Box(spacing=8)
            row.add(
                Gtk.Label(
                    label=MIC_NAMES[i] if i < 4 else ["Left out", "Right out"][i - 4]
                )
            )
            choices = Gtk.ComboBoxText()
            choices.append("auto", "Automatic")
            choices.append("", "Disconnected")
            for p in self.audio.ports(i < 4):
                choices.append(p, p)
            preferred = self.session["routes"][i]
            choices.set_active_id("auto" if preferred == "" else preferred or "")
            if preferred and preferred != self.routes[i]:
                choices.append("missing", f"{preferred} (unavailable)")
                choices.set_active_id("missing")
            choices.connect(
                "changed", lambda w, j=i: self.choose_route(j, w.get_active_id())
            )
            row.add(choices)
            area.add(row)
        buffer = Gtk.ComboBoxText()
        buffer.append("2", "40 ms · fast")
        buffer.append("4", "80 ms · stable")
        buffer.set_active_id(str(self.session.get("gpu_hops", 4)))

        def latency(w):
            hops = int(w.get_active_id())
            self.session["gpu_hops"] = hops
            self.audio.call("buffer", hops)
            self.save()
            self.test.refresh()

        buffer.connect("changed", latency)
        area.add(Gtk.Label(label="Character monitor buffer · Homer totals 45 ms fast / 90 ms stable", xalign=0))
        area.add(buffer)
        d.show_all()
        d.run()
        d.destroy()

    def transport_key(self, window, event):
        if (event.keyval != Gdk.KEY_space or isinstance(window.get_focus(), Gtk.Editable)
                or event.state & Gtk.accelerator_get_default_mod_mask()):
            return False
        self.play_pause()
        return True

    def play_pause(self):
        self.test.start_when_ready = False
        if self.loading and not self.ready:
            if self.open_audio():
                self.resume_playing = not self.resume_playing
            return
        if not self.open_audio():
            return
        if not self.ready:
            # With no songs, an eight-beat silent bed supplies the native clock.
            self.backing = dict(track="", start=0, end=4, loop=True, bpm=120, label="")
            self.master_clock = ClipClock(self.backing)
            self.audio.prepare_track(np.zeros(4 * 96000, dtype="i2"), None, self.backing)
            self.audio.call("stage_clock", *self.audio.clock_args(self.master_clock))
            self.audio.call("launch_at", -1, 0)
            self.duration = 4
            self.active_path = self.active_assignment = None
            self.updating = True
            self.seek.set_range(0, self.duration)
            self.updating = False
            self.ready = True
        on = not self.audio.call("playing")
        self.audio.call("transport", 0, on)

    def seek_change(self, w):
        if self.ready and not self.updating:
            self.cancel_launch()
            self.actions.clear()
            self.audio.call("transport", 1, round(w.get_value() * 48000))

    def sync_character_workers(self, requests):
        waiting = "GPU in use · direct vocal"
        owner = next((i for i, worker in enumerate(self.workers) if worker), None)
        for lane, request in enumerate(requests):
            if self.worker_requests[lane] != request:
                self.stop_worker(lane)
        self.worker_requests = requests
        # Keep the active singer when changing the buffer or switching mic inputs.
        if owner is not None and requests[owner] and not self.workers[owner]:
            self.start_worker(owner)
        for lane, request in enumerate(requests):
            if not request or self.workers[lane] or self.worker_status[lane] not in ("", waiting):
                continue
            if any(self.workers):
                self.worker_status[lane] = waiting
            else:
                self.start_worker(lane)

    def stop_worker(self, i):
        self.audio.call("character", i, 0)
        if self.workers[i]:
            self.workers[i].stop()
            self.workers[i] = None
        self.worker_status[i] = ""
        self.epochs[i] += 1

    def start_worker(self, i):
        voice = self.strips[i]["voice"]
        if voice < 3:
            return
        epoch = self.epochs[i] + 1
        self.epochs[i] = epoch

        def report(msg):
            def display():
                if self.closed or self.epochs[i] != epoch:
                    return False
                if msg == "ready":
                    self.worker_status[i] = "GPU live"
                    self.audio.call("character", i, epoch)
                else:
                    self.stop_worker(i)
                    self.worker_status[i] = msg
                return False

            GLib.idle_add(display)

        try:
            self.worker_status[i] = "Warming GPU · direct vocal"
            self.workers[i] = character_worker(
                self.audio,
                i,
                IDS[voice - 3],
                epoch,
                report,
                packets=1 if self.session.get("gpu_hops", 4) == 2 else 2,
            )
        except OSError as e:
            self.worker_status[i] = str(e)

    def voice(self, i, voice):
        self.audio.call("hear", i, -1)
        self.strips[i]["voice"] = voice
        self.configure()
        self.test.refresh()
        self.save()

    def monitor(self, i):
        if not self.live[i]:
            if not self.open_audio():
                return
            if not self.routes[i]:
                self.info = f"Choose a physical input for {MIC_NAMES[i]} under Audio I/O."
                return
        self.live[i] = not self.live[i]
        self.audio.call("live", i, self.live[i])
        self.test.refresh()

    def audition(self, i):
        if not self.open_audio():
            return
        id = (
            -1 if self.audio.call("hearing", i) else max(0, self.strips[i]["voice"] - 2)
        )
        if not self.audio.call("hear", i, id):
            self.info = "Recorded audition unavailable."

    def show_stage(self):
        self.stage.show_all()
        self.stage.present()

    def tick(self):
        if self.closed:
            return False
        self.sync_launch()
        self.apply_slot_actions()
        if self.loading and self.loading.done():
            self.install()
        if self.pending:
            p = self.pending
            beats = max(0, 4 * (self.master_clock.bar(p["boundary"]) - self.conductor_bar()))
            family, choice, _ = p["target"]
            n = self.nudges.get(p["slot"], 0) if p["slot"] in self.actions else 0
            move = f" · {n:+d} bar" if n else ""
            self.info = f"Queued · {self.songs[family]['title']} · {self.loops[family][choice].label}{move} · {beats:.1f} beats"
        self.refresh_loop_controls()
        self.test.refresh()
        self.audio.call("collect")
        self.updating = True
        pos = self.elapsed()
        if self.duration and self.backing["loop"]:
            pos %= self.duration
        self.seek.set_value(pos)
        self.updating = False
        update_label(self.time, f"{int(pos) // 60}:{int(pos) % 60:02}")
        self.seek.set_sensitive(self.ready)
        playing = self.resume_playing if self.loading and not self.ready else self.audio.call("playing")
        update_label(self.play, "Pause" if playing else "Play")
        for i, (_, _, _, live, hear, meter, label) in enumerate(self.cells):
            if self.ready and self.carrier_names[i] in self.songs:
                source = resolve(self.strips, i)
                leader = next(j for j, strip in enumerate(self.strips) if strip is source)
                self.show_midi_choices(i, self.carrier_names[i], leader)
            live.set_sensitive(not self.test.active[i])
            hear.set_sensitive(not self.test.active[i])
            update_label(live, "Stop live" if self.live[i] else "Live")
            update_label(
                hear, "Stop sample" if self.audio.call("hearing", i) else "Hear"
            )
            meter.set_value(10 * min(1.0, self.audio.call("peak", i)))
            status = self.worker_status[i]
            if self.strips[i]["voice"] == CHORD_TUNE and not self.audio.call("chorale_ready", i):
                status = "Chorale VST3 unavailable · muted"
            elif self.strips[i]["voice"] == TAL_VOCODER and not self.audio.call("tal_ready", i):
                status = "TAL-Vocoder-2 unavailable · muted"
            elif status == "GPU live":
                status += f" · {self.audio.call('wet', i)} converted · {self.audio.call('misses', i)} late"
            update_label(label, " · ".join(filter(None, (self.test.messages[i], status))))
        route = ""
        if self.audio.opened:
            route = f"48 kHz · {self.audio.call('status', 0)} frames · {self.audio.call('status', 1)} xruns"
            if self.audio.call("status", 2):
                route = "Audio server disconnected"
            elif not any(self.routes[4:]):
                route += " · Choose an output in Audio I/O"
        current = ""
        if playing and self.backing_slot is None:
            current = "Playing without backing"
        elif self.selected_mix and self.backing_slot is not None:
            family, choice, mode = self.selected_mix
            state = "Playing" if playing else "Paused"
            current = f"{state} · {self.songs[family]['title']} · {self.loops[family][choice].label}"
            if mode == "carrier":
                name = self.carrier_names[self.playing_slot - 1]
                if name in self.songs and name in self.audio.backing_ids:
                    source = resolve(self.strips, self.playing_slot - 1)
                    leader = next(i for i, strip in enumerate(self.strips) if strip is source)
                    clip, _, _ = self.pane_transport(leader)
                    assigned = self.songs[name]
                    if clip and clip.get("backingScoreId") and family_key(clip) == family_key(assigned):
                        name = clip["label"]
                    else:
                        name = assigned["label"]
                current += f" · MIDI carrier · {name}" if name else " · No carrier score for this assignment"
        update_label(self.status, "    ".join(v for v in [current, self.info, route] if v))
        if self.stage.get_visible():
            for pane in self.stage.panes:
                pane.queue_draw()
        return True

    def close(self, *_):
        if self.closed:
            return
        self.closed = True
        self.test.close()
        self.jobs.close()
        for i in range(4):
            self.stop_worker(i)
        self.flush()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.audio.close()
        self.stage.destroy()
        if Gtk.main_level():
            Gtk.main_quit()


if __name__ == "__main__":
    app = Desk()
    Gtk.main()
