"""Bindings for the single-library desk, using the installed runtime's media and workers."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import weakref

import numpy as np

from backing_loops import PRESETS, family_key
from carrier import prepare_score
from midi_versions import VERSIONS, arrangements
from vocoder_bass import PARTS
from audio import Audio, BUILD, DATA, MEDIA, ROOT, D, F, FP, Hop, I, IDS, L, P, S, SP, U, decode


class TalParameter(C.Structure):
    _fields_ = [("id", U), ("steps", I), ("flags", I), ("initial", F), ("name", C.c_char * 128)]


class DeskAudio(Audio):
    def tal_parameters(self, lane):
        count = self.call("tal_count", lane)
        return [self.call("tal_info", lane, i).contents for i in range(count)]

    def __init__(self):
        engine = os.environ.get("KARAOKE_DESK_ENGINE")
        if not engine:
            build = Path(__file__).resolve().parents[1] / "scripts/build-engine.sh"
            engine = subprocess.check_output([build, BUILD], text=True).strip()
        self.lib = C.CDLL(engine)
        strings = C.POINTER(S)
        hop = C.POINTER(Hop)
        specs = {
            "new": (P, ()),
            "free": (None, (P,)),
            "open": (S, (P,)),
            "ports": (strings, (P, I)),
            "ports_free": (None, (strings,)),
            "route": (I, (P, I, I, S)),
            "status": (I, (P, I)),
            "note": (None, (P, S, D, D, I)),
            "score": (None, (P, S, D, SP, I)),
            "backing_score": (None, (P, S, S, S, C.POINTER(I), I, D, D, D, C.POINTER(D), I, D, I, I, FP)),
            "score_variant": (None, (P, S, I, S)),
            "midi_version": (None, (P, I, I)),
            "bass_note": (None, (P, S, D, D, I)),
            "lead_note": (None, (P, S, D, D, I)),
            "score_unison": (None, (P, S, F)),
            "stereo_width": (None, (P, I, F)),
            "tal_source_level": (None, (P, I, F)),
            "glide": (None, (P, I, F)),
            "lead_level": (None, (P, I, F)),
            "score_gains": (None, (P, S, FP)),
            "sample": (None, (P, I, FP, I)),
            "tal_count": (I, (P, I)),
            "tal_info": (C.POINTER(TalParameter), (P, I, I)),
            "tal_display": (S, (P, I, I, F)),
            "tal_set": (None, (P, I, I, F)),
            "chorale_ready": (I, (P, I)),
            "tal_ready": (I, (P, I)),
            "has_carrier": (I, (P, S)),
            "config": (None, (P, I, I, S, I, I, FP)),
            "live": (None, (P, I, I)),
            "vocoder_shape": (None, (P, I, F, F, I)),
            "vocoder": (None, (P, I, F, F, F, F)),
            "reverb": (None, (P, I, F)),
            "hear": (I, (P, I, I)),
            "hearing": (I, (P, I)),
            "peak": (F, (P, I)),
            "master": (None, (P, F, I)),
            "buffer": (None, (P, I)),
            "collect": (None, (P,)),
            "character": (None, (P, I, U)),
            "take": (I, (P, I, hop)),
            "put": (I, (P, I, hop)),
            "wet": (I, (P, I)),
            "misses": (I, (P, I)),
            "render": (None, (P, C.POINTER(FP), FP, FP, U)),
            "tune_recording": (None, (SP, I, I)),
            "test_lanes": (None, (P, U)),
            "mute_lanes": (None, (P, U)),
            "stage_backing_enabled": (None, (P, I)),
            "stage_carrier_backing": (None, (P, I)),
            "backing_enabled": (None, (P, I)),
            "active_lane": (None, (P, I)),
            "stage_lane": (None, (P, I)),
            "test_source": (
                None,
                (P, I, SP, I, I, C.POINTER(D), C.POINTER(D), I, C.POINTER(D), I),
            ),
            "tempo": (None, (P, D)),
            "carrier_clock": (None, (P, I, I, D)),
            "score_grid": (None, (P, S, D, D, C.POINTER(D), I)),
            "transport": (None, (P, I, L)),
            "playing": (I, (P,)),
            "prepare": (L, (P, SP, I, SP, I, I, I, I, D, I, I)),
            "stage_clock": (None, (P, C.POINTER(D), C.POINTER(D), I, S)),
            "pane_clock": (None, (P, I, D, C.POINTER(D), C.POINTER(D), I, S, C.POINTER(D), C.POINTER(D), I, S)),
            "launch_at": (I, (P, L, L)),
            "launch": (I, (P, L)),
            "cancel": (None, (P,)),
            "active": (I, (P,)),
            "clock": (L, (P,)),
            "track": (L, (P, SP, I, SP, I, I, I, I, D)),
        }
        for name, (result, args) in specs.items():
            f = getattr(self.lib, "desk_" + name)
            f.restype, f.argtypes = result, args
        self.ptr = self.lib.desk_new()
        self._free_engine = weakref.finalize(self, self.lib.desk_free, self.ptr)
        self.opened = False
        self.imported = set()
        self.backing_ids = {}
        self.version_labels = {}
        self.version_ids = {}
        self.catalog_loops = {}
        self.written = set()
        self.duets = set()

    def close(self):
        self._free_engine()
        self.ptr = None

    def resources(self):
        self.load_catalog_scores(BUILD / "sounds", MEDIA / "sounds")
        dry_preview = ROOT / "assets/voice-check.flac"
        if dry_preview.is_file():
            self.add_sample(0, dry_preview)
        for slot, ident in enumerate(IDS, 1):
            if ident == "homer":
                preview = Path(__file__).with_name("homer-preview.flac")
                if preview.is_file():
                    self.add_sample(slot, preview)
                    continue
            stem = "cookie-monster" if ident == "cookie" else ident
            path = DATA / "tracks/voice-transform/characters" / (stem + "-freestyle.wav")
            if path.is_file():
                self.add_sample(slot, path)

    def load_catalog_scores(self, prepared, bundled):
        def catalog(directory):
            path = directory / "scores.json"
            return json.loads(path.read_text()) if path.is_file() else {}

        local, shipped = catalog(prepared), catalog(bundled)
        loops = Path(__file__).with_name("carrier-loops.json")
        authored = json.loads(loops.read_text()) if loops.is_file() else {}
        self.written.update(authored)
        for name, loop in authored.items():
            self.load_carrier_loop(name, loop)
        for path in Path(__file__).with_name("vocoder-scores").glob("*.json"):
            name = path.stem
            self.written.add(name)
            part = json.loads(path.read_text())
            key = name.encode()
            for field, method in (("notes", "note"), ("bass", "bass_note")):
                for start, end, pitch in part[field]:
                    self.call(method, key, start / 1000, end / 1000, pitch)
            self.call("score", key, part["loop"], None, 0)
            bars = PRESETS.get(name.removesuffix(" - Ah"), {}).get("downbeats", [])
            self.call("score_grid", key, part["bpm"], part["beatZero"], (D * len(bars))(*bars), len(bars))
        for name in shipped.keys() | local.keys():
            # Retain a short catalog loop superseded by a full written arrangement.
            alias = name + " / original loop" if name in self.written else name
            key = alias.encode()
            if alias != name:
                self.catalog_loops[name] = alias
            if " / " in name:
                self.duets.add(name)
            entries = (local.get(name, {}), shipped.get(name, {}))
            score = next((entry for entry in entries if entry.get("events")), None)
            if score:
                events = score["events"]
                for i, (start, notes) in enumerate(events):
                    end = events[i + 1][0] if i + 1 < len(events) else score["duration"]
                    for note in notes:
                        self.call("note", key, start, end, note)
                self.call("score", key, score["duration"], None, 0)
                if alias == name:
                    self.load_bass_part(name, score["duration"])
                continue

            # Older libraries sometimes keep only an eight-second audio reference.
            # The engine transcribes it here and retains just its note changes.
            for directory, entry in zip((prepared, bundled), entries):
                clip = entry.get("audio")
                if not clip:
                    continue
                reference = directory / clip
                if not reference.is_file():
                    reference = bundled / clip
                if reference.is_file():
                    pcm = decode(reference, 2, seconds=8)
                    self.call("score", key, len(pcm) / 96000, pcm.ctypes.data_as(SP), len(pcm))
                    if alias == name:
                        self.load_bass_part(name, len(pcm) / 96000)
                    break

    def load_bass_part(self, name, duration, *, recording=False):
        pattern = PARTS.get(name)
        if not pattern:
            return
        key = name.encode()
        for start, end, pitch in pattern.excerpt(duration):
            self.call("bass_note", key, start, end, pitch)
        if not recording:
            # Some lyric editions run at a different tempo from the carrier excerpt.
            self.call("score_grid", key, pattern.bpm, 0, None, 0)

    def load_carrier_loop(self, name, loop):
        key = name.encode()
        tick = 60 / (loop["bpm"] * loop["ticksPerBeat"])
        length = loop["lengthTicks"]
        bar = 4 * loop["ticksPerBeat"]
        sections = [(0, length, 0)]
        if "sections" in loop:
            sections = [(first * bar, stop * bar, transpose) for first, stop, transpose in loop["sections"]]
        gains = (F * 128)(*([1.] * 128))
        for first, stop, transpose in sections:
            for base in range(first, stop, length):
                for start, end, pitch in loop["notes"]:
                    if base + start < stop:
                        self.call("note", key, (base + start) * tick, min(base + end, stop) * tick,
                                  pitch + transpose)
            for pitch, gain in loop["gains"].items():
                gains[int(pitch) + transpose] = gain
        if "leadFile" in loop:
            for start, end, pitch in json.loads(Path(__file__).with_name(loop["leadFile"]).read_text()):
                self.call("lead_note", key, start * tick, end * tick, pitch)
        self.call("score_unison", key, loop.get("unison", 0))
        self.call("score_gains", key, gains)
        self.call("score", key, 0 if "sections" in loop else length * tick, None, 0)
        # The riff starts on album bar zero, including in cuts starting on an odd bar.
        self.call("score_grid", key, loop["bpm"], 0, None, 0)

    def import_scores(self, songs):
        # A full recording score may coexist with a legacy four-bar audition.
        scores = {}
        for key, song in songs.items():
            if song.get("score"):
                score_id = song.get("scoreId", key)
                # Practice cuts can carry local test-singing notes but share
                # their full recording's vocoder score and measured tempo map.
                if key == score_id or score_id not in scores:
                    scores[score_id] = song
        previous = self.imported.copy()
        for key in scores.keys() - previous:
            score = scores[key]["score"]
            if "bassNotesMs" in score:
                for start, end, pitch in score["bassNotesMs"]:
                    self.call("bass_note", key.encode(), start / 1000, end / 1000, pitch)
            else:
                self.load_bass_part(key, score["duration"], recording=True)
        super().import_scores(scores)
        for key in self.imported - previous:
            song = scores[key]
            bars = song.get("downbeats", [])
            self.call("score_grid", key.encode(), song["bpm"], song.get("beat_zero", 0),
                      (D * len(bars))(*bars), len(bars))

        # Each recorded cut can have a different backing, including cuts whose
        # written vocal line lives on an album MIDI clock.
        for key, song in songs.items():
            backing = song.get("carrierBacking")
            if not backing or not backing.is_file():
                continue
            part = prepare_score(backing)
            original = song.get("scoreId", key)
            local = song.get("score", {})
            keep_written = (key in self.written or "scoreId" in song or "scoreFile" in song
                            or (not local.get("transcription")
                                and any(len(pitches) > 1 for _, pitches in local.get("events", ()))))
            layers = arrangements(backing, part)
            song["backingScoreId"] = self.load_arrangements(key, song, original, part, layers,
                                                            keep_written, family_key(song))
            for duet in self.duets:
                if duet.startswith(key + " / "):
                    self.load_arrangements(duet, song, duet, part, layers, True, "")

    def load_arrangements(self, key, song, original, part, layers, authored, family):
        keep = authored and not part.get("rendered")
        focus = part
        if part.get("arrangement"):
            keep = key in ("Around the World", "Doin' It Right - Vocoder") or key in self.duets
            if keep:
                focus = dict(part, notes=[])
        extra = layers.get("extra", [])
        full = dict(focus, notes=focus["notes"] + extra,
                    pan={**layers.get("pan", {}), **focus.get("pan", {})})
        main = self.load_backing_score(key, song, original, full, keep, family)
        versions = {"all": VERSIONS["all"]}

        def add(mode, score, source, written, label, corrections=False):
            ident = self.load_backing_score(key, song, source, score, written, family,
                                            version=mode, corrections=corrections)
            if self.call("has_carrier", ident.encode()):
                self.call("score_variant", main.encode(), list(VERSIONS).index(mode), ident.encode())
                versions[mode] = label

        loop = self.catalog_loops.get(key, original)
        written = bool(self.call("has_carrier", loop.encode()))
        if extra:
            add("focus", focus, original, keep, "Focused part", True)
            add("extra", dict(part, notes=extra, pan=layers.get("pan", {})), "", False,
                layers.get("label", VERSIONS["extra"]))
        if written and (loop != original or full["notes"] or song.get("carrierNotesMs")):
            add("original", dict(part, notes=[]), loop, True, VERSIONS["original"])
        if "earlier" in layers and (layers["earlier"] != full["notes"] or authored != keep):
            add("earlier", dict(part, notes=layers["earlier"], pan={}), original,
                authored and not part.get("rendered"), VERSIONS["earlier"], True)
        for slot, mode in enumerate(VERSIONS):
            if slot and mode not in versions:
                self.call("score_variant", main.encode(), slot, b"")
        self.version_labels[key] = versions
        return main

    def load_backing_score(self, key, song, original, part, keep_written, family, *,
                           version="all", corrections=True):
        # The clip and every version use their own recording's tempo and splice map.
        extra = song.get("carrierNotesMs", []) if corrections else []
        bars = song.get("downbeats", ())
        settings = [part["source"], bool(part.get("rendered")), extra, original, family, keep_written,
                    song["bpm"], song.get("beat_zero", 0), bars, song.get("album_bar_offset", 0)]
        settings.extend([version, part["notes"], part.get("pan", {})])
        stamp = hashlib.sha256(json.dumps(settings, separators=(",", ":")).encode()).hexdigest()
        ident = f"{key} / backing {stamp}"
        ids = self.backing_ids if version == "all" else self.version_ids
        entry = key if version == "all" else (key, version)
        if ids.get(entry) != ident:
            notes = part["notes"] + extra
            notes = np.ascontiguousarray(notes, dtype=np.int32).reshape(-1, 3)
            pan = (F * 128)()
            for pitch, position in part.get("pan", {}).items():
                pan[int(pitch)] = max(-1, min(1, position))
            self.call("backing_score", ident.encode(), original.encode(), family.encode(),
                      notes.ctypes.data_as(C.POINTER(I)), len(notes), part["duration"], song["bpm"],
                      song.get("beat_zero", 0), (D * len(bars))(*bars), len(bars),
                      song.get("album_bar_offset", 0), keep_written, original not in self.imported, pan)
            ids[entry] = ident
        return ident

    def tune_recording(self, pcm, mask):
        tuned = np.array(pcm, dtype=np.int16, copy=True)
        self.lib.desk_tune_recording(tuned.ctypes.data_as(SP), len(tuned), mask)
        return tuned

    def test_source(self, lane, pcm=None, song=None, backing=None):
        if pcm is None or not len(pcm):
            self.call("test_source", lane, None, 0, 0, None, None, 0, None, 0)
            return
        spec = []
        grids = []
        for item in (song, backing):
            spec.extend([item[k] for k in ("start", "end", "bpm")])
            spec.extend([item.get("beat_zero", 0), item.get("cycle", 0), item["loop"]])
            grid = item.get("downbeats", [])
            grids.append((D * len(grid))(*grid))
        a, b = grids
        self.call(
            "test_source",
            lane,
            pcm.ctypes.data_as(SP),
            len(pcm),
            song["track"] == backing["track"],
            (D * 12)(*spec),
            a,
            len(a),
            b,
            len(b),
        )

    def prepare_track(self, bed, guide, region):
        start, end = region["start"], region["end"]
        a, b = region.get("vocal_window", (start, end))
        return self.call(
            "prepare",
            bed.ctypes.data_as(SP) if bed is not None else None,
            len(bed) // 2 if bed is not None else round(end * 48000),
            guide.ctypes.data_as(SP) if guide is not None else None,
            len(guide) // 2 if guide is not None else 0,
            round(start * 48000),
            round(end * 48000),
            True,
            region["bpm"],
            round(a * 48000),
            round(b * 48000),
        )

    @staticmethod
    def clock_args(clock, lyric_track=None):
        if clock is None:
            return None, None, 0, None
        r = clock.region
        a, b = r.get("vocal_window", (r["start"], r["end"]))
        spec = (D * 9)(r["start"], r["end"], r["bpm"], r.get("beat_zero", 0),
                       clock.origin, a, b, lyric_track == r["track"], r.get("album_bar_offset", 0))
        bars = r.get("downbeats", [])
        return spec, (D * len(bars))(*bars), len(bars), r.get("backingScoreId", "").encode()
