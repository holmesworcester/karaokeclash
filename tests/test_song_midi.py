"""Verify the recording clock, repeated cuts and chord pitches delivered to the desk."""

import sys
import tempfile
import unittest
from bisect import bisect_right
from pathlib import Path
from unittest.mock import Mock

import mido

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from backing_loops import calibrated_catalog
from desk import carrier_id
from moran_chords import write_midi
from stand_by_me import arrangement, chord_score


def sounding(events, time):
    index = bisect_right([a for a, _ in events], time) - 1
    return set(events[index][1]) if index >= 0 else set()


def midi_frames(path, track=None):
    midi = mido.MidiFile(path)
    if track is not None:
        midi.tracks[:] = [midi.tracks[0], next(t for t in midi.tracks if t.name == track)]
    held, events, when = set(), [], 0
    for msg in midi:
        when += msg.time
        if msg.type == "note_on" and msg.velocity:
            held.add(msg.note)
        elif msg.type in ("note_on", "note_off"):
            held.remove(msg.note)
        else:
            continue
        events.append((when, held.copy()))
    return events


class SongMidi(unittest.TestCase):
    def test_stand_by_me_cut_phase_and_nine_bar_chorus_restart(self):
        bar = 240 / 126

        def score(key, count):
            return chord_score({"id": key, "bpm": 126, "bars": count}, count * bar)["events"]

        main = score("hook-1", 36)
        for lap in range(4):
            for i, expected in enumerate(["A", "A", "A", "A", "F#", "F#", "D", "E", "A"]):
                notes = sounding(main, (lap * 9 + i + 0.25) * bar)
                self.assertEqual(len(notes), 4)
                root = {"A": 45, "F#": 42, "D": 38, "E": 40}[expected]
                self.assertEqual(min(notes), root)
        self.assertEqual(
            [min(sounding(score("teaser", 4), (i + 0.1) * bar)) for i in range(4)], [45, 45, 45, 42]
        )
        self.assertEqual(
            [min(sounding(score("hook-2", 32), (i + 0.1) * bar)) for i in range(8)],
            [45, 45, 45, 42, 42, 38, 40, 45],
        )

    def test_orchestral_midi_splices_refrain_and_preserves_tied_chord_voices(self):
        # Three distinctive lead notes stand in for the first chorus, second
        # chorus and final cadence on the 80-bar source score.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi = mido.MidiFile(ticks_per_beat=480)
            for i in range(9):
                track = midi.add_track(str(i))
                previous = 0
                for bar, pitch in [(24, 69), (48, 71), (72, 73)]:
                    track.append(mido.Message("note_on", note=pitch, time=bar * 1920 - previous))
                    track.append(mido.Message("note_off", note=pitch, time=480))
                    previous = bar * 1920 + 480
                track.append(mido.MetaMessage("end_of_track", time=80 * 1920 - previous))
            midi.save(root / "source.mid")
            grid = [1.392 + i * 2.014 for i in range(89)]
            duration = grid[-1] + 2.54
            arrangement(root / "source.mid", grid, duration, root)
            lead = midi_frames(root / "orchestra.mid", "Clarinet")
            for bar, note in [(24, 69), (48, 71), (72, 71), (80, 73)]:
                self.assertEqual(sounding(lead, grid[bar] + 0.12), {note})
            held = midi_frames(root / "chords.mid", "Chord pitches")
            self.assertEqual(sounding(held, grid[0] - 0.02), set())
            for i, notes in enumerate(
                [{45, 57, 61, 64}] * 2
                + [{42, 57, 61, 66}] * 2
                + [{38, 57, 62, 66}, {40, 56, 59, 64}, {45, 57, 61, 64}]
            ):
                self.assertEqual(sounding(held, grid[i] + 0.002), notes)
            self.assertAlmostEqual(mido.MidiFile(root / "orchestra.mid").length, duration, delta=0.002)

    def test_moran_midi_and_catalog_agree_on_recording_and_mashup_routing(self):
        title = "My Heart Will Go On"
        song = calibrated_catalog({title: {}})[title]
        audio = Mock()
        audio.call.side_effect = lambda _, key: key == (title + " / Milkshake").encode()
        for lyrics, expected in [(title, title), ("Milkshake", title + " / Milkshake"), ("", title)]:
            self.assertEqual(carrier_id({"anchor": title, "lyrics": lyrics}, audio), expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "moran.mid"
            write_midi(path)
            midi = midi_frames(path)
            events = song["score"]["events"]
            for start, expected in events:
                self.assertEqual(sounding(midi, start + 0.002), set(expected))
            for time, notes in [
                (62, {37, 52, 59, 63, 68}),
                (192.9, {41, 51, 56, 60, 65}),
                (238.8, {44, 55, 58, 63}),
                (256.5, {44, 55, 58, 60, 63}),
            ]:
                self.assertEqual(sounding(events, time), notes)
            self.assertEqual(sounding(midi, 260.25), set())
