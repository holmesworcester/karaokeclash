"""Import an offbeat performance, stretch its practice cuts, and hear the bass on a live mic."""

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import FP, PARAMS, F
from clip_clock import ClipClock
from desk_audio import DeskAudio
from import_bass import cut_bass, recording_bass
from import_job import BAR, publish
from playback import catalog


def write_pcm(path, samples, rate):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, rate, 0, "NONE", ""))
        wav.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())


def note_at(notes, ms):
    return [pitch for start, end, pitch in notes if start <= ms < end]


def fixture(folder):
    grid = [0, 1.97, 4.04, 6.00, 8.08]
    rate = 24000
    time = np.arange(round(grid[-1] * rate)) / rate
    bass = np.zeros_like(time)
    drums = np.zeros_like(time)
    expected = []
    for i, key in enumerate([47, 45, 40, 47]):
        beat = (grid[i + 1] - grid[i]) / 4
        for j in range(4):
            pos = time - (grid[i] + j * beat)
            hit = np.maximum(pos, 0)
            drums += np.where(
                (pos >= 0) & (pos < 0.19),
                0.32
                * np.exp(-26 * hit)
                * np.sin(2 * np.pi * (67 * hit + 1.8 * (1 - np.exp(-24 * hit)))),
                0,
            )
            if i == 3:  # The keyboard continues over a bass break.
                continue
            start, end = 0.46 * beat, 0.93 * beat
            env = np.clip(np.minimum((pos - start) / 0.007, (end - pos) / 0.018), 0, 1)
            phase = 2 * np.pi * 440 * 2 ** ((key - 69) / 12) * pos
            bass += 0.17 * env * (np.sin(phase) + 0.28 * np.sin(2 * phase))
            expected.append((grid[i] + j * beat + start, grid[i] + j * beat + end, key - 12))
    # Keep the bass in stereo opposition. Summing to mono would lose the part.
    low = np.column_stack((bass, -bass))
    kit = np.column_stack((drums, 0.78 * drums))
    chord = sum(0.036 * np.sin(2 * np.pi * f * time) for f in (329.63, 392, 523.25))
    guide = 0.1 * np.sin(2 * np.pi * 270 * time)
    for name, pcm in (
        ("bass", low),
        ("drums", kit),
        ("backing", low + kit + chord[:, None]),
        ("guide", np.column_stack((guide, guide))),
    ):
        write_pcm(folder / (name + ".wav"), pcm, rate)
    meta = {
        "title": "Offbeat fixture",
        "artist": "Band",
        "key": "A minor",
        "bpm": 120,
        "meter": 4,
        "downbeats": grid,
        "review": "Chorus begins at bar two.",
        "rows": [[["Verse"], [[400, 750]]], [["Hook"], [[2310, 2620]]]],
        "teaser": {"start_bar": 0, "bars": 2, "voice_ms": [300, 900], "label": "Verse"},
        "hooks": [{"start_bar": 1, "bars": 2, "voice_ms": [2220, 2850], "label": "Hook"}],
    }
    (folder / "analysis.json").write_text(json.dumps(meta))
    return meta, expected


class ImportBass(unittest.TestCase):
    def test_separated_and_mixed_stereo_offbeats_keep_the_original_rests(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            meta, expected = fixture(folder)
            short = folder / "truncated-bass.wav"
            write_pcm(short, np.zeros((24000, 2)), 24000)
            for stem in (folder / "bass.wav", None, short):
                notes = recording_bass(
                    folder / "backing.wav",
                    folder / "drums.wav",
                    stem,
                    {"duration": 8.08, "events": []},
                    meta["bpm"],
                )
                self.assertEqual(len(notes), len(expected))
                for (a, b, k), (on, off, pitch) in zip(notes, expected):
                    self.assertEqual(k, pitch)
                    self.assertAlmostEqual(a, on * 1000, delta=50)
                    self.assertAlmostEqual(b, off * 1000, delta=50)
            # A bass-less pad can supply one secure root. A single-note lead
            # and the pause between chords should not acquire invented notes.
            rate = 12000
            t = np.arange(2 * rate) / rate
            x = np.stack([0.1 * np.sin(2 * np.pi * 330 * t)] * 2, axis=1)
            x[:rate] += sum(0.07 * np.sin(2 * np.pi * f * t[:rate]) for f in [392, 523.25])[:, None]
            write_pcm(folder / "pad.wav", x, rate)
            write_pcm(folder / "drums.wav", np.zeros_like(x), rate)
            score = {
                "duration": 2,
                "events": [[0, [64, 67, 72]], [0.3, [60, 64, 67, 72]], [1, []], [1.2, [64]]],
            }
            self.assertEqual(
                recording_bass(folder / "pad.wav", folder / "drums.wav", None, score, 120),
                [[0, 1000, 36]],
            )
            self.assertEqual(
                recording_bass(
                    folder / "pad.wav", folder / "drums.wav", folder / "drums.wav", score, 120
                ),
                [],
            )

    def test_deep_bass_keeps_its_register_and_can_carry_an_otherwise_empty_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rate = 16000
            t = np.arange(rate) / rate
            sub = 0.16 * np.sin(2 * np.pi * 55 * t) * ((t > 0.2) & (t < 0.8))
            write_pcm(folder / "bass.wav", np.column_stack((sub, sub)), rate)
            score = {"duration": 1, "events": []}
            notes = recording_bass(None, None, folder / "bass.wav", score, 120)
            self.assertEqual(note_at(notes, 500), [33])
            self.assertFalse(note_at(notes, 900))
            score["bassNotesMs"] = notes
            desk = DeskAudio()
            self.addCleanup(desk.close)
            desk.import_scores({"bass only": {"score": score, "bpm": 120}})
            self.assertTrue(desk.call("has_carrier", b"bass only"))

    def test_cut_clips_crossing_notes_at_both_ends_on_each_repeat(self):
        notes = [[200, 1700, 33], [2300, 3900, 28], [5050, 7100, 31]]
        result = cut_bass(notes, [1, 2.5, 5.5], 4, 2)
        self.assertEqual(
            result,
            [
                [0, 933, 33],
                [1733, 2933, 28],
                [3700, 4000, 31],
                [4000, 4933, 33],
                [5733, 6933, 28],
                [7700, 8000, 31],
            ],
        )

    def test_add_song_publishes_bass_and_native_wet_voice_plays_the_repeated_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            imports = folder / "songs"
            target = publish(folder, folder / "backing.wav", imports, 4, sections=True)
            with patch("library.IMPORTS", imports):
                songs = {k: v for k, v in catalog().items() if k.startswith(target.name + "/")}
            full, hook, teaser, section = (
                songs[f"{target.name}/{name}"] for name in ("full", "hook-1", "teaser", "section-1")
            )
            self.assertEqual(note_at(full["score"]["bassNotesMs"], 2300), [33])
            for cut, pitches in [
                (hook, [33, 28, 33, 28]),
                (teaser, [35, 33]),
                (section, [35, 33, 28, None]),
            ]:
                for bar, pitch in enumerate(pitches):
                    self.assertEqual(
                        note_at(cut["score"]["bassNotesMs"], (bar + 0.18) * BAR * 1000),
                        [] if pitch is None else [pitch],
                    )
                    self.assertFalse(
                        note_at(cut["score"]["bassNotesMs"], (bar + 0.04) * BAR * 1000)
                    )
            audio = DeskAudio()
            self.addCleanup(audio.close)
            audio.import_scores(songs)
            audio.import_scores(songs)
            clock = ClipClock(hook)
            pcm = np.zeros(round(hook["end"] * 48000) * 2, "i2")
            self.assertGreater(audio.prepare_track(pcm, None, hook), 0)
            audio.call("stage_clock", *audio.clock_args(clock))
            audio.call("launch_at", -1, 0)
            key = hook.get("backingScoreId", hook["id"]).encode()
            audio.call("config", 0, 1, key, 0, 0, (F * len(PARAMS))(*(p[3] for p in PARAMS)))
            audio.call("live", 0, 1)
            audio.call("backing_enabled", False)
            audio.call("carrier_clock", 0, 0, 126)
            audio.call("vocoder", 0, 100, 50, 5, 40)
            audio.call("transport", 1, round(2 * BAR * 48000))
            audio.call("transport", 0, 1)
            t = np.arange(48000) / 48000
            mic = (0.12 * np.sin(2 * np.pi * 226 * t)).astype("f4")
            inputs = (FP * 4)()
            inputs[0] = mic.ctypes.data_as(FP)
            left, right = np.zeros((2, len(t)), "f4")
            audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(t))

            def sub(a, b):
                part = left[round(a * 48000) : round(b * 48000)]
                return abs(
                    np.mean(
                        part * np.hanning(len(part)) * np.exp(-2j * np.pi * 55 * t[: len(part)])
                    )
                )

            self.assertGreater(sub(0.29, 0.41), 0.004)
            self.assertGreater(sub(0.29, 0.41), 4 * sub(0.05, 0.17))
