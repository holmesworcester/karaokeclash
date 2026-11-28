import json
from bisect import bisect_right
import ctypes
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from backing_loops import backing_loops
from audio import FP, PARAMS
from clip_clock import ClipClock
from desk_audio import DeskAudio
from starred_arrangements import BAR, RATE, RECIPES, arrangement, midi_score, prepare, score_excerpt


def cuts_of(name):
    recipe = json.loads((RECIPES / (name + ".json")).read_text())
    return {c["id"]: arrangement(c["sections"], recipe["rows"]) for c in recipe["cuts"]}


def words_of(rows):
    return [word.lower() for words, _ in rows for word in words]


class StarredArrangements(unittest.TestCase):
    def test_verve_recording_has_verse_rests_violin_returns_and_a_separate_bass(self):
        recipe = json.loads((RECIPES / "bitter-sweet-symphony.json").read_text())
        grid = recipe["full_grid"]
        self.assertGreater(grid[-1], 356)
        self.assertGreater(recipe["rows"][-1][1][-1][1], 350000)
        score = midi_score(RECIPES / recipe["midi"], 357.266667)
        events = score["events"]

        def sounding(t):
            return set(events[bisect_right([start for start, _ in events], t) - 1][1])

        def at(bar, beat):
            return sounding(grid[bar] + beat * (grid[bar+1] - grid[bar]) / 4)

        self.assertEqual(at(0, 1.2), {52, 64, 68})
        self.assertEqual(at(0, 2.7), {47, 62, 66, 69})
        self.assertNotIn(83, at(24, 1.2))  # The opening vocal verse drops the high hook.
        for bar in (8, 44, 68, 100):
            self.assertIn(83, at(bar, 1.2))
            self.assertIn(78, at(bar, 2.7))
            self.assertIn(86, at(bar+1, 1.3))
            self.assertIn(85, at(bar+1, 3.3))
        first_bass = next(t for t, notes in events if any(n < 45 for n in notes))
        self.assertAlmostEqual(first_bass, 45.05, delta=.05)
        for t, key in ((45.3, 40), (46.65, 35), (48.2, 33), (123.7, 28), (127, 33)):
            self.assertIn(key, sounding(t))

    def test_verve_practice_clock_plays_album_violin_through_following_mic(self):
        recipe = json.loads((RECIPES / "bitter-sweet-symphony.json").read_text())
        grid = recipe["full_grid"]
        full_id = recipe["slug"] + "/full"
        full = dict(bpm=85.6, downbeats=grid, score=midi_score(RECIPES / recipe["midi"], 357.266667))
        main_grid = recipe["cuts"][0]["sections"][0]["grid"]
        excerpt = score_excerpt(full["score"], main_grid)
        cut = dict(track="practice", score=excerpt, scoreId=full_id, start=0, end=excerpt["duration"],
                   loop=True, bpm=126, album_bar_offset=float(np.interp(main_grid[0], grid, np.arange(len(grid)))))
        audio = DeskAudio()
        self.addCleanup(audio.close)
        audio.import_scores({full_id: full, "practice": cut})
        self.assertTrue(audio.prepare_track(np.zeros(round(cut["end"]*RATE)*2, "i2"), None, cut))
        audio.call("stage_clock", *audio.clock_args(ClipClock(cut)))
        audio.call("launch_at", -1, 0)
        values = (ctypes.c_float * len(PARAMS))(*(p[3] for p in PARAMS))
        audio.call("config", 1, 1, full_id.encode(), 0, 0, values)
        audio.call("carrier_clock", 1, 0, full["bpm"])
        audio.call("vocoder", 1, 100, 0, 5, 40)
        audio.call("glide", 1, 0)
        audio.call("live", 1, 1)
        audio.call("tempo", 126)
        audio.call("transport", 0, 1)
        n = 2*RATE
        speech = np.random.default_rng(85).uniform(-.18, .18, n).astype("f4")
        inputs = (FP * 4)()
        inputs[1] = speech.ctypes.data_as(FP)
        left, right = np.zeros((2, n), "f4")
        audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), n)

        def tone(beat, pitch):
            start = round(beat*60/126*RATE)
            clip = left[start:start+4800] * np.hanning(4800)
            hz = 440*2**((pitch-69)/12)
            return abs(np.dot(clip, np.exp(-2j*np.pi*hz*np.arange(len(clip))/RATE)))

        self.assertGreater(tone(1.18, 83), 3*tone(2.68, 83))
        self.assertGreater(tone(2.68, 78), 3*tone(1.18, 78))

    def test_saved_singing_edits(self):
        taio = cuts_of("dynamite")
        self.assertEqual(len(taio["main"][0]) - 1, 4)
        self.assertEqual(
            words_of(taio["main"][1]),
            [w for w in ("dance", "plans", "brands", "hands") for _ in range(4)],
        )
        closing = cuts_of("closing-time")
        self.assertEqual(words_of(closing["teaser"][1]), ["closing", "time"])
        main = " ".join(words_of(closing["main"][1]))
        self.assertIn("finish your whiskey or beer", main)
        self.assertIn("your brothers or your sisters come", main)
        self.assertNotIn("take me home", main)
        self.assertEqual(len(closing["main"][0]) - 1, 32)
        tiffany = words_of(cuts_of("i-think-were-alone-now")["main"][1])
        self.assertEqual(tiffany[:3], ["running", "just", "as"])
        self.assertEqual(tiffany[-9:], "we tumble to the ground and then you say".split())
        verve = words_of(cuts_of("bitter-sweet-symphony")["main"][1])
        self.assertEqual(verve[:4], ["million", "different", "people", "from"])
        self.assertTrue("mould" in verve or "mold" in verve)
        self.assertEqual(verve[-5:], ["no"] * 5)
        offspring = cuts_of("pretty-fly")
        self.assertEqual(
            words_of(offspring["teaser"][1]), ["getting", "a", "tattoo", "yeah", "he's"]
        )
        self.assertIn("for a white guy", " ".join(words_of(offspring["main"][1])))
        self.assertIn("debate", offspring)
        rihanna = cuts_of("dont-stop-the-music")
        self.assertEqual(
            words_of(rihanna["main"][1]), "chest to chest and now we're face to face".split()
        )
        self.assertEqual(words_of(rihanna["title"][1])[-4:], ["don't", "stop", "the", "music"])
        self.assertEqual(words_of(cuts_of("stayin-alive")["main"][1]), ["ah", "ha", "ha", "ha"])
        spin = cuts_of("you-spin-me-round")
        self.assertEqual(set(spin), {"main", "teaser"})
        self.assertEqual(words_of(spin["main"][1]).count("record"), 2)

    def test_two_nonadjacent_source_bars_become_one_repeatable_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            time = np.arange(6 * RATE) / RATE
            for stem, gain in [("instrumental", 4100), ("drums", 1900), ("clean-vocal", 1000)]:
                samples = np.where(
                    time < 2.4,
                    gain * np.sin(2 * np.pi * 180 * time),
                    gain * np.sin(2 * np.pi * 360 * time),
                ).astype("<i2")
                with wave.open(str(root / (stem + ".wav")), "wb") as wav:
                    wav.setparams((2, 2, RATE, 0, "NONE", ""))
                    wav.writeframes(np.repeat(samples, 2).tobytes())
            (root / "master.info.json").write_text('{"id":"fixture"}')
            first = dict(grid=[0.2, 1.8], voice_ms=[290, 900], lines=[[0, 0, 1]])
            last = dict(grid=[3.0, 4.85], voice_ms=[3240, 3850], lines=[[1, 0, 1]])
            recipe = dict(
                slug="fixture",
                recording="fixture",
                artist="Singer",
                title="Studio",
                rows=[[["One"], [[310, 810]]], [["Two"], [[3300, 3720]]]],
                cuts=[
                    dict(id="main", label="Two then one", sections=[last, first]),
                    dict(id="teaser", label="One", sections=[first]),
                ],
            )
            song = prepare(recipe, root, root / "imports")
            cuts = json.loads((song / "song.json").read_text())["cuts"]
            catalog = {
                c["id"]: dict(
                    c, id=c["id"], title="Studio", loopFamily="full", label=c["label"], cuts=cuts
                )
                for c in cuts
            }
            choices = backing_loops(catalog)["full"]
            self.assertEqual(choices["main"].source, "main")
            self.assertEqual(choices["teaser"].source, "teaser")
            main = catalog["main"]
            self.assertEqual(words_of(main["rows"]["lead"]), ["two", "one"])
            for name in ("backing", "drums", "guide"):
                with wave.open(str(song / main["audio"] / (name + ".wav"))) as wav:
                    self.assertEqual(wav.getnframes(), round(2 * BAR * RATE))
                    pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")[::2]
                for bar, hz in enumerate((360, 180)):
                    at = round((bar + 0.24) * BAR * RATE)
                    sample = pcm[at : at + 4800]
                    self.assertAlmostEqual(
                        np.argmax(abs(np.fft.rfft(sample))) * RATE / len(sample), hz, delta=10
                    )
                    end = round((bar + 0.85) * BAR * RATE)
                    self.assertEqual(bool(np.any(pcm[end : end + 960])), name != "guide")
