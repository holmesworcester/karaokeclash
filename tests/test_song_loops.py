"""Exercise the two reported pickups and the importer's bar/word clocks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audience import Pane
from backing_loops import backing_loops, calibrated_catalog, family_key, phrase_groups
from import_job import BAR, RATE, publish, validate
import library
from test_singing import prepare_vocal


class SongLoops(unittest.TestCase):
    def test_milkshake_teaser_stops_after_two_words_until_the_next_pass(self):
        title = "Milkshake"
        # The four-bar chorus starts the next phrase at 714 ms.
        rows = [(["My", "milkshake"], [[0, 214], [286, 690]]),
                (["brings", "all", "the", "boys", "to", "the", "yard"],
                 [[714, 952], [1048, 1095], [1167, 1238], [1286, 1452],
                  [1500, 1571], [1595, 1714], [1786, 2048]])]
        songs = calibrated_catalog({title: dict(id=title, title=title, track=title,
            start=0, end=0, loop=False, bpm=126, beat_zero=0, rows={"lead": rows})})
        song = songs[title]
        modes = backing_loops(songs)[title]
        teaser = modes["teaser"].region(song, 4 * BAR)
        main = modes["main"].region(song, 4 * BAR)
        pane = Pane(None, 0)
        self.addCleanup(pane.destroy)
        length = teaser["end"] - teaser["start"]
        self.assertAlmostEqual(length, 4 * BAR, places=4)
        self.assertAlmostEqual(length - teaser["vocal_window"][1], 6.919, places=3)
        for lap in (0, length, 2 * length):
            line = pane.lines_at(song, teaser, lap + .4, None)[0][0]
            self.assertEqual(list(line[2]), ["My", "milkshake"])
            for rest in (.715, 2.0, length - .01):
                self.assertEqual(pane.lines_at(song, teaser, lap + rest, None)[0], [])
        self.assertEqual(list(pane.lines_at(song, main, 1.3, None)[0][0][2]), rows[1][0])

    def test_installed_milkshake_recording_and_two_word_guide(self):
        songs = calibrated_catalog(library.catalog())
        if "studio-milkshake/full" not in songs:
            self.skipTest("Requires the private Kelis recording")
        families = backing_loops(songs)
        self.assertEqual([k for k in families if songs[k]["title"] == "Milkshake"], ["Milkshake"])
        modes = families["Milkshake"]
        self.assertEqual(modes["main"].source, "Milkshake")
        full = songs[modes["full"].source]
        with wave.open(str(full["bed"])) as backing:
            length = backing.getnframes() / backing.getframerate()
        region = modes["full"].region(full, length)
        self.assertEqual(region["start"], 0)
        self.assertAlmostEqual(region["end"], 182.813, places=2)
        last_words, last_spans = full["rows"]["lead"][-1]
        self.assertEqual(last_words[-1], "charge")
        self.assertGreater(last_spans[-1][0], 171000)
        teaser = songs[modes["teaser"].source]
        self.assertEqual([words for words, _ in teaser["rows"]["lead"]], [["My", "milkshake"]])
        voice = prepare_vocal(teaser, source="extracted")
        self.assertEqual(len(voice), round(4 * BAR * RATE))
        self.assertGreater(np.std(voice[:round(.69 * RATE)]), 100)
        self.assertFalse(voice[round(.7 * RATE):].any())
        for path in (teaser["bed"], teaser["bed"].with_name("drums.wav")):
            with wave.open(str(path)) as stem:
                self.assertEqual(stem.getnframes(), len(voice))
                for bar in range(4):
                    stem.setpos(round((bar + .5) * BAR * RATE))
                    self.assertGreater(np.std(np.frombuffer(stem.readframes(RATE // 2), "<i2")), 50)

    def test_two_times_doo_teaser_and_answered_chorus(self):
        # Excerpts from the original video score. The teaser shares a row with
        # the count, and the later chorus has sung pickups at both loop edges.
        rows = [
            (["Two", "times"], [[45949, 46340], [46409, 47212]]),
            (["Two", "times,", "do,", "do-do-do,", "do,", "do-do-do"],
             [[105048, 105332], [105508, 105972], [106428, 106652], [106889, 108192],
              [108288, 108635], [108749, 110128]]),
            (["Do,"], [[110128, 110352]]),
            (["Three", "times,", "do,", "do-do-do"],
             [[160408, 160888], [160888, 161553], [162508, 162652], [162968, 164000]]),
            (["Two", "times", "(two", "times,", "gimme", "more)"],
             [[164108, 164293], [164568, 165128], [165128, 165293], [165488, 165972],
              [166888, 167328], [167328, 167788]]),
            (["Three", "times", "(three", "times,", "more", "and", "more)"],
             [[167788, 168092], [168269, 168728], [168728, 169053], [169188, 169652],
              [170568, 170908], [170908, 171048], [171048, 171488]]),
            (["Two", "times", "(two", "times,", "gimme", "more)"],
             [[171488, 171752], [171968, 172428], [172428, 172692], [172888, 173353],
              [174728, 175053], [175548, 175772]]),
            (["Easy", "done,", "done,", "but", "give", "me", "more"],
             [[176028, 176568], [176568, 177028], [177028, 177248], [177248, 177468],
              [177468, 177728], [177728, 178120], [178188, 178673]]),
            (["Two", "times"], [[178868, 179132], [179348, 179908]]),
        ]
        title = "2 Times"
        songs = calibrated_catalog({title: dict(id=title, track=title, bpm=130, beat_zero=2.15,
                                               start=0, end=0, loop=False, rows={"lead": rows})})
        song = songs[title]
        modes = backing_loops(songs)[title]
        pane = Pane(None, 0)
        self.addCleanup(pane.destroy)
        for mode, phrases in (("teaser", [rows[1][0][2:]]), ("main", [w for w, _ in rows[4:8]])):
            with self.subTest(mode=mode):
                region = modes[mode].region(song, 226.8)
                length = region["end"] - region["start"]
                lane = pane.timeline(song, region, None, 0)[1][0]
                self.assertEqual([list(entry[3]) for entry in lane], phrases)
                for lap in (0, length):
                    for entry in lane:
                        visible, _ = pane.lines_at(song, region, lap + entry[-1] - .05, None)
                        self.assertEqual(visible[0][2], entry[3])
                    if mode == "teaser":
                        self.assertGreater(region["end"] - region["vocal_window"][1], 2.7)
                        self.assertEqual(pane.lines_at(song, region, lap + length - 1, None)[0], [])

    def test_longer_studio_chorus_joins_the_original_chant_and_supplies_a_short_teaser(self):
        title = "One Week"
        chant = dict(id=title, title=title, label=title, track=title, start=0, end=4*BAR,
                     loop=True, bpm=126, rows={"lead": [(["Hold"], [[40, 320]])]})
        chorus_id = "studio-one-week/chorus"
        chorus = dict(chant, id=chorus_id, track=chorus_id, cuts=[{"id": "chorus"}],
                      loopFamily=title, loopRole="main", end=10*BAR, teaserBars=4,
                      teaserWindowMs=[1582, 3622], rows={"lead": [
                          (["It's", "been", "one", "week"], [[1598, 1800], [1800, 2100],
                            [2100, 2300], [2300, 2450]]),
                          (["sorry"], [[16600, 17600]])]})
        songs = calibrated_catalog({title: chant, chorus_id: chorus})
        families = backing_loops(songs)
        self.assertEqual(list(families), [title])
        self.assertEqual(families[title]["full"].source, title)
        for mode, bars in (("teaser", 4), ("main", 10)):
            loop = families[title][mode]
            self.assertEqual(loop.source, chorus_id)
            region = loop.region(songs[loop.source], 10*BAR)
            self.assertAlmostEqual(region["end"] - region["start"], bars*BAR, places=4)
            pane = Pane(None, 0)
            self.addCleanup(pane.destroy)
            words = {w for _, w, *_ in pane.timeline(chorus, region, None, 0)[2]}
            self.assertIn("It's", words)
            self.assertEqual("sorry" in words, mode == "main")
            if mode == "teaser":
                self.assertAlmostEqual(region["end"] - region["vocal_window"][1], 4, places=2)

    def test_songs_with_a_sung_edition_select_sung_teaser_and_main(self):
        songs = calibrated_catalog(library.catalog())
        sung_families = {family_key(song) for song in songs.values() if phrase_groups(song)}
        for key, modes in backing_loops(songs).items():
            if family_key(songs[key]) in sung_families:
                for mode in ("teaser", "main"):
                    with self.subTest(song=key, mode=mode):
                        self.assertTrue(phrase_groups(songs[modes[mode].source]))

    def test_reviewed_opening_and_chorus_on_the_installed_editions(self):
        # Word landmarks and independent drum / keyboard attacks from the masters.
        cases = [
            ("Blue Monday", 130.29, 2.238, 129.325, 136.690, 131108, 133352,
             134568, 131108, 158952, 160588),
            ("Blue (Da Ba Dee)", 127.99, 2.281, .406, 7.907, 1768, 4552,
             4929, 33848, 62852, 83268),
        ]
        openings = {
            "Blue Monday": (["How", "does", "it", "feel"],
                            [[131108, 131773], [132108, 132452], [132568, 132672], [132808, 133352]]),
            "Blue (Da Ba Dee)": (["Yo", "listen", "up", "here's", "a", "story"],
                                 [[1768, 2053], [2229, 2952], [3129, 3233], [3488, 3912],
                                  [4008, 4088], [4088, 4552]]),
        }
        for key, bpm, origin, down, wrap, attack, release, next_word, chorus, tail, verse in cases:
            with self.subTest(key=key):
                rows = [openings[key], (["next"], [[next_word, next_word+250]]),
                        (["refrain"], [[tail-300, tail]]), (["verse"], [[verse, verse+350]])]
                if chorus != attack:
                    rows.append((["chorus"], [[chorus, chorus+300]]))
                songs = calibrated_catalog({key: dict(id=key, track=key, bpm=bpm, beat_zero=origin, start=0,
                                                      end=0, loop=False, rows={"lead": rows})})
                modes = backing_loops(songs)[key]
                t = modes["teaser"].region(songs[key], 210)
                self.assertAlmostEqual(t["start"], down, delta=.015)
                self.assertAlmostEqual(t["end"], wrap, delta=.015)
                lo, hi = t["vocal_window"]
                self.assertLessEqual(lo, attack/1000)
                self.assertGreaterEqual(hi, release/1000)
                self.assertLess(hi, next_word/1000)
                self.assertGreater(t["end"]-hi, 2.9)
                pane = Pane(None, 0)
                self.addCleanup(pane.destroy)
                for lap in (0, t["end"] - t["start"]):
                    position = lap + attack / 1000 - t["start"] + .02
                    visible, _ = pane.lines_at(songs[key], t, position, None)
                    self.assertEqual(list(visible[0][2]), openings[key][0])
                    gap = lap + hi - t["start"] + .1
                    self.assertEqual(pane.lines_at(songs[key], t, gap, None)[0], [])
                m = modes["main"].region(songs[key], 210)
                lo, hi = m.get("vocal_window", (m["start"], m["end"]))
                self.assertLessEqual(lo, chorus/1000)
                self.assertGreaterEqual(hi, tail/1000)
                self.assertLessEqual(hi, verse/1000)
                if key.startswith("Blue ("):
                    self.assertGreater(m["start"], next_word/1000)

    def test_legacy_import_finds_opening_before_the_prepared_chorus(self):
        full = dict(id="old/full", cuts=["full", "hook-1"], start=0, end=12, loop=False,
                    bpm=240, downbeats=[.10, 1.14, 2.18, 3.16, 4.09, 5.13, 6.12, 7.10,
                                        8.11, 9.12, 10.13, 11.11],
                    rows={"lead": [(["Opening"], [[5030, 6180]]), (["Chorus"], [[8650, 9250]])]})
        hook = dict(full, id="old/hook-1", label="Song · chorus", loop=True, start=0, rows={"lead": [(["Chorus"], [[0, 530]])]})
        modes = backing_loops({s["id"]: s for s in (full, hook)})[full["id"]]
        self.assertEqual(modes["main"].source, hook["id"])
        self.assertEqual(modes["teaser"].source, full["id"])
        r = modes["teaser"].region(full, 12)
        lo, hi = r["vocal_window"]
        self.assertIn(r["start"], full["downbeats"])
        self.assertIn(r["end"], full["downbeats"])
        self.assertLess(lo, 5.030)
        self.assertGreater(hi, 6.180)
        self.assertLess(hi, 8.650)
        self.assertGreater(r["end"] - hi, .25)

    def test_recorded_teaser_keeps_the_attack_and_decay_outside_highlight_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            guide = Path(tmp) / "guide.wav"
            with wave.open(str(guide), "wb") as wav:
                wav.setparams((1, 2, RATE, 0, "NONE", ""))
                wav.writeframes(np.full(RATE, 2000, "<i2").tobytes())
            voice = prepare_vocal(dict(guide=guide, rows={"lead": [(["Feel"], [[300, 650]])]},
                                       teaserWindowMs=[250, 800]))
            self.assertTrue(np.all(voice[:12000] == 0))
            self.assertTrue(np.all(voice[12000:38400] == 2000))
            self.assertTrue(np.all(voice[38400:] == 0))

    def test_import_preserves_pickup_and_gate_on_every_repeat(self):
        grid = [.1, 1.11, 2.08, 3.10, 4.12, 5.13, 6.11, 7.10, 8.12]
        teaser = dict(label="Opening", start_bar=0, bars=4, voice_ms=[780, 1320])
        chorus = dict(label="Chorus", start_bar=4, bars=3, voice_ms=[4730, 6650])
        meta = dict(title="Fixture", artist="Singer", bpm=240, meter=4, key="F",
                    review="Measured clicks and tones", downbeats=grid, teaser=teaser, hooks=[chorus],
                    rows=[[["Here", "again"], [[810, 960], [1010, 1300]]],
                          [["Next"], [[1490, 1800]]], [["Old", "tail"], [[3900, 4100], [4200, 4320]]],
                          [["Our", "chorus"], [[4800, 5060], [5210, 6600]]],
                          [["New", "verse"], [[6850, 7050], [7120, 7450]]]])
        for invalid, message in [(dict(teaser, voice_ms=[900, 1320]), "cuts through"),
                                 (dict(teaser, voice_ms=[780, 4000]), "cuts through"),
                                 (dict(teaser, bars=2, voice_ms=[1400, 1870]), "vocal-free beat")]:
            with self.assertRaisesRegex(ValueError, message):
                validate(dict(meta, teaser=invalid), 8.3)
        with self.assertRaisesRegex(ValueError, "lyric line"):
            validate(dict(meta, hooks=[dict(chorus, voice_ms=[5140, 6650])]), 8.3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "analysis.json").write_text(json.dumps(meta))
            t = np.arange(round(8.3 * RATE)) / RATE
            for name, hz in [("backing", 150), ("drums", 90), ("guide", 420)]:
                signal = 3000*np.sin(t * 2*np.pi*hz)
                with wave.open(str(root/(name+".wav")), "wb") as f:
                    f.setparams((2, 2, RATE, 0, "NONE", ""))
                    f.writeframes(np.repeat(signal.astype("<i2"), 2).tobytes())
            target = publish(root, root/"backing.wav", root/"songs", 8)
            with patch.object(library, "IMPORTS", target.parent):
                songs = library.catalog()
            modes = backing_loops(songs)[target.name + "/full"]
            for mode, label in [("teaser", ["Here", "again"]), ("main", ["Our", "chorus"])]:
                s = songs[modes[mode].source]
                r = modes[mode].region(s, s["end"])
                self.assertTrue(all(row[0] == label for row in s["rows"]["lead"]))
                with wave.open(str(s["guide"])) as f:
                    voice = np.frombuffer(f.readframes(f.getnframes()), "<i2")[::2]
                with wave.open(str(s["bed"])) as f:
                    bed = np.frombuffer(f.readframes(f.getnframes()), "<i2")[::2]
                self.assertEqual(len(voice), len(bed))
                self.assertAlmostEqual(len(voice)/RATE, (4 if mode == "teaser" else 9)*BAR, places=4)
                for lap in range(1 if mode == "teaser" else 3):
                    start = round(lap*3*BAR*RATE)
                    for beat, sounding in [(0.1, False), (0.84, True), (1.8 if mode == "teaser" else 2.85, False)]:
                        at = start+round(beat*BAR*RATE)
                        self.assertGreater(np.max(abs(bed[at:at+480])), 2000)
                        self.assertEqual(np.max(abs(voice[at:at+480])) > 500, sounding)
                if mode == "teaser":
                    self.assertGreater(r["end"] - r["vocal_window"][1], BAR)
