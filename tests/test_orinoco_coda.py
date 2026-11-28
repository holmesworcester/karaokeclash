"""Exercise the starred arrangement on the desk's actual cut and singer clocks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audience import Pane
from backing_loops import backing_loops, calibrated_catalog
from orinoco_coda import BAR, FIRST, RATE, SOURCE_BAR, SOURCE_START, assemble, song_record


class OrinocoCoda(unittest.TestCase):
    def test_written_harmony_follows_each_recording_cut(self):
        verse, coda, teaser = "Orinoco Flow", "studio-orinoco-flow/coda", "studio-orinoco-flow/teaser"
        songs = calibrated_catalog({key: {} for key in (verse, coda, teaser)})
        def held(key, bar):
            return [keys for t, keys in songs[key]["score"]["events"] if t <= bar * BAR][-1]
        f, c, g = [41, 53, 57, 60], [48, 55, 60, 64], [43, 55, 59, 62]
        for start in (0, 9, 18):
            self.assertEqual([held(verse, start + b) for b in (.1, .6, 1.1)], [f, c, g])
        for key in (coda, teaser):
            self.assertEqual([held(key, b) for b in (.1, 1.1, 1.6, 2.1, 20.1)], [g, f, c, g, g])

    def test_repeat_uses_the_recorded_title_responses_and_downbeat(self):
        # Give every source bar its own stereo fingerprint. The ending of the
        # video differs from the KAR, so a continuous video excerpt fails here.
        source = np.zeros((40 * RATE, 2), dtype="<i2")
        for bar in range(15):
            start = round((FIRST - SOURCE_START + bar * SOURCE_BAR) * RATE)
            end = round((FIRST - SOURCE_START + (bar + 1) * SOURCE_BAR) * RATE)
            source[start:end] = [bar + 1, -bar - 1]
        result = assemble(source)
        self.assertEqual(len(result), round(21 * SOURCE_BAR * RATE))
        centers = [round((i + .5) * SOURCE_BAR * RATE) for i in range(21)]
        numbers = [1, 2, 3, 6, 5, 6, 7, 8, 7, 8, 11, 12, 13, 14, 7, 8, 7, 8, 11, 12, 13]
        np.testing.assert_array_equal(result[centers], np.column_stack([numbers, -np.array(numbers)]))

    def test_one_library_row_two_laps_and_only_six_teaser_words(self):
        title = "Orinoco Flow"
        songs = {title: dict(id=title, title=title, track=title, start=0, end=20,
                             loop=False, bpm=126)}
        record = song_record({})
        for cut in record["cuts"]:
            key = "studio-orinoco-flow/" + cut["id"]
            songs[key] = dict(record | cut, id=key, track=key)
        songs = calibrated_catalog(songs)
        library = backing_loops(songs)
        self.assertEqual(list(library), [title])
        modes = library[title]
        self.assertEqual(modes["full"].source, title)
        pane = Pane(None, 0)
        self.addCleanup(pane.destroy)
        for mode, bars in [("teaser", 4), ("main", 21)]:
            choice = modes[mode]
            song = songs[choice.source]
            region = choice.region(song, 40.0)
            length = region["end"] - region["start"]
            self.assertAlmostEqual(length, bars * BAR, places=4)
            words = [w for _, w, _, _ in sorted(pane.timeline(song, region, None, 0)[2],
                                                 key=lambda word: word[2])]
            if mode == "main":
                self.assertEqual(words[:4], ["deep", "sea", "of", "clouds"])
                self.assertEqual(words[-2:], ["Ross", "and"])
                self.assertEqual(words.count("Orinoco"), 4)
                for lap in (0, length):
                    self.assertEqual(pane.lines_at(song, region, lap + .1, None)[0][0][2][0], "deep")
            else:
                expected = "We can sail we can sail".split()
                self.assertEqual(words, expected)
                lo, hi = region["vocal_window"]
                self.assertGreater(region["end"] - hi, 4.3)
                for lap in (0, length):
                    t = lap - region["start"]
                    self.assertEqual(list(pane.lines_at(song, region, t + lo + .16, None)[0][0][2]),
                                     expected)
                    self.assertEqual(pane.lines_at(song, region, t + hi + .07, None)[0], [])
