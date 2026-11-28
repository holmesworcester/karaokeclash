"""Check the Cowgirl vocal layers, cut selection, and repeating singer cues."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from cowgirl_loops import BAR, DOWNBEAT, RATE, RECORDING, matched_chant, prepare
from desk import Desk
from desk_audio import DeskAudio
import library


class CowgirlLoops(unittest.TestCase):
    def test_each_shifted_chant_retrigger_moves_out_of_the_main_guide(self):
        rng = np.random.default_rng(5)
        frames = round(4 * BAR * RATE)
        chant = sosfilt(butter(2, .14, output="sos"), rng.normal(0, .025, frames))
        backing = np.zeros(frames)
        for i in range(11):
            start = round(i * 1.5 * BAR / 4 * RATE)
            end = min(frames, round((i + 1) * 1.5 * BAR / 4 * RATE))
            lag = round(.004 * RATE * np.cos(i))
            backing[start:end] = (.32 + i/20) * np.take(chant, np.arange(start, end) - lag, mode="wrap")
        singer = .02 * np.sin(2 * np.pi * 222 * np.arange(frames) / RATE)
        estimate = matched_chant(chant, singer + backing)
        self.assertLess(np.linalg.norm(estimate - backing), .07 * np.linalg.norm(backing))

    def test_operator_buttons_select_the_foreground_words_and_recorded_layers(self):
        frames = round(4 * BAR * RATE)
        axis = np.arange(frames) / RATE
        chant = np.random.default_rng(7).uniform(-.025, .025, frames)
        hook = .05 * np.sin(2 * np.pi * 220 * axis)
        verse = .035 * np.sin(2 * np.pi * 570 * np.arange(round(32 * BAR * RATE)) / RATE)
        album_voice = verse.copy()
        origin = DOWNBEAT + 133 * BAR
        for onset in (277986, 285553, 293121, 300687):
            sample = round(((onset - 23) / 1000 - origin) * RATE)
            album_voice[sample:sample+frames] += .7 * chant
        drums = np.stack([.02 * np.cos(2 * np.pi * 61 * axis)] * 2, axis=1)
        bed = drums + .01

        def excerpt(path, first_bar, bars, channels):
            if path.name == "clean-vocal.wav":
                return {15: chant, 47: hook + .7 * chant, 133: album_voice}[first_bar][:, None]
            return drums if path.name == "drums.wav" else bed

        # These onsets and releases come from the local album edition.
        lines = [("I'm invisible", [[89289, 89468], [89468, 90732]]),
                 ("I'm invisible", [[94669, 95128], [95128, 96392]]),
                 ("I scream", [[254728, 254788], [254788, 255572]]),
                 ("beautiful dream", [[268488, 269208], [269208, 269652]])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "studio-cowgirl"
            target.mkdir()
            (root / "reference.info.json").write_text(json.dumps({"id": RECORDING}))
            (root / "reviewed.lrc").write_text("\n".join("[00:00]" + line for line, _ in lines))
            (root / "timing.json").write_text(json.dumps([times for _, times in lines]))
            with patch("cowgirl_loops.excerpt", side_effect=excerpt):
                prepare(root, target)
            with patch.object(library, "IMPORTS", root):
                songs = {k: s for k, s in library.catalog().items() if s["title"] == "Cowgirl"}
            songs.setdefault("Cowgirl", dict(id="Cowgirl", track="Cowgirl", title="Cowgirl", label="Cowgirl",
                                             start=0, end=0, loop=False, bpm=126.92))
            with patch("desk.catalog", return_value=songs), patch("desk.STATE", root / "session"), \
                    patch.object(DeskAudio, "resources"):
                desk = Desk()
                try:
                    self.assertEqual(list(desk.library.items), ["Cowgirl"])
                    desk.cells[0][0].assign("Cowgirl")
                    desk.install()
                    pane = desk.stage.panes[0]
                    for mode, bars, first_words in (("teaser", 4, ["Everything", "everything"]),
                                                    ("main", 4, ["I'm", "invisible"]),
                                                    ("full", 32, ["I", "scream"])):
                        with self.subTest(mode=mode):
                            desk.loop_controls[1][0][mode].clicked()
                            desk.install()
                            song = songs[desk.test.song_key(0)]
                            self.assertEqual(song["id"], "studio-cowgirl/" + mode)
                            self.assertAlmostEqual(desk.duration, bars * BAR, delta=1/RATE)
                            when = song["rows"]["lead"][0][1][0][0]/1000 + .03
                            for lap in (0, desk.duration):
                                visible, _ = pane.lines_at(song, desk.backing, lap + when, None)
                                self.assertEqual(list(visible[0][2]), first_words)
                                if mode == "teaser":
                                    self.assertFalse(pane.lines_at(song, desk.backing, lap + 2, None)[0])
                                if mode == "full":
                                    rows = song["rows"]["lead"]
                                    self.assertEqual(len(rows), 22)
                                    self.assertEqual([w.lower().strip(",") for w in rows[15][0]], ["everything"] * 7)
                                    self.assertEqual(rows[-1][0], ["Am,", "an", "eraser", "of", "love"])
                                    for words, spans in rows:
                                        self.assertEqual(len(words), len(spans))
                                        for word, (a, b) in zip(words, spans, strict=True):
                                            at = lap + (a + b)/2000
                                            current, _ = pane.lines_at(song, desk.backing, at, None)
                                            self.assertTrue(any(word == w and (a, b) == tuple(times)
                                                                for line in current for w, times in zip(line[2], line[3])))
                finally:
                    desk.window.destroy()
            def read(mode, stem):
                return wavfile.read(target / mode / (stem + ".wav"))[1].astype(float) / 32768

            np.testing.assert_allclose(read("main", "guide"), hook, atol=.001)
            np.testing.assert_allclose(read("main", "backing")[:, 0], (bed[:, 0] + .7 * chant), atol=.001)
            full = read("full", "backing")[:, 0]
            phase = (np.arange(len(full)) + round(2 * BAR * RATE)) % frames
            def second(t):
                return slice(round((t-origin)*RATE), round((t+1-origin)*RATE))
            for t in (254, 279, 295, 307):
                idx = second(t)
                np.testing.assert_allclose(full[idx], (bed[:, 0] + .7 * chant + .55 * hook)[phase[idx]], atol=.001)
            response = second(287)
            np.testing.assert_allclose(full[response], (bed[:, 0] + .55 * hook)[phase[response]], atol=.001)
            self.assertFalse(read("teaser", "guide")[int(1.42 * RATE):].any())
            for mode in ("teaser", "main"):
                np.testing.assert_allclose(read(mode, "drums")[:frames], drums, atol=1/32768)
            np.testing.assert_allclose(read("full", "drums")[:, 0], drums[phase, 0], atol=1/32768)
            dry = read("full", "guide")
            self.assertFalse(dry[:3*RATE].any())
            np.testing.assert_allclose(dry[4*RATE:5*RATE], 1.65 * verse[4*RATE:5*RATE], atol=1/32768)
            np.testing.assert_allclose(dry[second(279)], 1.65 * verse[second(279)], atol=.002)
            np.testing.assert_allclose(dry[response], 1.65 * album_voice[response], atol=1/32768)
            self.assertFalse(dry[second(291)].any())
            self.assertGreater(dry[second(306.9)].std(), .01)
            self.assertFalse(dry[second(309)].any())
