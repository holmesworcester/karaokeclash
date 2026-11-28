"""Assign the two Mac vocal parts through the Linux desk and render their shared clock."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import FP
from desk import Desk
from desk_audio import DeskAudio

LEAD = "Doin' It Right"
ROBOT = LEAD + " - Vocoder"
BACKING_TITLE = LEAD + " (backing)"
RATE = 48000
FULL_START = 56.629


def mac_parts():
    # Word boundaries from the 89 BPM Mac prompts, including the shortened return.
    def line(text, edges):
        return text.split(), list(zip(edges, edges[1:]))

    lead = dict(id=LEAD, title=LEAD, label=LEAD, artist="Daft Punk", track=LEAD,
                bpm=89, beat_zero=240/89, start=0, end=0, loop=False,
                rows={"lead": [line("If you do it right", [67416, 67901, 68420, 68919, 69425, 70052]),
                               line("If you lose your way tonight", [88989, 89663, 90337, 91011, 91685, 92360, 93647]),
                               line("If you do it right", [175281, 175766, 176285, 176784, 177290, 177917]),
                               line("That's how you know the magic's right",
                                    [244719, 245393, 246067, 246742, 247416, 248090, 249438, 250166])]})
    robot = dict(lead, id=ROBOT, part="Vocoder", rows={"lead": [
        line("Doing it right", [2697, 3054, 3202, 3667]),
        line("Doing it right", [56629, 56987, 57135, 57600]),
        line("Doing it right", [67416, 67773, 67921, 68387]),
        line("Doing it right", [175281, 175638, 175787, 176252]),
        line("Everybody will be dancing tonight", [238348, 238928, 239117, 239272, 239656, 240000])]})
    return {LEAD: lead, ROBOT: robot}


class DoinItRight(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        # Only the shared accompaniment exists. A second playback path must fail.
        bed = root / "private/linux-four/playback" / LEAD / "backing.wav"
        bed.parent.mkdir(parents=True)
        with wave.open(str(bed), "wb") as wav:
            wav.setparams((1, 2, 6000, 0, "NONE", ""))
            wav.writeframes(bytes(1520800 * 2))
        for ctx in (patch("playback.DATA", root), patch("desk.STATE", root / "session"),
                    patch("desk.catalog", side_effect=mac_parts), patch.object(DeskAudio, "resources")):
            ctx.start()
            self.addCleanup(ctx.stop)
        d = self.desk = Desk()
        self.addCleanup(d.window.destroy)
        d.open_audio = lambda: True
        d.library.search.set_text(LEAD)
        self.assertEqual([r.get_accessible().get_name() for r in d.library.items.values()
                          if r.get_child_visible()], [LEAD + " · Daft Punk", BACKING_TITLE + " · Daft Punk"])

    def assign(self, lane, key):
        d = self.desk
        d.library.rows.select_row(d.library.items[key])
        d.cells[lane][0].clicked()

    def render(self, frames, signal=None):
        inputs = (FP * 4)()
        if signal is not None:
            inputs[1] = signal.ctypes.data_as(FP)
        left, right = np.zeros((2, frames), dtype="f4")
        self.desk.audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), frames)
        return left

    def seek_recording(self, seconds):
        self.desk.seek.set_value(seconds - FULL_START)

    def assert_parts(self, recording_time, words):
        d = self.desk
        for lane, (key, expected) in enumerate(zip((LEAD, ROBOT), words)):
            backing, position, _ = d.pane_transport(lane)
            # Clocks can count previous laps differently after switching sources.
            distance = (backing["start"] + position - recording_time + d.duration / 2) % d.duration - d.duration / 2
            self.assertAlmostEqual(distance, 0, delta=1/RATE)
            pane = d.stage.panes[lane]
            song = pane.song_for(backing, None)
            self.assertEqual(song["id"], key)
            self.assertEqual(d.test.song_key(lane), key)
            lines, _ = pane.lines_at(song, backing, position, None, lead_ms=0)
            self.assertEqual([" ".join(row[2]) for row in lines], [expected] if expected else [])

    def test_lead_first_and_backing_join_during_full_playback(self):
        d = self.desk
        self.assign(0, LEAD)
        d.install()
        d.loop_controls[1][0]["full"].clicked()
        d.tempo.set_value(126)
        d.play.clicked()
        self.seek_recording(67.5)
        self.assign(1, ROBOT)
        d.loop_controls[2][0]["full"].clicked()
        self.assert_parts(67.5, ("If you do it right", "Doing it right"))
        self.assertNotIn(2, d.actions)
        for at, words in [(FULL_START, (None, "Doing it right")),
                          (90., ("If you lose your way tonight", None)),
                          (175.4, ("If you do it right", "Doing it right")),
                          (177.4, ("If you do it right", None)),
                          (239.4, (None, "Everybody will be dancing tonight")),
                          (250., ("That's how you know the magic's right", None))]:
            self.seek_recording(at)
            self.render(960)
            self.assert_parts(at + .02 * 126 / 89, words)

        # Hand the backing to the robot mid-song on the next quarter-bar boundary.
        d.quant.set_active_id(".25")
        self.seek_recording(67.45)
        d.source_buttons[2].clicked()
        remaining = d.pending["boundary"] - d.elapsed()
        frames = round(remaining * RATE * 89 / 126) + 48
        self.render(frames)
        d.tick()
        self.assertEqual(d.playing_slot, 2)
        self.assert_parts(67.45 + frames / RATE * 126 / 89, ("If you do it right", "Doing it right"))
        d.seek.set_value(d.duration - .01)
        self.render(round(.1 * RATE))
        self.assert_parts(FULL_START - .01 + .1 * 126 / 89, (None, "Doing it right"))

    def test_return_from_teaser_joins_the_other_singers_full_pass(self):
        d = self.desk
        self.assign(0, LEAD)
        d.install()
        self.assign(1, ROBOT)
        d.loop_controls[2][0]["teaser"].clicked()
        d.tempo.set_value(89)
        d.quant.set_active_id(".25")
        d.play.clicked()
        self.seek_recording(175.18)
        d.loop_controls[2][0]["full"].clicked()
        boundary = d.master_clock.seconds(d.actions[2][3])
        self.assertEqual(d.slot_clocks[2][1], "teaser")
        self.render(round((boundary - d.elapsed() + .04) * RATE))
        d.tick()
        self.assertEqual(d.slot_clocks[2][1], "full")
        self.assert_parts(FULL_START + d.elapsed(), ("If you do it right", "Doing it right"))

    def test_full_starts_with_one_vocoder_cycle_and_plays_to_the_last_frame(self):
        d = self.desk
        self.assign(1, ROBOT)
        self.assign(0, LEAD)
        for slot in (1, 2):
            d.loop_controls[slot][0]["full"].clicked()
        d.install()
        self.assertEqual(d.playing_slot, 2)
        first_lead = mac_parts()[LEAD]["rows"]["lead"][0][1][0][0] / 1000
        for slot in (1, 2):
            region = d.slot_region(slot, "full")
            self.assertEqual(region["start"], FULL_START)
            self.assertAlmostEqual(first_lead - region["start"], 4 * 240 / 89, delta=.001)
            self.assertEqual(round(region["end"] * RATE), 12166400)
        self.assertEqual(d.elapsed(), 0)
        self.assert_parts(FULL_START, (None, "Doing it right"))
        d.strips[1].update(voice=1, pulse=False, delay=False, compressor=False, reverb=0)
        d.configure()
        d.audio.call("live", 1, 1)
        d.tempo.set_value(89)
        d.play.clicked()
        # Full-wet vocoding changes an incoming mic only while a chord is scored.
        signal = np.random.default_rng(89).uniform(-.1, .1, 4000).astype("f4")
        for at, singing in [(FULL_START + .03, True), (67.45, True), (90., False),
                            (175.30, True), (177.4, False), (245., False)]:
            with self.subTest(at=at):
                self.seek_recording(at)
                pcm = self.render(len(signal), signal)
                delta = np.sqrt(np.mean((pcm[480:] - signal[480:]) ** 2))
                if singing:
                    self.assertGreater(delta, .01)
                else:
                    self.assertLess(delta, .0001)
        self.seek_recording(67.5)
        self.assert_parts(67.5, ("If you do it right", "Doing it right"))
