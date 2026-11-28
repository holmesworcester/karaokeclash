"""Measure the layered ATC carrier at the chorus entrance in the native desk mix."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import F, FP
from desk import Desk, Gtk
from desk_audio import DeskAudio

RATE = 48000
SONG = "Around the World"
DOWNBEAT = 1.0064


def level(pcm, beat, start, end, midi):
    window = pcm[round(start * beat * RATE):round(end * beat * RATE)]
    t = np.arange(len(window)) / RATE
    hz = 440 * 2 ** ((midi - 69) / 12)
    return abs(np.sum(window * np.hanning(len(window)) * np.exp(-2j * np.pi * hz * t))) / len(window)


class ATCCarrier(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        bed = root / "private/linux-four/playback" / SONG / "backing.wav"
        bed.parent.mkdir(parents=True)
        with wave.open(str(bed), "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", ""))
            wav.writeframes(bytes(2 * 8000 * 225))

        # Older installations still supply the original plucked, single-note carrier.
        (root / "scores.json").write_text(json.dumps({SONG: {"duration": 8 * 60/126,
            "events": [[0, [76]], [.14087, []], [.23810, [69]], [.37897, []]]}}))
        song = dict(id=SONG, title=SONG, label=SONG, artist="ATC", track=SONG,
                    bpm=126, beat_zero=DOWNBEAT, start=0, end=0, loop=False,
                    rows={"lead": [(["La"], [[23831, 24083]])]})
        for ctx in (patch("desk.catalog", return_value={SONG: song}), patch("desk.STATE", root / "session"),
                    patch("playback.DATA", root),
                    patch.object(DeskAudio, "resources", lambda a: a.load_catalog_scores(root, root))):
            ctx.start()
            self.addCleanup(ctx.stop)
        d = self.desk = Desk()
        self.addCleanup(d.window.destroy)
        d.open_audio = lambda: True
        d.strips[0].update(voice=1, compressor=False)
        d.library.rows.select_row(d.library.items[SONG])
        d.cells[0][0].clicked()
        d.install()
        d.audio.call("vocoder", 0, 100, 0, 5, 40)
        d.audio.call("live", 0, 1)

    def control(self, name, root=None):
        def walk(w):
            yield w
            if isinstance(w, Gtk.Container):
                for child in w.get_children():
                    yield from walk(child)
        return next(w for w in walk(root or self.desk.window)
                    if w.get_accessible().get_name() == "Red " + name)

    def chorus(self, mode, bpm, bar=12, singer=None):
        d = self.desk
        d.tempo.set_value(bpm)
        d.loop_controls[1][0][mode].clicked()
        d.install()
        # The sung entrance is album bar 12. Teaser and Main start on bar 11.
        entrance = DOWNBEAT + bar * 240/126 - d.backing["start"]
        d.audio.call("transport", 1, round(entrance * RATE))
        d.play.clicked()
        frames = round(16 * 60/bpm * RATE)
        signal = np.random.default_rng(126).uniform(-.16, .16, frames).astype("f4")
        if singer is not None:
            signal = singer(np.arange(frames) / RATE).astype("f4")
        inputs = (FP * 4)()
        inputs[0] = signal.ctypes.data_as(FP)
        left, right = np.zeros((2, frames), dtype="f4")
        d.audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), frames)
        d.play.clicked()
        np.testing.assert_array_equal(left, right)
        return left

    def test_bass_pedal_and_plucked_riff_follow_album_phase_at_two_tempos(self):
        self.control("vocoder glide").set_value(0)
        self.control("vocal lead").set_value(0)
        for mode, bpm in (("full", 126), ("main", 110), ("teaser", 126)):
            with self.subTest(mode=mode, bpm=bpm):
                pcm = self.chorus(mode, bpm)
                beat = 60/bpm
                for lap in (0, 8):
                    def tone(start, end, midi):
                        return level(pcm, beat, lap + start, lap + end, midi)
                    # Through the gap after the upper C5, bass A2 and pedal E3 continue.
                    self.assertGreater(tone(1.33, 1.49, 45), .0004)
                    self.assertGreater(tone(1.33, 1.49, 52), .0002)
                    # C4 in the new held chord also contributes a C5 harmonic.
                    self.assertGreater(tone(1.05, 1.27, 72), tone(1.33, 1.49, 72))
                    # The bass moves A2, E2, F2, C3, G2 across the two bars.
                    for at, pitch in ((0,45), (2,40), (4,41), (6,48), (7,43)):
                        self.assertGreater(tone(at + .1, at + .85, pitch),
                                           3 * tone(at + .1, at + .85, pitch-2), (lap, at, pitch))
                    # E3 releases at F2. Measure a long window to resolve F2's second harmonic.
                    self.assertGreater(tone(.1, 1.9, 52), 5 * tone(4.1, 5.9, 52))

    def test_inline_stereo_width_is_saved_per_mic_and_resets(self):
        from desk import settings
        width = self.control("vocoder stereo width")
        self.assertTrue(width.get_mapped())
        self.assertEqual(width.get_value(), 100)
        width.set_value(24)
        self.desk.flush()
        saved = settings()["strips"]
        self.assertEqual([s["stereo_width"] for s in saved], [24, 100, 100, 100])
        self.control("reset vocoder controls").clicked()
        self.assertEqual(width.get_value(), 100)

    def test_glide_knob_bends_riff_through_the_gap_and_preserves_beat(self):
        knob = self.control("vocoder glide")
        self.assertEqual(knob.get_value(), 100)
        self.control("vocal lead").set_value(0)
        for mode, bpm in (("full", 126), ("main", 110)):
            knob.set_value(100)
            gliding = self.chorus(mode, bpm)
            knob.set_value(0)
            stepped = self.chorus(mode, bpm)
            # Between E5 and A4, the glide crosses C#5. The first attack still lands on E5.
            beat = 60/bpm
            self.assertGreater(level(gliding, beat, .36, .46, 73), 2 * level(stepped, beat, .36, .46, 73))
            self.assertGreater(level(gliding, beat, .05, .23, 76), 2 * level(gliding, beat, .05, .23, 73))

    def test_lead_dial_follows_the_recording_and_ends_with_the_sung_phrase(self):
        lead = self.control("vocal lead")
        self.assertEqual(lead.get_value(), 70)
        for bar in (12, 85):
            lead.set_value(0)
            off = self.chorus("full", 126, bar)
            lead.set_value(100)
            on = self.chorus("full", 126, bar)
            diff = on-off
            def rms(a,b):
                return np.mean(diff[round(a*60/126*RATE):round(b*60/126*RATE)]**2)**.5
            # Both choruses enter with two sung notes then leave a full beat of air.
            self.assertGreater(rms(.1,.9), .003)
            self.assertLess(rms(3.2,4.3), .04*rms(.1,.9))
            root = 45 if bar == 12 else 46
            self.assertGreater(level(on, 60/126, .1, 1.75, root),
                               3 * level(on, 60/126, .1, 1.75, root-1))

    def test_full_chord_carries_two_vowel_shapes_and_falls_silent_with_singer(self):
        beat = 60/126
        def singer(t):
            # A pitched glottal source with the first two resonances of "ah" and "ee".
            def vowel(f1, f2):
                y = np.zeros_like(t)
                for h in range(1, 30):
                    hz = 170*h
                    gain = np.exp(-.5*((hz-f1)/120)**2) + .85*np.exp(-.5*((hz-f2)/180)**2)
                    y += .07*gain*np.sin(2*np.pi*hz*t)
                return y
            return (t < 15*beat) * np.where(t < 8*beat, vowel(750,1200), vowel(320,2400))
        self.control("vocoder preset").set_active_id("Vowel stack")
        pcm = self.chorus("full", 126, singer=singer)
        def formant_ratio(start):
            window = pcm[round(start*beat*RATE):round((start+5)*beat*RATE)]
            power = abs(np.fft.rfft(window*np.hanning(len(window))))**2
            hz = np.fft.rfftfreq(len(window), 1/RATE)
            return power[(hz>1950)&(hz<2850)].sum() / power[(hz>800)&(hz<1600)].sum()
        self.assertGreater(formant_ratio(9), 5 * formant_ratio(1))
        self.assertLess(np.max(abs(pcm[-2400:])), .002)

    def test_middle_e_has_its_own_quieter_synth_level(self):
        normal = self.chorus("full", 126)
        gains = (F * 128)(*([1.] * 128))
        for bass in (45, 40, 41, 48, 43):
            gains[bass] = .75
        self.desk.audio.call("score_gains", SONG.encode(), gains)
        louder = self.chorus("full", 126)
        # Compare the E to the root in the same chord to remove overall level differences.
        def balance(pcm):
            return level(pcm, 60/126, .1, 1.9, 52) / level(pcm, 60/126, .1, 1.9, 45)
        self.assertLess(balance(normal), .85 * balance(louder))
