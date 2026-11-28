"""Compare the two mic inputs at the output of the production mixer."""

import ctypes as C
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import FP, PARAMS, SP, VOICES
from desk import Desk
from desk_audio import DeskAudio


class MicEffects(unittest.TestCase):
    def test_carrier_assignment_and_character_rehearsal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t = np.arange(48000) / 48000
            guide = (2400 * np.sin(2*np.pi*300*t)).astype("<i2")
            for name, samples in (("guide", guide), ("bed", np.zeros_like(guide))):
                with wave.open(str(root / (name + ".wav")), "wb") as file:
                    file.setparams((1, 2, 48000, 0, "NONE", ""))
                    file.writeframes(samples.tobytes())
            song = dict(id="A", title="A", label="A", track="A", start=0, end=1,
                        bpm=120, loop=True, guide=root / "guide.wav", rows={}, key="D minor",
                        score=dict(events=[[0, [69]]], duration=1))
            bed = root / "bed.wav"
            with patch("desk.catalog", return_value={"A": song}), patch("desk.STATE", root / "session"), \
                    patch("desk.playback_files", return_value=(bed, bed)), patch.object(DeskAudio, "resources"):
                desk = Desk()
                try:
                    desk.open_audio = lambda: True
                    desk.cells[0][0].assign("A")
                    self.assertEqual(desk.strips[0]["voice"], 0)
                    desk.strips[0]["anchor"] = "autotune"
                    desk.configure()
                    self.assertIn("D minor", desk.cells[0][1].get_accessible().get_name())
                    # A carrier on a later strip must still be configured.
                    desk.set_song_assignment(2, "anchor", "mic:1")
                    desk.cells[1][1].assign("A")
                    self.assertEqual(desk.cells[1][2].get_label(), "Dry")
                    self.assertEqual(desk.cells[2][2].get_label(), "Dry")
                    desk.voice(1, 1)
                    desk.voice(2, 1)
                    self.assertEqual(desk.cells[1][2].get_label(), "Vocoder Clear")
                    self.assertEqual(desk.cells[2][2].get_label(), "Vocoder Clear")
                    self.assertEqual(desk.cells[2][2].get_tooltip_text(), "Vocoder notes from A")
                    # With a natural Hear assignment, explicitly choosing the
                    # vocoder on Red uses Red's own displayed song score.
                    desk.set_song_assignment(0, "anchor", "natural")
                    desk.voice(0, 1)
                    self.assertEqual(desk.cells[0][2].get_tooltip_text(), "Vocoder notes from A")
                    desk.voice(0, 0)
                    desk.install()
                    desk.test.refresh()
                    desk.test.buttons[1].clicked()

                    def spectrum():
                        timeout = time.monotonic() + 5
                        while any(desk.test.pending) and time.monotonic() < timeout:
                            desk.test.refresh()
                            time.sleep(.005)
                        self.assertFalse(any(desk.test.pending))
                        self.assertTrue(desk.audio.call("playing"))
                        left, right = np.zeros((2, 48000), "f4")
                        desk.audio.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
                        return abs(np.fft.rfft(left * np.hanning(len(left))))

                    desk.mix_controls[1].set_active_id("mute")
                    wet = spectrum()
                    self.assertGreater(wet[440], 15)
                    desk.voice(1, 0)
                    dry = spectrum()
                    self.assertGreater(dry[300], 300)
                    self.assertLess(dry[440], 1)

                    # A busy GPU leaves the unconverted guide on the strip.
                    # Its selected character must not silently apply a MIDI carrier.
                    with patch("desk.character_worker", side_effect=OSError("GPU in use")):
                        desk.voice(1, 3)
                        desk.cells[1][1].assign("A")
                        self.assertEqual(desk.worker_status[1], "GPU in use")
                        mix = spectrum()
                        self.assertGreater(mix[300], 300)
                        self.assertLess(mix[440], .01 * mix[300])
                finally:
                    desk.window.destroy()

    def test_switching_effects_releases_and_reacquires_the_score_on_live_audio(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        t = np.arange(24000) / 48000
        mic = (.065 * np.sin(2*np.pi*300*t)).astype("f4")
        inputs = (FP * 4)()
        inputs[1] = mic.ctypes.data_as(FP)
        values = (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS))
        for key, note in [(b"red", 69), (b"green", 81)]:
            audio.call("note", key, 0, 1, note)
            audio.call("score", key, 1, None, 0)
        audio.prepare_track(np.zeros(96000, "i2"), None, dict(start=0, end=1, bpm=120))
        audio.call("launch_at", -1, 0)
        audio.call("live", 1, 1)
        audio.call("transport", 0, 1)
        # A half-second block contains 150 input cycles, so changes below are
        # driven by control transitions on one mixer with continuous input.
        for voice, key in [(1, b"red"), (3, b"red"), (3, b"green"), (2, b"green"),
                           (1, b"green"), (0, b"green"), (1, b"red"), (1, b"missing")]:
            with self.subTest(voice=VOICES[voice], key=key):
                audio.call("config", 1, voice, key, 0, 0, values)
                left, right = np.zeros((2, len(t)), "f4")
                audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(t))
                spectrum = np.abs(np.fft.rfft(left * np.hanning(len(t))))
                if voice == 1 and key in (b"red", b"green"):
                    hz = 440 if key == b"red" else 880
                    self.assertGreater(spectrum[hz // 2], 5)
                else:
                    self.assertLess(spectrum[440 // 2], 1)
                    self.assertLess(spectrum[880 // 2], 1)
                    self.assertGreater(left.std(), .01)

    def test_blue_monday_bass_rests_do_not_cut_syllables(self):
        from test_singing import spoken_word

        audio = DeskAudio()
        self.addCleanup(audio.close)
        # Select original note lengths to hear the dry syllable in the bass rest.
        audio.call("glide", 0, 0)
        # First note and first rest of the installed Blue Monday MIDI score.
        audio.call("note", b"Blue Monday", 0, .11561, 41)
        audio.call("note", b"Blue Monday", .23121, .34634, 41)
        audio.call("score", b"Blue Monday", 1, None, 0)
        region = dict(track="Blue Monday", start=0, end=1, bpm=130.29)
        audio.prepare_track(np.zeros(96000, "i2"), None, region)
        audio.call("launch_at", -1, 0)
        values = (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS))
        audio.call("config", 0, 1, b"Blue Monday", 0, 0, values)
        audio.call("live", 0, 1)
        audio.call("transport", 0, 1)
        word = spoken_word("sister").astype("f4")
        speech = .15 * np.resize(word, 24000) / max(abs(word))
        inputs = (FP * 4)()
        inputs[0] = speech.ctypes.data_as(FP)
        left, right = np.zeros((2, len(speech)), "f4")
        audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
        rest = slice(6000, 10800)
        self.assertGreater(speech[rest].std(), .001)
        # The master limiter can still release during a rest.
        self.assertGreater(left[rest].std() / speech[rest].std(), .7)
        self.assertGreater(np.corrcoef(left[rest], speech[rest])[0, 1], .99)
        held = slice(480, 5200)
        # The consonant's amplitude contour survives on the modulated upper
        # bands, even though its original waveform is absent from a wet voice.
        def envelopes(pcm):
            bright = np.diff(pcm[held])
            return np.sqrt(np.mean(bright[:len(bright)//240*240].reshape(-1, 240)**2, axis=1))
        self.assertGreater(np.corrcoef(envelopes(speech)[:-1], envelopes(left)[1:])[0, 1], .65)
        self.assertGreater(np.std(left[held] - speech[held]), .002)

    def test_vocal_matches_mic_through_each_effect(self):
        t = np.arange(48000) / 48000
        # Off-key, with harmonics to excite the vocoder and a gap for effect tails.
        pcm = (1800 * (np.sin(2*np.pi*276*t) + .4*np.sin(2*np.pi*552*t))
               * (t < .65)).astype("i2")
        dry = pcm.astype("f4") / 32768
        song = dict(track="guide", start=0, end=1, loop=True, bpm=120)
        bed = np.zeros(2 * len(t), "i2")
        values = (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS))
        carrier = np.repeat((12000 * np.sin(2*np.pi*400*t)).astype("i2"), 2)

        def render(test, anchor=b"notes", voice=0, mask=0, flags=0, reverb=0):
            audio = DeskAudio()
            try:
                audio.call("note", b"notes", 0, 1, 69)  # A4, 440 Hz
                audio.call("score", b"notes", 1, None, 0)
                audio.call("note", b"Doin' It Right", 0, 1, 64)
                audio.call("score", b"Doin' It Right", 1, None, 0)
                audio.call("score", b"recording", 1, carrier.ctypes.data_as(SP), len(carrier))
                audio.prepare_track(bed, None, song)
                audio.call("launch_at", -1, 0)
                audio.call("config", 2, voice, anchor, mask, flags, values)
                audio.call("reverb", 2, reverb)
                audio.call("live", 2, int(not test))
                inputs = (FP * 4)()
                if test:
                    audio.test_source(2, pcm, song, song)
                    audio.call("test_lanes", 1 << 2)
                else:
                    inputs[2] = dry.ctypes.data_as(FP)
                audio.call("transport", 0, 1)
                left, right = np.zeros((2, len(t)), "f4")
                audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(t))
                np.testing.assert_array_equal(left, right)
                return left
            finally:
                audio.close()

        for test in (False, True):
            np.testing.assert_array_equal(render(test, anchor=b"missing", voice=1), render(test))
            for voice in (0, 2, *range(3, len(VOICES))):
                baseline = render(test, anchor=b"natural", voice=voice)
                for anchor in (b"notes", b"recording"):
                    with self.subTest(rehearsal=test, voice=VOICES[voice], anchor=anchor):
                        np.testing.assert_array_equal(render(test, anchor=anchor, voice=voice), baseline)
        for name, settings in (
            ("MIDI carrier", dict(voice=1)),
            ("transcribed carrier", dict(voice=1, anchor=b"recording")),
            ("Technologic", dict(voice=2, anchor=b"natural")),
            ("pitch", dict(mask=1)),
            ("pulse", dict(flags=1)),
            ("delay", dict(flags=2)),
            ("compression", dict(flags=4)),
            ("reverb", dict(reverb=.5)),
            ("combined", dict(voice=1, mask=1, flags=7, reverb=.5)),
        ):
            with self.subTest(effect=name):
                mic = render(False, **settings)
                rehearsal = render(True, **settings)
                np.testing.assert_allclose(rehearsal, mic, rtol=0, atol=1e-6)
                self.assertGreater(rehearsal.std(), .001)
                self.assertGreater((rehearsal - dry).std(), .001)
                if settings.get("voice") and settings.get("anchor") != b"natural":
                    hz = 392 if settings.get("anchor") == b"recording" else 440  # G4 or A4
                    # The carrier frequency is absent from the guide.
                    window = slice(4800, 24000)
                    signal = rehearsal[window] * np.hanning(19200)
                    self.assertGreater(abs(np.fft.rfft(signal))[round(hz * 19200 / 48000)], 5)
