"""Exercise the scored wet bass on the same native transport as the live mics."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import F, FP, PARAMS
from clip_clock import ClipClock
from desk import carrier_id
from desk import TAL_VOCODER
from desk_audio import DeskAudio

RATE = 48000


def amplitude(pcm, begin, end, pitch):
    samples = pcm[round(begin * RATE) : round(end * RATE)]
    hz = 440 * 2 ** ((pitch - 69) / 12)
    return abs(
        np.mean(
            samples
            * np.hanning(len(samples))
            * np.exp(-2j * np.pi * hz * np.arange(len(samples)) / RATE)
        )
    )


class VocoderBass(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = DeskAudio()
        self.addCleanup(self.audio.close)
        # Legacy audio-only installs supply the two-bar Fireflies arpeggiator as a clip.
        riff = 0.24 * np.sin(2 * np.pi * 622.25 * np.arange(8 * 8000) / 8000)
        with wave.open(str(self.root / "riff.wav"), "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", ""))
            wav.writeframes(np.rint(32767 * riff).astype("<i2").tobytes())
        scores = {"2Fly - Rap": {"audio": "riff.wav"}}
        for anchor, bpm in [("Lights Burn Dimmer", 106), ("My Heart Will Go On", 131)]:
            scores[anchor + " / Fergalicious"] = {
                "events": [[0, [59]], [6 * 60 / bpm, [62]]],
                "duration": 16 * 60 / bpm,
            }
        (self.root / "scores.json").write_text(json.dumps(scores))
        self.audio.load_catalog_scores(self.root, self.root)

    def render(
        self, key, bpm, zero, start, length, voice=1, wet=100, speed=1, mute_after=None
    ):
        a = self.audio
        region = dict(
            track=key,
            start=start,
            end=start + 4 * length,
            loop=True,
            bpm=bpm,
            beat_zero=zero,
        )
        clock = ClipClock(region)
        bed = np.zeros(2 * round(region["end"] * RATE), "i2")
        self.assertGreater(a.prepare_track(bed, None, region), 0)
        a.call("stage_clock", *a.clock_args(clock))
        self.assertGreater(a.call("launch_at", -1, 0), 0)
        a.call("carrier_clock", 0, 0, bpm)
        a.call(
            "config",
            0,
            voice,
            key.encode(),
            0,
            0,
            (F * len(PARAMS))(*(p[3] for p in PARAMS)),
        )
        a.call("vocoder", 0, wet, 50, 5, 40)
        a.call("live", 0, 1)
        a.call("backing_enabled", False)
        a.call("tempo", bpm * speed)
        a.call("transport", 1, 0)
        a.call("transport", 0, 1)
        t = np.arange(round(length * RATE)) / RATE
        voice_in = (
            0.09 * np.sin(2 * np.pi * 230 * t) + 0.03 * np.sin(2 * np.pi * 460 * t)
        ).astype("f4")
        if mute_after is not None:
            voice_in[t > mute_after] = 0
        inputs = (FP * 4)()
        inputs[0] = voice_in.ctypes.data_as(FP)
        left, right = np.zeros((2, len(t)), "f4")
        a.call(
            "render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(t)
        )
        np.testing.assert_array_equal(left, right)
        return left, voice_in

    def test_september_offbeats_use_the_bass_for_the_selected_album_section(self):
        # Bar 17 has D in the verse. Bar 33 has A at the same place in the chorus.
        beat = 60 / 130
        for bar, pitch, speed, mode in [
            (17, 26, 1, 1),
            (33, 33, 0.85, -1),
            (34, 28, 1, TAL_VOCODER),
        ]:
            with self.subTest(bar=bar, speed=speed, mode=mode):
                room = beat / speed
                pcm, _ = self.render(
                    "Cry for You",
                    130,
                    2.704,
                    2.704 + 4 * bar * beat,
                    3 * room,
                    voice=mode,
                    speed=speed,
                )
                for i in (0, 1, 2):
                    body = amplitude(pcm, (i + 0.57) * room, (i + 0.90) * room, pitch)
                    self.assertGreater(body, 0.006)
                    self.assertGreater(
                        body,
                        3 * amplitude(pcm, (i + 0.12) * room, (i + 0.40) * room, pitch),
                    )
                    self.assertGreater(
                        body,
                        2
                        * amplitude(
                            pcm, (i + 0.57) * room, (i + 0.90) * room, pitch + 5
                        ),
                    )

    def test_vocal_controls_and_silence_gate_the_wet_bass(self):
        beat = 60 / 130
        start = 2.704 + 32 * 4 * beat
        wet, _ = self.render("Cry for You", 130, 2.704, start, 1.5, mute_after=0.5)
        self.assertGreater(amplitude(wet, 0.265, 0.43, 35), 0.006)
        self.assertLess(np.max(np.abs(wet[RATE:])), 0.0001)
        dry, original = self.render("Cry for You", 130, 2.704, start, 1.5, wet=0)
        np.testing.assert_allclose(dry[1000:], original[1000:], atol=1e-6)

    def test_imported_loop_and_legacy_mashup_receive_their_own_pitches(self):
        key = "studio-cowgirl/teaser"
        self.audio.import_scores(
            {key: {"bpm": 126.92, "score": {"events": [[0, [62]]], "duration": 7.5638}}}
        )
        pcm, _ = self.render(key, 126.92, 0, 0, 0.65)
        self.assertGreater(amplitude(pcm, 0.10, 0.60, 31), 0.006)
        for anchor, bpm in [("Lights Burn Dimmer", 106), ("My Heart Will Go On", 131)]:
            target = carrier_id(
                {"anchor": anchor, "lyrics": "Fergalicious"}, self.audio
            )
            pcm, _ = self.render(target, bpm, 0, 5 * 60 / bpm, 1.6)
            self.assertGreater(amplitude(pcm, 0.06, 0.40, 31), 0.006)
            self.assertGreater(amplitude(pcm, 0.75, 1.20, 38), 0.006)

    def test_audio_only_excerpt_keeps_its_riff_and_gets_bass_at_the_room_tempo(self):
        # The chant was recorded at 106 BPM. The lyric edition runs at 90.
        pcm, _ = self.render("2Fly - Rap", 90, 0, 0, 2.6)
        for begin, end, pitch in [(0.15, 1.15, 34), (1.50, 2.45, 39)]:
            self.assertGreater(amplitude(pcm, begin, end, pitch), 0.006)
            self.assertGreater(amplitude(pcm, begin, end, 75), 0.00012)

    def test_direct_scores_handle_songs_outside_the_chant_bank(self):
        for name in [
            "Lights Burn Dimmer",
            "Wicked Game",
            "O Superman",
            "O Superman - Ah",
        ]:
            self.assertTrue(self.audio.call("has_carrier", name.encode()), name)
        self.assertEqual(
            carrier_id(
                {"anchor": "Lights Burn Dimmer", "lyrics": "Wicked Game"}, self.audio
            ),
            "Lights Burn Dimmer",
        )
