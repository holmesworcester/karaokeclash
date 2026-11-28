"""Check actual stereo output for backing ownership, follow links and test vocals."""
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import FP
from desk import Desk, VOCODER_HOLD, settings
from desk_audio import DeskAudio


class AudioRouting(unittest.TestCase):
    # Each song and stem has a distinct stereo sample. The guide is mono.
    STEMS = {
        "red": ((100, 220), (310, 460), 1100),
        "blue": ((510, 630), (710, 890), 2100),
        "green": ((920, 1050), (1220, 1370), 3100),
    }

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = root = Path(temp.name)
        songs, paths = {}, {}
        for name, (full, drums, guide) in self.STEMS.items():
            files = []
            for stem, frame in (("full", full), ("drums", drums), ("guide", (guide,))):
                path = root / f"{name}-{stem}.wav"
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((len(frame), 2, 48000, 0, "NONE", ""))
                    wav.writeframes(np.asarray(frame, dtype="<i2").tobytes() * 48000)
                files.append(path)
            songs[name] = dict(id=name, title=name, label=name, track=name, bpm=240,
                               start=0, end=1, loop=True, guide=files[2], rows={})
            paths[name] = files[:2]
        for ctx in (patch("desk.catalog", return_value=songs), patch("desk.STATE", root / "session"),
                    patch("desk.playback_files", side_effect=lambda s: paths[s["id"]]),
                    patch.object(DeskAudio, "resources")):
            ctx.start()
            self.addCleanup(ctx.stop)
        d = self.desk = Desk()
        self.addCleanup(d.window.destroy)
        d.tempo.set_value(240)
        d.open_audio = lambda: True
        for strip in d.strips:
            strip["compressor"] = False
        d.cells[0][1].assign("red")
        d.install()
        d.cells[1][0].assign("blue")
        self.follow(1, 0)
        d.quant.set_active_id(".25")

    def follow(self, lane, leader):
        d = self.desk
        d.set_song_assignment(lane, "anchor", f"mic:{leader}")
        d.configure()
        d.assignment_changed(lane + 1)

    def render(self, frames):
        left, right = np.zeros((2, frames), dtype="f4")
        self.desk.audio.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), frames)
        return np.rint(np.stack((left, right), axis=1) * 32768).astype(int)

    def assert_output(self, pcm, stereo, guide=0):
        expected = np.broadcast_to(np.array(stereo) + guide, pcm.shape)
        np.testing.assert_array_equal(pcm, expected)

    def test_carrier_backing_plays_scored_pitches_and_rests_without_a_mic(self):
        d = self.desk
        # Use a different score ID from the backing asset, as on imported recordings.
        d.songs["red"].update(scoreId="red MIDI", score={
            "duration": 1, "events": [[0, [57]], [.3, []], [.5, [69]], [.8, []]]})
        d.audio.import_scores(d.songs)
        d.cells[0][0].assign("red")
        d.set_song_assignment(0, "anchor", "natural")
        d.configure()
        d.mix_controls[1].set_active_id("carrier")
        d.play.clicked()

        def assert_pitch(hz):
            pcm = self.render(4800)
            np.testing.assert_array_equal(pcm[:, 0], pcm[:, 1])
            bins = abs(np.fft.rfft(pcm[:, 0] * np.hanning(len(pcm))))
            self.assertGreater(pcm[:, 0].std(), 500)
            self.assertEqual(10 * np.argmax(bins), hz)

        assert_pitch(220)
        d.play.clicked()
        self.assert_output(self.render(256), (0, 0))
        d.play.clicked()
        d.tempo.set_value(120)
        d.seek.set_value(.4)
        self.assert_output(self.render(4800), (0, 0))
        d.seek.set_value(.52)
        assert_pitch(440)
        d.seek.set_value(.99)
        self.render(960)  # Cross the loop seam at half speed.
        assert_pitch(220)
        d.tick()
        self.assertIn("MIDI carrier · red MIDI", d.status.get_text())
        d.flush()
        self.assertEqual(settings()["strips"][0]["playback_mode"], "carrier")
        # Switching the accompaniment leaves the playhead and live voice settings intact.
        frame = d.audio.call("clock")
        bar = d.conductor_bar()
        d.mix_controls[1].set_active_id("full")
        self.assertEqual(d.audio.call("clock"), frame % round(d.duration * 48000))
        self.assertAlmostEqual(d.conductor_bar(), bar)
        self.assert_output(self.render(256), self.STEMS["red"][0])

    def test_carrier_backing_follows_the_selected_mic_and_reports_a_missing_score(self):
        d = self.desk
        d.audio.call("note", b"red", 0, 1, 60)
        d.audio.call("score", b"red", 1, None, 0)
        d.configure()
        d.mix_controls[2].set_active_id("carrier")
        d.play.clicked()
        d.source_buttons[2].clicked()
        d.install()
        self.assert_output(self.render(12000), self.STEMS["red"][0])
        self.assertGreater(self.render(3000)[:, 0].std(), 500)
        d.tick()
        self.assertIn("MIDI carrier · red", d.status.get_text())
        d.mix_controls[2].set_active_id("mute")
        self.assert_output(self.render(256), (0, 0))
        d.cells[1][1].assign("blue")
        d.mix_controls[2].set_active_id("carrier")
        d.tick()
        self.assertIn("No carrier score", d.status.get_text())
        self.assert_output(self.render(256), (0, 0))

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/"linux-four"/"sweet-dreams-arp.json").is_file(),
                         "requires the installed recording chart")
    def test_following_mic_vocodes_sweet_dreams_on_drop_and_rejoins_the_recording(self):
        d = self.desk
        title = "Sweet Dreams"
        songs = dict(d.songs)
        songs[title] = dict(id=title, track=title, label=title, title=title, start=0, end=0,
                            loop=False, bpm=126, beat_zero=1.0064, rows={})
        path = self.root / "sweet.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((2, 2, 48000, 0, "NONE", ""))
            wav.writeframes(bytes(5 * 48000 * 4))

        def listen(seconds):
            t = np.arange(round(seconds * 48000)) / 48000
            voice = (.12 * np.sin(2*np.pi*190*t)).astype("f4")
            inputs = (FP * 4)()
            inputs[1] = voice.ctypes.data_as(FP)
            left, right = np.zeros((2, len(t)), "f4")
            d.audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(t))
            return left

        def level(pcm, hz):
            t = np.arange(len(pcm)) / 48000
            return abs(np.dot(pcm * np.hanning(len(t)), np.exp(-2j*np.pi*hz*t))) / len(t)

        with patch("desk.catalog", return_value=songs), patch("desk.playback_files", return_value=(path, path)):
            d.reload_songs()
            d.tempo.set_value(126)
            d.voice(1, 1)
            d.audio.call("vocoder", 1, 100, 0, 5, 40)
            d.audio.call("glide", 1, 0)
            d.audio.call("live", 1, 1)
            d.cells[0][0].assign(title)
            d.set_song_assignment(0, "anchor", "natural")
            d.configure()
            d.assignment_changed(1)
            self.assertIn(title, d.cells[1][2].get_tooltip_text())
            # The scoring selection must reach the follower before the new bed installs.
            preview = listen(.69)
            self.assertGreater(level(preview[1440:8880], 65.406), .001)
            self.assertGreater(level(preview[24480:32000], 130.813), .001)
            d.install()
            d.tick()
            d.audio.call("transport", 1, 0)
            d.play.clicked()
            # The recording begins with about one second of rest, then its C2 arp.
            playing = listen(1.23)
            self.assertLess(level(playing[2400:14400], 65.406), .0001)
            self.assertGreater(level(playing[2400:14400], 190), .015)
            self.assertGreater(level(playing[51000:58000], 65.406), .001)
            d.voice(1, VOCODER_HOLD)
            self.assertGreater(level(listen(.17)[960:], 65.406), .001)

    def test_new_test_vocal_prepares_while_first_backing_is_loading(self):
        d = Desk()
        self.addCleanup(d.window.destroy)
        d.open_audio = lambda: True
        d.cells[0][0].assign("red")
        d.test.refresh()
        d.test.buttons[0].clicked()
        self.assertFalse(d.ready)
        self.assertTrue(d.test.pending[0])
        # Wait for the real guide decoder while withholding the backing launch.
        d.test.jobs[("red", "auto")][0].result(timeout=5)
        self.assertFalse(d.audio.call("playing"))
        d.install()
        d.test.refresh()
        self.assertEqual(d.test.messages[0], "Test vocal")
        self.assertTrue(d.audio.call("playing"))

    def test_red_to_blue_changes_backing_at_global_boundary_even_while_following_red(self):
        d = self.desk
        d.mix_controls[1].set_active_id("stripped")
        d.install()
        d.play.clicked()
        self.assert_output(self.render(137), self.STEMS["red"][1])
        d.source_buttons[2].clicked()
        d.install()
        # At 240 BPM the first quarter-bar ends exactly at frame 12000.
        pcm = self.render(12000 - 137 + 32)
        self.assert_output(pcm[:-32], self.STEMS["red"][1])
        self.assert_output(pcm[-32:], self.STEMS["blue"][0])
        d.tick()
        self.assertEqual(d.playing_slot, 2)
        self.assertEqual(d.backing["id"], "blue")
        self.assertEqual(d.strips[1]["anchor"], "mic:0")

    def test_backing_and_test_vocal_stems_can_be_muted_independently(self):
        d = self.desk
        d.test.buttons[1].clicked()
        deadline = time.monotonic() + 3
        while d.test.pending[1] and time.monotonic() < deadline:
            d.test.refresh()
            time.sleep(.002)
        self.assertEqual(d.test.messages[1], "Test vocal")
        self.assert_output(self.render(128), self.STEMS["red"][0], self.STEMS["blue"][2])
        d.mix_controls[2].set_active_id("mute")
        self.assert_output(self.render(64), self.STEMS["red"][0], self.STEMS["blue"][2])
        d.mix_controls[2].set_active_id("stripped")
        d.source_buttons[2].clicked()
        d.install()
        self.render(12000)
        d.tick()
        self.assert_output(self.render(64), self.STEMS["blue"][1], self.STEMS["blue"][2])
        d.mix_controls[2].set_active_id("mute")
        self.assert_output(self.render(64), (0, 0), self.STEMS["blue"][2])
        d.test.buttons[1].clicked()
        self.assert_output(self.render(64), (0, 0))
        d.mix_controls[2].set_active_id("stripped")
        self.assert_output(self.render(64), self.STEMS["blue"][1])

    def test_switch_vocal_source_while_two_mics_rehearse_the_same_song(self):
        d = self.desk
        song = d.songs["red"]
        song["rows"] = {"lead": [(["Ready"], [[0, 1000]])]}
        d.cells[1][0].assign("red")
        lead, other = d.test.sources[:2]
        self.assertTrue(lead.get_mapped())
        self.assertEqual(other.get_active_id(), "auto")

        def ready():
            deadline = time.monotonic() + 5
            while any(d.test.pending) and time.monotonic() < deadline:
                d.test.refresh()
                time.sleep(.002)
            self.assertEqual(d.test.messages[:2], ["Test vocal"] * 2)

        lead.set_active_id("extracted")
        d.test.buttons[0].clicked()
        d.test.buttons[1].clicked()
        ready()
        bed, _, guide = self.STEMS["red"]
        self.assert_output(self.render(4800), bed, 2 * guide)
        d.mix_controls[1].set_active_id("mute")
        self.assert_output(self.render(64), (0, 0), 2 * guide)
        at = d.audio.call("clock")
        with patch.dict("os.environ", KARAOKE_TTS_MODEL_DIR=str(self.root)):
            lead.set_active_id("computer")
            ready()
        self.assertTrue(d.audio.call("playing"))
        self.assertEqual(d.audio.call("clock"), at)
        self.assertEqual(other.get_active_id(), "auto")
        # The second mic still supplies its guide. The first now sings an
        # audible, time-varying waveform through both stereo channels.
        voice = self.render(20000) - guide
        self.assertGreater(voice[:, 0].std(), 500)
        np.testing.assert_allclose(voice[:, 0], voice[:, 1], rtol=0, atol=1)
        d.flush()
        self.assertEqual([s["test_source"] for s in settings()["strips"]], ["computer", "auto", "auto", "auto"])
        lead.set_active_id("extracted")
        ready()
        self.render(96000)  # Let the room limiter release after the louder computer voice.
        self.assert_output(self.render(1000), (0, 0), 2 * guide)
        d.mix_controls[1].set_active_id("full")
        self.assert_output(self.render(1000), bed, 2 * guide)

    def test_nested_follow_and_leader_reassignment_do_not_replace_selected_backing(self):
        d = self.desk
        d.cells[2][0].assign("green")
        self.follow(2, 1)
        d.source_buttons[3].clicked()
        d.install()
        d.play.clicked()
        self.assert_output(self.render(64), self.STEMS["green"][0])
        d.cells[0][1].assign("blue")
        d.install()
        self.assert_output(self.render(14000), self.STEMS["green"][0])
        self.assertEqual(d.test.song_key(2), "green")
        # A directly assigned Hear song still supplies the backing on that pane.
        d.cells[2][1].assign("red")
        d.install()
        self.render(12000)
        d.tick()
        self.assert_output(self.render(64), self.STEMS["red"][0])
        self.assertEqual(d.test.song_key(2), "green")

    def test_empty_following_slot_supplies_no_backing_and_reselecting_recovers(self):
        d = self.desk
        self.follow(3, 1)
        d.play.clicked()
        d.source_buttons[4].clicked()
        d.install()
        self.assert_output(self.render(64), (0, 0))
        self.assertTrue(d.audio.call("playing"))
        d.source_buttons[1].clicked()
        d.install()
        self.assert_output(self.render(64), self.STEMS["red"][0])

    def test_selecting_new_backing_from_silence_never_previews_the_last_song(self):
        d = self.desk
        d.play.clicked()
        d.source_buttons[4].clicked()
        self.assert_output(self.render(500), (0, 0))
        d.source_buttons[2].clicked()
        d.install()
        pcm = self.render(12000 - 500 + 64)
        self.assert_output(pcm[:-64], (0, 0))
        self.assert_output(pcm[-64:], self.STEMS["blue"][0])

    def test_replacing_queued_blue_with_green_never_leaks_blue(self):
        d = self.desk
        d.cells[2][0].assign("green")
        d.play.clicked()
        d.source_buttons[2].clicked()
        d.install()
        self.assert_output(self.render(500), self.STEMS["red"][0])
        d.source_buttons[3].clicked()
        d.install()
        pcm = self.render(12000 - 500 + 64)
        self.assert_output(pcm[:-64], self.STEMS["red"][0])
        self.assert_output(pcm[-64:], self.STEMS["green"][0])

    def test_failed_blue_backing_cannot_play_red_under_the_blue_indicator(self):
        d = self.desk
        path = self.root / "blue-full.wav"
        original = path.read_bytes()
        d.play.clicked()
        path.unlink()
        d.source_buttons[2].clicked()
        for failure in ("missing", "corrupt"):
            with self.subTest(failure=failure):
                if failure == "corrupt":
                    path.write_bytes(b"invalid recording")
                    d.select_source()
                    d.install()
                d.tick()
                self.assertIn("without backing", d.status.get_text())
                self.assertTrue(d.source_buttons[2].get_active())
                self.assert_output(self.render(128), (0, 0))
        path.write_bytes(original)
        d.select_source()
        d.install()
        self.render(12000)
        d.tick()
        self.assert_output(self.render(64), self.STEMS["blue"][0])
