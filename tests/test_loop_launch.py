"""Drive loop launches from two mic panes through the real desk mixer."""
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import D, FP, SP
from clip_clock import ClipClock
from desk import Desk
from desk_audio import DeskAudio
from routing import SongLibrary


class LoopLaunch(unittest.TestCase):
    def test_constant_headache_song_assignment_and_tmf_buttons(self):
        title = "Constant Headache"
        chorus = title + " - Chant"
        duration = 16 * 60 / 126
        full_duration = 184.306
        guitar = dict(id=title, title=title, label=title + " · Guitar", part="Guitar",
                      track=title, start=0, end=0, loop=False, bpm=98.5, beat_zero=6.122,
                      rows={}, cue=["Instrumental"])
        chant = dict(guitar, id=chorus, label=title + " · Chorus", part="Chorus", track=chorus,
                     bpm=126, beat_zero=0, loop=True, end=duration, rows={"lead": [
                         (["But", "I'm", "a", "constant", "headache"], [(223, 505), (581, 751),
                          (751, 816), (891, 1573), (1644, 2322)]),
                         (["They"], [(3990, 4230)])]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def files(song):
                bed = root / (song["id"] + ".wav")
                return bed, bed

            for song, seconds, rate in ((guitar, full_duration, 8000), (chant, duration, 48000)):
                with wave.open(str(files(song)[0]), "wb") as stream:
                    stream.setparams((1, 2, rate, 0, "NONE", ""))
                    stream.writeframes(bytes(round(seconds * rate) * 2))
            with patch("desk.catalog", return_value={title: guitar, chorus: chant}), \
                    patch("desk.STATE", root / "state"), patch("desk.playback_files", side_effect=files), \
                    patch.object(DeskAudio, "resources"):
                desk = Desk()
                try:
                    desk.library.search.set_text(title)
                    matches = [row for row in desk.library.items.values() if row.get_child_visible()]
                    self.assertEqual([row.get_accessible().get_name() for row in matches], [title])
                    desk.library.rows.select_row(matches[0])
                    # Exercise See on the first mic and Hear alone on the second.
                    for slot, field in ((1, 0), (2, 1)):
                        target = desk.cells[slot - 1][field]
                        target.clicked()
                        desk.source_buttons[slot].set_active(True)
                        desk.install()
                        modes = desk.loop_controls[slot][0]
                        for mode, source in (("teaser", chorus), ("main", chorus), ("full", title)):
                            with self.subTest(slot=slot, mode=mode):
                                modes[mode].clicked()
                                desk.install()
                                desk.refresh_loop_controls()
                                self.assertEqual(desk.selected_mix[:2], (title, mode))
                                self.assertTrue(modes[mode].get_style_context().has_class("playing"))
                                self.assertEqual(desk.source_key(slot), title)
                                self.assertEqual(desk.test.song_key(slot - 1), source)
                                self.assertAlmostEqual(desk.duration, full_duration if mode == "full" else duration, places=4)
                                if mode == "teaser":
                                    self.assertEqual(desk.backing["vocal_window"], (0, 3.990))
                                else:
                                    self.assertNotIn("vocal_window", desk.backing)
                        pane = desk.stage.panes[slot - 1]
                        sheet = pane.song_for(desk.backing, None, desk.source_key(slot))
                        self.assertEqual(len(sheet["rows"]["lead"]), 20)
                        self.assertNotIn("Instrumental", sheet["cue"])
                        # Word sweeps on the matched guide, including the first
                        # and final chorus releases and the long instrumental outro.
                        for at, word in [(27.5, "hear"), (47.1, "constant"), (67.5, "mine"),
                                         (72., None), (86.35, "you"), (146.5, "mine"), (170., None)]:
                            desk.seek.set_value(at)
                            backing, pos, _ = desk.pane_transport(slot - 1)
                            lines, _ = pane.lines_at(sheet, backing, pos, None, lead_ms=0)
                            swept = [w for _, _, words, spans, ms in lines
                                     for w, (a, b) in zip(words, spans) if a <= ms < b]
                            self.assertEqual(swept, [word] if word else [], (slot, at))
                finally:
                    desk.window.destroy()

    def test_inactive_mic_rewinds_inside_loop_on_shared_half_bar(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        region = dict(track="bed", start=0, end=1.6, bpm=600)
        master = ClipClock(region)
        silence = np.zeros(76800 * 2, dtype="i2")
        self.assertTrue(audio.prepare_track(silence, None, region))
        audio.call("stage_clock", *audio.clock_args(master))
        audio.call("launch_at", -1, 0)
        source = dict(region, track="mic", end=.8)
        # Rewinding from halfway through bar one wraps into bar two.
        audio.call("pane_clock", 1, .5, *audio.clock_args(ClipClock(source), "mic"),
                   *audio.clock_args(ClipClock(source, 1), "mic"))
        vocal = np.repeat(np.array([1000, -1000], dtype="i2"), 19200)
        spec = (D * 12)(*([0, .8, 600, 0, 0, 1] * 2))
        audio.call("test_source", 1, vocal.ctypes.data_as(SP), len(vocal),
                   True, spec, None, 0, None, 0)
        audio.call("test_lanes", 1 << 1)
        audio.call("live", 1, 1)
        audio.call("transport", 0, 1)
        left, right = np.zeros((2, 9601), dtype="f4")
        audio.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
        self.assertGreater(left[-2], 0)
        self.assertLess(left[-1], 0)

    def test_launch_on_outgoing_grid_with_replacement_and_named_cuts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 48000
            bar = .4
            frames = int(bar * sample_rate)
            data = np.repeat(np.arange(24, dtype="i2") * 50 + 100, frames)
            paths = [root / "full.wav", root / "stripped.wav"]
            for path, gain in zip(paths, (1, 2)):
                with wave.open(str(path), "wb") as stream:
                    stream.setparams((2, 2, sample_rate, 0, "NONE", ""))
                    stream.writeframes(np.repeat(data * gain, 2).tobytes())

            song = dict(id="A", title="A", label="A", track="A", bpm=600, beat_zero=0,
                        key="C major", start=0, end=0, loop=False, loopBar=6,
                        teaserVoiceBar=1, rows={"lead": [[["Here"], [[2850, 3050]]]]},
                        loops=[dict(id="bridge", label="Bridge", firstBar=20, bars=2)])
            with patch("desk.catalog", return_value={"A": song}), patch("desk.STATE", root / "state"), \
                    patch("desk.playback_files", return_value=paths), patch.object(DeskAudio, "resources"):
                desk = Desk()
                self.addCleanup(desk.window.destroy)
                desk.tempo.set_value(song["bpm"])
                desk.open_audio = lambda: True
                self.assertIsInstance(desk.library, SongLibrary)
                self.assertEqual(len(desk.source_buttons), 4)
                self.assertEqual(len(desk.loop_controls), 4)
                self.assertTrue(all(not w.get_sensitive() for c in desk.loop_controls.values() for w in c[0].values()))
                desk.play.clicked()
                self.assertTrue(desk.audio.call("playing"))
                desk.play.clicked()
                self.assertFalse(desk.audio.call("playing"))
                desk.library.rows.select_row(desk.library.items["A"])
                desk.cells[2][0].clicked()
                desk.install()
                self.assertEqual(desk.shown_source, (3, "A"))
                self.assertEqual(desk.extra_loops(3), [("bridge", "Bridge")])
                modes, back, forward, extra = desk.loop_controls[3]

                def render(n):
                    left, right = np.zeros((2, n), dtype="f4")
                    desk.audio.call("render", None, left.ctypes.data_as(FP),
                                    right.ctypes.data_as(FP), n)
                    np.testing.assert_array_equal(left, right)
                    return np.round(left * 32768).astype(int)

                # Enter the sparse teaser, then replace a queued Main with Bridge.
                modes["teaser"].clicked()
                self.assertAlmostEqual(desk.duration, 4*bar)
                lo, hi = desk.backing["vocal_window"]
                self.assertLess(lo, 2.850)
                self.assertGreater(hi, 3.050)
                self.assertGreater(desk.backing["end"] - hi, bar)
                desk.play.clicked()
                render(3*frames + 400)
                modes["main"].clicked()
                self.assertEqual(desk.selected_mix[1], "teaser")
                self.assertAlmostEqual(desk.pending["boundary"], 4*bar)
                extra.get_popover().popup()
                rows = extra.get_popover().get_child().get_children()[1].get_child().get_child()
                rows.emit("row-activated", rows.get_row_at_index(0))
                desk.tick()
                self.assertEqual(desk.pending["target"][1], "bridge")
                remaining = frames - 400
                actual = render(remaining + 500)
                np.testing.assert_array_equal(actual[:remaining], 100 + 9*50)
                np.testing.assert_array_equal(actual[remaining:], 100 + 20*50)
                desk.sync_launch()
                self.assertEqual(desk.selected_mix[1], "bridge")
                self.assertAlmostEqual(desk.elapsed(), 500 / sample_rate)

                # Assigning another mic source preserves Mic 3's loop choice.
                desk.cells[0][0].assign("A")
                desk.quant.set_active_id(".25")
                desk.source_buttons[1].clicked()
                self.assertEqual(desk.pending["target"][1], "full")
                self.assertAlmostEqual(desk.pending["boundary"], bar / 4)
                render(int(frames / 4))
                desk.sync_launch()
                self.assertEqual(desk.selected_mix[1], "full")
                desk.source_buttons[3].clicked()
                self.assertEqual(desk.pending["target"][1], "bridge")
                desk.choose_loop(3, "full")
                self.assertIsNone(desk.pending)
                self.assertEqual(desk.selected_mix[1], "full")

                # A mix change keeps the clock and the same region.
                pos = desk.elapsed()
                desk.mix_controls[3].set_active_id("stripped")
                desk.install()
                self.assertAlmostEqual(desk.elapsed(), pos)
                np.testing.assert_array_equal(render(10), 200)
                # Both sources take their arrow taps on the same fractional grid.
                # The sample before that edge is still in the outgoing bar.
                desk.quant.set_active_id(".5")
                desk.seek.set_value(.04)
                desk.tick()
                playing = desk.conductor_bar()
                edge = desk.master_clock.next_edge(desk.elapsed(), ".5")
                until = round((edge - desk.elapsed()) * sample_rate)
                forward.clicked()
                inactive_forward = desk.loop_controls[1][2]
                inactive_forward.clicked()
                self.assertAlmostEqual(desk.pending["boundary"], edge)
                self.assertAlmostEqual(desk.actions[1][3], desk.actions[3][3])
                before = int(edge / bar + 1e-6)
                actual = render(until + 1)
                self.assertEqual(actual[-2], 200 + int((edge - 1/sample_rate) / bar)*100)
                self.assertEqual(actual[-1], 200 + ((before + 1) % 24)*100)
                desk.tick()
                self.assertNotIn(1, desk.actions)
                self.assertAlmostEqual(desk.conductor_bar(), playing + (until+1)/frames, places=5)
                inactive = desk.slot_clocks[1][2]
                self.assertAlmostEqual(inactive.bar(inactive.seconds(desk.conductor_bar())),
                                       desk.conductor_bar())
                # Opposing taps before the edge cancel without rewinding the grid.
                forward.clicked()
                back.clicked()
                self.assertIsNone(desk.pending)
                self.assertNotIn(3, desk.actions)
                # Selecting an empty backing pane keeps the native transport
                # running, leaves the mic song in time, and silences only the bed.
                desk.source_buttons[2].clicked()
                pos = desk.elapsed()
                self.assertTrue(desk.audio.call("playing"))
                np.testing.assert_array_equal(render(960), 0)
                self.assertAlmostEqual(desk.elapsed(), pos + .02)
                desk.tick()
                desk.play.clicked()
                desk.play.clicked()
                self.assertTrue(desk.audio.call("playing"))
                desk.source_buttons[3].clicked()
                self.assertGreater(render(1)[0], 0)
                desk.close()
