"""Exercise the visible toggle, per-strip FX and source selection on the production mixer."""

from pathlib import Path
import sys
import ctypes as C
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import Hop, PARAMS
from desk import Desk
from desk_audio import DeskAudio, FP


class VocalToggle(unittest.TestCase):
    def test_independent_vocal_toggles_and_corner_source_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def wav(name, data):
                path = root / (name + ".wav")
                with wave.open(str(path), "wb") as file:
                    file.setparams((1, 2, 48000, 0, "NONE", ""))
                    file.writeframes(data.astype("<i2").tobytes())
                return path

            bed = wav("backing", np.zeros(48000))
            songs = {}
            for i, hz in enumerate((300, 700, 1100, 1500)):
                name = str(i)
                guide = wav(name, 1200 * np.sin(2*np.pi*hz*np.arange(48000)/48000))
                songs[name] = dict(id=name, title=name, label=name, track=name, start=0,
                                   end=1, loop=True, bpm=120, guide=guide, rows={})
            with patch("desk.catalog", return_value=songs), patch("desk.STATE", root / "session"), \
                    patch("desk.playback_files", return_value=(bed, bed)), patch.object(DeskAudio, "resources"):
                desk = Desk()
                self.addCleanup(desk.window.destroy)
                self.addCleanup(desk.close)
                desk.tempo.set_value(120)
                desk.open_audio = lambda: True
                desk.cells[0][0].assign("0")
                desk.install()
                desk.test.refresh()
                self.assertFalse(any(button.get_sensitive() for button in desk.test.buttons[1:]))
                desk.test.buttons[0].clicked()
                self.assertEqual(desk.test.active, [True, False, False, False])
                self.assertEqual(desk.test.keys[1:], [None] * 3)
                for lane in range(1, 4):
                    desk.cells[lane][lane % 2].assign(str(lane))
                    desk.test.refresh()
                    desk.test.buttons[lane].clicked()
                for strip in desk.strips:
                    strip["compressor"] = False
                desk.configure()
                desk.install()
                def ready():
                    until = time.monotonic() + 10
                    while time.monotonic() < until:
                        desk.test.refresh()
                        if not any(desk.test.pending):
                            break
                        time.sleep(.005)
                    self.assertEqual(desk.test.messages, ["Test vocal" if a else "" for a in desk.test.active])
                ready()
                self.assertTrue(desk.audio.call("playing"))
                for i in range(1, 5):
                    b = desk.source_buttons[i]
                    self.assertFalse(b.get_mode())
                    self.assertEqual(b.get_label(), "♫")
                    self.assertTrue(b.get_style_context().has_class("backing-note"))
                    if i:
                        self.assertGreater(b.get_allocation().x, desk.loop_controls[i][1].get_allocation().x)
                def render():
                    left, right = np.zeros((2, 48000), "f4")
                    desk.audio.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
                    return left
                # Red supplies Blue's Hear song. Blue rehearses the words on its
                # own screen, including when it has a different explicit Hear song.
                desk.cells[1][0].assign("1")
                for anchor in ("0", "mic:0"):
                    desk.set_song_assignment(1, "anchor", anchor)
                    desk.configure()
                    ready()
                    self.assertEqual(desk.test.song_key(1), "1")
                    self.assertEqual(desk.test.keys[1][0][0], "1")
                baseline = render()
                spectrum = np.abs(np.fft.rfft(baseline))
                for hz in (300, 700, 1100, 1500):
                    self.assertGreater(spectrum[hz], 300)
                # With its backing muted, Green's test vocal still goes through
                # Green's pulse effect. The other three mic signals stay dry.
                desk.mix_controls[3].set_active_id("mute")
                desk.strips[2].update(pulse=True)
                desk.strips[2]["values"][:3] = [100, 50, 100]
                desk.configure()
                changed = np.abs(np.fft.rfft(render()))
                self.assertLess(changed[1100], .5 * spectrum[1100])
                for hz in (300, 700, 1500):
                    self.assertAlmostEqual(changed[hz]/spectrum[hz], 1, delta=.05)
                # One live GPU peer receives the dry words in native packets.
                # The other requested character retains its direct mic until the GPU frees up.
                from gi.repository import GLib

                peers = {}
                def start(audio, lane, name, epoch, notify, packets):
                    peer = Mock()
                    peers[lane] = peer
                    notify("ready")
                    return peer

                def wet_spectrum(lane, gain):
                    context = GLib.MainContext.default()
                    while context.pending():
                        context.iteration(False)
                    ready()
                    left, right = np.zeros((2, 48000), "f4")
                    pending = []
                    for pos in range(0, len(left), 960):
                        desk.audio.call("render", None, left[pos:].ctypes.data_as(FP),
                                       right[pos:].ctypes.data_as(FP), 960)
                        hop = Hop()
                        while desk.audio.call("take", lane, C.byref(hop)):
                            np.ctypeslib.as_array(hop.audio)[:] *= gain
                            pending.append(Hop.from_buffer_copy(hop))
                        # Two input hops and one hop of compute, as on the stable mic.
                        if pos % 1920 == 0 and len(pending) >= 3:
                            for item in pending[:2]:
                                self.assertTrue(desk.audio.call("put", lane, C.byref(item)))
                            del pending[:2]
                    self.assertGreater(desk.audio.call("wet", lane), 30)
                    tail = left[9600:]
                    return abs(np.fft.rfft(tail)) * len(left) / len(tail)

                with patch("desk.character_worker", side_effect=start) as worker:
                    desk.voice(0, 3)
                    desk.voice(1, 4)
                    mix = wet_spectrum(0, .25)
                    self.assertEqual(worker.call_count, 1)
                    self.assertIn("direct vocal", desk.worker_status[1])
                    self.assertAlmostEqual(mix[240]/spectrum[300], .25, delta=.03)
                    self.assertAlmostEqual(mix[560]/spectrum[700], 1, delta=.03)
                    desk.voice(0, 0)
                    mix = wet_spectrum(1, .6)
                    peers[0].stop.assert_called_once()
                    self.assertEqual(worker.call_count, 2)
                    self.assertAlmostEqual(mix[560]/spectrum[700], .6, delta=.03)
                    desk.voice(0, 3)
                    self.assertIn("direct vocal", desk.worker_status[0])
                    desk.session["gpu_hops"] = 2
                    desk.audio.call("buffer", 2)
                    desk.test.refresh()
                    self.assertEqual(worker.call_args.args[1], 1)
                    self.assertEqual(worker.call_args.kwargs["packets"], 1)
                    self.assertIn("direct vocal", desk.worker_status[0])
                    desk.voice(0, 0)
                    desk.voice(1, 0)
                desk.test.buttons[0].clicked()
                ready()
                without_first = np.abs(np.fft.rfft(render()))
                self.assertLess(without_first[300], 5)
                self.assertGreater(without_first[700], 300)
                for button in desk.test.buttons[1:]:
                    button.clicked()
                self.assertFalse(any(desk.test.active))
                self.assertTrue(desk.audio.call("playing"))
                self.assertLess(np.abs(render()[-4800:]).max(), .001)
                # A missing vocal stem falls back to the words on that mic.
                songs["synthetic"] = dict(songs["0"], id="synthetic", title="synthetic",
                    label="synthetic", guide=root/"missing.wav",
                    rows={"lead": [[["Hello"], [[100, 600]]]]})
                desk.songs.update(songs)
                desk.refresh_lists()
                desk.cells[0][0].assign("synthetic")
                desk.test.buttons[0].clicked()
                ready()
                self.assertGreater(desk.test.dry_sources[desk.test.keys[0][0]].std(), 100)
                desk.source_buttons[3].clicked()
                desk.source_buttons[3].clicked()
                self.assertEqual([i for i,b in desk.source_buttons.items() if b.get_active()], [3])

    def test_empty_mics_pass_only_their_enabled_physical_input(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        values = (C.c_float * 14)(*[p[3] for p in PARAMS])
        for lane in range(4):
            audio.call("config", lane, 0, b"natural", 0, 0, values)
        # Only lane 1 is rehearsing. Lane 2 is live and lanes 3 and 4 are idle.
        audio.call("test_lanes", 1)
        audio.call("live", 1, 1)
        pcm = (np.sin(2*np.pi*300*np.arange(48000)/48000)*1000).astype("i2")
        bed = np.tile(np.int16(150), 2*len(pcm))
        song = dict(track="A", start=0, end=1, loop=True, bpm=120)
        audio.prepare_track(bed, None, song)
        audio.call("stage_lane", 0)
        audio.call("launch_at", -1, 0)
        audio.test_source(0, pcm, song, song)
        audio.call("transport", 0, 1)
        capture = np.array([np.sin(2*np.pi*f*np.arange(48000)/48000)*.03
                            for f in (500, 700, 900, 1100)], dtype="f4")
        inputs = (FP * 4)(*(x.ctypes.data_as(FP) for x in capture))
        def spectrum():
            left, right = np.zeros((2, 48000), "f4")
            audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
            return abs(np.fft.rfft(left))
        mix = spectrum()
        self.assertTrue(all(mix[f] > 100 for f in (0, 300, 700)))
        self.assertTrue(all(mix[f] < 1 for f in (500, 900, 1100)))
        audio.call("mute_lanes", 3)
        mix = spectrum()
        self.assertLess(mix[0], 1)  # The selected backing's DC signal is gone.
        self.assertTrue(all(mix[f] > 100 for f in (300, 700)))  # Test and live mic.
        self.assertTrue(all(mix[f] < 1 for f in (500, 900, 1100)))
        audio.call("master", -6, 1)
        self.assertLess(spectrum().max(), .001)
        self.assertTrue(audio.call("playing"))
