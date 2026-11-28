"""Single-carrier, cue timing and audition behavior on the Python and native paths."""

import ctypes as C
from io import BytesIO
from queue import Empty, Queue
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import Audio, Hop, PARAMS
from character_monitor import HomerMonitor, character_worker
from desk_audio import DeskAudio, FP
from homer_live import PrimedInput
from homer_timbre import HomerTimbre


class HomerSound(unittest.TestCase):
    def test_real_right_context_uses_the_existing_five_ms_and_drains_at_eof(self):
        raw = np.arange(2 * 320, dtype="<i2").tobytes()
        stream = PrimedInput(BytesIO(raw), 80)
        self.assertEqual(stream.read(800), bytes(160) + raw[:640])
        self.assertEqual(stream.read(640), raw[640:])
        self.assertEqual(stream.read(640), b"")
        mic = np.full(960, .1, dtype="f4")
        model = np.zeros_like(mic)
        model[480] = .6
        shaped = HomerTimbre(model_delay=240).process(mic, model)
        np.testing.assert_array_equal(shaped, model)

    def test_stable_fallback_matches_the_models_ten_ms_delay_across_packets(self):
        source = np.random.default_rng(17).normal(0, .1, 2880)
        delayed = np.concatenate((np.zeros(480), source[:-480]))
        shaper = HomerTimbre(model_delay=480)
        unity = shaper.previous.copy()
        # Prime the fallback's history while the model is on time, then miss two packets.
        shaper.hold(source[:960], unity, render=False)
        for start in (960, 1920):
            wet = shaper.hold(source[start:start+960], unity)
            np.testing.assert_allclose(wet, delayed[start:start+960], atol=1e-7)

    def test_hear_example_loads_and_plays_without_a_legacy_homer_recording(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        with patch.object(Audio, "resources"):
            audio.resources()
        self.assertTrue(audio.call("hear", 0, 1))
        left, right = np.zeros((2, 48000), "f4")
        audio.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
        self.assertGreater(left.std(), .01)
        np.testing.assert_array_equal(left, right)

    def test_fresh_model_keeps_its_waveform_at_five_ms_and_stops_in_mic_silence(self):
        t = np.arange(48000) / 48000
        source = .14 * np.sin(2 * np.pi * 120 * t)
        model = .18 * np.sin(2 * np.pi * 233 * t) + .1 * np.sin(2 * np.pi * 699 * t)
        source[-4800:] = 0
        shaper = HomerTimbre()
        rendered = np.concatenate([shaper.process(src, wet) for src, wet in
                                   zip(source.reshape(-1, 960), model.reshape(-1, 960))])
        # Compare samples, not just pitch: a source-driven filter could not
        # reproduce the model's harmonics or phase on this microphone.
        np.testing.assert_allclose(rendered[480:43200], model[240:42960], atol=1e-6)
        self.assertLess(abs(rendered[-3600:]).max(), 1e-6)

    def test_stale_worker_result_cannot_shape_a_new_epoch_or_replay_its_history(self):
        class Transport:
            def call(self, op, lane, pointer):
                return 1
        monitor = HomerMonitor(Transport(), 2)
        h = Hop()
        buffer = np.ctypeslib.as_array(h.audio)
        h.epoch, h.sequence = 3, 0
        buffer[:] = .2
        self.assertEqual(monitor.call("take", 0, C.byref(h)), 1)
        h.epoch = 4
        buffer[:] = 0
        monitor.call("take", 0, C.byref(h))
        h.epoch = 3
        self.assertFalse(monitor.call("put", 0, C.byref(h)))
        h.epoch = 4
        buffer[:] = .7
        self.assertTrue(monitor.call("put", 0, C.byref(h)))
        self.assertFalse(buffer.any())

    def test_an_inference_stall_keeps_the_current_mic_and_last_learned_tone(self):
        class Device:
            def __init__(self):
                self.input, self.output = Queue(), Queue()

            def call(self, op, lane, pointer):
                if op == "put":
                    self.output.put(Hop.from_buffer_copy(C.cast(pointer, C.POINTER(Hop)).contents))
                    return 1
                try:
                    hop = self.input.get_nowait()
                except Empty:
                    return 0
                C.memmove(pointer, C.byref(hop), C.sizeof(Hop))
                return 1

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", XDG_RUNTIME_DIR=tmp):
            block = Path(tmp) / "stall"
            command = [sys.executable, "-I", "-u", "-c", """
import sys, time
from pathlib import Path
import numpy as np
print('READY 16000', file=sys.stderr, flush=True)
while raw := sys.stdin.buffer.read(640):
    while Path(sys.argv[1]).exists():
        time.sleep(.002)
    sys.stdout.buffer.write((np.frombuffer(raw, '<i2') * .25).astype('<i2').tobytes())
    sys.stdout.buffer.flush()
""", str(block)]
            device, started = Device(), Event()
            worker = character_worker(device, 0, "homer", 1, lambda msg: started.set(),
                                      command=command, packets=1)
            try:
                self.assertTrue(started.wait(5))
                tone = .16 * np.sin(2*np.pi*150*np.arange(960)/48000)
                for seq in range(10):
                    if seq == 5:
                        block.touch()
                    # Alternate polarity during the stall to detect a replay of old PCM.
                    source = -tone if seq % 2 and seq >= 5 else tone
                    hop = Hop()
                    hop.epoch, hop.sequence = 1, seq
                    np.ctypeslib.as_array(hop.audio)[:] = source
                    device.input.put(hop)
                    result = device.output.get(timeout=2)
                    self.assertEqual(result.sequence, seq)
                    if seq >= 5:
                        wet = np.ctypeslib.as_array(result.audio)[480:]
                        current = source[240:720]
                        self.assertGreater(np.corrcoef(wet, current)[0, 1], .99)
                        self.assertTrue(.12 < wet.std() / current.std() < .5)
                self.assertGreaterEqual(worker.held, 5)
            finally:
                worker.stop()

    def test_hear_plays_only_the_character_example_then_restores_the_microphone(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        live = np.full(960, .1, "f4")
        silent = np.zeros_like(live)
        sample = np.full(1920, -.15, "f4")
        inputs = (FP * 4)(live.ctypes.data_as(FP), *[silent.ctypes.data_as(FP)] * 3)
        values = (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS))
        audio.call("config", 0, 3, b"natural", 0, 0, values)
        audio.call("live", 0, 1)
        audio.call("sample", 1, sample.ctypes.data_as(FP), len(sample))
        def render():
            left, right = np.zeros_like(live), np.zeros_like(live)
            audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
            return left
        dry_level = render()[-100:].mean()
        self.assertGreater(dry_level, .01)
        audio.call("character", 0, 1)
        self.assertTrue(audio.call("hear", 0, 1))
        self.assertAlmostEqual(render()[-100:].mean() / dry_level, -1.5, delta=.01)
        # The live worker keeps receiving the actual mic while its output is covered.
        hop = Hop()
        self.assertTrue(audio.call("take", 0, C.byref(hop)))
        np.testing.assert_allclose(np.ctypeslib.as_array(hop.audio), live)
        render()
        render()
        self.assertAlmostEqual(render()[-100:].mean(), dry_level, delta=.001)
