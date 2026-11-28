"""Pace the production mixer and Homer worker from recorded mic input on the Radeon."""

import ctypes as C
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from audio import PARAMS, ROOT, decode
from character_monitor import character_worker
from desk_audio import DeskAudio, FP
from homer_timbre import monitor_delay


def render_paced(audio, source):
    # Match the launcher's 64-frame PipeWire quantum and drain the monitor.
    step = 64
    padded = np.pad(source, (0, -len(source) % step)).astype("f4")
    result = np.empty_like(padded)
    zero, right = np.zeros(step, "f4"), np.empty(step, "f4")
    inputs = (FP * 4)(*[zero.ctypes.data_as(FP)] * 4)
    wake_lag = []
    start = time.perf_counter()
    for pos in range(0, len(padded), step):
        scheduled = start + pos / 48000
        time.sleep(max(0, scheduled - time.perf_counter()))
        wake_lag.append(time.perf_counter() - scheduled)
        inputs[0] = padded[pos:].ctypes.data_as(FP)
        audio.call("render", inputs, result[pos:].ctypes.data_as(FP), right.ctypes.data_as(FP), step)
    return result[:len(source)], max(wake_lag)


def capture(source, packets, factory=character_worker):
    audio = DeskAudio()
    hop = 960
    ready = threading.Event()
    messages = []
    def status(message):
        messages.append(message)
        ready.set()

    worker = None
    try:
        with patch.dict(os.environ, KARAOKE_GPU_TIMINGS="1"):
            worker = factory(audio, 0, "homer", 1, status, packets=packets)
        if not ready.wait(90) or messages != ["ready"]:
            raise RuntimeError(f"Homer initialization failed: {messages}")
        audio.call("config", 0, 3, b"natural", 0, 0, (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS)))
        audio.call("master", 0., 0)
        audio.call("buffer", packets * 2)
        audio.call("live", 0, 1)
        audio.call("character", 0, 1)
        result, clock_lag = render_paced(audio, np.pad(source, (0, hop * 6)))
        if messages != ["ready"] or not worker.timings:
            raise RuntimeError(f"Homer stopped during replay: {messages}")
        report = dict(packets=packets, converted=audio.call("wet", 0), late=audio.call("misses", 0),
                      held_tone=getattr(worker, "held", 0),
                      compute_p50_ms=round(float(np.median(worker.timings)), 2),
                      compute_p95_ms=round(float(np.percentile(worker.timings, 95)), 2),
                      compute_max_ms=round(max(worker.timings), 2),
                      clock_lag_max_ms=round(1000 * clock_lag, 2))
        return result, report
    finally:
        if worker:
            worker.stop()
        audio.close()


class LiveHomer(unittest.TestCase):
    def test_deadlines_and_sustained_pitch_on_both_monitor_settings(self):
        from scipy.io import wavfile

        out = Path("build/evidence/homer-speech")
        out.mkdir(parents=True, exist_ok=True)
        speech = decode(ROOT / "assets/voice-check.flac", 1, seconds=6, floating=True)
        t = np.arange(48000) / 48000
        vowel = sum(.15 / i * np.sin(2*np.pi*125*i*t) for i in range(1, 13))
        source = np.r_[np.zeros(9600), speech, np.zeros(9600), vowel, np.zeros(9600)]
        for packets in (1, 2):
            with self.subTest(packets=packets):
                rendered, report = capture(source, packets)
                offset = packets * 1920 + monitor_delay(packets)
                result = rendered[offset:offset + len(source)]
                wavfile.write(out / f"live-{packets * 40}ms.wav", 48000, result)
                (out / f"live-{packets * 40}ms.json").write_text(json.dumps(report, indent=2))
                print(report, flush=True)
                self.assertGreater(report["converted"], len(source) // 960 * .97)
                self.assertLessEqual(report["late"], max(2, .01 * report["converted"]))
                # Most packets must carry a fresh prediction, even in fast mode.
                self.assertLess(report["held_tone"], report["converted"] * (.5 if packets == 1 else .1))
                self.assertLess(report["compute_p95_ms"], packets * 20)
                held = result[-48000:-19200]
                self.assertGreater(held.std(), .005)
                # A held note must repeat at the singer's period, including across GPU hops.
                period = 48000 // 125
                self.assertGreater(np.corrcoef(held[:-period], held[period:])[0, 1], .85)
