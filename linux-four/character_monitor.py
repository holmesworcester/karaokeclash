"""Keep a single audible Homer voice even when a model prediction arrives late."""

from collections import deque
import ctypes as C
from pathlib import Path
import threading
import time

import numpy as np

from audio import CharacterWorker, Hop, LAB, ROOT
from homer_timbre import HomerTimbre, monitor_delay


class HomerMonitor:
    """Pair the model result with the exact input packet that produced it."""

    def __init__(self, audio, packets, model_delay=0):
        self.audio = audio
        self.shaper = HomerTimbre(model_delay=model_delay)
        self.pending = [None] * packets
        self.next_hop = None

    def call(self, op, lane, pointer):
        if op == "take" and not self.audio.call(op, lane, pointer):
            return 0
        hop = C.cast(pointer, C.POINTER(Hop)).contents
        key = (hop.epoch, hop.sequence)
        slot = hop.sequence % len(self.pending)
        samples = np.ctypeslib.as_array(hop.audio)
        if op == "take":
            self.pending[slot] = key, samples.copy()
            return 1

        captured = self.pending[slot]
        if captured is None or captured[0] != key:
            return 0
        self.pending[slot] = None
        if key != self.next_hop:
            self.shaper.reset()
        samples[:] = np.clip(self.shaper.process(captured[1], samples), -1, 1)
        self.next_hop = hop.epoch, hop.sequence + 1
        return self.audio.call(op, lane, pointer)


class LiveHomer:
    """Clock microphone playback independently of GPU inference on a separate thread."""

    def __init__(self, audio, lane, epoch, notify, command, packets, model_delay):
        self.audio, self.lane, self.epoch = audio, lane, epoch
        # A request is issued at the end of its 20 ms packet. Reserve time
        # before native playout to return the mic even when inference stalls.
        self.wait = (2 * packets - 1) * .020 - (.001 if packets == 1 else .004)
        self.requests = deque(maxlen=6)
        self.results = deque(maxlen=6)
        self.stop_event = threading.Event()
        self.model = HomerMonitor(self, packets, model_delay)
        self.worker = CharacterWorker(self.model, lane, "homer", epoch, notify, command=command, packets=packets)
        self.timings = self.worker.timings
        self.held = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def call(self, op, lane, pointer):
        # These two bounded, single-producer queues connect the GPU and monitor threads.
        if op == "take":
            if not self.requests:
                return 0
            C.memmove(pointer, C.byref(self.requests.popleft()), C.sizeof(Hop))
        else:
            hop = Hop.from_buffer_copy(C.cast(pointer, C.POINTER(Hop)).contents)
            self.results.append((hop, self.model.shaper.previous))
        return 1

    def run(self):
        pending = deque()
        ready = {}
        monitor = HomerTimbre(model_delay=self.model.shaper.model_delay)
        response = monitor.previous
        last_estimate = -1
        next_sequence = None
        last_sample = 0.
        last_held = None
        incoming = Hop()
        while self.worker.thread.is_alive() and not self.stop_event.is_set():
            while self.audio.call("take", self.lane, C.byref(incoming)):
                if incoming.epoch == self.epoch:
                    hop = Hop.from_buffer_copy(incoming)
                    pending.append((hop, time.monotonic() + self.wait))
                    self.requests.append(hop)
            while self.results:
                result, estimate = self.results.popleft()
                if pending and result.sequence >= pending[0][0].sequence:
                    ready[result.sequence] = result, estimate
                elif result.sequence > last_estimate:
                    response, last_estimate = estimate, result.sequence

            while pending:
                hop, deadline = pending[0]
                current = ready.pop(hop.sequence, None)
                if current is None and time.monotonic() < deadline:
                    break
                pending.popleft()
                if hop.sequence != next_sequence:
                    monitor.reset()
                next_sequence = hop.sequence + 1
                if current is not None:
                    result, response = current
                    last_estimate = hop.sequence
                source = np.ctypeslib.as_array(hop.audio)
                predicted = monitor.hold(source, response, render=current is None)
                if current is None:
                    self.held += 1
                    # The GPU can still be reading this input. Keep its copy immutable.
                    result = Hop.from_buffer_copy(hop)
                    np.ctypeslib.as_array(result.audio)[:] = np.clip(predicted, -1, 1)
                # The live model and the emergency mic filter have different
                # waveform phases. Remove the step only when switching paths.
                held = current is None
                samples = np.ctypeslib.as_array(result.audio)
                if last_held is not None and held != last_held:
                    n = monitor.step
                    samples[:n] = np.clip(samples[:n] + (last_sample - samples[0]) * (1 - monitor.fade), -1, 1)
                last_sample, last_held = float(samples[-1]), held
                self.audio.call("put", self.lane, C.byref(result))
            self.stop_event.wait(.0006)

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.worker.stop()


def character_worker(audio, lane, voice, epoch, notify, command=None, packets=2):
    if voice == "homer":
        model_delay = monitor_delay(packets) if command is None else 0
        if command is None:
            command = [LAB / "run-lab", LAB / "gpu/bin/python", Path(__file__).with_name("homer_live.py"),
                       ROOT / "tools/voice_transform", LAB / "framework-rvc", str(packets)]
        return LiveHomer(audio, lane, epoch, notify, command, packets, model_delay)
    return CharacterWorker(audio, lane, voice, epoch, notify, command=command, packets=packets)
