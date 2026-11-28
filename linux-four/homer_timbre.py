"""Play the converted character and retain an estimate of its tone for late packets."""

import numpy as np


DELAY_FRAMES = 240  # 5 ms at the desk's 48 kHz rate.


def monitor_delay(packets):
    """Real right context on the 48 kHz desk clock: 5 ms fast or 10 ms stable."""
    return packets * DELAY_FRAMES


class HomerTimbre:
    def __init__(self, *, model_delay=0):
        self.model_delay = model_delay
        self.step = DELAY_FRAMES
        self.size = 1536  # 32 ms of spectral history.
        self.window = np.hanning(self.size)
        bins = np.arange(-9, 10)
        kernel = np.exp(-.5 * (bins / 2.4) ** 2)
        self.kernel = kernel / kernel.sum()
        distance = np.abs(np.fft.fftfreq(self.size) * self.size)
        self.taper = np.clip((self.step - distance) / (self.step / 3), 0, 1)
        self.fade = (np.arange(self.step) + 1) / self.step
        self.reset()

    def reset(self):
        self.history = np.zeros((2, self.size))
        self.previous = np.ones(self.size // 2 + 1)
        self.monitor_response = self.previous
        self.log_gain = None
        self.gate = 0.
        self.extra_history = np.zeros(max(0, self.model_delay - self.step))

    def align_source(self, source):
        if not len(self.extra_history):
            return source
        samples = np.concatenate((self.extra_history, source))
        self.extra_history = samples[len(source):].copy()
        return samples[:len(source)]

    def hold(self, source, response, *, render=True):
        """Continue the current microphone with the last estimated tone during a stall."""
        h = self.step
        frame = np.concatenate((self.history[0, len(source):], self.align_source(source)))
        self.history[0] = frame
        previous, self.monitor_response = self.monitor_response, response
        if not render:
            return
        spectrum = np.fft.rfft(frame)
        old, new = np.fft.irfft([spectrum * previous, spectrum * response])
        start = self.size - len(source) - h
        out = new[start:-h].copy()
        out[:h] = old[start:start+h] + self.fade * (out[:h] - old[start:start+h])
        return out

    def process(self, source, character):
        """Keep the model's waveform, including its harmonic phase and articulation."""
        n, h = self.size, self.step
        if len(source) != len(character) or len(source) != 960:
            raise ValueError("Expected paired 20 ms microphone and character packets")
        data = np.concatenate((self.history, np.stack((self.align_source(source), character))), axis=1)
        self.history = data[:, -n:].copy()

        # Only the fallback needs a transfer function. It never replaces a fresh prediction.
        analysis = self.history.copy()
        if self.model_delay:
            analysis[0] = data[0, -n-h:-h]
        power = abs(np.fft.rfft(analysis * self.window) / 3) ** 2
        original, target = (np.convolve(p, self.kernel, mode="same") for p in power)
        gain = np.clip(.55 * np.log((target + .0001) / (original + .0001)), -2.8, 2.8)
        self.log_gain = gain if self.log_gain is None else .2 * self.log_gain + .8 * gain
        self.previous = np.fft.rfft(np.fft.irfft(np.exp(self.log_gain)) * self.taper)

        # The model can buzz after the singer stops. Close only at the silence floor.
        audio = character.copy() if self.model_delay else data[1, n-h:-h].copy()
        for i in range(0, len(source), h):
            frame = data[0, n-h+i:n+i+h]
            gate = min(1., np.max(np.abs(frame)) / .001)
            audio[i:i+h] *= self.gate + self.fade * (gate - self.gate)
            self.gate = gate
        return audio
