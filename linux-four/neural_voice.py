"""Render an English lyric line in context and recover each cue's source audio."""

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock

import numpy as np

from audio import DATA


# eSpeak's US diphthongs and affricates have single tokens in Kokoro's vocabulary.
_IPA = {"aɪ": "I", "aʊ": "W", "eɪ": "A", "oʊ": "O", "ɔɪ": "Y",
        "dʒ": "ʤ", "tʃ": "ʧ", "ɜːɹ": "ɜɹ", "ɜː": "ɜɹ", "ɚ": "əɹ", "ɾ": "T", "ː": ""}
_IPA_PATTERN = re.compile("|".join(_IPA))
_LOAD = Lock()


@lru_cache(maxsize=1024)
def pronunciations(text):
    raw = subprocess.check_output(["espeak-ng", "-q", "-v", "en-us", "--ipa", "--stdin"],
                                  input=text, text=True)
    return _IPA_PATTERN.sub(lambda m: _IPA[m[0]], raw).split()


def word_phones(words):
    # Connected speech gives unstressed function words their usual pronunciation.
    # Numbers, acronyms and multiword cues can expand. Resolve their ownership
    # individually when the phonemizer's boundaries differ from the lyric sheet.
    connected = pronunciations(" ".join(words))
    if len(connected) != len(words) or any(not w.strip(".,!?;:…").isalpha() for w in words):
        individual = [" ".join(pronunciations(w)) if any(c.isalnum() for c in w) else ""
                      for w in words]
        if len(connected) != len(words) or any(len(p.split()) != 1 for p in individual):
            connected = individual
    result = []
    for phones, word in zip(connected, words, strict=True):
        mark = re.search(r"[,.?!:;…]+[\"”’)]*$", word)
        result.append(phones + (mark[0][0] if mark and phones else ""))
    return result


class NeuralVoice:
    def __init__(self, directory, ort):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(directory / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.vocab = json.loads(metadata["kokoro_config"])["vocab"]
        self.styles = np.fromfile(directory / "am_echo.bin", "<f4").reshape(510, 1, 256)
        self.lock = Lock()

    def words(self, words):
        # Multiple mic strips share both the model and its three CPU threads.
        with self.lock:
            return self._words(tuple(words))

    @lru_cache(maxsize=48)
    def _words(self, words):
        groups = [[self.vocab[p] for p in phones if p in self.vocab] for phones in word_phones(words)]
        count = sum(map(len, groups)) + len(groups) - 1
        if count > 510:
            if len(words) == 1:
                return (None,)  # A single unbroken cue exceeds the model's context.
            mid = len(words) // 2
            return self._words(words[:mid]) + self._words(words[mid:])
        if not all(groups):
            # Keep punctuation-only cues silent and preserve the other cue indices.
            pronounceable = tuple(w for w, group in zip(words, groups) if group)
            spoken = iter(self._words(pronounceable)) if pronounceable else iter(())
            return tuple(next(spoken) if group else np.empty(0, "i2") for group in groups)

        ids = [0]
        spaces = []
        for group in groups:
            if len(ids) > 1:
                spaces.append(len(ids))
                ids.append(self.vocab[" "])
            ids.extend(group)
        ids.append(0)
        audio, durations = self.session.run(["waveform", "duration"], {
            "input_ids": np.array([ids], np.int64), "style": self.styles[count - 1],
            "speed": np.ones(1, "f4"),
        })
        # Kokoro predicts one duration per input token in 25 ms frames.
        # Upstream's timestamp convention offsets the acoustic onset by 3 frames.
        # Cut halfway through inter-word spaces and keep the complete first/last consonants.
        ends = np.cumsum(durations)
        active = np.flatnonzero(abs(audio) > .002)
        if not len(active):
            return tuple(np.empty(0, "i2") for _ in groups)
        first = max(0, active[0] - 120)
        last = min(len(audio), active[-1] + 121)
        cuts = [int((ends[i] - durations[i] / 2 - 3) * 600) for i in spaces]
        bounds = np.clip([first, *cuts, last], first, last)
        # The mixer and time stretcher both take 48 kHz mono PCM.
        pcm = np.empty(len(audio) * 2, np.float32)
        pcm[::2] = audio
        pcm[1:-1:2] = (audio[:-1] + audio[1:]) * .5
        pcm[-1] = audio[-1]
        pcm = np.clip(np.rint(pcm * 32768), -32768, 32767).astype("<i2")
        return tuple(pcm[a*2:b*2] for a, b in zip(bounds, bounds[1:]))


@lru_cache(maxsize=1)
def _load(directory):
    if not all((directory / f).is_file() for f in ("model.onnx", "am_echo.bin")):
        return None
    if not shutil.which("espeak-ng"):
        return None
    try:
        import onnxruntime
    except ImportError:
        return None
    return NeuralVoice(directory, onnxruntime)


def installed_voice():
    directory = Path(os.environ.get("KARAOKE_TTS_MODEL_DIR", DATA / "private/linux-four/kokoro-v1"))
    with _LOAD:
        return _load(directory)
