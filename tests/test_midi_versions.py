"""Compare actual room renders while switching MIDI arrangements mid-song."""
import ctypes as C
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import F, FP, PARAMS
from clip_clock import ClipClock
from desk_audio import DeskAudio
from midi_versions import VERSIONS


class MidiVersions(unittest.TestCase):
    def test_following_mic_plays_its_selected_version_on_each_recording_clock(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        params = (F * len(PARAMS))(*(p[3] for p in PARAMS))
        regions = []
        for name, low, high, old in (("album", 48, 66, 58), ("teaser", 50, 68, 60)):
            region = dict(id=name, track=name, bpm=120, start=0, end=2, loop=True)
            part = dict(source=[1], duration=2, arrangement="Bass", notes=[[0, 2000, low]])
            variants = dict(extra=[[0, 2000, high]], earlier=[[0, 2000, old]], label="Keys")
            if name == "teaser":
                region.update(carrierNotesMs=variants.pop("extra"), carrierNotesLabel="Backing vocals")
            region["backingScoreId"] = audio.load_arrangements(name, region, name, part, variants, False, "song")
            regions.append(region)
        self.assertEqual(audio.version_labels["album"],
                         dict(all="All parts", focus="Focused part", extra="Keys", earlier="Earlier extraction"))
        self.assertEqual(audio.version_labels["teaser"],
                         dict(all="All parts", focus="Instruments", extra="Backing vocals", earlier="Earlier extraction"))
        first, second = ClipClock(regions[0]), ClipClock(regions[1], .5)
        audio.prepare_track(np.zeros(4*48000, "i2"), None, first.region)
        audio.call("stage_clock", *audio.clock_args(first))
        audio.call("stage_lane", 1)
        audio.call("stage_carrier_backing", True)
        audio.call("launch_at", -1, 0)
        audio.call("pane_clock", 0, .5, *audio.clock_args(first), *audio.clock_args(second))
        audio.call("carrier_clock", 1, 0, 120)
        audio.call("config", 1, 1, first.region["backingScoreId"].encode(), 0, 0, params)
        audio.call("glide", 1, 0)
        audio.call("tempo", 120)
        audio.call("transport", 0, 1)

        def hear(played, silent):
            pcm = np.zeros((2, 12000), "f4")
            audio.call("render", None, *(ch.ctypes.data_as(FP) for ch in pcm), pcm.shape[1])
            sample = pcm[0, 2400:]
            t = np.arange(len(sample)) / 48000
            def level(k):
                return abs(np.mean(sample*np.hanning(len(sample))*np.exp(-2j*np.pi*440*2**((k-69)/12)*t)))
            self.assertGreater(min(map(level, played)), 6*max(map(level, silent)))
        hear([48, 66], [50, 58, 68])
        for version, present, absent in (("focus", [48], [58, 66]), ("extra", [66], [48, 58]),
                                          ("earlier", [58], [48, 66]), ("all", [50, 68], [48, 66]),
                                          ("focus", [50], [48, 60, 68]), ("extra", [68], [50, 60]),
                                          ("earlier", [60], [50, 68])):
            audio.call("midi_version", 1, list(VERSIONS).index(version))
            hear(present, absent)
        self.assertEqual(audio.call("clock"), 96000)
