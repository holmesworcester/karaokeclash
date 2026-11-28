"""Compare live Homer with the earlier, listener-preferred conversion."""

import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np
from scipy.io import wavfile

from audio import CharacterWorker, LAB, ROOT, decode
from homer_monitor_hardware import capture
from homer_timbre import monitor_delay
from homer_listening import listening_page


class HomerIdentity(unittest.TestCase):
    def test_identity_and_signal_quality_against_the_old_voice(self):
        out = Path("build/evidence/homer-identity-check").resolve()
        out.mkdir(parents=True, exist_ok=True)
        fixture = Path(__file__).resolve().parent / "fixtures/homer"
        paths = {"lyrics": fixture / "lyrics.flac", "reader": ROOT / "assets/voice-check.flac",
                 "singing": LAB / "framework-rvc/source/falcon.wav"}
        sources = {name: decode(path, 1, seconds=8) for name, path in paths.items()}
        gap = np.zeros(24000, dtype=np.int16)
        positions, parts = {}, [gap]
        offset = len(gap)
        for name, pcm in sources.items():
            positions[name] = slice(offset, offset + len(pcm))
            parts.extend([pcm, gap])
            offset += len(pcm) + len(gap)
        joined = np.concatenate(parts)
        mic = joined.astype("f4") / 32768.
        previous, old_timing = capture(mic, 2, factory=CharacterWorker)
        live, timing = capture(mic, 2)
        (out / "timing.json").write_text(json.dumps({"old": old_timing, "live": timing}, indent=2))
        latency = 3840 + monitor_delay(2)
        cases = {}
        for name, clip in positions.items():
            cases[name] = {}
            for kind, samples in (("old", previous[3840:][clip]),
                                  ("live", live[latency:][clip])):
                path = out / f"{kind}-{name}.wav"
                if kind == "old" and name == "lyrics":
                    samples = decode(fixture / "old-lyrics.flac", 1)
                wavfile.write(path, 48000, samples)
                cases[name][kind] = path
        listening_page(out, cases, LAB / "refs/homer/dearbaby.wav")
        candidates = [clip for versions in cases.values() for clip in versions.values()]
        candidates.append(fixture / "rejected-lyrics.flac")
        report = out / "scores.json"
        scorer = Path(__file__).with_name("score_homer.py").resolve()
        subprocess.run([LAB / "run-lab", LAB / "env/bin/python", scorer, "--lab", LAB,
                        "--dnsmos", os.environ["KARAOKE_DNSMOS"], "--output", report, *candidates], check=True)
        scores = json.loads(report.read_text())["clips"]
        self.assertGreater(scores["old-lyrics"]["homer_cosine"] - scores["rejected-lyrics"]["homer_cosine"], .025)
        # A dry or repeatedly late microphone cannot stand in for either converter.
        for replay in (old_timing, timing):
            self.assertGreater(replay["converted"], .97 * len(mic) / 960, replay)
            self.assertLessEqual(replay["late"], max(2, .01 * replay["converted"]), replay)
        self.assertLess(timing["held_tone"], timing["converted"] * .10, timing)
        identity, signal = [], []
        for name in sources:
            old, new = scores[f"old-{name}"], scores[f"live-{name}"]
            identity.append(new["homer_cosine"] - old["homer_cosine"])
            signal.append(new["dnsmos_sig"] - old["dnsmos_sig"])
        self.assertGreaterEqual(min(identity), -.02, identity)
        self.assertGreaterEqual(min(signal), -.10, signal)
        self.assertGreaterEqual(float(np.mean(identity)), 0, identity)
        self.assertGreater(float(np.mean(signal)), .035, signal)
