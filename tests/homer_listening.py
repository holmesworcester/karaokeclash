"""Create a private, blind listening test from the same WAVs as the numerical scores."""

import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile


def listening_page(folder, cases, reference):
    public = folder / "listen"
    public.mkdir(exist_ok=True)
    for name, versions in cases.items():
        clips = {}
        for kind, path in versions.items():
            cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path)]
            stats = subprocess.run(cmd + ["-af", "loudnorm=print_format=json", "-f", "null", "-"],
                                   check=True, capture_output=True, text=True).stderr
            measured, _ = json.JSONDecoder().raw_decode(stats[stats.rindex("{"):])
            loudness = float(measured["input_i"])
            if not np.isfinite(loudness):
                raise ValueError(f"Silent listening clip: {path}")
            pcm = subprocess.check_output(cmd + ["-v", "error", "-ar", "48000", "-ac", "1", "-f", "f32le", "-"])
            clips[kind] = np.frombuffer(pcm, "<f4") * 10 ** ((-20 - loudness) / 20)
        if len({len(pcm) for pcm in clips.values()}) != 1:
            raise ValueError(f"Comparison lengths do not match for {name}")
        # One common trim keeps the peaks intact and preserves the loudness match.
        trim = min(1., .96 / max(float(abs(pcm).max()) for pcm in clips.values()))
        for kind, pcm in clips.items():
            wavfile.write(public / f"{kind}-{name}.wav", 48000, np.rint(pcm * trim * 32767).astype("<i2"))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", reference,
                    "-ar", "48000", "-ac", "1", public / "reference.wav"], check=True)
    template = Path(__file__).with_name("homer-listening.html").read_text()
    (public / "index.html").write_text(template.replace("/*CASES*/[]", json.dumps(list(cases))))
