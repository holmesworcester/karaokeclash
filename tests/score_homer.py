"""Score actual audio against reserved Homer recordings and the independent DNSMOS model."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
import torch
import torchaudio


def audio(path):
    pcm = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                                   "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
    x = np.frombuffer(pcm, "<f4").copy()
    if len(x) < 16000 or not np.isfinite(x).all() or np.max(np.abs(x)) < .0001:
        raise ValueError(f"Expected at least one second of audible mono audio in {path}")
    return x


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def embedding(model, pcm):
    feat = torchaudio.compliance.kaldi.fbank(
        torch.from_numpy(pcm).reshape(1, -1) * 32768, num_mel_bins=80,
        dither=0, sample_frequency=16000, window_type="hamming",
    )
    centered = (feat - feat.mean(0)).unsqueeze(0).numpy()
    vector = model.run(None, {model.get_inputs()[0].name: centered})[0].ravel()
    return vector / np.linalg.norm(vector)


def signal_quality(model, pcm):
    # The non-personalized Microsoft P.835 model takes 9.01 s. Repeat short clips
    # and average its calibrated speech-signal ratings over the overlapping windows.
    width = 144160
    while len(pcm) < width:
        pcm = np.tile(pcm, 2)
    ratings = []
    for pos in range(0, len(pcm) - width + 1, 16000):
        scores = model.run(None, {model.get_inputs()[0].name: pcm[None, pos:pos+width]})[0]
        ratings.append(np.polyval([-.08397278, 1.22083953, .0052439], scores[0, 0]))
    return float(np.mean(ratings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--dnsmos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("clips", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    speaker = session(args.lab / "models/verifier.onnx")
    quality = session(args.dnsmos)
    targets = sorted((args.lab / "refs/heldout/homer").glob("*.wav"))
    if len(targets) < 2:
        raise ValueError("At least two reserved Homer recordings are required")
    refs = np.stack([embedding(speaker, audio(f)) for f in targets])
    center = refs.mean(0)
    center /= np.linalg.norm(center)
    result = {"references": [f.name for f in targets], "clips": {}}
    for clip in args.clips:
        pcm = audio(clip)
        voice = embedding(speaker, pcm)
        scores = {"homer_cosine": float(voice @ center), "dnsmos_sig": signal_quality(quality, pcm),
                  "per_reference_cosine": (refs @ voice).tolist()}
        if clip.stem in result["clips"]:
            raise ValueError(f"Duplicate clip label {clip.stem}")
        result["clips"][clip.stem] = scores
        print(f"{clip.stem:32s} Homer {scores['homer_cosine']:.3f}  SIG {scores['dnsmos_sig']:.2f}", flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
