"""Prepare the three Cowgirl performance parts from the remastered album stems."""

import argparse
import json
from pathlib import Path
import tempfile
import wave

import numpy as np
from scipy.signal import correlate

from audio import decode
from carrier import carrier

RATE = 48000
BPM = 126.92
BAR = 240 / BPM
DOWNBEAT = .055
RECORDING = "-JbGBLZghVE"


def excerpt(path, first_bar, bars, channels):
    start = DOWNBEAT + first_bar * BAR
    frames = round(bars * BAR * RATE)
    fade = round(.003 * RATE)
    audio = decode(path, channels, start=start, seconds=(frames + fade + 2) / RATE,
                   floating=True).reshape(-1, channels)
    if len(audio) < frames + fade:
        raise ValueError(f"{path.name} ends inside the selected passage")
    clip = audio[:frames].copy()
    # Carry the end of the previous pass across the first three milliseconds.
    ramp = np.linspace(0, 1, fade)[:, None]
    clip[:fade] = audio[frames:frames + fade] * (1 - ramp) + clip[:fade] * ramp
    return clip


def matched_chant(exposed, voices):
    """Fit the exposed sample to its eleven retriggers beneath the sung hook."""
    result = np.empty_like(voices)
    for word in range(11):
        start = round(word * 1.5 * BAR / 4 * RATE)
        end = min(len(voices), round((word + 1) * 1.5 * BAR / 4 * RATE))
        # Measure the lag on the body of each word, away from either retrigger.
        a, b = start + round(.055 * RATE), start + round(.395 * RATE)
        radius = round(.035 * RATE)
        reference = exposed[a:b]
        match = correlate(voices[a-radius:b+radius], reference, mode="valid", method="fft")
        lag = int(match.argmax()) - radius
        ref = np.take(exposed, np.arange(start, end) - lag, mode="wrap")
        gain = np.clip(match.max() / max(reference @ reference, 1e-9), 0, 1.5)
        result[start:end] = gain * ref
    return result


def relative_rows(rows, first_bar, lo_ms, hi_ms):
    origin = round((DOWNBEAT + first_bar * BAR) * 1000)
    phrases = [[words, [[a-origin, b-origin] for a, b in times]]
               for words, times in rows if lo_ms <= times[0][0] and times[-1][1] <= hi_ms]
    if not phrases:
        raise ValueError(f"No reviewed words in the album passage starting at {lo_ms / 1000:g}s")
    return phrases


def vocal_window(length, lo, hi):
    frames = np.arange(length)
    fade = round(.005 * RATE)
    return np.clip(np.minimum(frames - lo * RATE / 1000, hi * RATE / 1000 - frames) / fade, 0, 1).astype("f4")


def write_cut(root, role, label, bars, bed, drums, guide, rows, **options):
    directory = root / role
    directory.mkdir()
    frames = round(bars * BAR * RATE)
    for name, data in (("backing", bed), ("drums", drums), ("guide", guide[:, None])):
        if len(data) != frames:
            raise ValueError(f"{role} {name} has the wrong length")
        with wave.open(str(directory / (name + ".wav")), "wb") as wav:
            wav.setparams((data.shape[1], 2, RATE, 0, "NONE", ""))
            wav.writeframes(np.clip(np.rint(data * 32768), -32768, 32767).astype("<i2").tobytes())
    score = carrier(directory / "backing.wav", np.arange(bars + 1) * BAR)
    return dict(id=role, loopRole=role, label=label, audio=role, bars=bars, start=0,
                end=frames/RATE, bpm=BPM, beat_zero=0, loop=True, rows={"lead": rows},
                score=score, **options)


def full_verse(root, vocal, exposed, bed, drums, chant, hook):
    first_bar, bars = 133, 32
    origin = (DOWNBEAT + first_bar * BAR) * 1000
    sheet = json.loads(Path(__file__).with_name("cowgirl-verse.json").read_text())
    verse = relative_rows([(line.split(), spans) for line, spans in sheet], first_bar, origin, origin + bars*BAR*1000)
    recording = excerpt(vocal, first_bar, bars, 1)[:, 0]
    dry = recording.copy()

    # Four more chant cycles enter underneath the album lead. Align each
    # exposed template to its measured onset before taking it out of the mic.
    for onset in (277986, 285553, 293121, 300687):
        start = round((onset - 23 - origin) * RATE / 1000)
        part = dry[start:start+len(exposed)]
        part -= matched_chant(exposed, part)

    def window(lo, hi):
        return vocal_window(len(dry), lo - origin, hi - origin)

    # During the seven-word response the sample becomes the foreground part.
    # Leave the other retriggers in the backing. Keep only the first syllable
    # of the penultimate album response before the final "an eraser of love".
    response = window(285535, 290499)
    lead_window = window(254668, 285496) + window(293180, 305620) + window(306660, 308233)
    guide = dry * lead_window + recording * response
    # The four-bar backing starts on album bar 47. Match its bar phase to
    # the longer passage so the sampled pattern meets the foreground response.
    offset = round(((first_bar - 47) % 4) * BAR * RATE)
    frame = (np.arange(len(dry)) + offset) % len(bed)
    layered = (bed + .55 * hook[:, None])[frame] - (chant[frame] * response)[:, None]
    return write_cut(root, "full", "I scream", bars, layered, drums[frame], 1.65 * guide, verse)


def prepare(source, root):
    if json.loads((source / "reference.info.json").read_text())["id"] != RECORDING:
        raise ValueError("Expected the 8:29 remastered album recording of Cowgirl")
    words = [line.split("]", 1)[1].split() for line in (source / "reviewed.lrc").read_text().splitlines()
             if line.startswith("[")]
    rows = list(zip(words, json.loads((source / "timing.json").read_text()), strict=True))
    vocal = source / "clean-vocal.wav"
    backing = source / "instrumental.wav"
    drums = source / "drums.wav"
    exposed = excerpt(vocal, 15, 4, 1)[:, 0]
    chant_words = [[["Everything", "everything"], [[23, 698], [728, 1407]]]]
    teaser = write_cut(root, "teaser", "Everything, everything", 4, excerpt(backing, 15, 4, 2),
                       excerpt(drums, 15, 4, 2), exposed * vocal_window(len(exposed), 0, 1420), chant_words,
                       teaserWindowMs=[0, 1420])

    voices = excerpt(vocal, 47, 4, 1)[:, 0]
    chant = matched_chant(exposed, voices)
    bed = excerpt(backing, 47, 4, 2) + chant[:, None]
    beat = excerpt(drums, 47, 4, 2)
    lead = voices - chant
    hook = relative_rows(rows, 47, 88950, 96495)
    main = write_cut(root, "main", "I'm invisible", 4, bed, beat, lead, hook)

    full = full_verse(root, vocal, exposed, bed, beat, chant, lead)
    record = dict(title="Cowgirl", artist="Underworld", key="G Mixolydian (+ B♭)",
                  loopFamily="Cowgirl", cuts=[full, main, teaser])
    (root / "song.json").write_text(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("imports", type=Path)
    args = parser.parse_args()
    args.imports.mkdir(parents=True, exist_ok=True)
    target = args.imports / "studio-cowgirl"
    with tempfile.TemporaryDirectory(dir=args.imports.parent) as tmp:
        staged = Path(tmp) / target.name
        staged.mkdir()
        prepare(args.source, staged)
        staged.rename(target)
    print(target)
