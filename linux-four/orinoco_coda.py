"""Prepare the starred Orinoco Flow coda from the official video for the local library."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

import numpy as np

from import_job import BAR, BPM, RATE, carrier, check_stems, warped_rows, warped_stem

RECORDING = "LTrk4X9ACtw"
PRACTICE = "2241-8-909312-delay"
FIRST = 156.528   # First consonant of "deep" on the downbeat of the third verse.
SOURCE_BAR = 240 / 115
SOURCE_START = 150
BARS = 21
# (start bar on the video clock relative to FIRST, length in bars).
# Splice just before the beat-four pickups. Reuse the recorded "seen" and
# title response where the video sings a different answer from the KAR.
EDITS = [(0, 2.72), (4.72, 1), (3.72, 4), (5.72, 2),
         (9.72, 4), (5.72, 2), (5.72, 2), (9.72, 3.28)]

# Onset and release in milliseconds of the video, from its separated lead.
LINES = [
    ("deep sea of clouds", [(156528, 157028), (157028, 157368), (157368, 157528), (157528, 158088)]),
    ("To the island of the moon", [(158088, 158348), (158348, 158532), (158708, 159072),
                                   (159248, 159368), (159368, 159572), (159688, 160080)]),
    ("Carry me on the waves", [(160168, 160728), (160728, 160990), (161230, 161548),
                               (161548, 161748), (161748, 162255)]),
    ("Carry me on the waves", [(164368, 164888), (164888, 165140), (165413, 165708),
                               (165708, 165908), (165908, 166423)]),
    ("To the lands I've never seen", [(166428, 166668), (166668, 166968), (166968, 167328),
                                      (167328, 167488), (167488, 167928), (167928, 168288)]),
    ("We can sail we can sail", [(168568, 168768), (168768, 169028), (169028, 169581),
                                 (169608, 169808), (169808, 170013), (170013, 170556)]),
    ("With the Orinoco flow", [(170590, 170832), (170848, 171042),
                               (171104, 172125), (172128, 172591)]),
    ("We can steer we can near", [(176848, 177148), (177148, 177348), (177348, 177944),
                                  (177948, 178168), (178168, 178435), (178448, 178946)]),
    ("With Rob Dickins at the wheel", [(178948, 179228), (179228, 179490), (179508, 180082),
                                       (180088, 180268), (180268, 180476), (180505, 181053)]),
    ("We can sigh say goodbye", [(181088, 181288), (181288, 181528), (181528, 182080),
                                 (182088, 182360), (182368, 183040)]),
    ("Ross and his dependencies", [(183048, 183464), (183468, 183662),
                                   (183688, 183924), (183928, 185101)]),
]


def chord_score(coda):
    """The video verse restarts its F/C/G figure at each nine-bar verse cut.

    The coda starts three source bars into verse three. Its splices all move
    by whole two-bar figures, so the G bar remains first throughout the coda.
    """
    f, c, g = [41, 53, 57, 60], [48, 55, 60, 64], [43, 55, 59, 62]
    bars = BARS if coda else 27
    events = []
    for bar in range(bars):
        phase = bar + 1 if coda else bar % 9
        start = bar * BAR
        if phase % 2:
            events.append([start, g])
        else:
            events.extend([[start, f], [start + BAR / 2, c]])
    return {"duration": bars * BAR, "events": events}


def sections():
    start = 0
    for source_bar, bars in EDITS:
        yield FIRST + source_bar * SOURCE_BAR, start * SOURCE_BAR, bars * SOURCE_BAR
        start += bars


def edited_rows():
    rows = []
    for src, dest, length in sections():
        for line, spans in LINES:
            words, times = [], []
            for word, (a, b) in zip(line.split(), spans, strict=True):
                # A source edit may be immediately before the consonant of the next line.
                if src * 1000 - .01 <= a < (src + length) * 1000:
                    words.append(word)
                    times.append([round((dest - src) * 1000 + a),
                                  round((dest + min(length, b / 1000 - src)) * 1000)])
            if words:
                rows.append([words, times])
    return rows


def assemble(pcm):
    """Use identical sample boundaries and a 3 ms crossfade on all three stems."""
    fade = round(.003 * RATE)
    ramp = np.linspace(0, 1, fade)[:, None]
    parts = []
    for source, dest, duration in sections():
        start = round((source - SOURCE_START) * RATE)
        count = round((dest + duration) * RATE) - round(dest * RATE)
        section = pcm[start:start + count + fade]
        if len(section) != count + fade:
            raise ValueError("Source stem ends inside the selected coda")
        parts.append(section)
    joined = []
    for i, part in enumerate(parts):
        seam = parts[i-1][-fade:] * (1 - ramp) + part[:fade] * ramp
        joined.extend((np.rint(seam).astype("<i2"), part[fade:-fade]))
    return np.concatenate(joined)


def render(source, work, out, separator, model):
    if json.loads((source / "reference.info.json").read_text())["id"] != RECORDING:
        raise ValueError("This cut uses the 3:43 official music video")
    for stem, name in [("backing", "instrumental.wav"), ("guide", "clean-vocal.wav")]:
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(SOURCE_START), "-i", source / name,
                        "-t", "40", "-c:a", "pcm_f32le", work / (stem + ".wav")], check=True)
    subprocess.run([sys.executable, separator, work / "backing.wav", model, work / "drums.wav"],
                   check=True)
    grid = np.arange(BARS + 1) * SOURCE_BAR
    for stem in ("backing", "drums", "guide"):
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", work / (stem + ".wav"),
                                       "-ar", str(RATE), "-ac", "2", "-f", "s16le", "-"])
        pcm = assemble(np.frombuffer(raw, "<i2").reshape(-1, 2))
        path = work / "arranged.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((2, 2, RATE, 0, "NONE", ""))
            wav.writeframes(pcm.tobytes())
        warped_stem(path, out / (stem + ".wav"), grid, BARS)
    check_stems(out / "backing.wav", out / "guide.wav")
    return carrier(out / "backing.wav", np.arange(BARS + 1) * BAR)


def song_record(score):
    rows = warped_rows(edited_rows(), np.arange(BARS + 1) * SOURCE_BAR, BARS)
    def seconds(bar):
        return round(bar * BAR * RATE) / RATE
    main = dict(id="coda", label="21-bar coda", start=0, end=seconds(BARS), bars=BARS,
                bpm=BPM, beat_zero=0, loop=True, audio=".", rows={"lead": rows}, score=score,
                loopRole="main")
    # A four-bar backing around the six words, gated in the rests on either side.
    def on_desk(ms):
        return round((ms - 1000 * FIRST) * BAR / SOURCE_BAR)
    teaser = dict(main, id="teaser", label="We can sail", start=seconds(5), end=seconds(9),
                  bars=4, loopRole="teaser", teaserWindowMs=[on_desk(168522), on_desk(170576)])
    return dict(title="Orinoco Flow", artist="Enya", key="", loopFamily="Orinoco Flow",
                review=dict(video=RECORDING, practice=PRACTICE), cuts=[main, teaser])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "separator", "drums_model", "imports"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    args.imports.mkdir(parents=True, exist_ok=True)
    target = args.imports / "studio-orinoco-flow"
    with tempfile.TemporaryDirectory(dir=args.imports.parent) as tmp:
        work = Path(tmp)
        out = work / target.name
        out.mkdir()
        score = render(args.source, work, out, args.separator, args.drums_model)
        (out / "song.json").write_text(json.dumps(song_record(score), ensure_ascii=False))
        out.rename(target)
    print(target)
