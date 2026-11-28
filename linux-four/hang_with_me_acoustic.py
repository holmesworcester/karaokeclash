"""Arrange the piano changes on Robyn's Body Talk Pt. 1 recording for held synth chords and drum machine."""
import argparse
import json
from pathlib import Path
import tempfile
import wave

import numpy as np
from audio import decode
from carrier import save_rendered_score

RATE = 48000
BPM = 103
BEAT = 60 / BPM
DOWNBEAT = .30
RECORDING = "6Q4PL30HKH8"
EDITION = "studio-hang-with-me-acoustic"
TITLE = "Hang With Me (Acoustic) – Synth Chords"

# Bass first. Keep common upper notes in the same octave through each inversion.
VOICINGS = {
    "Bm": (47, 54, 59, 62), "A": (45, 52, 57, 61), "D": (38, 54, 57, 62),
    "G": (43, 55, 59, 62), "G/B": (47, 55, 59, 62), "D/A": (45, 54, 57, 62),
    "D/F#": (42, 54, 57, 62), "F#7": (42, 52, 58, 61),
    "F#7sus4": (42, 52, 59, 61), "F#m": (42, 54, 57, 61), "D5": (38, 50, 57, 62),
    "rest": (),
}
# Durations are in quarter notes. The pre-chorus anticipates the next
# downbeat. The first chorus has a two-bar secondary dominant before B minor.
RIFF = [("Bm", 2), ("A", 2), ("D", 4)]
PRE = ([("G", 3), ("D/A", 1), ("A", 3), ("D/A", 1)] * 2
       + [("G/B", 2), ("D/A", 2), ("A", 1), ("G", 1), ("D/F#", 2)])
HOOK = [("A", 2), ("Bm", 2), ("G", 2), ("D", 2), ("A", 4)]
CHORUS = HOOK + [("A", 2), ("D", 2)] + HOOK
TURN = [("F#7", 8)]
FORM = [
    ("Intro", RIFF), ("First verses", RIFF * 8),
    ("Pre-chorus", PRE + RIFF * 2), ("First chorus", CHORUS + TURN),
    ("Interlude", RIFF * 2), ("Quiet verse", [("D5", 8)] * 3 + RIFF),
    ("Second pre-chorus", PRE + RIFF),
    ("Chorus with break", CHORUS + [("A", 1), ("rest", 6), ("D/F#", 1)]),
    ("Final chorus", CHORUS + [("F#7sus4", 2), ("F#7", 6)]),
    ("Outro", RIFF * 2 + [("F#m", 4), ("A", 4), ("D", 16)]),
]
# First bar, exclusive last bar, first lyric row, exclusive last row.
CUTS = {"full": (0, 85, 0, 48), "main": (27, 40, 16, 23), "teaser": (2, 6, 0, 1)}


def chord_events(first_bar, stop_bar):
    beat = 0
    for _, section in FORM:
        for symbol, length in section:
            start, stop = max(beat, 4 * first_bar), min(beat + length, 4 * stop_bar)
            if start < stop:
                yield start - 4 * first_bar, stop - 4 * first_bar, symbol
            beat += length


def drum_loop():
    """A quiet kick on quarters, rim on two and four, and alternating closed hats."""
    rng = np.random.default_rng(103)
    t = np.arange(round(.32 * RATE)) / RATE
    kick = .23 * np.exp(-t / .082) * np.sin(2*np.pi*(47*t + 1.75*(1-np.exp(-t/.028))))
    rim = .068*np.exp(-t/.027)*(np.sin(2*np.pi*184*t) + .48*np.sin(2*np.pi*413*t))
    rim += .043*np.exp(-t/.041)*rng.normal(size=len(t))
    hat = .022*np.exp(-t/.026)*np.diff(rng.normal(size=len(t)+1))
    return kick, rim, hat


def instruments(first_bar, stop_bar):
    frames = round((stop_bar - first_bar) * 4 * BEAT * RATE)
    beat_track = np.zeros(frames, "f4")
    kick, rim, hat = drum_loop()
    for eighth in range((stop_bar - first_bar) * 8):
        start = round(eighth * BEAT * RATE / 2)
        n = min(len(hat), frames-start)
        hit = hat[:n] * (.64 if eighth % 2 else 1.)
        if eighth % 2 == 0:
            hit = hit + kick[:n]
        if eighth % 4 == 2:
            hit = hit + rim[:n]
        beat_track[start:start+n] += hit

    keys = np.zeros((frames, 2), "f4")
    events = []
    for start, stop, symbol in chord_events(first_bar, stop_bar):
        notes = VOICINGS[symbol]
        events.append([round(start * BEAT, 6), list(notes)])
        if not notes:
            continue
        a, b = (round(beat * BEAT * RATE) for beat in (start, stop))
        t = np.arange(b-a)/RATE
        # Soft key attack, sustained body, short release. A little stereo spread
        # in the inner voices leaves the bass and the singer in the middle.
        env = np.minimum(t/.013, 1) * np.minimum((len(t)-1-np.arange(len(t)))/(.055*RATE), 1)
        body = .67 + .33*np.exp(-t/.36)
        bright = .18 + .22*np.exp(-t/.24)
        for i, note in enumerate(notes):
            phase = 2*np.pi*440*2**((note-69)/12)*t
            tone = np.sin(phase) + bright*np.sin(2*phase) + .15*np.sin(3*phase) + .065*np.sin(4*phase)
            level = .157 if i == 0 else .099
            pan = (0, -.20, .20, .06)[i]
            sound = level*env*body*tone
            keys[a:b, 0] += sound*(1-pan)
            keys[a:b, 1] += sound*(1+pan)
    return keys + beat_track[:, None], np.column_stack((beat_track, beat_track)), events


def save_synth_score(folder, events, duration):
    spans = []
    for i, (start, pitches) in enumerate(events):
        end = events[i+1][0] if i+1 < len(events) else duration
        spans.extend((start, end, pitch) for pitch in pitches)
    save_rendered_score(folder / "backing.wav", duration, spans)


def source_rows(source):
    lines = [line.split("]", 1)[1].split() for line in (source/"acoustic.lrc").read_text().splitlines()
             if "]" in line and line.split("]", 1)[1].strip()]
    times = json.loads((source/"timing.json").read_text())
    if len(times) != 48 or len(lines) != 48 or any(len(w) != len(t) for w, t in zip(lines, times)):
        raise ValueError("Expected the 48 aligned acoustic lyric lines from LRCLIB 579528")
    # The aligner placed the first chorus 'me' on its release and missed three
    # held verse/outro vowels. End them at the decay on the isolated lead.
    for row, onset, release in [(3, 21028, 23150), (7, 39708, 42375),
                                (22, 88625, 91720), (47, 186548, 190440)]:
        times[row][-1] = [onset, release]
    return list(zip(lines, times))


def write_wav(path, samples):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1 if samples.ndim == 1 else samples.shape[1], 2, RATE, 0, "NONE", ""))
        wav.writeframes(np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2").tobytes())


def prepare(source, target):
    if json.loads((source/"album.info.json").read_text())["id"] != RECORDING:
        raise ValueError("Expected Robyn’s 3:18 acoustic album recording")
    rows = source_rows(source)
    singing = decode(source/"clean-vocal.wav", 1, floating=True)
    last_release = rows[-1][1][-1][1]/1000
    if len(singing) < round(last_release*RATE):
        raise ValueError("The acoustic vocal ends before the final held title")
    cuts = []
    for role, (first, stop, lo, hi) in CUTS.items():
        backing, drums, events = instruments(first, stop)
        folder = target / role
        folder.mkdir()
        origin = DOWNBEAT + 4*first*BEAT
        offset = round(origin * RATE)
        lead = np.zeros(len(backing), "f4")
        clip = singing[offset:offset+len(lead)]
        lead[:len(clip)] = clip
        phrases = [(words, [[a-round(origin*1000), b-round(origin*1000)] for a,b in spans])
                   for words, spans in rows[lo:hi]]
        window = {}
        if role != "full":
            # Preserve the breath and release around this passage. Adjacent lyric
            # lines in these same bars should not sound through Test vocal.
            start = round((phrases[0][1][0][0]/1000-.13)*RATE)
            end = round((phrases[-1][1][-1][1]/1000+.13)*RATE)
            frame = np.arange(len(lead))
            lead *= np.clip(np.minimum(frame-start, end-frame)/(.012*RATE), 0, 1)
            if role == "teaser":
                window["teaserWindowMs"] = [round(start*1000/RATE), round(end*1000/RATE)]
        for name, data in [("backing", backing), ("drums", drums), ("guide", lead)]:
            write_wav(folder/(name+".wav"), data)
        length = len(backing)/RATE
        save_synth_score(folder, events, length)
        cuts.append(dict(id=role, label=role.title(), loopRole=role, audio=role, start=0, end=length,
                         beat_zero=0, bpm=BPM, bars=stop-first, loop=True, rows={"lead": phrases},
                         score={"events": events, "duration": length}, **window))
    data = dict(title=TITLE, artist="Robyn", key="D major", cuts=cuts)
    (target/"song.json").write_text(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("imports", type=Path)
    args = parser.parse_args()
    args.imports.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.imports.parent) as tmp:
        staged = Path(tmp) / EDITION
        staged.mkdir()
        prepare(args.source, staged)
        staged.rename(args.imports / EDITION)
