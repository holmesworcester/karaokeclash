"""Prepare the two Ben E. King editions from the reviewed 1961 transfer.

The source MIDI is the 80-bar ensemble score (first “Stand by me” on
https://jmboucher.fr/accueil/mes-partitions/) exported with MuseScore.
Preparation uses python3-mido, FluidSynth and a General MIDI soundfont.
"""

import argparse
import json
import subprocess
import tempfile
import wave
from copy import deepcopy
from pathlib import Path

import numpy as np
from import_job import RATE, warped_stem
from carrier import save_rendered_score

TITLE = "Stand by Me"
EDITION = "stand-by-me-orchestral"
CHORDS = {
    "A": (45, 57, 61, 64),
    "F#m": (42, 57, 61, 66),
    "D": (38, 57, 62, 66),
    "E": (40, 56, 59, 64),
}
FORM = ("A", "A", "F#m", "F#m", "D", "E", "A", "A")
# Source bar, bars per passage, bars on the installed cut (including repeats).
CUTS = {"hook-1": (22, 9, 36), "hook-2": (71, 8, 32), "teaser": (7, 4, 4)}
BAR_TICKS = 1920
# Bring the ensemble up to the transfer's measured A4 = 445.86 Hz.
BEND = 938


def chord_score(cut, duration):
    if cut["id"] == "full":
        bars = enumerate(cut["downbeats"])
    else:
        first, phrase, count = CUTS[cut["id"]]
        if cut["bars"] != count:
            raise ValueError("Expected the reviewed pickup and repeat layout")
        bars = ((first + i % phrase, i * 240 / cut["bpm"]) for i in range(count))
    events = []
    for bar, start in bars:
        notes = list(CHORDS[FORM[bar % 8]])
        if not events or events[-1][1] != notes:
            events.append([round(start, 6), notes])
    return {"duration": duration, "events": events}


def read_notes(track):
    tick = 0
    sounding, notes = {}, []
    for msg in track:
        tick += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        key = (msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity:
            sounding[key] = (tick, msg.velocity)
        else:
            start, velocity = sounding.pop(key)
            notes.append((start, tick, msg.note, velocity))
    if sounding:
        raise ValueError("Unreleased note in ensemble score")
    return sorted(notes)


def arrangement(source, downbeats, seconds, out):
    import mido

    midi = mido.MidiFile(source)
    if midi.ticks_per_beat != 480 or len(midi.tracks) != 9 or midi.length != 160:
        raise ValueError("Expected the 80-bar, 120 BPM ensemble score")
    # One pickup measure puts the first measured downbeat on a MIDI bar line.
    grid = [0, *downbeats]
    step = downbeats[-1] - downbeats[-2]
    last_tick = round((len(grid) - 1 + (seconds - grid[-1]) / step) * BAR_TICKS)
    clock = mido.MidiTrack()
    clock.name = "1961 recording clock"
    clock.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    clock.append(mido.MetaMessage("key_signature", key="A"))
    for i, length in enumerate([*np.diff(grid), step]):
        clock.append(mido.MetaMessage("set_tempo", tempo=round(length * 250000), time=BAR_TICKS if i else 0))
    clock.append(mido.MetaMessage("end_of_track", time=last_tick - (len(grid) - 1) * BAR_TICKS))

    def part(name, channel, program, volume, pan, notes):
        track = mido.MidiTrack()
        track.name = name
        track.append(mido.Message("program_change", channel=channel, program=program))
        for control, value in [(7, volume), (10, pan), (91, 32)]:
            track.append(mido.Message("control_change", channel=channel, control=control, value=value))
        if channel != 9:
            track.append(mido.Message("pitchwheel", channel=channel, pitch=BEND))
        events = []
        for a, b, pitch, velocity in notes:
            events.append((a, 1, mido.Message("note_on", channel=channel, note=pitch, velocity=velocity)))
            events.append((b, 0, mido.Message("note_off", channel=channel, note=pitch)))
        # Fade into the end of the transfer, including the orchestral release.
        fade = 84 * BAR_TICKS
        for tick in range(fade, last_tick, BAR_TICKS // 4):
            expression = round(127 * (last_tick - tick) / (last_tick - fade))
            events.append(
                (tick, 0, mido.Message("control_change", channel=channel, control=11, value=expression))
            )
        previous = 0
        for tick, _, msg in sorted(events, key=lambda item: item[:2]):
            track.append(msg.copy(time=tick - previous))
            previous = tick
        return track

    def extend(notes):
        # Insert a complete refrain after the string interlude for the longer
        # record. Clip tied notes at both splice boundaries before repeating.
        result = []
        dest = BAR_TICKS
        for lo, hi in [(0, 72), (48, 56), (72, 80)]:
            for a, b, pitch, vel in notes:
                start, stop = max(a, lo * BAR_TICKS), min(b, hi * BAR_TICKS)
                if stop > start:
                    result.append((dest + start - lo * BAR_TICKS, dest + stop - lo * BAR_TICKS, pitch, vel))
            dest += (hi - lo) * BAR_TICKS
        return result

    # Keep the scored bass figure, clarinet melody and string interlude. Give
    # the piano the same plain triads as the sustained strings on its offbeats.
    parts = []
    patch = [
        (0, "Plucked bass", 0, 32, 114, 62),
        (1, "Piano", 1, 0, 86, 49),
        (3, "Violins", 2, 48, 90, 44),
        (4, "Flute", 3, 73, 83, 76),
        (5, "Clarinet", 4, 71, 94, 69),
        (6, "Cello", 5, 42, 95, 82),
        (7, "Double bass", 6, 43, 83, 58),
        (8, "Triangle", 9, 0, 88, 83),
    ]
    for idx, name, channel, program, volume, pan in patch:
        notes = read_notes(midi.tracks[idx])
        if idx == 1:
            strikes = sorted({(a, b, v) for a, b, _, v in notes})
            notes = [
                (a, b, pitch, v) for a, b, v in strikes for pitch in CHORDS[FORM[(a // BAR_TICKS) % 8]][1:]
            ]
        notes = extend(notes)
        if idx == 0:
            # Resolve the score's inverted final A onto its root into the fade.
            notes = [(a, b, 45 if a >= 88 * BAR_TICKS else pitch, v) for a, b, pitch, v in notes]
            notes.append((89 * BAR_TICKS, last_tick, 45, 58))
        parts.append(part(name, channel, program, volume, pan, notes))

    # Tie shared voices through changes. The guide for pitch effects uses the
    # same harmony, with its own bass note under the three upper voices.
    held = []
    active = {}
    for bar in range(90):
        tick = (bar + 1) * BAR_TICKS
        if tick >= last_tick:
            break
        pitches = set(CHORDS[FORM[bar % 8]])
        for pitch in active.keys() - pitches:
            held.append((active.pop(pitch), tick, pitch, 66))
        for pitch in pitches - active.keys():
            active[pitch] = tick
    held.extend((a, last_tick, p, 66) for p, a in active.items())
    strings = part("Held strings", 7, 49, 81, 54, [n for n in held if n[2] >= 55])
    for file, tracks in [
        ("orchestra", [*parts, strings]),
        ("percussion", [parts[-1]]),
        ("chords", [part("Chord pitches", 0, 48, 96, 64, held)]),
    ]:
        result = mido.MidiFile(type=1, ticks_per_beat=480)
        result.tracks.extend([clock, *tracks])
        result.save(out / (file + ".mid"))


def wave_length(path):
    with wave.open(str(path)) as wav:
        if wav.getframerate() != RATE or wav.getnchannels() != 2 or wav.getsampwidth() != 2:
            raise ValueError(f"Expected stereo desk PCM: {path}")
        return wav.getnframes()


def save_orchestra_scores(out, cuts):
    import mido

    starts, notes = {}, []
    now = 0.
    for msg in mido.MidiFile(out / "orchestra.mid"):
        now += msg.time
        if msg.type not in ("note_on", "note_off") or msg.channel == 9:
            continue
        key = (msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity:
            starts[key] = now
        else:
            notes.append((starts.pop(key), now, msg.note))
    if starts:
        raise ValueError("Unreleased orchestral MIDI notes")
    for cut in cuts:
        folder = out / cut["audio"]
        duration = wave_length(folder / "backing.wav") / RATE
        if cut["id"] == "full":
            save_rendered_score(folder / "backing.wav", duration, notes)
            continue
        grid = next(c["downbeats"] for c in cuts if c["id"] == "full")
        first, size, bars = CUTS[cut["id"]]
        grid = grid[first:first + size+1]
        local = np.arange(size+1) * 240 / cut["bpm"]
        passage = [(float(np.interp(a, grid, local)), float(np.interp(b, grid, local)), key)
                   for a, b, key in notes if a < grid[-1] and b > grid[0]]
        repeated = [(a+offset*local[-1], b+offset*local[-1], key)
                    for offset in range(bars//size) for a, b, key in passage]
        save_rendered_score(folder / "backing.wav", duration, repeated)


def build(original, midi, soundfont, stage):
    doc = json.loads((original / "song.json").read_text())
    cuts = {c["id"]: c for c in doc["cuts"]}
    if (doc["artist"], doc["title"]) != ("Ben E. King", TITLE) or cuts.keys() != {"full", *CUTS}:
        raise ValueError("Expected the reviewed Ben E. King studio edition")
    grid = cuts["full"]["downbeats"]
    if len(grid) != 89 or grid[:2] != [1.392, 3.398] or grid[-1] != 177.981:
        raise ValueError("The orchestration uses a different recording clock")
    for c in cuts.values():
        folder = original / c["audio"]
        frames = wave_length(folder / "guide.wav")
        if frames != wave_length(folder / "backing.wav") or frames != wave_length(folder / "drums.wav"):
            raise ValueError(f"Unequal stem lengths for {c['id']}")
        c["score"] = chord_score(c, frames / RATE)

    edition = deepcopy(doc)
    edition["title"] = TITLE + " (orchestral)"
    edition["review"] = (
        "Ensemble adapted from JM Boucher’s first Stand by Me score, with "
        "held triads and a repeated refrain fitted to the studio recording."
    )
    renders = stage / "renders"
    renders.mkdir()
    arrangement(midi, grid, cuts["full"]["score"]["duration"], renders)
    out = stage / EDITION
    out.mkdir()
    for c in edition["cuts"]:
        ident = c["id"]
        clip = out / ident
        clip.mkdir()
        old = original / c["audio"]
        (clip / "guide.wav").hardlink_to(old / "guide.wav")
        c.update(audio=ident, end=c["score"]["duration"])
        if ident == "hook-1":
            c["loopRole"] = "main"
    full = out / "full"
    frames = wave_length(full / "guide.wav")
    for instrument, name in [("orchestra", "backing"), ("percussion", "drums")]:
        rendered = renders / (name + ".wav")
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "-g",
                ".8",
                "-r",
                str(RATE),
                "-F",
                rendered,
                soundfont,
                renders / (instrument + ".mid"),
            ],
            check=True,
        )
        with wave.open(str(rendered)) as src, wave.open(str(full / (name + ".wav")), "wb") as dst:
            dst.setparams(src.getparams())
            if src.getnframes() < frames:
                raise ValueError("Orchestral render ends early")
            dst.writeframes(src.readframes(frames))
    for ident, (start, span, count) in CUTS.items():
        for name in ["backing.wav", "drums.wav"]:
            warped_stem(full / name, out / ident / name, grid[start : start + span + 1], count)
            if wave_length(out / ident / name) != wave_length(out / ident / "guide.wav"):
                raise ValueError("Orchestral cut and guide have different clocks")
    (out / "orchestra.mid").hardlink_to(renders / "orchestra.mid")
    (out / "chords.mid").hardlink_to(renders / "chords.mid")
    save_orchestra_scores(out, edition["cuts"])
    (out / "song.json").write_text(json.dumps(edition, ensure_ascii=False))
    return doc, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "ensemble_midi", "soundfont"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    original = args.original.resolve()
    with tempfile.TemporaryDirectory(dir=original.parent.parent) as tmp:
        stage = Path(tmp)
        initial = (original / "song.json").read_bytes()
        record, new = build(original, args.ensemble_midi, args.soundfont, stage)
        fixed = stage / "corrected.json"
        fixed.write_text(json.dumps(record, ensure_ascii=False))
        if (original / "song.json").read_bytes() != initial:
            raise RuntimeError("Original edition changed during preparation")
        new.rename(original.parent / EDITION)
        fixed.replace(original / "song.json")
