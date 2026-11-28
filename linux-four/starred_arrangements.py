"""Build the reviewed practice cuts from sample-aligned private recording stems."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import tempfile
import wave

import numpy as np

from import_job import (
    BAR,
    BPM,
    RATE,
    carrier,
    check_stems,
    duration,
    normalize,
    warped_rows,
    warped_stem,
)


RECIPES = Path(__file__).with_name("starred-arrangements")
STEMS = {"backing": "instrumental.wav", "drums": "drums.wav", "guide": "clean-vocal.wav"}


def midi_score(path, seconds):
    """Read the arrangement on the MIDI file's tempo map, retaining overlapping voices."""
    import mido

    held = Counter()
    events = []
    now = 0.0
    for message in mido.MidiFile(path):
        now += message.time
        if now > seconds or message.type not in ("note_on", "note_off") or message.channel == 9:
            continue
        attack = message.type == "note_on" and message.velocity
        when = round(now, 6)
        held[message.note] += 1 if attack else -1
        pitches = sorted(key for key, count in held.items() if count > 0)
        if events and when == events[-1][0]:
            events[-1][1] = pitches
        else:
            events.append([when, pitches])
    return {"duration": seconds, "events": events}


def score_excerpt(score, grid):
    """Use the same source bar warp as the three practice stems."""
    events = score["events"]
    starts = [t for t, _ in events]
    first = bisect_right(starts, grid[0])
    last = bisect_right(starts, grid[-1] - .5 / RATE)
    beats = np.arange(len(grid)) * BAR
    out = [[0, events[first - 1][1] if first else []]]
    out.extend([round(float(np.interp(t, grid, beats)), 6), notes] for t, notes in events[first:last])
    return {"duration": round(beats[-1] * RATE) / RATE, "events": out}


def arrangement(parts, rows):
    """Put word intervals and bar lines on the same clock as the spliced source."""
    grid, lyrics = [0.0], []
    for part in parts:
        source = part["grid"]
        lo, hi = part["voice_ms"]
        if len(source) < 2 or any(b <= a for a, b in zip(source, source[1:])):
            raise ValueError("Expected ordered source bar boundaries")
        if not source[0] * 1000 - 1 <= lo < hi <= source[-1] * 1000 + 1:
            raise ValueError("Vocal gate falls outside source bars")
        offset = grid[-1] - source[0]
        for i, first, stop in part["lines"]:
            words, spans = rows[i]
            chosen = spans[first:stop]
            if not chosen or chosen[0][0] < lo or chosen[-1][1] > hi:
                raise ValueError(f"Vocal gate cuts into selected words on line {i}")
            lyrics.append(
                [
                    words[first:stop],
                    [[round(a + offset * 1000), round(b + offset * 1000)] for a, b in chosen],
                ]
            )
        grid.extend(b + offset for b in source[1:])
    return grid, warped_rows(lyrics, grid, len(grid) - 1)


def splice(source, dest, parts, vocal):
    """Crossfade at the original downbeats without introducing an extra frame at joins."""
    overlap = round(0.003 * RATE)
    ramp = np.linspace(0, 1, overlap)[:, None]
    with wave.open(str(source)) as src, wave.open(str(dest), "wb") as out:
        out.setparams((2, 2, RATE, 0, "NONE", ""))
        sections = []
        elapsed = 0.0
        for part in parts:
            grid = part["grid"]
            start = round(grid[0] * RATE)
            end = elapsed + grid[-1] - grid[0]
            frames = round(end * RATE) - round(elapsed * RATE)
            elapsed = end
            src.setpos(start)
            pcm = np.frombuffer(src.readframes(frames + overlap), "<i2").reshape(-1, 2).copy()
            if len(pcm) != frames + overlap:
                raise ValueError("The recording ends inside this passage")
            if vocal:
                lo, hi = (max(0, round(t * RATE / 1000) - start) for t in part["voice_ms"])
                pcm[:lo] = 0
                pcm[hi:] = 0
                # Smooth a clipped voiced sample at a word boundary.
                fade = min(round(0.002 * RATE), (min(hi, len(pcm)) - lo) // 2)
                if fade > 0:
                    gain = np.linspace(0, 1, fade)[:, None]
                    pcm[lo : lo + fade] = (pcm[lo : lo + fade] * gain).round()
                    tail = min(hi, len(pcm))
                    pcm[tail - fade : tail] = (pcm[tail - fade : tail] * (1 - gain)).round()
            sections.append(pcm)
        for i, pcm in enumerate(sections):
            join = sections[i - 1][-overlap:] * (1 - ramp) + pcm[:overlap] * ramp
            out.writeframesraw(np.rint(join).astype("<i2").tobytes())
            out.writeframesraw(pcm[overlap:-overlap].tobytes())


def prepare(recipe, source, imports):
    if json.loads((source / "master.info.json").read_text())["id"] != recipe["recording"]:
        raise ValueError("The selected bars use a different recording")
    imports.mkdir(parents=True, exist_ok=True)
    target = imports / recipe["slug"]
    with tempfile.TemporaryDirectory(dir=imports.parent) as temp:
        work = Path(temp)
        out = work / "song"
        out.mkdir()
        seconds = duration(source / "instrumental.wav")
        for name, filename in STEMS.items():
            path = source / filename
            if abs(duration(path) - seconds) > 0.04:
                raise ValueError(f"Wrong recording clock for {filename}")
            normalize(path, work / (name + ".wav"), round(seconds * RATE))
        check_stems(work / "backing.wav", work / "guide.wav")
        cuts = []
        scored = midi_score(RECIPES / recipe["midi"], seconds) if "midi" in recipe else None
        if downbeats := recipe.get("full_grid"):
            full = out / "recording"
            full.mkdir()
            for name in STEMS:
                (full / (name + ".wav")).hardlink_to(work / (name + ".wav"))
            cuts.append(
                dict(
                    id="full",
                    label="Full song",
                    audio="recording",
                    start=0,
                    end=seconds,
                    loop=False,
                    bpm=240 / float(np.median(np.diff(downbeats))),
                    downbeats=downbeats,
                    rows={"lead": recipe["rows"]},
                    score=scored or carrier(full / "backing.wav", downbeats),
                )
            )
        for choice in recipe["cuts"]:
            ident, parts = choice["id"], choice["sections"]
            grid, rows = arrangement(parts, recipe["rows"])
            bars = len(grid) - 1
            folder = out / ident
            folder.mkdir()
            for name in STEMS:
                joined = work / "joined.wav"
                splice(work / (name + ".wav"), joined, parts, name == "guide")
                warped_stem(joined, folder / (name + ".wav"), grid, bars)
            cut = dict(
                id=ident,
                label=choice["label"],
                audio=ident,
                start=0,
                end=round(bars * BAR * RATE) / RATE,
                bpm=BPM,
                beat_zero=0,
                bars=bars,
                loop=True,
                rows={"lead": rows},
            )
            if scored:
                # These excerpts come from one continuous passage of the album MIDI.
                (section,) = parts
                source_grid = section["grid"]
                cut["score"] = score_excerpt(scored, source_grid)
                cut["scoreId"] = recipe["slug"] + "/full"
                cut["album_bar_offset"] = float(np.interp(source_grid[0], downbeats, np.arange(len(downbeats))))
            else:
                cut["score"] = carrier(folder / "backing.wav", np.arange(bars + 1) * BAR)
            if ident in ("teaser", "main"):
                cut["loopRole"] = ident
            if ident == "teaser":
                if len(parts) != 1:
                    raise ValueError("Expected one teaser phrase")
                g = parts[0]["grid"]
                cut["teaserWindowMs"] = [
                    round(np.interp(t / 1000, g, np.arange(len(g))) * BAR * 1000)
                    for t in parts[0]["voice_ms"]
                ]
            if ident == "main" and not recipe.get("full_grid"):
                cuts.append(dict(cut, id="full", label="Full passage"))
            cuts.append(cut)
        record = {k: recipe[k] for k in ("artist", "title")}
        record.update(cuts=cuts, review={"recording": recipe["recording"]}, key=recipe.get("key", ""))
        (out / "song.json").write_text(json.dumps(record, ensure_ascii=False))
        out.rename(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song", help="Recipe filename without .json")
    parser.add_argument("source", type=Path)
    parser.add_argument("imports", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            json.loads((RECIPES / (args.song + ".json")).read_text()), args.source, args.imports
        )
    )
