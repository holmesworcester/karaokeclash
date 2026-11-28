"""Loop choices for the playable backing library, on each source's own clock."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path

from moran_chords import score as moran_score

PRESET_FILE = Path(__file__).with_name("loop-presets.json")
PRESETS = json.loads(PRESET_FILE.read_text()) if PRESET_FILE.is_file() else {}


def calibrated_catalog(songs):
    """Apply the desk's authored presets and measured album clocks to runtime editions."""
    result = {}
    for key, song in songs.items():
        preset = PRESETS.get(key, {})
        if preset.get("supersededBy") in songs:
            continue
        song = dict(song, **preset)
        if key == "My Heart Will Go On":
            song["score"] = moran_score()
            song["scoreId"] = "Tony Moran mix / full recording"
        if key == "Orinoco Flow" or key in ("studio-orinoco-flow/coda", "studio-orinoco-flow/teaser"):
            from orinoco_coda import chord_score
            coda = key != "Orinoco Flow"
            song["score"] = chord_score(coda)
            song["scoreId"] = "Orinoco Flow / " + ("coda chords" if coda else "verse chords")
        if "scoreFile" in preset:
            song["score"] = json.loads(Path(__file__).with_name(preset["scoreFile"]).read_text())
        if "lyricsFile" in preset:
            song["rows"] = json.loads(Path(__file__).with_name(preset["lyricsFile"]).read_text())
        if "firstLyricMs" in song:
            # Prepared cuts can have a preceding word already truncated to zero.
            rows = {}
            for role, phrases in song.get("rows", {}).items():
                rows[role] = []
                for words, spans in phrases:
                    part = [(w, span) for w, span in zip(words, spans)
                            if span[0] >= song["firstLyricMs"]]
                    if part:
                        rows[role].append(tuple(zip(*part)))
            song["rows"] = rows
        result[key] = song
    return result


def phrase_groups(song):
    """Rows sharing audio time (e.g. a call and its echo) need one vocal gate."""
    spans = sorted((min(a for a, _ in times) / 1000, max(b for _, b in times) / 1000)
                   for lane in song.get("rows", {}).values() for _, times in lane if times)
    groups = []
    for start, end in spans:
        if groups and start < groups[-1][1]:
            groups[-1] = (groups[-1][0], max(end, groups[-1][1]))
        else:
            groups.append((start, end))
    return groups


def bar_edges(song, start, end):
    grid = song.get("downbeats")
    if grid:
        first = bisect_left(grid, start - .5 / 48000)
        return first, grid[first:bisect_right(grid, end + .5 / 48000)]
    origin = song.get("beat_zero", 0)
    bar = 240 / song["bpm"]
    first = math.ceil((start - .5 / 48000 - origin) / bar)
    stop = math.floor((end + .5 / 48000 - origin) / bar) + 1
    return first, [origin + i * bar for i in range(first, stop)]


def vocal_window(groups, first, last, start, end):
    # Open and close in the adjacent rests. Leave room for the actual breath
    # and release even if the millisecond score ends on the last voiced frame.
    a, b = groups[first][0], groups[last][1]
    previous = groups[first - 1][1] if first else start
    following = groups[last + 1][0] if last + 1 < len(groups) else end
    return max(start, a - .35, (previous + a) / 2), min(end, b + .35, (following + b) / 2)


def opening_on_grid(groups, edges, first, count, preferred):
    """Find a whole opening on a legacy edition with no authored teaser."""
    available = [(a, b) for a, b in groups if edges[0] <= a and b <= edges[-1]]
    if not available:
        return first, count, None
    a, b = next((span for span in available if span[1] > preferred), available[0])
    left = bisect_right(edges, a) - 1
    right = bisect_left(edges, b)
    # Include at least a beat of accompaniment after the phrase when available.
    beat = (edges[right] - edges[right - 1]) / 4
    stop = min(len(edges) - 1, bisect_left(edges, b + beat))
    count = max(count, stop - left)
    first = min(left, max(first, stop - count), len(edges) - 1 - count)
    return first, count, a


@dataclass(frozen=True)
class Loop:
    label: str
    source: str
    bars: int = 0
    first_bar: int = 0
    fragment: bool = False
    main: bool = False
    start_ms: int | None = None

    def region(self, song, duration):
        start = song["start"]
        end = min(song["end"] or duration, duration) if song["loop"] else duration
        if self.start_ms is not None and start <= self.start_ms / 1000 < end:
            start = self.start_ms / 1000
        region = dict(song)
        authored = song.get("teaserWindowMs") if self.fragment else None
        groups = phrase_groups(song) if self.main or (self.fragment and not authored) else []
        target = None
        if self.bars or self.fragment or self.main:
            offset, edges = bar_edges(song, start, end)
            if len(edges) >= 2:
                count = min(self.bars or len(edges) - 1, len(edges) - 1)
                first = max(0, min(self.first_bar - offset, len(edges) - 1 - count))
                if self.fragment and groups and self.bars:
                    preferred = min(first + song.get("teaserVoiceBar", 0), len(edges) - 2)
                    first, count, target = opening_on_grid(groups, edges, first, count, edges[preferred])
                start, end = edges[first], edges[first + count]
        if start > 0 and not song.get("downbeats"):
            region["album_bar_offset"] = (start-song.get("beat_zero", 0))*song["bpm"]/240
            region["beat_zero"] = start
        start = round(start * 48000) / 48000
        end = round(end * 48000) / 48000
        if authored:
            region["vocal_window"] = max(start, authored[0] / 1000), min(end, authored[1] / 1000)
        elif groups:
            inside = [i for i, (a, b) in enumerate(groups) if start <= a and b <= end]
            if inside:
                first, last = inside[0], inside[-1]
                if self.fragment:
                    if target is not None:
                        first = min(inside, key=lambda i: abs(groups[i][0] - target))
                    last = first
                crosses = any(a < edge < b for a, b in groups for edge in (start, end))
                if self.fragment or crosses:
                    region["vocal_window"] = vocal_window(groups, first, last, start, end)
            else:
                region["vocal_window"] = (start, start)
        return dict(region, start=start, end=end, loop=True)


def family_key(song):
    key = song["id"]
    if song.get("loopFamily"):
        return song["loopFamily"]
    if song.get("cuts"):
        base, sep, _ = key.rpartition("/")
        return base + sep + "full"
    return song.get("title", key)


def backing_loops(songs):
    """Include all assignable editions, including the single-library chant stems."""
    groups = {}
    for s in songs.values():
        groups.setdefault(family_key(s), []).append(s)
    result = {}
    for key, entries in groups.items():
        full = songs.get(key, entries[0])
        key = full["id"]
        cuts = [s for s in entries if s["id"] != key]
        first = full.get("loopBar", 0)
        choices = {
            "teaser": Loop("Teaser", key, full.get("teaserBars", 4),
                           full.get("teaserBar", first), fragment=True),
            "main": Loop("Main", key, full.get("mainBars", 16), first, main=True),
            "full": Loop("Full", key, start_ms=full.get("fullStartMs")),
        }
        if full.get("cuts") and cuts:
            chorus = next((s for s in cuts if s.get("loopRole") != "teaser"), None)
            if chorus:
                choices["main"] = Loop("Main", chorus["id"], main=True)
            # Earlier imports have no dedicated teaser. Find their opening line
            # on the intact recording, including any pickup before the chorus.
            choices["teaser"] = Loop("Teaser", key, 4, fragment=True)
        for extra in full.get("loops", []):
            choices[extra["id"]] = Loop(
                extra["label"], key, extra["bars"], extra["firstBar"]
            )
        for s in cuts:
            if s.get("cuts") and s.get("loopRole") == "teaser":
                choices["teaser"] = choices[s["id"]] = Loop("Teaser", s["id"], fragment=True)
                continue
            if role := s.get("loopRole"):
                if role == "teaser":
                    # Extend the existing hook entrance to four bars on the full master.
                    choices[role] = Loop(
                        "Teaser",
                        key,
                        4,
                        round((s["start"] - full["beat_zero"]) * full["bpm"] / 240),
                        fragment=True,
                    )
                else:
                    choices[role] = Loop(role.title(), s["id"], main=role == "main")
                    if role == "main" and s.get("teaserWindowMs"):
                        choices["teaser"] = Loop("Teaser", s["id"], s.get("teaserBars", 0),
                                                 fragment=True)
            choices[s["id"]] = Loop(
                s.get("part") or s["label"].split(" · ", 1)[-1], s["id"]
            )
        result[key] = choices
    return result


def launch_boundary(backing, frame, quantum):
    """Return the next breakpoint on the outgoing clip clock, in audio frames."""
    rate = 48000
    length = round((backing["end"] - backing["start"]) * rate)
    lap, offset = divmod(frame, length)
    if quantum == "cycle":
        return (lap + 1) * length
    bars = float(quantum)
    now = backing["start"] + offset / rate
    grid = backing.get("downbeats", [])
    if len(grid) > 1 and grid[0] <= now < grid[-1]:
        i = bisect_right(grid, now) - 1
        phase = i + (now - grid[i]) / (grid[i + 1] - grid[i])
        mark = (math.floor(phase / bars + 1e-8) + 1) * bars
        j = int(mark)
        when = (
            grid[j] + (mark - j) * (grid[j + 1] - grid[j])
            if j + 1 < len(grid)
            else grid[-1] + (mark - len(grid) + 1) * 240 / backing["bpm"]
        )
    else:
        origin = grid[-1] if grid and now >= grid[-1] else backing.get("beat_zero", 0)
        if backing.get("performance") and backing["performance"]["loop"]:
            origin = backing["start"]
        step = bars * 240 / backing["bpm"]
        when = origin + (math.floor((now - origin) / step + 1e-8) + 1) * step
    delta = round((when - backing["start"]) * rate)
    return lap * length + min(length, max(offset + 1, delta))
