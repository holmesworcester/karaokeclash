"""Place a looping source on the desk's continuous musical clock."""
import bisect
from dataclasses import dataclass
import math


def grid_position(song, value, inverse=False):
    """Convert between fractional bars and recording seconds on the runtime's grid."""
    grid = song.get("downbeats")
    if grid and len(grid) >= 2:
        i = math.floor(value) if inverse else bisect.bisect_right(grid, value) - 1
        i = max(0, min(len(grid) - 2, i))
        length = grid[i + 1] - grid[i]
        return grid[i] + (value - i) * length if inverse else i + (value - grid[i]) / length
    origin, bar = song.get("beat_zero", 0), 240 / song.get("bpm", 120)
    return origin + value * bar if inverse else (value - origin) / bar


def launch_origin(choice, bar):
    """Teaser/Main share bar phase even when launched halfway through a bar."""
    if choice not in ("teaser", "main"):
        return bar
    if abs(bar - round(bar)) < .00005:
        return round(bar)
    return math.floor(bar)


@dataclass
class ClipClock:
    region: dict
    origin: float = 0.0  # global bar at the beginning of this pass

    def convert(self, value, inverse=False):
        r = self.region
        first = grid_position(r, r["start"])
        bars = grid_position(r, r["end"]) - first
        # A sample-rounded four-bar clip still occupies exactly four musical bars.
        if abs(bars - round(bars)) < 0.00005 and round(bars) > 0:
            bars = round(bars)
        length = r["end"] - r["start"]
        if inverse:
            laps, phase = divmod(value - self.origin, bars)
            return laps * length + grid_position(r, first + phase, inverse=True) - r["start"]
        laps, phase = divmod(value, length)
        return self.origin + laps * bars + grid_position(r, r["start"] + phase) - first

    def bar(self, seconds):
        return self.convert(seconds)

    def seconds(self, bar):
        return self.convert(bar, inverse=True)

    def next_edge(self, elapsed, interval):
        if interval == "cycle":
            length = self.region["end"] - self.region["start"]
            return (math.floor(elapsed / length) + 1) * length
        step = float(interval)
        return self.seconds((math.floor(self.bar(elapsed) / step + 1e-9) + 1) * step)
