"""A plain, fitted Pango lyric layout shared by every audience quadrant."""

from clip_clock import ClipClock, grid_position
import bisect
import heapq
import math
from typing import NamedTuple
import cairo
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gtk, Gdk, Pango, PangoCairo

from backing_loops import family_key
from library import lyric_position


# The same 7-point bands identify the microphones on the Mac control and audience windows.
MIC_NAMES = ("Red", "Blue", "Green", "Gold")
MIC_COLORS = (0xB65C49, 0x477BA3, 0x4F8564, 0xAD7D30)
PANE_BACKGROUNDS = (0x22100F, 0x0D1924, 0x0D1D14, 0x211908)
LYRIC_READY = 0x909090
LYRIC_SWEPT = 0xFFFFFF


def rgb(color):
    return ((color >> 16) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255)


def mic_panel(index, content):
    panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    panel.get_accessible().set_name(MIC_NAMES[index])
    style = Gtk.CssProvider()
    style.load_from_data(f"box {{ border-top: 7px solid #{MIC_COLORS[index]:06x}; }}".encode())
    panel.get_style_context().add_provider(style, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    panel.pack_start(content, True, True, 0)
    return panel


def fit(context, words, width, height, size):
    layout = PangoCairo.create_layout(context)
    layout.set_width(max(1, int(width)) * Pango.SCALE)
    layout.set_wrap(Pango.WrapMode.WORD_CHAR)
    layout.set_alignment(Pango.Alignment.CENTER)
    layout.set_text(" ".join(words), -1)
    font = Pango.FontDescription.from_string("Inter, sans-serif Semi-Bold")
    lo, hi = 1, size
    while lo < hi:
        mid = (lo + hi + 1) // 2
        font.set_absolute_size(mid * Pango.SCALE)
        layout.set_font_description(font)
        w, h = layout.get_pixel_size()
        if w <= width and h <= height:
            lo = mid
        else:
            hi = mid - 1
    font.set_absolute_size(lo * Pango.SCALE)
    layout.set_font_description(font)
    return layout


def word_boxes(layout, words):
    """Collect visual runs in singing order, including words split by wrapping."""
    offset = 0
    result = []
    for word in words:
        runs = []
        for char in word:
            rect = layout.index_to_pos(offset)
            offset += len(char.encode())
            x, y, w, h = (n / Pango.SCALE for n in (rect.x, rect.y, rect.width, rect.height))
            if runs and runs[-1][1] == y and runs[-1][0] + runs[-1][2] == x and runs[-1][2] * w >= 0:
                runs[-1][2] += w
            else:
                runs.append([x, y, w, h])
        result.append(runs)
        offset += 1
    return result


def bar_at(song, seconds):
    """Use the recording's measured downbeats, extending its last interval if needed."""
    grid = song.get("downbeats")
    if grid and len(grid) >= 2:
        index = max(0, min(len(grid) - 2, bisect.bisect_right(grid, seconds) - 1))
        return max(0, index + math.floor((seconds - grid[index]) / (grid[index + 1] - grid[index])))
    length = 60 * song.get("meter", 4) / song["bpm"]
    return max(0, math.floor((seconds - song.get("beat_zero", 0)) / length))


READING_LINES = 4


class Phrase(NamedTuple):
    at: float
    lane: int
    role: str
    words: tuple
    spans: tuple
    number: int
    until: float


def loop_position(backing, elapsed):
    length = backing["end"] - backing["start"]
    if not backing["loop"] or length <= 0:
        return 0, elapsed
    # Both the native transport and its loop boundaries count audio frames.
    lap, frame = divmod(round(elapsed * 48000), max(1, round(length * 48000)))
    return lap, frame / 48000


def transport_bar(backing, elapsed):
    start, end = backing["start"], backing["end"]
    rounds, elapsed = loop_position(backing, elapsed)
    first = bar_at(backing, start)
    count = bar_at(backing, end - 0.5 / 48000) - first + 1
    return backing.get("turn_offset", 0) + rounds * count + bar_at(backing, start + elapsed) - first


def draw_section(ctx, caption, width, height):
    layout = fit(ctx, [caption], width - 8, height, min(24, height))
    _, h = layout.get_pixel_size()
    ctx.move_to(4, (height - h) / 2)
    PangoCairo.layout_path(ctx, layout)
    ctx.set_line_width(2)
    ctx.set_source_rgb(0.12, 0.12, 0.12)
    ctx.stroke_preserve()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()


class Pane(Gtk.DrawingArea):
    def __init__(self, desk, index):
        super().__init__()
        self.desk, self.index = desk, index
        self.type_scale = None
        self.cache = {}
        self.sheets = {}
        self.timelines = {}
        self.reading_context = None
        self.rotation = 0
        self.queued_layout = None
        self.connect("draw", self.draw_pane)

    def song_for(self, backing, pair, fallback=None):
        key = pair["lyrics"] if pair else self.desk.strips[self.index]["lyrics"]
        song = self.desk.songs.get(key, self.desk.songs.get(fallback))
        if song is None:
            return backing if pair else None
        if backing and song.get("id") and backing.get("id") and song["id"] != backing["id"]:
            family = family_key(song)
            if family == family_key(backing):
                # Teaser and Main may be separate recordings with different lyrics
                # and clocks. Follow the transport between the family's primary cuts.
                modes = getattr(self.desk, "loops", {}).get(family, {})
                primary = {loop.source for mode, loop in modes.items() if mode in ("teaser", "main", "full")}
                if song["id"] == family or {song["id"], backing["id"]} <= primary:
                    return self.desk.songs.get(backing["id"], backing)
        return song

    def lanes(self, song, paired):
        key = id(song), paired
        if key in self.sheets:
            return self.sheets[key][1]
        lanes = []
        for kind, rows in song.get("rows", {}).items():
            eligible = sorted((r for r in rows if r[0] and r[1] and (
                not song.get("performance") and not song.get("cuts")
                or song["start"] * 1000 <= r[1][0][0] < song["end"] * 1000
            )), key=lambda r: r[1][0][0])
            by_bar = {}
            if paired:
                for words, spans in eligible:
                    groups = {}
                    for word, span in zip(words, spans):
                        groups.setdefault(bar_at(song, span[0] / 1000), []).append((word, span))
                    for number, group in groups.items():
                        by_bar.setdefault(number, []).append(tuple(zip(*group)))
            lanes.append((kind, eligible, [r[1][0][0] for r in eligible], by_bar))
        if len(self.sheets) >= 4:
            self.sheets.clear()
        self.sheets[key] = song, lanes
        return lanes

    def lines_at(self, song, backing, elapsed, pair, lead_ms=1500):
        """Return each lane's reachable line and its word clock on this recording."""
        lanes = self.lanes(song, bool(pair))
        window = backing.get("vocal_window")
        start, end = backing["start"], backing["end"]
        _, local = loop_position(backing, elapsed)
        source_now = start + local
        same_edition = song is backing or (song.get("id") and song["id"] == backing.get("id"))
        if same_edition:
            seconds = source_now
        elif backing["loop"] and backing["track"] != song["track"] and not backing.get("downbeats"):
            # Pass an absolute recording clock into the runtime beat mapper. This
            # also works with older release runtimes that take elapsed literally.
            seconds = lyric_position(song, dict(backing, loop=False), source_now)
        else:
            seconds = lyric_position(song, backing, local)
        ms = 1000 * seconds
        singing = not pair or transport_bar(backing, elapsed) % 2 == int(self.index != pair["first"])
        turn = ("Sing" if singing else "Next bar") if pair else ""
        if window and not window[0] <= source_now < window[1]:
            return [], turn
        row_ms = ms
        if pair:
            wanted_bar = bar_at(song, ms / 1000) + int(not singing)
            clock = backing if backing["track"] == song["track"] else song
            if (not singing and clock["loop"] and clock["end"] > clock["start"]
                    and wanted_bar > bar_at(song, clock["end"] - 0.5 / 48000)):
                wanted_bar = bar_at(song, clock["start"])
                row_ms = clock["start"] * 1000

        reachable = None
        if (backing["loop"] and end > start) or (pair and singing):
            # A pair may split this source line on the backing's bar boundary.
            # Sweep only the words this singer can reach on the current pass.
            reachable = self.timeline(song, backing, pair if singing else None, elapsed - local)[2]
        visible = []
        for slot, (kind, rows, starts, by_bar) in enumerate(lanes):
            if pair:
                rows = by_bar.get(wanted_bar, [])
                starts = [spans[0][0] for _, spans in rows]
            i = bisect.bisect_right(starts, row_ms) - 1
            if i < 0 or row_ms > rows[i][1][-1][1] + 160:
                i += 1
            if i >= len(rows):
                continue
            words, spans = rows[i]
            if reachable is not None:
                fragment = [(w, span) for w, span in zip(words, spans)
                            if (slot, w, *span) in reachable]
                if not fragment:
                    continue
                words, spans = zip(*fragment)
            if window:
                # Both edges are on the backing clock. Convert their nearby
                # offsets to the assigned lyric edition's tempo before clipping.
                rate = 1 if song["track"] == backing["track"] else backing["bpm"] / song["bpm"]
                lo = ms + (window[0] - source_now) * 1000 * rate
                hi = ms + (window[1] - source_now) * 1000 * rate
                fragment = [(w, (a, min(b, hi))) for w, (a, b) in zip(words, spans)
                            if lo <= a < hi]
                if not fragment:
                    continue
                words, spans = zip(*fragment)
            preview = pair and not singing
            if ((preview or spans[0][0] - lead_ms <= row_ms)
                    and row_ms <= spans[-1][1] + 160):
                visible.append((slot, kind, words, spans, -math.inf if preview else row_ms))
        if not lanes and singing:
            cue = [x for x in song.get("cue", []) if x]
            if cue:
                visible.append((0, "", cue, [], row_ms))
        return visible, turn

    def entrance(self, song, backing, elapsed, pair, horizon):
        """Read the first reachable phrase from each lane's mapped word timings."""
        length = backing["end"] - backing["start"]
        looping = backing["loop"] and length > 0
        lap, local = loop_position(backing, elapsed)
        offset = lap * length
        found = {}
        for _ in range((3 if pair else 2) if looping else 1):
            _, lanes, _ = self.timeline(song, backing, pair, offset)
            for slot, entries in enumerate(lanes):
                if slot in found:
                    continue
                for phrase in entries:
                    if offset + phrase.at > elapsed + horizon:
                        break
                    if phrase.until > local:
                        found[slot] = (slot, phrase.role, phrase.words, phrase.spans, -math.inf)
                        break
            offset += length
            local = 0
            if offset > elapsed + horizon or len(found) == len(lanes):
                break
        return list(found.values())

    def timeline(self, song, backing, pair, offset):
        """Index sung intervals and previewable lines on one pass of the backing."""
        parity = transport_bar(backing, offset) % 2 if pair else 0
        key = id(song), id(backing), pair["first"] if pair else None, parity
        if key in self.timelines:
            return self.timelines[key][2]
        start, end = backing["start"], backing["end"]
        lo, hi = backing.get("vocal_window", (start, end if end > start else math.inf))
        lo, hi = max(start, lo), min(end, hi) if end > start else hi
        same = song is backing or (song.get("id") and song["id"] == backing.get("id"))
        cross = song["track"] != backing["track"]
        period = 0
        if not same and (cross or song["loop"]):
            period = song.get("cycle") or (song["end"] - song["start"] if song["loop"] else 0)

        # Invert lyric_position at the word endpoints. Downbeat interpolation is
        # piecewise linear, so a single BPM ratio would drift on measured grids.
        clock_start = start if cross and not backing["loop"] and not backing.get("downbeats") else 0
        origin = song["start"] if cross and not song.get("cycle") else 0
        if not cross and not same and song["loop"] and song.get("cycle"):
            origin = -song["start"]

        def backing_time(t):
            if not cross:
                return t - origin
            return grid_position(backing, grid_position(song, t - origin), inverse=True) + clock_start

        def word_time(t):
            if not cross or not math.isfinite(t):
                return t + origin
            return grid_position(song, grid_position(backing, t - clock_start), inverse=True) + origin

        first, last = word_time(lo), word_time(hi)
        ranges = []
        entries = []
        lanes = self.lanes(song, bool(pair))
        attacks = set()
        for slot, (role, rows, _, by_bar) in enumerate(lanes):
            if pair:
                rows = [r for group in by_bar.values() for r in group]
            for words, spans in rows:
                fragments = {}
                for word, span in zip(words, spans):
                    a, b = span
                    a, b = a / 1000, b / 1000
                    passes = range(1)
                    if period > 0 and math.isfinite(last):
                        base = 0 if song.get("cycle") else song["start"]
                        a, b = max(base, a), min(base + period, b)
                        passes = range(math.floor((first - b) / period) + 1,
                                       math.ceil((last - a) / period))
                    for n in passes:
                        left = backing_time(a + n * period)
                        if left < lo - 1e-8:
                            continue
                        right = min(hi, backing_time(b + n * period))
                        if left >= right:
                            continue
                        if pair:
                            # Work on this pass's recording clock. Adding a lap
                            # to an exact downbeat can round into the preceding bar.
                            turn = (parity + bar_at(backing, left) - bar_at(backing, start)) % 2
                            if turn != int(self.index != pair["first"]):
                                continue
                        attacks.add((slot, word, *span))
                        ranges.append((left - start, right - start))
                        fragments.setdefault(n, []).append((word, span, left - start, right - start))
                for fragment in fragments.values():
                    tokens, intervals, begins, ends = zip(*fragment)
                    entries.append((min(begins), slot, role, tokens, intervals, max(ends)))
        phrases = [[] for _ in lanes]
        number = 0
        releases = [-math.inf] * READING_LINES
        for at, lane, role, words, spans, until in sorted(entries, key=lambda r: r[0]):
            # A harmony can still be singing when the lead comes around again.
            # Leave its baseline occupied and take the next available one.
            for candidate in range(number, number + READING_LINES):
                if releases[candidate % READING_LINES] <= at:
                    number = candidate
                    break
            phrases[lane].append(Phrase(at, lane, role, words, spans, number, until))
            releases[number % READING_LINES] = until
            number += 1
        merged = []
        for a, b in sorted(ranges):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        if len(self.timelines) >= 8:
            self.timelines.clear()
        self.timelines[key] = song, backing, (merged, phrases, attacks)
        return merged, phrases, attacks

    def vocal_ranges(self, song, backing, pair, offset):
        return self.timeline(song, backing, pair, offset)[0]

    def phrase_pass(self, song, backing, pair, lap):
        """Return the timed phrases and the first baseline number of this pass."""
        length = backing["end"] - backing["start"]
        lanes = self.timeline(song, backing, pair, lap * length)[1]
        if not lap:
            return lanes, 0
        def width(lanes):
            return max((lane[-1].number + 1 for lane in lanes if lane), default=0)

        count = width(lanes)
        if pair and transport_bar(backing, length) % 2 != transport_bar(backing, 0) % 2:
            # A paired singer gets the opposite bars on every other lap.
            other = width(self.timeline(song, backing, pair, (lap + 1) * length)[1])
            period = count + other
            return lanes, (lap // 2 * period if period > READING_LINES else 0) + (other if lap % 2 else 0)
        # A short refrain fits on the screen. Its lines can stay there on repeat.
        return lanes, lap * count if count > READING_LINES else 0

    def reading_window(self, song, backing, elapsed, pair, visible, pending):
        """Keep each phrase on its numbered baseline from preview through release."""
        def key(lane, words, spans):
            return lane, spans[0][0], tuple(words)

        current = {key(row[0], row[2], row[3]): row for row in visible if row[3]}

        def lane_entries(lane, local, first_pass, incoming):
            i = bisect.bisect_left(lane, local, key=lambda p: p.at)
            if first_pass and i:
                prev = lane[i - 1]
                live = current.get(key(prev.lane, prev.words, prev.spans))
                if incoming:
                    keep = prev.until > local
                else:
                    keep = live is not None and live[4] >= prev.spans[0][0]
                if keep:
                    i -= 1
            for j in range(i, len(lane)):
                phrase = lane[j]
                row = (phrase.lane, phrase.role, phrase.words, phrase.spans, -math.inf)
                if first_pass and not incoming and j == i:
                    live = current.get(key(phrase.lane, phrase.words, phrase.spans))
                    # Repetitions on the same backing pass can share all source
                    # timestamps. Only the current occurrence gets the live sweep.
                    if live is not None and (phrase.at <= local or live[4] < phrase.spans[0][0]):
                        row = live
                yield phrase, row

        def candidates(part, region, now, stop, incoming):
            length = region["end"] - region["start"]
            looping = region["loop"] and length > 0
            lap, local = loop_position(region, now)
            for step in range((3 if pair else 2) if looping else 1):
                lanes, first = self.phrase_pass(part, region, pair, lap + step)
                streams = (lane_entries(lane, local, step == 0, incoming) for lane in lanes)
                for phrase, row in heapq.merge(*streams, key=lambda item: item[0].number):
                    if (lap + step) * length + phrase.at >= stop:
                        return
                    yield first + phrase.number, row
                local = 0

        context = id(song), id(backing), pair["first"] if pair else None
        if context != self.reading_context:
            self.rotation = (self.queued_layout[1] if self.queued_layout
                             and self.queued_layout[0] == context else 0)
            self.reading_context = context
        self.queued_layout = None
        placed = {self.rotation: row for row in visible if not row[3]}
        stop = pending["boundary"] if pending else math.inf
        for number, row in candidates(song, backing, elapsed, stop, False):
            slot = (self.rotation + number) % READING_LINES
            if slot in placed:
                previous = placed[slot]
                # A released harmony gets a short display hold, ending as soon
                # as another lane needs the same baseline for an actual attack.
                if not (previous[3] and previous[4] >= previous[3][-1][1]
                        and row[3][0][0] <= row[4] < row[3][-1][1]):
                    break
                del placed[slot]
            placed[slot] = row

        if pending:
            target = pending["region"]
            part = self.song_for(target, pair)
            if part:
                # Continue the same cycle at a queued launch, even when it seeks
                # into the middle of a phrase. Keep that phase when the cut lands.
                lap, local = loop_position(backing, stop)
                if lap and local == 0:
                    lap, local = lap - 1, backing["end"] - backing["start"]
                lanes, before = self.phrase_pass(song, backing, pair, lap)
                preceding = []
                for lane in lanes:
                    i = bisect.bisect_left(lane, local, key=lambda p: p.at)
                    if i:
                        preceding.append(lane[i - 1].number + 1)
                before += max(preceding, default=int(any(not row[3] for row in visible)))
                coming = candidates(part, target, pending.get("offset", 0), math.inf, True)
                for i, (number, row) in enumerate(coming):
                    if i == 0:
                        shift = (self.rotation + before - number) % READING_LINES
                        self.queued_layout = ((id(part), id(target), context[2]), shift)
                    slot = (shift + number) % READING_LINES
                    if slot in placed:
                        break
                    placed[slot] = row
        return placed.items()

    def section_at(self, song, backing, pair, elapsed, pending):
        length = backing["end"] - backing["start"]
        looping = backing["loop"] and length > 0
        lap, local = loop_position(backing, elapsed)
        offset = lap * length
        ranges = self.vocal_ranges(song, backing, pair, offset)
        i = bisect.bisect_right(ranges, (local, math.inf))
        previous = ranges[i - 1][1] if i else None
        if previous is not None and local < previous:
            return None
        entry = ranges[i][0] if i < len(ranges) else None
        section = "Intro" if previous is None else "Break" if entry is not None else "Outro"
        if entry is None and looping:
            # An odd-length loop swaps the two paired singers on alternate passes.
            for n in range(1, 3 if pair else 2):
                following = self.vocal_ranges(song, backing, pair, offset + n * length)
                if following:
                    entry = n * length + following[0][0]
                    break
        if pending:
            boundary = pending["boundary"] - offset
            if entry is None or entry >= boundary:
                target = pending["region"]
                incoming = self.song_for(target, pair)
                at = pending.get("offset", 0)
                lap, position = loop_position(target, at)
                duration = target["end"] - target["start"]
                following = self.vocal_ranges(incoming, target, pair, lap * duration) if incoming else []
                arrival = next((lap * duration + max(position, a) for a, b in following
                                if b > position), None)
                if arrival is None and incoming and target["loop"] and duration > 0:
                    for step in range(1, 3 if pair else 2):
                        after = (lap + step) * duration
                        following = self.vocal_ranges(incoming, target, pair, after)
                        if following:
                            arrival = after + following[0][0]
                            break
                entry = None
                if arrival is not None:
                    out_clock, in_clock = ClipClock(backing), ClipClock(target)
                    bars = in_clock.bar(arrival) - in_clock.bar(at)
                    entry = out_clock.seconds(out_clock.bar(pending["boundary"]) + bars) - offset
        if entry is None and not ranges:
            return None
        beat = 60 / backing.get("bpm", 120)
        if section == "Break" and entry is not None and entry - previous < max(2, 4 * beat):
            return None
        # Quantize to the native clock before rounding the displayed seconds.
        wait = max(0, round((entry - local) * 48000)) / 48000 if entry is not None else None
        return section, wait

    def draw_pane(self, widget, ctx):
        width, height = self.get_allocated_width(), self.get_allocated_height()
        ctx.set_source_rgb(*rgb(PANE_BACKGROUNDS[self.index]))
        ctx.paint()
        desk = self.desk
        pair = getattr(desk, "bar_pairs", {}).get(self.index)
        # The native mixer publishes the launch at an exact audio frame.
        if getattr(desk, "pending", None):
            desk.sync_launch()
        backing, elapsed, pending = (desk.pane_transport(self.index) if hasattr(desk, "pane_transport")
                                     else (desk.backing, desk.elapsed(), getattr(desk, "pending", None)))
        song = self.song_for(backing, pair)
        if song is None:
            self.get_accessible().set_name(MIC_NAMES[self.index])
            return False
        backing = backing or song
        length = backing["end"] - backing["start"]
        bar = 240 / backing.get("bpm", 120)
        edge = math.inf
        if backing["loop"] and length > 0:
            edge = length - elapsed % length
        target = backing
        if pending and pending["boundary"] - elapsed <= edge + 0.5 / 48000:
            edge = max(0, pending["boundary"] - elapsed)
            target = pending["region"]
        section = self.section_at(song, backing, pair, elapsed, pending)
        lead = 1500
        if section and section[1] is not None:
            lead = max(lead, 1000 * section[1] + 1)
        visible, turn = self.lines_at(song, backing, elapsed, pair, lead_ms=min(lead, edge * 1000))
        if pair and turn == "Next bar" and pending and target is pending["region"] and edge <= bar:
            incoming_song = self.song_for(target, pair)
            if incoming_song:
                pickup = max(0, target.get("vocal_window", (target["start"],))[0] - target["start"])
                warning = 480 / target.get("bpm", 120)
                if self.entrance(incoming_song, target, pending.get("offset", 0), pair, pickup + warning):
                    # The queued entrance supersedes the partner's ordinary next-bar cue.
                    visible = []
        rows = self.reading_window(song, backing, elapsed, pair, visible, pending)
        if pair:
            ctx.set_source_rgb(1, 1, 1)
            ctx.set_font_size(16)
            ctx.move_to(12, 21)
            ctx.show_text(turn)
        description = [MIC_NAMES[self.index], song["label"]]
        if section:
            label, wait = section
            if wait is not None and hasattr(desk, "tempo") and length > 0:
                clock = ClipClock(backing)
                bars = clock.bar(elapsed + wait) - clock.bar(elapsed)
                wait = round(bars * 240 / desk.tempo.get_value() * 48000) / 48000
            seconds = math.ceil(wait) if wait is not None else None
            caption = label if seconds is None else f"{label} · {seconds // 60}:{seconds % 60:02}"
            description.append(label if wait is None else f"{label} · Vocals in {math.ceil(wait)} s")
        if pair:
            description.append(turn)
        for _, row in rows:
            _, role, words, _, _ = row
            phrase = " ".join(words)
            description.append(f"{role} {phrase}".strip() if row in visible else f"Up next · {phrase}")
        accessible = self.get_accessible()
        description = " · ".join(description)
        if accessible.get_name() != description:
            accessible.set_name(description)
        header = max(1, min(40, height // 6))
        if section:
            draw_section(ctx, caption, width, header)
        capacity = READING_LINES
        available = max(1, height - header - 8)
        # Reserve the same four baselines through the count-in and the song.
        line_height = max(1, available * 0.90 / capacity)
        first_y = header + (available - capacity * line_height) / 2
        font_size = max(1, round(min(width / 12, line_height * 0.82)))
        # Choose type from the assigned lyric sheet once per pane size. A new
        # phrase or another trip around the loop must never reflow its neighbors.
        key = pair["lyrics"] if pair else desk.strips[self.index]["lyrics"]
        sheet = desk.songs.get(key, song)
        dimensions = width, line_height, font_size
        if not self.type_scale or self.type_scale[0] is not sheet or self.type_scale[1] != dimensions:
            shared_size = font_size
            for _, phrases, _, _ in self.lanes(sheet, False):
                for words, _ in phrases:
                    layout = fit(ctx, words, width - 28, line_height, shared_size)
                    shared_size = layout.get_font_description().get_size() // Pango.SCALE
            self.type_scale = sheet, dimensions, shared_size
        shared_size = self.type_scale[2]
        for slot, row in rows:
            _, _, words, spans, row_ms = row
            top = first_y + slot * line_height
            key = (tuple(words), width, line_height, shared_size)
            if key not in self.cache:
                if len(self.cache) >= 32:
                    self.cache.clear()
                layout = fit(ctx, words, width - 28, line_height, shared_size)
                self.cache[key] = layout, word_boxes(layout, words)
            layout, boxes = self.cache[key]
            _, h = layout.get_pixel_size()
            y = top + (line_height - h) / 2
            ctx.push_group()
            ctx.set_source_rgb(*rgb(LYRIC_READY))
            ctx.paint()
            for runs, (start, end) in zip(boxes, spans):
                if row_ms < start:
                    continue
                fraction = 1 if row_ms >= end else (row_ms - start) / (end - start)
                remaining = fraction * sum(abs(w) for _, _, w, _ in runs)
                for x, top, w, height in runs:
                    filled = min(remaining, abs(w))
                    ctx.rectangle(14 + x, y + top, filled if w >= 0 else -filled, height)
                    remaining -= filled
            ctx.set_source_rgb(*rgb(LYRIC_SWEPT))
            ctx.fill()
            fill = ctx.pop_group()
            ctx.move_to(14, y)
            PangoCairo.layout_path(ctx, layout)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.set_line_width(max(2, layout.get_font_description().get_size() / Pango.SCALE * 0.11))
            ctx.set_source_rgb(0, 0, 0)
            ctx.stroke_preserve()
            ctx.set_source(fill)
            ctx.fill()
        return False


class Audience(Gtk.Window):
    def __init__(self, desk):
        super().__init__(title="Audience · Karaoke Playground 4")
        self.set_default_size(1100, 730)
        grid = Gtk.Grid(
            column_homogeneous=True,
            row_homogeneous=True,
            column_spacing=2,
            row_spacing=2,
        )
        background = Gtk.CssProvider()
        background.load_from_data(b"grid { background-color: #525c54; }")
        grid.get_style_context().add_provider(
            background, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self.add(grid)
        self.panes = []
        self.full = False
        for i in range(4):
            pane = Pane(desk, i)
            pane.set_hexpand(True)
            pane.set_vexpand(True)
            grid.attach(mic_panel(i, pane), i % 2, i // 2, 1, 1)
            self.panes.append(pane)
        self.connect("delete-event", lambda *_: self.hide() or True)
        self.connect("key-press-event", self.key)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.connect(
            "button-press-event",
            lambda _, e: self.toggle()
            if e.type == Gdk.EventType.DOUBLE_BUTTON_PRESS
            else False,
        )

    def toggle(self):
        self.full = not self.full
        if self.full:
            self.fullscreen()
        else:
            self.unfullscreen()
        return True

    def key(self, _, event):
        if event.keyval == Gdk.KEY_F11 or (
            self.full and event.keyval == Gdk.KEY_Escape
        ):
            return self.toggle()
        return False
