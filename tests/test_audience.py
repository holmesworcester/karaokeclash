"""Render real audience panes using the installed app's lyric clock."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cairo
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from library import catalog
from backing_loops import Loop, backing_loops, calibrated_catalog
from audience import Gtk, PangoCairo, PANE_BACKGROUNDS, Pane, transport_bar


class AudienceRendering(unittest.TestCase):
    def test_four_lines_use_pane_height_and_long_phrases_fit(self):
        phrases = ["Buy it use it break it fix it", "Trash it change it",
                   "Mail upgrade it", "Charge it point it zoom it press it snap it work it"]
        song = dict(track="song", start=0, end=8, loop=False, label="Test", bpm=126,
                    rows={"lead": [(phrase.split(), [(1000*i + 40*j, 1000*i + 40*j + 38)
                              for j in range(len(phrase.split()))]) for i, phrase in enumerate(phrases)]})
        desk = SimpleNamespace(songs={0: song}, backing=song, strips=[{"lyrics": 0}]*4,
                               elapsed=lambda: 0.14)
        for width, height in [(959, 532), (480, 240)]:
            window = Gtk.OffscreenWindow()
            pane = Pane(desk, 0)
            pane.set_size_request(width, height)
            window.add(pane)
            window.show_all()
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            with patch("audience.PangoCairo.layout_path", wraps=PangoCairo.layout_path) as draw:
                pane.draw(cairo.Context(surface))
            sizes = [args[1].get_font_description().get_size() for args, _ in draw.call_args_list]
            self.assertEqual(len(sizes), 4)
            self.assertEqual(len(set(sizes)), 1)
            pixels = np.frombuffer(surface.get_data(), np.uint8).reshape(height, width, 4)
            y, x = np.where((pixels != pixels[0, 0]).any(axis=2))
            self.assertGreater(np.ptp(y), (height - min(40, height // 6)) * .75)
            self.assertGreater(np.ptp(x), width * .75)
            self.assertGreater(x.min(), 4)
            self.assertLess(x.max(), width - 4)
            self.assertGreaterEqual(len(pane.cache), 4)
            for (_, _, slot_height, _), (layout, _) in pane.cache.items():
                self.assertLessEqual(layout.get_pixel_size()[1], slot_height)
            window.destroy()

    def test_centered_outlined_words_and_timed_fill_on_each_microphone(self):
        song = dict(track="song", start=0, end=4, loop=False, label="Test",
                    rows={"lead": [(["HÉLLO", "WORLD"], [(1000, 1500), (1600, 2200)])]})
        desk = SimpleNamespace(songs={0: song}, backing=song, strips=[{"lyrics": 0}] * 4)
        for index, color in enumerate(PANE_BACKGROUNDS):
            window = Gtk.OffscreenWindow()
            pane = Pane(desk, index)
            pane.set_size_request(480, 240)
            window.add(pane)
            window.show_all()
            background = np.array([color & 255, (color >> 8) & 255, color >> 16])
            highlight = 255
            frames = {}
            for seconds in [0.5, 1.15, 1.35, 1.55, 1.9, 2.25, 1.25]:
                desk.elapsed = lambda: seconds
                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 480, 240)
                pane.draw(cairo.Context(surface))
                pixels = np.frombuffer(surface.get_data(), np.uint8).reshape(240, 480, 4)[:, :, :3]
                self.assertTrue((pixels[0] == background).all())
                # Inspect the fixed reading area independently of the countdown.
                lyric_ink = (pixels[42:] == 0).all(axis=2)
                y, x = np.where(lyric_ink)
                y += 42
                self.assertAlmostEqual((x.min() + x.max()) / 2, 240, delta=3)
                if frames:
                    self.assertEqual((y.min(), y.max()), baseline)
                else:
                    baseline = (y.min(), y.max())
                self.assertTrue((pixels == 0).all(axis=2).any())
                frames[seconds] = (pixels[42:] == highlight).all(axis=2)
                if seconds < 2.2:
                    self.assertTrue((pixels[42:] == 144).all(axis=2).any())
                # Playback may change the letters, but must never draw outside them.
                painted = (pixels[42:] != background).any(axis=2)
                if seconds == 0.5:
                    preview_ink = painted.copy()
                else:
                    self.assertFalse(painted[~preview_ink].any())
                if seconds == 1.15:
                    def luminance(bgr):
                        c = np.asarray(bgr) / 255
                        linear = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055)**2.4)
                        return linear @ [.0722, .7152, .2126]
                    ready = luminance([144]*3)
                    swept = luminance([255]*3)
                    self.assertGreaterEqual((ready + .05) / (luminance(background) + .05), 4.5)
                    self.assertGreaterEqual((swept + .05) / (ready + .05), 3)
            self.assertFalse(frames[0.5].any())
            previous = frames[0.5]
            for seconds in sorted(frames)[1:]:
                current = frames[seconds]
                self.assertTrue(current[previous].all())
                self.assertGreater(current.sum(), previous.sum())
                if previous.any():
                    self.assertGreater(np.where(current)[1].max(), np.where(previous)[1].max())
                previous = current
            window.destroy()


class AlternatingBars(unittest.TestCase):
    def test_shared_row_on_irregular_bars_seek_and_odd_loop(self):
        song = dict(track="pair", label="Duet", start=0, end=5.47, loop=True,
                    bpm=120, downbeats=[0, 1.8, 3.9, 5.47],
                    rows={"lead": [(["one", "two", "three"],
                                    [(100, 600), (1800, 2500), (3900, 4600)])]})
        pair = dict(first=3, second=1, lyrics="pair")
        desk = SimpleNamespace(songs={"pair": song}, backing=song,
                               bar_pairs={3: pair, 1: pair}, strips=[{"lyrics": None}] * 4)
        for i in (3, 1):
            pane = Pane(desk, i)
            window = Gtk.OffscreenWindow()
            pane.set_size_request(480, 240)
            window.add(pane)
            window.show_all()
            self.addCleanup(window.destroy)
            for time, first_sings, word, following in [(0.2, True, "one", "two"),
                    (1.8, False, "two", "three"), (4, True, "three", "one"),
                    (5.7, False, "one", "two"), (1.9, False, "two", "three"),
                    (5 * 5.47 + .2, False, "one", "two"), (6 * 5.47 + .2, True, "one", "two")]:
                desk.elapsed = lambda: time
                pane.draw(cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 480, 240)))
                name = pane.get_accessible().get_name()
                singing = first_sings == (i == 3)
                self.assertIn(" · Sing" if singing else " · Next bar", name)
                expected = word if singing else following
                actual = name.partition(" · lead ")[2].partition(" · Up next · ")[0]
                self.assertEqual(actual, expected)


class LaunchWarnings(unittest.TestCase):
    def setUp(self):
        self.full = dict(id="song", track="song", label="Song", start=0, end=24,
                         loop=False, bpm=120, rows={"lead": [
                             (["old", "ending"], [(5900, 6800), (6800, 8000)]),
                             (["wrong", "verse"], [(8100, 8600), (8600, 9000)]),
                             (["pickup", "sing", "hold"], [(16100, 16800), (16900, 17600), (17600, 18400)]),
                             (["after"], [(18500, 19000)]),
                         ]})
        self.outgoing = dict(self.full, loop=True, end=8)
        self.incoming = dict(self.full, loop=True, start=12, end=20, vocal_window=(16, 18))
        self.desk = SimpleNamespace(songs={"song": self.full}, strips=[dict(lyrics="song")]*4,
                                    pending=dict(boundary=8, region=self.incoming),
                                    sync_launch=lambda: None, backing=self.outgoing)
        self.window = Gtk.OffscreenWindow()
        self.pane = Pane(self.desk, 0)
        self.pane.set_size_request(480, 240)
        self.window.add(self.pane)
        self.window.show_all()
        self.addCleanup(self.window.destroy)

    def frame(self, seconds):
        self.desk.elapsed = lambda: seconds
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 480, 240)
        self.pane.draw(cairo.Context(surface))
        pixels = np.frombuffer(surface.get_data(), np.uint8).reshape(240, 480, 4)
        return self.pane.get_accessible().get_name(), pixels.copy()

    def drawn_lyrics(self, seconds, phrases):
        # Read the text at its final Cairo position on the singer's pane.
        lines = []
        original = PangoCairo.layout_path

        def record(ctx, layout):
            if layout.get_text() in phrases:
                x, y = ctx.get_current_point()
                lines.append((y, layout.get_text(), x, layout.get_line_count(),
                              layout.get_font_description().get_size()))
            original(ctx, layout)

        with patch("audience.PangoCairo.layout_path", record):
            self.frame(seconds)
        return sorted(lines)

    def test_milkshake_chorus_reads_down_the_pane_across_wraps_and_seeks(self):
        title = "Milkshake"
        # The clean CI runtime omits this recording. Use its eight measured
        # phrase intervals there and exercise the full word score when installed.
        recorded = catalog().get(title)
        if recorded is None:
            rows = [
                (["My", "milkshake"], [(0, 214), (286, 690)]),
                (["call"], [(714, 2048)]),
                (["And", "they're", "like"], [(2071, 2214), (2262, 2500), (2548, 2667)]),
                (["answer"], [(2881, 3905)]), (["fifth"], [(4024, 4595)]),
                (["answer"], [(4714, 5738)]), (["seventh"], [(5738, 6405)]),
                (["last"], [(6643, 7548)]),
            ]
            recorded = dict(id=title, track=title, label=title, bpm=126, cycle=16 * 60 / 126,
                            start=0, end=0, loop=False, rows={"lead": rows})
        songs = calibrated_catalog({title: recorded})
        song = songs[title]
        self.desk.songs = songs
        self.desk.strips[0]["lyrics"] = title
        self.desk.pending = None
        self.desk.backing = backing_loops(songs)[title]["main"].region(song, song["cycle"])
        length = self.desk.backing["end"] - self.desk.backing["start"]
        phrases = [" ".join(words) for words, _ in song["rows"]["lead"]]
        self.assertEqual(phrases[0], "My milkshake")
        self.assertEqual(phrases[2], "And they're like")
        times = [.1, .8, 2.1, 3.0, 4.1, 4.8, 5.9, 6.8]
        self.assertEqual(len(phrases), len(times))
        for lap, index in ([(lap, i) for lap in range(2) for i in range(8)]
                           + [(0, 3), (0, 0), (3, 7), (4, 0)]):
            with self.subTest(lap=lap, phrase=index):
                lines = self.drawn_lyrics(lap * length + times[index], phrases)
                upcoming = range(index, index + 4)
                expected = [phrases[i % 8] for i in sorted(upcoming, key=lambda i: i % 4)]
                self.assertEqual([line[1] for line in lines], expected)

    def singing_slot(self, seconds):
        # Only swept glyphs are white inside the four lyric bands. This
        # distinguishes repeated, identically spelled lines in a rendered frame.
        _, pixels = self.frame(seconds)
        white = (pixels[40:, :, :3] == 255).all(axis=2)
        y = np.flatnonzero(white.any(axis=1))
        self.assertTrue(len(y), f"No swept words at {seconds:.3f}s")
        # Midpoints between the four lyric baselines on a 480 x 240 pane.
        slots = np.unique(np.searchsorted([93, 136, 179], 40 + y))
        self.assertEqual(len(slots), 1, f"Multiple rows sweeping at {seconds:.3f}s")
        return slots[0]

    def check_recorded_lyric_flow(self, title, mode, bpm, start, end, onsets, releases):
        # CI has no private recordings. Keep the opening timings, repeated text,
        # and overlapping releases of the installed cut in the portable fixture.
        songs = calibrated_catalog(catalog())
        family = next((k for k, song in songs.items()
                       if k.endswith("/full") and song.get("title") == title), None)
        if family:
            choice = backing_loops(songs)[family][mode]
            song = songs[choice.source]
        else:
            choice = Loop(mode, title, start_ms=start)
            words = ["chant"] * 3 + ["echo", "title", "reply", "verse", "tail", "title"]
            rows = [([w], [(a, b)]) for w, a, b in zip(words, onsets, releases, strict=True)]
            song = dict(self.full, id=title, track=title, start=0, end=end, bpm=bpm,
                        rows={"lead": rows})
        self.desk.songs[choice.source] = song
        self.desk.strips[0]["lyrics"] = choice.source
        self.desk.backing = choice.region(song, end)
        self.desk.pending = None
        region = self.desk.backing
        windows = sorted((spans[0][0], spans[-1][1]) for _, spans in song["rows"]["lead"]
                         if region["start"] * 1000 <= spans[0][0] < region["end"] * 1000)
        self.assertGreaterEqual(len(windows), len(onsets))
        # Cold seek, every sung phrase, skipped frames, then backward seeks.
        for i in [6, *range(len(windows)), 4, 1, 7]:
            with self.subTest(title=title, mode=mode, phrase=i):
                attack, end = windows[i]
                if i + 1 < len(windows):
                    end = min(end, windows[i + 1][0])
                seconds = (attack + .55 * (end - attack)) / 1000 - region["start"]
                self.assertEqual(self.singing_slot(seconds), i % 4)

    def test_ms_new_booty_main_and_full_sweep_down_four_fixed_slots(self):
        self.check_recorded_lyric_flow(
            "Ms. New Booty", "main", 126, 0, 64.761895833,
            [1881, 3792, 5701, 7612, 9256, 11675, 13539, 15494, 16890],
            [3611, 5535, 7445, 9239, 11293, 13234, 15205, 17095, 18977])
        self.check_recorded_lyric_flow(
            "Ms. New Booty", "full", 97, 1559, 280.161,
            [1571, 4048, 6528, 9008, 11151, 14248, 16708, 19248, 21069],
            [3813, 6312, 8793, 11129, 13836, 16312, 18852, 21228, 23732])

    def test_whisper_song_main_and_full_sweep_down_four_fixed_slots(self):
        self.check_recorded_lyric_flow(
            "Wait (The Whisper Song)", "main", 126, 0, 64.761895833,
            [1187, 3817, 5047, 7297, 8900, 11410, 12689, 14907, 16898],
            [3010, 4937, 6815, 8790, 10636, 12530, 14424, 16513, 19013])
        self.check_recorded_lyric_flow(
            "Wait (The Whisper Song)", "full", 102, 0, 180.981,
            [8986, 11884, 13968, 16608, 18868, 21188, 23728, 25728, 27628],
            [11831, 13968, 16472, 18532, 20792, 23172, 25728, 27312, 30272])

    def test_interleaved_parts_read_in_time_order_even_when_both_have_a_visible_row(self):
        self.desk.pending = None
        self.desk.backing = self.full
        self.full["rows"] = {
            "answer": [(["reply"], [(1900, 2300)]), (["echo"], [(3100, 3400)])],
            "call": [(["held"], [(100, 2100)]), (["again"], [(2500, 2900)])],
        }
        phrases = ["held", "reply", "again", "echo"]
        for t in (.5, 1.95, .2):
            with self.subTest(t=t):
                lines = self.drawn_lyrics(t, phrases)
                self.assertEqual([line[1] for line in lines], phrases)

    def test_held_harmony_keeps_its_row_while_the_lead_wraps(self):
        self.desk.pending = None
        self.desk.backing = self.full
        calls = ["one", "two", "three", "four", "five"]
        self.full["rows"] = {
            "harmony": [(["hold"], [(100, 6000)])],
            "lead": [([word], [(i * 1000, i * 1000 + 800)])
                     for i, word in enumerate(calls, 1)],
        }
        lyrics = ["hold", *calls]
        first = self.drawn_lyrics(.5, lyrics)
        for i, call in enumerate(calls):
            t = i + 1.5
            drawn = self.drawn_lyrics(t, lyrics)
            self.assertEqual(drawn[0], first[0])
            self.assertIn("lead " + call, self.frame(t)[0])
            active = next(r for r in drawn if r[1] == call)
            self.assertEqual(active[0], first[1 + i % 3][0])

    def test_new_attack_replaces_the_previous_lanes_release_hold(self):
        self.desk.pending = None
        self.desk.backing = self.full
        self.full["rows"] = {
            "harmony": [(["hold"], [(100, 3950)])],
            "lead": [([str(i)], [(1000*i, 1000*i + 300)]) for i in range(1, 6)],
        }
        self.assertEqual(self.singing_slot(4.09), 0)
        self.assertIn("lead 4 · Up next · 5", self.frame(4.09)[0])

    def test_repeated_edition_only_sweeps_the_occurrence_being_sung(self):
        self.desk.pending = None
        self.desk.songs["song"] = dict(self.full, id="chant", track="chant", cycle=1,
                                       end=1, loop=True, rows={"lead": [(["chant"], [(100, 800)])]})
        self.desk.backing = dict(self.full, track="other", loop=True, end=5)
        for i in (0, 1, 4, 5, 2):
            self.assertEqual(self.singing_slot(i + .5), i % 4)

    def test_queued_pickup_appears_below_untimed_cue_and_holds_its_place(self):
        cue = dict(self.full, id="suite/full", cuts=["full"], rows={}, cue=["Listen"])
        cut = dict(self.full, id="suite/hook-1", cuts=["hook-1"],
                   rows={"lead": [(["pickup"], [(100, 600)])]})
        self.desk.songs = {s["id"]: s for s in (cut, cue)}
        self.desk.strips[0]["lyrics"] = cue["id"]
        self.desk.backing = cue
        self.desk.pending = dict(region=cut, boundary=2)
        lines = self.drawn_lyrics(1.5, ["Listen", "pickup"])
        self.assertEqual([r[1] for r in lines], ["Listen", "pickup"])
        self.desk.backing, self.desk.pending = cut, None
        self.assertEqual(self.drawn_lyrics(.4, ["pickup"]), lines[1:])

    def test_queued_move_on_another_mic_keeps_the_preview_across_laps(self):
        from clip_clock import ClipClock
        from desk import Desk

        words = ["one", "two", "three", "four", "five"]
        self.full["rows"] = {"lead": [([w], [(1000*i, 1000*i + 800)])
                                      for i, w in enumerate(words)]}
        old = ClipClock(dict(self.full, end=5, loop=True))
        new = ClipClock(dict(old.region), origin=-1.625)
        desk = self.desk
        desk.playing_slot, desk.ready, desk.pending = 2, True, None
        desk.source_key = lambda slot: "song"
        desk.slot_clocks = {1: ("song", "main", old)}
        desk.actions = {1: ("song", "main", new, 1)}
        desk.pane_transport = lambda index: Desk.pane_transport(desk, index)
        desk.conductor_bar = lambda: .7
        preview = self.drawn_lyrics(0, ["one"])
        self.assertEqual(len(preview), 1)
        desk.slot_clocks[1] = desk.actions.pop(1)[:3]
        desk.conductor_bar = lambda: 1
        self.assertEqual(self.drawn_lyrics(0, ["one"]), preview)
        self.assertEqual(self.singing_slot(0), 2)

    def test_paired_blue_teaser_sweeps_its_fragment_on_the_backing_grid(self):
        title = "Blue (Da Ba Dee)"
        sheet = dict(self.full, id=title, track=title, bpm=127.99, beat_zero=2.281,
                     rows={"lead": [(["Yo", "listen", "up", "here's", "a", "story"],
                                     [(1768, 2053), (2229, 2952), (3129, 3233),
                                      (3488, 3912), (4008, 4088), (4088, 4552)])]})
        self.desk.songs[title] = sheet
        self.desk.bar_pairs = {0: dict(lyrics=title, first=0, second=1)}
        self.desk.backing = dict(sheet, start=.4058541667, end=7.9064375, loop=True,
                                 beat_zero=.4058535042, vocal_window=(1.418, 4.74))
        self.desk.pending = None
        self.assertEqual(self.singing_slot(1.50), 0)
        self.assertIn("Sing · lead Yo listen", self.frame(1.50)[0])

    def test_cue_for_a_later_part_stays_below_the_intervening_phrases(self):
        self.desk.pending = None
        self.desk.backing = self.full
        self.full["rows"] = {
            "harmony": [(["answer"], [(1300, 1600)])],
            "lead": [(["one"], [(100, 400)]), (["two"], [(500, 700)]),
                     (["three"], [(800, 1000)])],
        }
        phrases = ["one", "two", "three", "answer"]
        self.assertEqual([r[1] for r in self.drawn_lyrics(.2, phrases)], phrases)

    def test_queued_seek_repeats_the_held_phrase_in_performance_order(self):
        self.full["rows"] = {"lead": [
            (["opening"], [(1000, 1500)]), (["hold"], [(2000, 3500)]),
            (["finish"], [(4000, 4500)])]}
        self.desk.backing = self.full
        target = dict(self.full)
        self.desk.pending = dict(boundary=3, region=target, offset=2.4)
        phrases = ["opening", "hold", "finish"]
        for t, expected in [(1.2, ["opening", "hold", "hold", "finish"]),
                            (2.7, ["hold", "hold", "finish"])]:
            self.assertEqual([r[1] for r in self.drawn_lyrics(t, phrases)], expected)
        preview = self.drawn_lyrics(2.7, phrases)[1:]
        self.desk.backing, self.desk.pending = target, None
        self.assertEqual([r[1] for r in preview], ["hold", "finish"])
        self.assertEqual(self.drawn_lyrics(2.4, phrases), preview)

    def test_prepared_chorus_starts_at_first_complete_word(self):
        # Exact opening timings from the installed chorus cut (private media).
        key = "Constant Headache - Chant"
        raw = dict(self.full, id=key, cycle=7.619, rows={"lead": [
            (["sheets"], [(0, 8)]),
            (["But", "I'm", "a", "constant", "headache"], [(223, 505), (581, 751),
             (751, 816), (891, 1573), (1644, 2322)])]})
        song = calibrated_catalog({key: raw})[key]
        self.desk.songs[self.desk.strips[0]["lyrics"]] = song
        self.desk.backing = dict(song, start=0, end=song["cycle"], loop=True)
        self.desk.pending = None
        for pos in (0, .223, song["cycle"]):
            name, _ = self.frame(pos)
            self.assertNotIn("sheets", name)
            self.assertIn("But I'm a constant headache", name)

    def test_constant_headache_loop_choices_use_the_prepared_chorus_clock(self):
        title = "Constant Headache"
        chorus = title + " - Chant"
        guitar = dict(self.full, id=title, title=title, track=title, rows={}, cue=["Instrumental"])
        chant = dict(guitar, id=chorus, track=chorus, part="Chorus", bpm=126, cycle=960/126,
                     rows={"lead": [(["sheets"], [(0, 8)]),
                                    (["But", "I'm", "a", "constant", "headache"], [(223, 505),
                                     (581, 751), (751, 816), (891, 1573), (1644, 2322)]),
                                    (["line"], [(3810, 3990)]), (["They"], [(3990, 4230)])]})
        songs = calibrated_catalog({title: guitar, chorus: chant})
        modes = backing_loops(songs)[title]
        self.desk.songs = songs
        self.desk.pending = None
        for assignment in (title, chorus):
            self.desk.strips[0]["lyrics"] = assignment
            for mode in ("teaser", "main", chorus):
                choice = modes[mode]
                self.assertEqual(choice.source, chorus)
                self.desk.backing = choice.region(songs[choice.source], chant["cycle"])
                for t in (0, .23, 3.85, 4.05, 7.61, 7.62, 11.7):
                    with self.subTest(assignment=assignment, mode=mode, t=t):
                        name, _ = self.frame(t)
                        self.assertNotIn("sheets", name)
                        self.assertNotIn("Instrumental", name)
                        self.assertIn("constant headache", name)
                        self.assertEqual("They" in name, mode != "teaser")

    def test_imported_loop_choices_follow_the_playing_recording(self):
        prefix = "import/"
        songs = {}
        for part, word in (("full", "Verse"), ("teaser", "Opening"), ("hook-1", "Chorus"), ("hook-2", "Coda")):
            key = prefix + part
            songs[key] = dict(self.full, id=key, track=key, title="Imported song", cuts=[part],
                              label="Imported song · " + part, rows={"lead": [([word], [(200, 590)])]})
        songs[prefix + "teaser"]["loopRole"] = "teaser"
        self.desk.songs = songs
        self.desk.loops = backing_loops(songs)
        self.desk.pending = None
        for assigned, playing, word in (("teaser", "hook-1", "Chorus"), ("hook-1", "teaser", "Opening"),
                                        ("hook-2", "teaser", "Coda")):
            self.desk.strips[0]["lyrics"] = prefix + assigned
            self.desk.backing = dict(songs[prefix + playing], start=0, end=4, loop=True)
            name, _ = self.frame(.25)
            self.assertIn("lead " + word, name)

    def test_only_words_starting_in_cut_appear_in_preview_and_playback(self):
        self.full["rows"]["lead"] = [
            (["tail", "in", "loop", "outside"], [(3700, 4100), (4200, 4500),
                                                  (7700, 8100), (8200, 8500)])]
        self.desk.backing = dict(self.full, start=4, end=8, loop=True)
        self.desk.pending = None
        for pos in (0, .3, 3.85, 4):
            name, _ = self.frame(pos)
            self.assertIn("in loop", name)
            self.assertNotIn("tail", name)
            self.assertNotIn("outside", name)

    def test_four_lines_advance_on_fast_phrases_and_seek(self):
        self.desk.pending = None
        self.desk.backing = self.full
        self.full["rows"]["lead"] = [
            ([word], [(1000 + i*350, 1300 + i*350)])
            for i, word in enumerate(["one", "two", "three", "four", "five", "six"])
        ]
        positions = [(1.1, 0), (1.45, 1), (1.8, 2), (2.15, 3), (2.5, 4), (2.85, 5), (1.1, 0)]
        phrases = [row[0][0] for row in self.full["rows"]["lead"]]
        for t, index in positions:
            expected = phrases[index:index + 4]
            name, _ = self.frame(t)
            self.assertEqual(name.split(" · lead ")[1].split(" · Up next · "), expected)
            lines = self.drawn_lyrics(t, phrases)
            physical = sorted(expected, key=lambda text: phrases.index(text) % 4)
            self.assertEqual([line[1] for line in lines], physical)
            self.assertEqual(self.singing_slot(t), index % 4)

    def test_different_length_lines_keep_their_layout_over_two_loop_cycles(self):
        self.desk.pending = None
        self.desk.backing = dict(self.full, start=4, end=12, loop=True)
        phrases = ["a", "one of the longer lines spanning several words", "b", "c",
                   "d", "e", "f", "last phrase"]
        self.full["rows"]["lead"] = [
            (phrase.split(), [(4200 + 1000*i + 30*j, 4225 + 1000*i + 30*j)
                              for j, _ in enumerate(phrase.split())])
            for i, phrase in enumerate(phrases)]
        typography = {}
        for step in range(164):
            lines = self.drawn_lyrics(step / 10, phrases)
            self.assertEqual(len(lines), 4)
            ordered = [line[1] for line in lines]
            # Each baseline has two alternating phrases in this eight-line loop.
            self.assertEqual([phrases.index(text) % 4 for text in ordered], [0, 1, 2, 3])
            for y, text, x, wraps, size in lines:
                shape = (x, y, wraps, size)
                self.assertEqual(typography.setdefault(text, shape), shape)
            if step in (79, 80, 159, 160):
                self.assertEqual(ordered[:2], phrases[:2])

    def test_five_line_loop_continues_down_through_wraps_and_seeks(self):
        phrases = ["first", "second", "third", "fourth", "last"]
        self.full["rows"] = {"lead": [([word], [(1000*i, 1000*i + 800)])
                                      for i, word in enumerate(phrases)]}
        self.desk.backing = dict(self.full, start=0, end=5, loop=True)
        self.desk.pending = None
        for i in [13, *range(12), 4, 20, 7]:
            with self.subTest(phrase=i):
                self.assertEqual(self.singing_slot(i + .4), i % 4)
                expected = [phrases[j % 5] for j in sorted(range(i, i + 4), key=lambda j: j % 4)]
                self.assertEqual([r[1] for r in self.drawn_lyrics(i + .4, phrases)], expected)

    def test_one_line_teaser_repeats_in_place(self):
        self.full["rows"] = {"lead": [(["chant"], [(200, 800)])]}
        self.desk.backing = dict(self.full, start=0, end=1, loop=True)
        self.desk.pending = None
        preview = self.drawn_lyrics(0, ["chant"])
        self.assertEqual(len(preview), 1)
        for t in (.6, 1, 1.6, 9.6):
            self.assertEqual(self.drawn_lyrics(t, ["chant"]), preview)

    def test_launch_preview_honors_exact_pickup_and_incoming_offset(self):
        self.full["rows"]["lead"] = [
            (["pickup"], [(16013, 16023)]), (["next"], [(16200, 16500)])]
        region = dict(self.full, start=16, end=20, loop=True)
        def entrance(offset, horizon):
            return [list(row[2]) for row in self.pane.entrance(
                self.full, region, offset, None, horizon)]
        self.assertEqual(entrance(0, .012), [])
        self.assertEqual(entrance(0, .014), [["pickup"]])
        self.assertEqual(entrance(.018, .001), [["pickup"]])
        self.assertEqual(entrance(.025, .18), [["next"]])

    def test_queued_preview_starts_at_requested_phrase(self):
        self.incoming = dict(self.full, start=0, end=24, loop=False)
        self.desk.pending.update(region=self.incoming, offset=16)
        name, _ = self.frame(7.5)
        self.assertIn("lead old ending · Up next · pickup sing hold · Up next · after", name)
        self.assertNotIn("wrong verse", name)

    def test_intro_countdown_clears_at_pickup_break_seek_and_outro(self):
        self.desk.pending = None
        self.desk.backing = self.full
        name, _ = self.frame(0)
        self.assertIn("Intro · Vocals in 6 s · lead old ending", name)
        # Only the top caption changes while counting down. The four rows
        # below it keep precisely the same pixels until the pickup.
        _, before = self.frame(3.91)
        _, after = self.frame(5.41)
        self.assertFalse(np.array_equal(before[:40], after[:40]))
        np.testing.assert_array_equal(before[40:], after[40:])
        _, pickup = self.frame(5.9)
        self.assertTrue((pickup[:40] == pickup[0, 0]).all())
        self.assertNotIn("Vocals in", self.frame(5.9)[0])
        self.assertNotIn("Break", self.frame(8.05)[0])  # a breath between lines
        self.assertIn("Break · Vocals in 7 s", self.frame(9.2)[0])
        self.assertNotIn("Break", self.frame(16.1)[0])
        self.assertIn("Outro", self.frame(19.1)[0])
        self.assertNotIn("Vocals in", self.frame(20)[0])
        name, _ = self.frame(13.1)
        self.assertIn("Break · Vocals in 3 s", name)

    def test_countdown_uses_backing_seconds_on_measured_grids(self):
        self.full.update(id="vocals", downbeats=[0, 4, 8, 12, 16])
        self.desk.backing = dict(self.full, id="backing", track="backing", downbeats=[0, 2, 5, 9, 12])
        self.desk.pending = None
        # The first attack at lyric time 5.9 maps to backing time 3.425.
        self.assertIn("Intro · Vocals in 3 s", self.frame(0.5)[0])
        self.assertIn("Intro · Vocals in 1 s", self.frame(3.42)[0])
        self.assertNotIn("Intro", self.frame(3.43)[0])

    def test_harmony_hold_and_untimed_cues(self):
        self.desk.pending = None
        self.desk.backing = self.full
        self.full["rows"]["harmony"] = [(["ooh"], [(8950, 14000)])]
        self.assertNotIn("Break", self.frame(11)[0])
        self.assertIn("Break · Vocals in 3 s", self.frame(14)[0])
        untimed = dict(self.full, rows={}, cue=["Sing along"])
        self.desk.songs["song"] = self.desk.backing = untimed
        name, _ = self.frame(2)
        self.assertIn("Sing along", name)
        self.assertNotIn("Vocals in", name)

    def test_queued_pickup_countdown_tracks_wall_seconds_after_a_room_tempo_edit(self):
        self.outgoing["vocal_window"] = (0, 0)
        self.incoming["bpm"] = 60
        self.desk.pending["offset"] = 2
        # One outgoing bar then .525 of an incoming bar before the first word.
        for bpm, seconds in [(150, 3), (75, 5)]:
            self.desk.tempo = SimpleNamespace(get_value=lambda: bpm)
            name, _ = self.frame(6)
            self.assertIn(f"Vocals in {seconds} s", name)
            self.assertIn("pickup", name)

    def test_countdown_to_loop_wrap_and_queued_pickup(self):
        self.desk.backing = dict(self.full, start=16, end=20, loop=True)
        self.desk.pending = None
        self.assertIn("Outro · Vocals in 1 s", self.frame(3.5)[0])
        self.assertIn("Intro · Vocals in 1 s", self.frame(4.0)[0])
        self.desk.pending = dict(boundary=4, region=self.incoming)
        self.assertIn("Outro · Vocals in 5 s", self.frame(3.5)[0])
        self.desk.pending = None
        self.desk.backing = self.incoming
        self.assertIn("Intro · Vocals in 5 s", self.frame(0)[0])
        self.assertNotIn("Vocals in", self.frame(4.11)[0])

    def test_outgoing_sweep_and_incoming_fragment_through_two_bar_pickup(self):
        name, pixels = self.frame(7.0)
        self.assertIn("lead old ending · Up next · pickup sing hold", name)
        # The old line retains its white sweep; the incoming line is gray.
        self.assertTrue((pixels[:140, :, :3] == 255).all(axis=2).any())
        self.assertTrue((pixels[95:, :, :3] == 144).all(axis=2).any())
        self.desk.pending = None
        self.desk.backing = self.incoming
        for seconds in (0.05, 1.8, 3.95):
            self.assertIn("Up next · pickup sing hold", self.frame(seconds)[0])
        name, _ = self.frame(4.7)
        self.assertIn("lead pickup sing hold", name)
        self.assertNotIn("Up next", name)
        name, _ = self.frame(6.3)
        self.assertNotIn("lead after", name)

    def test_pair_preview_uses_incoming_singers_bar_and_backing_fallback(self):
        answer = dict(self.full, id="answer", track="answer", label="Answer", start=0, end=8,
                      loop=True, rows={"lead": [(["Alpha"], [(200, 900)]),
                                                (["Beta"], [(2300, 2900)])]})
        self.desk.pending["region"] = answer
        # Mic 0 is second, with lyrics following the active backing for the pair.
        self.desk.bar_pairs = {0: dict(lyrics=None, first=1, second=0)}
        name, _ = self.frame(7.1)
        self.assertIn("Sing", name)
        self.assertIn("Up next · Beta", name)
        self.assertNotIn("Alpha", name)
        self.desk.pending = None
        self.desk.backing = answer
        name, _ = self.frame(2.5)
        self.assertIn("Sing · lead Beta", name)

        # A launch after an odd cycle transfers the first incoming bar to the partner.
        outgoing = dict(self.outgoing, end=6)
        answer["turn_offset"] = transport_bar(outgoing, 6)
        self.desk.backing = outgoing
        self.desk.pending = dict(boundary=6, region=answer)
        name, _ = self.frame(5.3)
        self.assertIn("Next bar · Up next · Alpha", name)
        self.desk.pending = None
        self.desk.backing = answer
        self.assertIn("Sing · lead Alpha", self.frame(0.4)[0])


if __name__ == "__main__":
    unittest.main()
