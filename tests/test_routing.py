from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from desk import Desk, Gtk, settings, strip
from backing_loops import backing_loops, calibrated_catalog
from dial import Dial
from library import catalog
from routing import SongLibrary, SongTarget


def descendants(widget):
    yield widget
    if isinstance(widget, Gtk.Container):
        for child in widget.get_children():
            yield from descendants(child)


class MicControls(unittest.TestCase):
    def test_library_lists_each_recording_once_and_searches_title_and_artist(self):
        songs = calibrated_catalog(catalog())
        # Two independently imported recordings of the same song retain their artist
        # and assignment ID, including when the library refreshes during a search.
        for prefix, artist in (("a", "Full Moon Quartet"), ("b", "Night Orchestra")):
            for cut in ("full", "chorus", "teaser"):
                key = f"{prefix}/{cut}"
                songs[key] = dict(id=key, title="Twin recordings", artist=artist,
                                  label=f"Twin recordings · {cut}", cuts=[cut],
                                  loopRole="teaser" if cut == "teaser" else None)
        loops = backing_loops(songs)
        library = SongLibrary()
        self.addCleanup(library.destroy)
        library.show_catalog(songs, loops)
        self.assertEqual(set(library.items), set(loops))
        for key, row in library.items.items():
            song = songs[key]
            expected = " · ".join(filter(None, (song["title"], song.get("artist"))))
            self.assertEqual(row.get_accessible().get_name(), expected)

        def visible():
            return {key for key, row in library.items.items() if row.get_child_visible()}

        library.search.set_text("Twin recordings")
        self.assertEqual(visible(), {"a/full", "b/full"})
        library.search.set_text("twin recordings full")
        self.assertEqual(visible(), {"a/full"})
        library.show_catalog(songs, loops)
        self.assertEqual(visible(), {"a/full"})
        for mode in ("teaser", "main"):
            library.search.set_text("twin recordings " + mode)
            self.assertEqual(visible(), set())

    def test_library_shows_titles_and_artists_and_keeps_edition_notes_searchable(self):
        albums = "Original studio album version from Under Construction."
        songs = {
            "work/full": dict(title="Work It", artist="Missy Elliott",
                              importRequest=dict(source="Missy Elliott - Work It", notes=albums)),
            "blue/full": dict(title="Blue (Da Ba Dee) [Original Studio Version]", artist="Eiffel 65"),
            "tipsy/full": dict(title="Tipsy (Club Mix) – Album Version", artist="J-Kwon"),
        }
        library = SongLibrary()
        self.addCleanup(library.destroy)
        library.show_catalog(songs, songs)
        for key, expected in (("work/full", ["Work It", "Missy Elliott"]),
                              ("blue/full", ["Blue (Da Ba Dee)", "Eiffel 65"]),
                              ("tipsy/full", ["Tipsy (Club Mix)", "J-Kwon"])):
            row = library.items[key]
            labels = [w.get_text() for w in descendants(row) if isinstance(w, Gtk.Label)]
            self.assertEqual(labels, expected)
            self.assertEqual(row.get_accessible().get_name(), " · ".join(expected))
        self.assertIn(albums, library.items["work/full"].get_tooltip_text())
        library.search.set_text("under construction")
        self.assertEqual([k for k, row in library.items.items() if row.get_child_visible()], ["work/full"])

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/"linux-four"/"loop-presets.json").is_file(),
                         "requires the installed recording chart")
    def test_get_low_has_one_recording_and_three_standard_loop_choices(self):
        # The private catalog has three legacy entries on one recording clock.
        # The clean release has no bundled media, so provide those entries here.
        bundled = {}
        for key, part, start, end in (
                ("Get Low", "Full song", 0, 334.066667),
                ("Get Low - Window to the Wall", "To the window to the wall", 29.156, 33.908475),
                ("Get Low - 369 to Wiggle", "3-6-9 to wiggle", 96.001833, 162.536486)):
            bundled[key] = dict(id=key, title="Get Low", part=part, label="Get Low · " + part,
                                track="get-low", bpm=101, start=start, end=end, loop=bool(start),
                                beat_zero=0 if start else 29.156 % (240 / 101))
        songs = calibrated_catalog(bundled)
        self.assertEqual(list(songs), ["Get Low"])
        loops = backing_loops(songs)
        self.assertEqual(set(loops["Get Low"]), {"teaser", "main", "full"})

        library = SongLibrary()
        self.addCleanup(library.destroy)
        library.show_catalog(songs, loops)
        library.search.set_text("Get Low")
        matches = [row.key for row in library.items.values() if row.get_child_visible()]
        self.assertEqual(matches, ["Get Low"])
        self.assertNotIn("Full song", library.items["Get Low"].get_accessible().get_name())

        song = songs["Get Low"]
        for mode, start, end in (("teaser", 27.112614, 36.617564), ("main", 96.023505, 162.558158),
                                 ("full", 0, song["end"])):
            with self.subTest(mode=mode):
                loop = loops["Get Low"][mode]
                self.assertEqual(loop.source, song["id"])
                region = loop.region(song, song["end"])
                self.assertAlmostEqual(region["start"], start, delta=1/48000)
                self.assertAlmostEqual(region["end"], end, delta=1/48000)
                if mode == "teaser":
                    self.assertEqual(region["vocal_window"], (29.169, 33.864))

    def test_drag_begin_renders_the_selected_library_row(self):
        library = SongLibrary()
        library.refill([("a", "First song", "Verse", ""), ("b", "Second song", "Chorus", "")])
        window = Gtk.Window()
        window.add(library)
        window.show_all()
        self.addCleanup(window.destroy)
        images = []
        for row in library.items.values():
            source = row.get_child()
            with patch.object(Gtk, "drag_set_icon_surface") as install:
                from gi.repository import Gdk
                context = Gdk.DragContext()
                source.emit("drag-begin", context)
                actual_context, surface = install.call_args.args
                self.assertEqual(actual_context, context)
                self.assertEqual(surface.get_width(), row.get_allocated_width() * row.get_scale_factor())
                images.append(bytes(surface.get_data()))
        self.assertNotEqual(*images)

    def test_launch_starts_with_empty_targets_even_after_a_saved_set(self):
        strips = [dict(strip("Old lyrics"), anchor="Old anchor", reverb=37) for _ in range(4)]
        routes = ["input-1", "", "", "", "", ""]
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "session.json"
            state.write_text(json.dumps(dict(strips=strips, backing="Old backing", routes=routes)))
            with patch("desk.STATE", state):
                session = settings()
        self.assertNotIn("backing", session)
        self.assertEqual(session["routes"], routes)
        for mic in session["strips"]:
            self.assertIsNone(mic["lyrics"])
            self.assertEqual(mic["anchor"], "natural")
            self.assertEqual(mic["reverb"], 37)

    def test_drop_click_and_exposed_dials(self):
        desk = Desk.__new__(Desk)
        desk.library = SongLibrary()
        desk.library.refill([("café", "Café", "Chorus", "Test")])
        desk.songs = {"café": {"label": "Café"}}
        desk.strips = [strip(None) for _ in range(4)]
        desk.bar_pairs = {}
        desk.hear_modes = []
        desk.pair_controls = []
        desk.cells = []
        desk.vocoder_panels = []
        desk.tal_panels = []
        desk.loop_controls = {}
        desk.loop_style = Gtk.CssProvider()
        desk.source_buttons = {}
        desk.mix_controls = {}
        desk.session = {"presets": {}}
        desk.midi_controls = []
        desk.preset_controls = []
        from test_mode import TestMode
        desk.test = TestMode(desk)
        desk.audio = Mock()
        desk.audio.tal_parameters.return_value = []
        desk.audio.call.return_value = 0  # This fixture has no carrier.
        desk.assignment_changed = Mock()
        desk.configure = Mock()
        desk.save = Mock()
        for index in range(4):
            window = Gtk.Window()
            pane = desk.cell(index)
            window.add(pane)
            window.show_all()
            hear, see = [w for w in descendants(pane) if isinstance(w, SongTarget)]
            for target, icon in [(hear, "audio-input-microphone-symbolic"), (see, "camera-video-symbolic")]:
                image, caption = target.get_child().get_children()
                self.assertEqual(image.get_icon_name()[0], icon)
                self.assertEqual(caption.get_text(), "Drag here")
                self.assertIn(target.role, target.get_accessible().get_name())
            data = Mock()
            data.get_data.return_value = "café".encode()
            context = Mock()
            with patch.object(Gtk, "drag_finish") as finish:
                hear.drop(hear, context, 0, 0, data, 0, 123)
                finish.assert_called_once_with(context, True, False, 123)
            self.assertEqual(desk.strips[index]["anchor"], "café")
            self.assertIsNone(desk.strips[index]["lyrics"])
            desk.library.rows.select_row(desk.library.items["café"])
            see.clicked()
            self.assertEqual(desk.strips[index]["lyrics"], "café")
            dials = [w for w in descendants(pane) if isinstance(w, Dial)]
            self.assertEqual(len(dials), 16)
            lead = next(w for w in dials if "vocal lead" in w.get_accessible().get_name())
            lead.set_value(35)
            self.assertEqual(desk.strips[index]["lead"], 35)
            self.assertTrue(all(dial.get_mapped() for dial in dials))
            room = next(w for w in dials if "reverb" in w.get_accessible().get_name())
            room.set_value(37)
            self.assertEqual(desk.strips[index]["reverb"], 37)
            window.destroy()

    def test_pair_reassign_and_unbind_keep_independent_mics(self):
        from copy import deepcopy
        song = dict(id="hook", track="hook", title="Hook", label="Hook")
        songs = {s["id"]: s for s in (song, dict(song, id="verse", label="Verse"))}
        with tempfile.TemporaryDirectory() as directory, patch("desk.catalog", return_value=songs), patch(
            "desk.STATE", Path(directory) / "state"
        ), patch("desk.Audio") as audio:
            desk = Desk()
            self.addCleanup(desk.window.destroy)
            try:
                desk.strips[0].update(lyrics="hook", voice=1, reverb=12)
                desk.strips[2].update(lyrics="verse", reverb=35)
                before = deepcopy(desk.strips)
                for window in (desk.window, desk.stage):
                    labels = [w.get_text() for w in descendants(window) if isinstance(w, Gtk.Label)]
                    for name in ("Red", "Blue", "Green", "Gold"):
                        self.assertNotIn(name, labels)
                        panel = next(w for w in descendants(window) if isinstance(w, Gtk.Box)
                                     and w.get_accessible().get_name() == name)
                        style = panel.get_style_context()
                        self.assertEqual(style.get_border(Gtk.StateFlags.NORMAL).top, 7)
                        color = style.get_property("border-top-color", Gtk.StateFlags.NORMAL)
                        self.assertEqual(color.to_string(), {
                            "Red": "rgb(182,92,73)", "Blue": "rgb(71,123,163)",
                            "Green": "rgb(79,133,100)", "Gold": "rgb(173,125,48)",
                        }[name])
                menu = desk.pair_controls[0].get_popover()
                menu.popup()
                rows = next(w for w in descendants(menu) if isinstance(w, Gtk.ListBox))
                partner = next(r for r in rows if r.key == 2)
                self.assertEqual(partner.get_child().get_text(), "Alternate with Green")
                rows.emit("row-activated", partner)
                self.assertEqual(desk.bar_pairs[2]["lyrics"], "hook")
                self.assertEqual(desk.strips, before)
                self.assertIn("Red", desk.pair_controls[2].get_label())
                with patch.object(desk, "assignment_changed"):
                    desk.cells[2][0].assign("verse")
                self.assertEqual(desk.bar_pairs[0]["lyrics"], "verse")
                room = next(w for w in descendants(desk.window) if isinstance(w, Dial)
                            and w.get_accessible().get_name() == "Green room reverb")
                room.set_value(40)
                audio.return_value.call.assert_any_call("reverb", 2, .4)
                audio.return_value.call.assert_any_call("reverb", 0, .12)
                desk.pair_panes(1, 2)
                self.assertNotIn(0, desk.bar_pairs)
                desk.pair_panes(2, None)
                self.assertFalse(desk.bar_pairs)
                before[2]["reverb"] = 40
                self.assertEqual(desk.strips, before)
            finally:
                desk.close()

    def test_output_selection_and_preferences_across_launches(self):
        ports = ["Dummy Output:playback_FL", "Dummy Output:playback_FR",
                 "Speakers:playback_FR", "Speakers:playback_FL", "recorder:input_FL"]
        def launch():
            desk = Desk.__new__(Desk)
            desk.session = settings()
            desk.routes = [""] * 6
            desk.audio = Mock()
            desk.audio.ports.side_effect = lambda capture: [] if capture else ports
            desk.audio.call.return_value = 0
            desk.save = desk.flush
            desk.autoroute()
            return desk

        with tempfile.TemporaryDirectory() as directory, patch(
            "desk.STATE", Path(directory) / "session.json"
        ), patch("desk.subprocess.run", return_value=Mock(stdout='node.description = "Speakers"')):
            desk = launch()
            self.assertEqual(desk.routes[4:], ["Speakers:playback_FL", "Speakers:playback_FR"])
            desk.choose_route(4, ports[0])
            desk.choose_route(5, "")
            desk.mute = Mock()
            desk.mute.get_active.return_value = True
            desk.ceiling = Mock()
            desk.ceiling.get_value.return_value = -12
            desk.master()
            disconnected = ports.pop(0)
            desk = launch()
            self.assertEqual(desk.routes[4:], ["Speakers:playback_FL", ""])
            self.assertEqual(desk.session["routes"][4:], [disconnected, None])
            self.assertEqual(desk.session["ceiling"], -12)
            self.assertTrue(desk.session["muted"])
            ports.append(disconnected)
            self.assertEqual(launch().routes[4:], [disconnected, ""])
            desk.choose_route(5, "auto")
            self.assertEqual(launch().routes[4:], [disconnected, "Speakers:playback_FR"])
