"""Exercise the voice picker across linked mic strips, song changes and relaunch."""
import sys
import json
import tempfile
import unittest
import wave
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import VOICES
from desk import CHORD_TUNE, TUNELET, TAL_VOCODER, VOCODER_HOLD, VOCODER_PRESETS, Desk, Gtk, strip
from dial import Dial
from desk_audio import DeskAudio


def choose(menu, key):
    popover = menu.get_popover()
    popover.popup()
    scroll = next(w for w in popover.get_child() if isinstance(w, Gtk.ScrolledWindow))
    rows = scroll.get_child().get_child()
    row = next(row for row in rows if row.key == key)
    if not row.get_sensitive():
        raise ValueError(f"Unavailable menu choice: {key}")
    rows.emit("row-activated", row)


class VoiceState(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name in ("bed", "drums"):
            with wave.open(str(root / (name + ".wav")), "wb") as wav:
                wav.setparams((2, 2, 48000, 0, "NONE", ""))
                wav.writeframes(bytes(48000 * 4))
        song = dict(id="notes", title="Notes", label="Notes", track="notes", start=0,
                    end=1, loop=True, bpm=120, rows={}, score=dict(events=[[0, [69]]], duration=1))
        songs = {"notes": song, "words": dict(song, id="words", track="words", score=None,
                                              title="Words", label="Words")}
        for context in (patch("desk.catalog", return_value=songs), patch("desk.STATE", root / "state"),
                        patch("desk.playback_files", return_value=(root / "bed.wav", root / "drums.wav")),
                        patch.object(DeskAudio, "resources")):
            context.start()
            self.addCleanup(context.stop)
        self.desk = self.open_desk()

    def open_desk(self):
        desk = Desk()
        self.addCleanup(desk.window.destroy)
        return desk

    def assert_voices(self, desk, voices):
        self.assertEqual([strip["voice"] for strip in desk.strips], voices)
        self.assertEqual([cell[2].get_label() for cell in desk.cells], [VOICES[v] for v in voices])

    def test_hear_button_shows_follow_target_and_resolved_song(self):
        d = self.desk
        mode = d.hear_modes[1]
        self.assertEqual(mode.get_label(), "Natural")
        d.cells[0][1].assign("notes")
        d.cells[1][0].assign("words")
        choose(mode, "mic:0")
        choose(d.cells[1][2], 3)
        self.assertEqual(mode.get_label(), "Follow Red")
        self.assertIn("Notes", mode.get_accessible().get_name())
        self.assertEqual(d.test.song_key(1), "words")
        choose(d.cells[1][2], 1)
        self.assertIn("notes", d.cells[1][2].get_tooltip_text())
        self.assertEqual(mode.get_label(), "Follow Red")
        d.cells[0][1].assign("words")
        self.assertIn("Words", mode.get_tooltip_text())
        self.assertIn("No vocoder score", d.cells[1][2].get_tooltip_text())
        choose(mode, "autotune")
        self.assertEqual(mode.get_label(), "Autotune")
        d.cells[1][1].assign("notes")
        self.assertEqual(mode.get_label(), "Song")
        self.assertIn("Notes", mode.get_tooltip_text())

    def test_clear_and_hold_expose_the_live_controls_on_the_selected_pane(self):
        d = self.desk
        choose(d.cells[1][2], 1)
        panel = d.vocoder_panels[1]
        self.assertIs(panel.get_toplevel(), d.window)
        self.assertTrue(panel.preset.get_mapped())
        self.assertTrue(panel.reset.get_mapped())
        self.assertTrue(panel.width.get_mapped())
        self.assertFalse(panel.source.get_mapped())
        self.assertFalse(panel.menu.get_mapped())
        self.assertFalse(d.vocoder_panels[0].inline.get_mapped())
        for i, control in panel.controls.items():
            self.assertTrue(control.get_mapped())
            if i != 6:
                self.assertIsInstance(control, Dial)
        panel.preset.set_active_id("Tight speech")
        self.assertEqual(d.strips[1]["vocoder"], VOCODER_PRESETS["Tight speech"])
        panel.controls[7].set_value(0)
        panel.controls[0].set_value(82)
        panel.controls[6].set_active_id("-1")
        panel.width.set_value(34)
        d.preset_controls[1].bindings["lead"][3](42)
        self.assertEqual(panel.preset.get_active_id(), "Custom")
        self.assertEqual(d.strips[0]["vocoder"], strip(None)["vocoder"])

        # The right-click preset belongs to the dial on this pane.
        panel.controls[0].emit("popup-menu")
        self.assertEqual(d.preset_controls[1].scope.get_active_id(), "vocoder.0")
        d.preset_controls[1].get_popover().popdown()
        choose(d.cells[1][2], TAL_VOCODER)
        self.assertIn("silent", d.cells[1][2].get_tooltip_text())
        self.assertTrue(panel.inline.get_mapped())
        self.assertTrue(d.tal_panels[1].get_mapped())
        self.assertEqual(panel.labels[0].get_text(), "Wet level")
        self.assertTrue(panel.source.get_mapped())
        self.assertEqual(panel.source.get_value(), 0)
        panel.source.set_value(63)
        self.assertEqual(d.strips[0]["tal_source_level"], 0)
        panel.source.emit("popup-menu")
        self.assertEqual(d.preset_controls[1].scope.get_active_id(), "tal_source_level")
        d.preset_controls[1].get_popover().popdown()
        self.assertFalse(panel.controls[1].get_mapped())
        self.assertTrue(panel.controls[5].get_mapped())
        choose(d.cells[1][2], VOCODER_HOLD)
        self.assertFalse(panel.source.get_mapped())
        self.assertTrue(panel.controls[1].get_mapped())
        self.assertEqual(panel.labels[0].get_text(), "Dry / Wet")
        self.assertFalse(d.tal_panels[1].get_mapped())
        self.assertEqual(panel.controls[0].get_value(), 82)
        choose(d.cells[0][2], CHORD_TUNE)
        harmony = d.vocoder_panels[0]
        harmony.menu.get_popover().popup()
        self.assertEqual(harmony.labels[0].get_text(), "Harmony level")
        self.assertFalse(harmony.controls[1].get_mapped())
        self.assertFalse(harmony.controls[7].get_mapped())
        harmony.menu.get_popover().popdown()
        choose(d.cells[2][2], TAL_VOCODER)
        choose(d.cells[3][2], TUNELET)
        d.window.destroy()

        restored = self.open_desk()
        self.assertEqual([cell[2].get_label() for cell in restored.cells],
                         ["Chord Tune (Chorale VST3)", "Vocoder Hold", "TAL Vocoder", "Chord Tune (Tunelet)"])
        panel = restored.vocoder_panels[1]
        self.assertTrue(panel.controls[0].get_mapped())
        self.assertEqual(panel.preset.get_active_id(), "Custom")
        self.assertEqual(panel.controls[0].get_value(), 82)
        self.assertEqual(panel.controls[7].get_value(), 0)
        self.assertEqual(panel.controls[6].get_active_id(), "-1")
        self.assertEqual(panel.width.get_value(), 34)
        self.assertEqual(restored.strips[0]["lead"], 70)
        self.assertEqual(restored.strips[1]["lead"], 42)
        panel.preset.set_active_id("Smooth vowels")
        self.assertEqual(restored.strips[1]["vocoder"], VOCODER_PRESETS["Smooth vowels"])
        panel.reset.clicked()
        self.assertEqual(panel.preset.get_active_id(), "Balanced")
        self.assertEqual(restored.strips[1]["vocoder"], strip(None)["vocoder"])
        self.assertEqual(panel.width.get_value(), 100)
        choose(restored.cells[1][2], TAL_VOCODER)
        self.assertTrue(panel.source.get_mapped())
        self.assertEqual(panel.source.get_value(), 63)
        choose(restored.cells[1][2], 0)
        self.assertFalse(panel.inline.get_mapped())
        self.assertTrue(panel.menu.get_mapped())

    def test_all_tal_parameters_are_reachable_and_restore_per_mic(self):
        d = self.desk
        choose(d.cells[0][2], TAL_VOCODER)
        menu = d.tal_panels[0]
        self.assertTrue(menu.get_visible())
        self.assertTrue(menu.get_mapped())
        by_name = {p.name.decode(): i for i, p in enumerate(menu.params)}
        self.assertEqual(len(menu.midi), 16*130)
        self.assertEqual(set(menu.sliders) | set(menu.midi.values()), set(range(len(menu.params))))
        self.assertEqual(len(menu.params)-len(menu.midi), 38)
        for band in range(11):
            i = by_name[f"vocoderband{band:02}"]
            self.assertEqual(menu.sliders[i].get_accessible().get_name(), f"Band {band+1}")
        volume = by_name["volume"]
        self.assertIsInstance(menu.sliders[volume], Dial)
        self.assertTrue(menu.sliders[volume].get_mapped())
        route = by_name["inputmode"]
        menu.sliders[volume].set_value(.23)
        menu.sliders[route].set_active(0)
        self.assertEqual(d.strips[1]["tal_params"], {})
        d.window.destroy()
        restored = self.open_desk()
        menu = restored.tal_panels[0]
        self.assertTrue(menu.get_mapped())
        self.assertAlmostEqual(menu.sliders[volume].get_value(), .23, places=3)
        self.assertEqual(menu.sliders[route].get_active(), 0)
        menu.reset()
        self.assertEqual(restored.strips[0]["tal_params"], {})
        self.assertEqual(menu.sliders[route].get_active(), 2)
        choose(restored.cells[0][2], CHORD_TUNE)
        self.assertFalse(menu.get_visible())

    def test_pane_and_local_presets_recall_across_mics_and_relaunch(self):
        d = self.desk
        source, dest = d.preset_controls[:2]
        d.cells[1][0].assign("words")
        d.cells[0][0].assign("notes")
        d.audio.version_labels["notes"] = {"all": "All parts", "focus": "Focused part"}
        d.configure()
        midi = d.midi_controls[0]
        self.assertTrue(midi.get_mapped())
        self.assertEqual(midi.get_active_id(), "all")
        midi.set_active_id("focus")
        choose(d.cells[0][2], TAL_VOCODER)
        tal = d.tal_panels[0]
        volume = next(i for i, p in enumerate(tal.params) if p.name == b"volume")
        dial = tal.sliders[volume]
        dial.set_value(.357)
        dial.emit("popup-menu")
        self.assertEqual(source.scope.get_active_id(), f"tal.{tal.params[volume].id}")
        source.name.set_text("Quiet output")
        source.write.clicked()
        source.scope.set_active_id("pane")
        source.name.set_text("Wide robot")
        source.write.clicked()
        source.get_popover().popdown()
        dest.get_popover().popup()
        dest.scope.set_active_id("pane")
        dest.names.set_active_id("Wide robot")
        dest.recall.clicked()
        self.assertEqual(d.strips[1]["lyrics"], "words")
        self.assertEqual(d.strips[1]["voice"], TAL_VOCODER)
        self.assertEqual(d.strips[1]["midi_version"], "focus")
        self.assertEqual(d.midi_controls[1].get_active_id(), "all")
        target = d.tal_panels[1]
        self.assertAlmostEqual(target.sliders[volume].get_value(), .357)
        target.sliders[volume].set_value(.95)
        d.strips[1]["lead"] = 23
        target.sliders[volume].emit("popup-menu")
        dest.names.set_active_id("Quiet output")
        dest.recall.clicked()
        self.assertAlmostEqual(target.sliders[volume].get_value(), .357)
        self.assertEqual(d.strips[1]["lead"], 23)
        self.assertEqual(d.strips[2]["tal_params"], {})
        d.window.destroy()
        fresh = self.open_desk()
        self.assertIn("Wide robot", fresh.session["presets"]["pane"])
        self.assertEqual(fresh.strips[0]["midi_version"], "focus")
        choose(fresh.cells[0][2], VOCODER_HOLD)
        self.assertFalse(fresh.tal_panels[0].get_visible())
        self.assertTrue(fresh.midi_controls[0].get_visible())

    def test_existing_vocoder_tweaks_survive_the_new_controls(self):
        from desk import STATE, settings
        state = self.desk.session
        state["strips"][0]["vocoder"] = [55, 50, 2, 16]
        state["strips"][1]["vocoder"] = [100, 100, 15, 300]
        state["strips"][2]["vocoder"] = [86, 60, 8, 45, 2, 65, -1]
        STATE.write_text(json.dumps(state))
        restored = settings()["strips"]
        self.assertEqual(restored[0]["vocoder"], strip(None)["vocoder"])
        self.assertEqual(restored[1]["vocoder"], [100,100,15,300,0,55,0,100])
        self.assertEqual(restored[2]["vocoder"], [86,60,8,45,2,65,-1,100])

    def test_picker_changes_only_one_voice_with_a_score_and_linked_mics(self):
        d = self.desk
        d.cells[0][1].assign("notes")
        d.cells[0][0].assign("words")
        d.set_song_assignment(1, "anchor", "mic:0")
        d.set_song_assignment(2, "anchor", "mic:1")
        d.configure()
        controls = deepcopy(d.strips)
        expected = [0] * 4
        # Include repeated selections and both directions of the reported transition.
        for lane, voice in [(1, 1), (1, 3), (0, 1), (2, 2), (1, 3), (1, 1), (0, 0), (1, 0)]:
            with self.subTest(lane=lane, voice=VOICES[voice]):
                choose(d.cells[lane][2], voice)
                expected[lane] = voice
                self.assert_voices(d, expected)
                self.assertEqual(d.strips, [dict(strip, voice=v) for strip, v in zip(controls, expected)])

    def test_assignment_follow_backing_and_relaunch_preserve_explicit_voices(self):
        d = self.desk
        voices = [3, 0, 1, 2]
        for i, v in enumerate(voices):
            choose(d.cells[i][2], v)
        d.set_song_assignment(1, "anchor", "mic:0")
        d.set_song_assignment(2, "anchor", "mic:1")
        for key in ("notes", "words", "notes"):
            d.cells[0][1].assign(key)
            d.cells[1][0].assign(key)
            d.source_buttons[2].clicked()
            d.mix_controls[2].set_active_id("mute")
            d.mix_controls[2].set_active_id("stripped")
            d.install()
            d.tick()
            self.assertEqual(d.playing_slot, 2)
            self.assertEqual(d.selected_mix[2], "stripped")
            self.assert_voices(d, voices)
            self.assertEqual(d.strips[1]["anchor"], "mic:0")
            self.assertEqual(d.strips[2]["anchor"], "mic:1")
        d.window.destroy()
        restored = self.open_desk()
        self.assert_voices(restored, voices)
        self.assertTrue(all(s["lyrics"] is None for s in restored.strips))
        restored.cells[1][1].assign("notes")
        self.assert_voices(restored, voices)
