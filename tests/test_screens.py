from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview import present_desk


class ScreenPlacement(unittest.TestCase):
    def test_single_monitor_opens_only_the_operator_window(self):
        desk = Gtk.Window()
        stage = Gtk.Window()
        self.addCleanup(stage.destroy)
        self.addCleanup(desk.destroy)
        app = Mock(window=desk, stage=stage, show_stage=stage.show_all)
        desk.show_all()
        present_desk(app)
        self.assertTrue(desk.get_visible())
        self.assertFalse(stage.get_visible())

    def test_audience_uses_the_other_monitor_when_operator_is_on_the_second(self):
        app = Mock()
        display = app.window.get_display.return_value
        screens = [object(), object()]
        display.get_monitor_at_window.return_value = screens[1]
        display.get_n_monitors.return_value = 2
        display.get_monitor.side_effect = screens.__getitem__
        present_desk(app)
        app.stage.fullscreen_on_monitor.assert_called_once_with(app.window.get_screen(), 0)
        self.assertTrue(app.stage.full)
        app.show_stage.assert_called_once()
        app.window.present.assert_called_once()
