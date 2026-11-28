import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import Audio, FP
from desk import Desk, Gdk
from import_job import publish
from playback import playback_files


def write_tones(path, seconds, frequencies):
    t = np.arange(round(seconds * 48000)) / 48000
    signal = sum(2000 * np.sin(2 * np.pi * f * t) for f in frequencies)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 48000, 0, "NONE", ""))
        stream.writeframes(np.repeat(signal.astype("<i2"), 2).tobytes())


def tones(pcm):
    peaks = abs(np.fft.rfft(pcm))
    return {f for f in (90, 220, 700) if peaks[round(f * len(pcm) / 48000)] > peaks.max() * .2}


class Playback(unittest.TestCase):
    def test_import_and_play_each_window_on_the_real_mixer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, frequencies in [("backing", [90, 220]), ("guide", [700]), ("drums", [90])]:
                write_tones(root / (name + ".wav"), 8, frequencies)
            meta = dict(title="Test", artist="Singer", key="C major", bpm=120, meter=4,
                        review="Fixture", downbeats=[0, 2, 4, 6, 8],
                        rows=[[["Hey"], [[500, 900]]]],
                        teaser=dict(label="Hey", start_bar=0, bars=4, voice_ms=[400, 1000]),
                        hooks=[dict(label="Hook", start_bar=0, bars=4, voice_ms=[0, 8000])])
            (root / "analysis.json").write_text(json.dumps(meta))
            songdir = publish(root, root / "backing.wav", root / "songs", 4)
            record = json.loads((songdir / "song.json").read_text())
            songs = {}
            for cut in record["cuts"]:
                key = cut["id"]
                songs[key] = dict(record, **cut, track=key, title="Test", artist="Singer",
                                  bed=songdir / cut["audio"] / "backing.wav")
                for path in playback_files(songs[key]):
                    with wave.open(str(path)) as wav:
                        self.assertEqual(wav.getnframes(), round(cut["end"] * 48000))

            with patch("desk.catalog", return_value=songs), patch("desk.STATE", root / "session"), patch.object(Audio, "resources"):
                desk = Desk()
                self.addCleanup(desk.window.destroy)
                try:
                    desk.open_audio = lambda: True
                    space = Gdk.Event.new(Gdk.EventType.KEY_PRESS)
                    space.keyval = Gdk.KEY_space
                    space.state = Gdk.ModifierType(0)

                    def render():
                        left, right = np.zeros((2, 48000), "f4")
                        desk.audio.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
                        return tones(left)

                    for i in range(4):
                        lyrics = desk.cells[i][0]
                        self.assertTrue(lyrics.assign("hook-1"))
                        desk.source_buttons[i+1].clicked()
                        self.assertEqual([j for j,b in desk.source_buttons.items() if b.get_active()], [i+1])
                    self.assertEqual(len(desk.source_buttons), 4)
                    desk.install()
                    desk.play.grab_focus()
                    desk.window.emit("key-press-event", space)
                    self.assertTrue(desk.audio.call("playing"))
                    render()
                    pos = desk.elapsed()
                    desk.mix_controls[4].set_active_id("stripped")
                    desk.install()
                    self.assertEqual(desk.elapsed(), pos)
                    self.assertEqual(render(), {90})
                    desk.mix_controls[2].set_active_id("mute")
                    self.assertEqual(render(), {90})
                    pos = desk.elapsed()
                    desk.mix_controls[4].set_active_id("mute")
                    self.assertEqual(render(), set())
                    self.assertAlmostEqual(desk.elapsed(), pos + 1)
                    desk.mix_controls[4].set_active_id("stripped")
                    self.assertEqual(render(), {90})
                    desk.stage.emit("key-press-event", space)
                    self.assertFalse(desk.audio.call("playing"))
                    desk.jobs.source.grab_focus()
                    self.assertFalse(desk.transport_key(desk.window, space))
                    desk.cells[0][0].assign("full")
                    previous_time = desk.elapsed()
                    desk.source_buttons[1].clicked()
                    desk.mix_controls[1].set_active_id("full")
                    desk.install()
                    self.assertAlmostEqual(desk.elapsed(), previous_time * 126 / 120, places=4)
                    desk.play.clicked()
                    self.assertEqual(render(), {90, 220})
                    desk.seek.set_value(7.9)
                    self.assertEqual(render(), {90, 220})
                    self.assertTrue(desk.audio.call("playing"))
                finally:
                    desk.close()
