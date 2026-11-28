"""Listen to live Homer on the operator’s test strip via an independent local ASR."""

from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

from desk import Desk
from desk_audio import DeskAudio
from gi.repository import GLib
from homer_monitor_hardware import render_paced


class HardwareSpeech(unittest.TestCase):
    def test_homer_sings_the_timed_words_from_the_operator_button(self):
        evidence = Path("build/evidence/homer-speech").resolve()
        evidence.mkdir(parents=True, exist_ok=True)
        lines = [
            ("Turn the lights down low.", [300, 570, 685, 1000, 1260, 1800]),
            ("Give the drummer one more beat.", [2200, 2420, 2530, 2940, 3150, 3400, 3950]),
            ("Wait for the count, then bring it back.", [4300, 4620, 4780, 4890, 5300, 5490, 5750, 5850, 6350]),
            ("Meet me by the station.", [7000, 7250, 7500, 7750, 7950, 8500]),
            ("We can take the early train.", [8900, 9130, 9300, 9560, 9700, 10100, 10650]),
            ("Grab your coat before we go.", [11000, 11350, 11600, 12000, 12300, 12500, 13050]),
        ]
        rows = [[text.split(), list(zip(edges, edges[1:]))] for text, edges in lines]
        song = dict(id="Rehearsal", title="Rehearsal", label="Rehearsal", track="Rehearsal",
                    start=0, end=14, bpm=120, loop=True, rows={"lead": rows})
        with tempfile.TemporaryDirectory() as tmp:
            bed = Path(tmp) / "backing.wav"
            with wave.open(str(bed), "wb") as wav:
                wav.setparams((1, 2, 48000, 0, "NONE", ""))
                wav.writeframes(bytes(14 * 48000 * 2))
            with patch("desk.catalog", return_value={song["id"]: song}), \
                    patch("test_singing.spoken_word", side_effect=AssertionError("unexpected Flite fallback")), \
                    patch("desk.STATE", Path(tmp) / "state"), \
                    patch("desk.playback_files", return_value=(bed, bed)), patch.object(DeskAudio, "resources"):
                desk = Desk()
                self.addCleanup(desk.window.destroy)
                self.addCleanup(desk.close)
                desk.open_audio = lambda: True  # Off-device native render, no speaker or mic connection.
                desk.cells[0][0].assign(song["id"])
                desk.strips[0].update(voice=3, compressor=False, reverb=0, pulse=False, delay=False)
                desk.install()
                desk.test.refresh()
                t = time.monotonic()
                desk.test.buttons[0].clicked()
                loop = GLib.MainContext.default()
                while not desk.audio.call("playing") and time.monotonic() - t < 120:
                    while loop.pending():
                        loop.iteration(False)
                    desk.test.refresh()
                    time.sleep(.025)
                elapsed = time.monotonic() - t
                self.assertEqual(desk.test.messages[0], "Test vocal")
                self.assertIn((song["id"], "auto"), desk.test.dry_sources)
                self.assertEqual(desk.worker_status[0], "GPU live")
                self.assertTrue(desk.audio.call("playing"))
                left, _ = render_paced(desk.audio, np.zeros(14 * 48000, "f4"))
                converted = desk.audio.call("wet", 0)
                held = desk.workers[0].held
                self.assertGreater(converted, 680)
                self.assertLess(held, .1 * converted)
                capture = evidence / "homer.wav"
                with wave.open(str(capture), "wb") as wav:
                    wav.setparams((1, 2, 48000, 0, "NONE", ""))
                    wav.writeframes(np.clip(left * 32768, -32768, 32767).astype("<i2").tobytes())

        asr = evidence / "asr.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", capture, "-ar", "16000", asr], check=True)
        prefix = evidence / "recognized"
        subprocess.run([os.environ.get("KARAOKE_WHISPER_CLI", "whisper-cli"), "-m", os.environ["KARAOKE_WHISPER_MODEL"],
                        "-l", "en", "-t", "4", "-bs", "3", "-nt", "-otxt", "-of", prefix, asr], check=True)
        expected = re.findall(r"[a-z]+", " ".join(text for text, _ in lines).lower())
        heard = re.findall(r"[a-z]+", prefix.with_suffix(".txt").read_text().lower())
        match = SequenceMatcher(None, expected, heard).ratio()
        report = dict(seconds_of_audio=len(left) / 48000, button_to_play_seconds=elapsed,
                      live_packets=converted, held_packets=held,
                      word_sequence_match=round(match, 3), recognized=" ".join(heard))
        (evidence / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report)
        self.assertGreaterEqual(match, .90, report)


if __name__ == "__main__":
    unittest.main()
