"""Check the acoustic harmony in the rendered backing and the desk's three vocal cuts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
import library
from backing_loops import backing_loops, calibrated_catalog
from carrier import read_score
from desk import Desk
from desk_audio import DeskAudio
from hang_with_me_acoustic import BEAT, CUTS, EDITION, RATE, RECORDING, TITLE, instruments, prepare
from playback import playback_files


class HangWithMeAcoustic(unittest.TestCase):
    def test_chorus_voicings_and_steady_pulse_in_the_audio(self):
        backing, drums, events = instruments(28, 37)
        notes = (backing-drums).mean(axis=1)

        def level(beat, pitch):
            start = round((beat + .2)*BEAT*RATE)
            y = notes[start:start+round(.6*BEAT*RATE)]
            phase = 2*np.pi*440*2**((pitch-69)/12)*np.arange(len(y))/RATE
            return abs(np.dot(y*np.hanning(len(y)), np.exp(-1j*phase)))/len(y)

        # A, Bm, G, D and the two-bar F# dominant. Its raised third
        # distinguishes this turnaround from a diatonic F# minor chord.
        for beat, bass, third in [(0,45,61), (2,47,62), (4,43,59), (6,38,54), (28,42,58), (34,42,58)]:
            self.assertGreater(level(beat, bass), .014)
            self.assertGreater(level(beat, third), .01)
        self.assertGreater(level(34,58), 12*level(34,57))
        self.assertIn([round(28*BEAT, 6), [42,52,58,61]], events)
        # Even through the later vocal break there is a downstroke every quarter.
        backing, drums, _ = instruments(64, 66)
        for beat in range(8):
            pos = round(beat*BEAT*RATE)
            self.assertGreater(drums[pos+240:pos+1600,0].std(), .07)
            self.assertLess(drums[pos+5900:pos+7000,0].std(), .04)
        np.testing.assert_array_equal(backing[round(2*BEAT*RATE):round(6*BEAT*RATE)],
                                      drums[round(2*BEAT*RATE):round(6*BEAT*RATE)])

    def test_library_buttons_use_their_own_audio_score_and_lyric_clock(self):
        # Fixture takes on the acoustic recording clock, with the previous
        # phrase extending into Main's first bar and the next verse after Teaser.
        rows = [(["Phrase"], [[i, i+100]]) for i in np.linspace(5508, 186000, 48, dtype=int).tolist()]
        rows[0] = (["Opening"], [[5508,8220]])
        rows[1] = (["Next"], [[10168,12852]])
        rows[15] = (["Pickup"], [[62929,64510]])
        times = [(64928,67490), (67848,71160), (71548,73500), (74568,76730),
                 (77148,80505), (81309,82692), (84589,91720)]
        for i, (a,b) in enumerate(times,16):
            rows[i] = (["Chorus" if i==16 else "Response"], [[a,b]])
        # One sustained tone exposes ungated neighboring vocals as well as
        # incorrect file offsets. The mix itself is rendered by the real synth.
        guide = np.full(192*RATE, .125, "f4")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / EDITION
            target.mkdir()
            (root/"album.info.json").write_text(json.dumps({"id":RECORDING}))
            with patch("hang_with_me_acoustic.source_rows", return_value=rows), \
                    patch("hang_with_me_acoustic.decode", return_value=guide):
                prepare(root, target)
            full_score = read_score(target / "full/backing.wav")
            self.assertTrue(full_score["rendered"])
            def pitches(beat):
                t = beat * BEAT * 1000
                return {key for a, b, key in full_score["notes"] if a <= t < b}
            self.assertEqual(pitches(.5), {47, 54, 59, 62})
            self.assertEqual(pitches(2.5), {45, 52, 57, 61})
            self.assertEqual(pitches(64 * 4 + 4), set())
            with patch.object(library, "IMPORTS", root):
                # The downloadable runtime starts with no bundled recordings.
                # Exercise adding the acoustic edition beside the studio edition there too.
                studio = dict(id="Hang with Me", title="Hang with Me", artist="Robyn",
                              label="Hang with Me", bpm=115, beat_zero=2.812, start=0, end=0, loop=False)
                songs = calibrated_catalog({studio["id"]: studio, **library.catalog()})
            full = EDITION + "/full"
            choices = backing_loops(songs)
            self.assertIn("Hang with Me", choices)
            self.assertIn(full, choices)
            with patch("desk.catalog", return_value={k:v for k,v in songs.items() if v["title"]==TITLE}), \
                    patch("desk.STATE", root/"session"), patch.object(DeskAudio, "resources"):
                desk = Desk()
                try:
                    self.assertEqual(list(desk.library.items), [full])
                    desk.cells[0][0].assign(full)
                    for role in ("teaser", "main", "full"):
                        desk.loop_controls[1][0][role].clicked()
                        desk.install()
                        song = songs[desk.test.song_key(0)]
                        self.assertEqual(song["id"], EDITION+"/"+role)
                        first, stop, *_ = CUTS[role]
                        self.assertAlmostEqual(desk.duration, (stop-first)*4*BEAT, delta=1/RATE)
                        self.assertTrue(desk.audio.call("has_carrier", song["id"].encode()))
                        for path in (*playback_files(song), song["guide"]):
                            with wave.open(str(path)) as wav:
                                self.assertEqual(wav.getnframes(), round(desk.duration*RATE))
                        when = song["rows"]["lead"][0][1][0][0]/1000+.05
                        for lap in (0, desk.duration):
                            lines, _ = desk.stage.panes[0].lines_at(song, desk.backing, lap+when, None)
                            self.assertEqual(lines[0][2][0], "Chorus" if role=="main" else "Opening")
                        if role != "full":
                            with wave.open(str(song["guide"])) as wav:
                                pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
                            self.assertEqual(abs(pcm[:RATE//4]).max(), 0)
                            self.assertEqual(pcm[round(when*RATE)], 4096)
                            self.assertEqual(abs(pcm[-RATE//2:]).max(), 0)
                finally:
                    desk.window.destroy()
