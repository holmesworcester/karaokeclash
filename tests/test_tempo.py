"""Render clips, singer audio and carrier scores against a single editable room tempo."""
import csv
import ctypes as C
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audience import Pane, grid_position
from backing_loops import backing_loops, calibrated_catalog
from clip_clock import ClipClock, launch_origin
from desk import TUNELET, Desk
from desk_audio import DeskAudio, FP
from audio import PARAMS
from library import catalog
from playback import playback_files

RATE = 48000


def render(audio, n):
    left, right = np.zeros((2, n), dtype="f4")
    audio.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), n)
    return left, right


def write_wav(path, pcm):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, RATE, 0, "NONE", ""))
        wav.writeframes(pcm.astype("<i2").tobytes())


class RoomTempo(unittest.TestCase):
    def test_get_low_over_atc_words_and_measured_kicks_share_the_next_downbeat(self):
        # Drum-stem attacks and lyric alignments are independent recording landmarks.
        cases = [("Get Low", 101, .6411485, 334.0667, 29.499, 96.032, 29.180, "To"),
                 ("Around the World", 126, 1.0064, 230.476, 23.864, 25.772, 23.831, "La")]
        for name, bpm, zero, duration, first_kick, later_kick, word_time, word in cases:
            with self.subTest(song=name):
                s = calibrated_catalog({name: dict(id=name, track=name, start=0., end=duration,
                    loop=False, bpm=bpm, beat_zero=zero, rows={"lead": [
                        ([word], [[round(word_time*1000), round((word_time+.4)*1000)]])]})})[name]
                regions = backing_loops({name: s})[name]
                teaser = regions["teaser"].region(s, duration)
                main = regions["main"].region(s, duration)
                clock = ClipClock(teaser)
                for r, kick in [(teaser, first_kick), (main, later_kick)]:
                    beat = ClipClock(r).bar(kick-r["start"])*4
                    self.assertAlmostEqual(beat, 4*round(beat/4), delta=.025)
                self.assertAlmostEqual(clock.bar(first_kick-teaser["start"]), 1, delta=.005)
                # Render the recorded kick timing at room tempo and the sung pickup
                # through a separate native pane. Stereo separates their evidence.
                region = dict(teaser, start=0., end=teaser["end"]-teaser["start"], beat_zero=0.)
                region["vocal_window"] = tuple(t-teaser["start"] for t in teaser["vocal_window"])
                region["loop"] = True
                frames = round(region["end"]*RATE)
                stem = np.zeros((frames, 2), dtype="i2")
                kick_at = round((first_kick-teaser["start"])*RATE)
                stem[kick_at:kick_at+2200, 1] = 3000
                singer = np.zeros(frames, dtype="i2")
                word_at = round((word_time-teaser["start"])*RATE)
                singer[word_at:word_at+9500] = 1600
                a = DeskAudio()
                self.addCleanup(a.close)
                a.call("tempo", 120)
                a.prepare_track(stem.ravel(), None, region)
                a.call("stage_clock", *a.clock_args(ClipClock(region)))
                a.call("launch_at", -1, 0)
                a.test_source(0, singer, region, region)
                a.call("pane_clock", 0, float("inf"), *a.clock_args(ClipClock(region), name),
                       *a.clock_args(None))
                a.call("test_lanes", 1)
                a.call("transport", 0, 1)
                x, y = render(a, round(2.2*RATE))
                kick = np.flatnonzero(y-x > .02)[0]/RATE
                pickup = np.flatnonzero(x > .015)[0]/RATE
                self.assertAlmostEqual(kick, 2., delta=.025)
                self.assertAlmostEqual(pickup, clock.bar(word_time-teaser["start"])*2, delta=.025)
                pane = Pane(None, 0)
                self.addCleanup(pane.destroy)
                rows, _ = pane.lines_at(s, teaser, clock.seconds((pickup+.035)/2), None)
                self.assertEqual(list(rows[0][2]), [word])
                self.assertAlmostEqual(clock.bar(a.call("clock")/RATE), 1.1, delta=.00002)

    def test_installed_get_low_atc_and_2fly_drum_attacks_at_120_bpm(self):
        from scipy.io import wavfile
        from scipy.signal import butter, sosfilt
        songs = calibrated_catalog(catalog())
        families = backing_loops(songs)
        keys = ("Get Low", "Around the World", "2Fly")
        if not all(key in families and playback_files(songs[key])[1].exists() for key in keys):
            self.skipTest("Studio drum stems are available on the provisioned Linux desk")
        for key in keys:
            for mode in ("teaser", "main"):
                with self.subTest(song=key, mode=mode):
                    item = families[key][mode]
                    song = songs[item.source]
                    hz, pcm = wavfile.read(playback_files(song)[1], mmap=True)
                    self.assertEqual(hz, RATE)
                    r = item.region(song, len(pcm)/RATE)
                    start, end = [round(r[k]*RATE) for k in ("start", "end")]
                    cut = dict(r, start=0, end=(end-start)/RATE, beat_zero=0)
                    a = DeskAudio()
                    try:
                        a.prepare_track(pcm[start:end].ravel(), None, cut)
                        a.call("stage_clock", *a.clock_args(ClipClock(cut)))
                        a.call("tempo", 120)
                        a.call("launch_at", -1, 0)
                        a.call("transport", 0, 1)
                        left, _ = render(a, round(6.2*RATE))
                    finally:
                        a.close()
                    kicks = sosfilt(butter(2, (70,350), fs=RATE, btype="bandpass", output="sos"), left)
                    for boundary in (2, 4, 6):
                        window = kicks[round((boundary-.085)*RATE):round((boundary+.11)*RATE)]
                        env = np.sqrt(np.mean(window[:len(window)//96*96].reshape(-1,96)**2, axis=1))
                        attack = np.flatnonzero(env > env.max()*.2)[0]*.002 - .085
                        self.assertAlmostEqual(attack, 0, delta=.025)

    def test_global_tempo_edit_keeps_a_pending_fractional_handoff_and_pitch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            songs, paths = {}, {}
            for name, bpm, pitch in [("A", 120, 330), ("B", 100, 440)]:
                bar = 240/bpm
                sample = np.arange(round(6*bar*RATE))
                pcm = np.column_stack((2000*np.sin(2*np.pi*pitch*sample/RATE),
                                       1000*(1+np.floor(sample/(bar*RATE)))))
                path = root/(name+".wav")
                write_wav(path, pcm)
                paths[name] = path, path
                songs[name] = dict(id=name, track=name, title=name, label=name, start=0.,
                                   end=6*bar, bpm=bpm, loop=True, rows={}, mainBars=4)
            with patch("desk.STATE", root/"state"), patch("desk.catalog", return_value=songs), \
                 patch("desk.playback_files", side_effect=lambda s: paths[s["id"]]), \
                 patch.object(DeskAudio, "resources"):
                d = Desk()
                self.addCleanup(d.window.destroy)
                d.open_audio = lambda: True
                d.tempo.set_value(120)
                d.cells[0][0].assign("A")
                d.install()
                d.cells[1][0].assign("B")
                d.quant.set_active_id(".5")
                d.choose_loop(1, "main")
                d.play.clicked()
                render(d.audio, RATE//4)
                d.choose_loop(2, "teaser")
                d.source_buttons[2].clicked()
                d.install()
                self.assertAlmostEqual(d.actions[2][3], .5)
                self.assertAlmostEqual(d.conductor_bar(), .125)
                d.tempo.set_value(150)
                self.assertAlmostEqual(d.conductor_bar(), .125)
                # .375 bar at the new tempo. Switching to B must not change 150 BPM.
                frames = round(.375*240/150*RATE)
                render(d.audio, frames-1)
                d.sync_launch()
                self.assertEqual(d.playing_slot, 1)
                render(d.audio, 2)
                d.sync_launch()
                self.assertEqual(d.playing_slot, 2)
                self.assertEqual(d.tempo.get_value(), 150)
                self.assertAlmostEqual(d.elapsed(), 1.2, delta=2/RATE)
                x, y = render(d.audio, RATE)
                spectrum = abs(np.fft.rfft(x*np.hanning(RATE)))
                self.assertAlmostEqual(np.argmax(spectrum), 440, delta=1)
                # The first recording bar ends .8 seconds after the mid-bar handoff.
                onset = np.flatnonzero(y > 1500/32768)[0]/RATE
                self.assertAlmostEqual(onset, .8, delta=.025)
                self.assertAlmostEqual(d.conductor_bar(), 1.125, delta=.00003)
                d.flush()
                from desk import settings
                self.assertEqual(settings()["bpm"], 150)

    def test_native_transport_warps_each_measured_bar_and_preserves_wrap_phase(self):
        a = DeskAudio()
        self.addCleanup(a.close)
        grid = [0., 1.75, 4.06, 5.8]
        song = dict(track="drifting band", start=0., end=grid[-1], downbeats=grid, bpm=120)
        stem = np.zeros((round(grid[-1]*RATE), 2), dtype="i2")
        for i, start in enumerate(grid[:-1]):
            stem[round(start*RATE):round(grid[i+1]*RATE)] = 1000*(i+1)
        a.prepare_track(stem.ravel(), None, song)
        a.call("stage_clock", *a.clock_args(ClipClock(song)))
        a.call("launch_at", -1, 0)
        a.call("tempo", 150)
        a.call("transport", 0, 1)
        for lap in range(2):
            for i in range(3):
                signal, _ = render(a, round(1.6*RATE))
                self.assertAlmostEqual(np.median(signal)*32768, 1000*(i+1), delta=1)
                elapsed = a.call("clock")/RATE
                self.assertAlmostEqual(elapsed, lap*grid[-1]+grid[i+1], delta=1/RATE)

    def test_harmonizer_follows_leaders_note_changes_on_the_global_beat(self):
        a = DeskAudio()
        self.addCleanup(a.close)
        # A 100 BPM backing, a 120 BPM carrier and a 150 BPM room. Red's
        # half-bar skip at the first room beat should take Blue to C5.
        song = dict(track="slow", start=0., end=4.8, loop=True, bpm=100)
        leader = dict(song, track="score", end=4., bpm=120)
        a.prepare_track(np.zeros(round(4.8*RATE)*2, dtype="i2"), None, song)
        a.call("stage_clock", *a.clock_args(ClipClock(song)))
        a.call("launch_at", -1, 0)
        a.call("pane_clock", 0, .25, *a.clock_args(ClipClock(leader)),
               *a.clock_args(ClipClock(leader, -.5)))
        a.call("note", b"carrier", 0., .1, 69)
        a.call("note", b"carrier", 1.5, 1.6, 72)
        a.call("score", b"carrier", 4., None, 0)
        a.call("carrier_clock", 1, 0, 120.)
        a.call("config", 1, TUNELET, b"carrier", 0, 0,
               (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS)))
        a.call("live", 1, 1)
        a.call("tempo", 150)
        a.call("transport", 0, 1)
        axis = np.arange(RATE)/RATE
        tone = (.1*np.sin(2*np.pi*200*axis)).astype("f4")
        inputs = (FP * 4)()
        inputs[1] = tone.ctypes.data_as(FP)
        x, y = np.zeros((2, RATE), dtype="f4")
        a.call("render", inputs, x.ctypes.data_as(FP), y.ctypes.data_as(FP), RATE)
        for start, end, expected in [(.15, .35, 440), (.46, .76, 523.25)]:
            v = x[round(start*RATE):round(end*RATE)]
            spectrum = abs(np.fft.rfft(v*np.hanning(len(v)), n=RATE))
            self.assertAlmostEqual(np.argmax(spectrum), expected, delta=2)

    def test_legacy_prepared_main_discards_partial_bars_at_both_ends(self):
        from backing_loops import Loop
        song = dict(start=0., end=8.3, loop=True, bpm=120, downbeats=[.15, 2.12, 4.14, 6.2, 8.19])
        region = Loop("Main", "legacy", main=True).region(song, 8.3)
        self.assertEqual((region["start"], region["end"]), (.15, 8.19))
        self.assertAlmostEqual(ClipClock(region).bar(10*(8.19-.15)), 40)

    def test_all_installed_teaser_and_main_edges_follow_the_source_grid(self):
        # The redistributable runtime contains no studio recordings. Use their
        # WAV lengths and source clocks there, then overlay every installed
        # edition (including private imports) when the media is available.
        reference, lengths = {}, {}
        with Path(__file__).with_name("fixtures").joinpath("catalog-clocks.csv").open() as f:
            for row in csv.DictReader(f):
                key, family = row["edition"], row["family"]
                song = dict(id=key, title=family, label=key, track=key, loopFamily=family,
                            start=float(row["start"]), end=float(row["end"]), loop=bool(int(row["loop"])),
                            beat_zero=float(row["beat_zero"]), bpm=float(row["bpm"]))
                reference[key] = song
                lengths[key] = int(row["frames"])/RATE
        installed = catalog()
        songs = calibrated_catalog(reference | installed)
        loops = backing_loops(songs)
        for key, choices in loops.items():
            for mode in ("teaser", "main"):
                with self.subTest(song=key, loop=mode):
                    item = choices[mode]
                    song = songs[item.source]
                    duration = lengths.get(item.source)
                    if item.source in installed:
                        path = playback_files(song)[0]
                        if path.exists():
                            duration = Desk.wave_duration(path)
                    duration = duration or song["end"] or song["cycle"]
                    region = item.region(song, duration)
                    a, b = (grid_position(song, region[field]) for field in ("start", "end"))
                    self.assertAlmostEqual(a, round(a), delta=.00005)
                    self.assertAlmostEqual(b, round(b), delta=.00005)
                    self.assertGreaterEqual(round(b-a), 1)
                    for edge in (0, 7.25, 12.5):
                        clock = ClipClock(region, launch_origin(mode, edge))
                        length = region["end"] - region["start"]
                        count = round(b-a)
                        self.assertAlmostEqual(clock.bar(100*length), clock.origin+100*count, delta=.00005)
                        self.assertAlmostEqual(clock.bar(clock.seconds(edge)), edge, delta=.00005)
