"""Exercise stereo transcription, cache refresh and sample-clock routing through the native synth."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import F, FP, PARAMS
from backing_loops import calibrated_catalog
from carrier import SR, prepare_score, read_score, recover_chord_tones, transcription
from stand_by_me import save_orchestra_scores
from carrier_audit import source_confirmation
from clip_clock import ClipClock
from desk_audio import DeskAudio
from playback import catalog


def play_note(pcm, key, start, stop, level, pan, partials):
    a, b = round(SR*start), round(SR*stop)
    t = np.arange(b-a)/SR
    phase = 2*np.pi*440*2**((key-69)/12)*t
    env = np.minimum(1, np.minimum(t/.014, (t[-1]-t)/.035))*np.exp(-.5*t)
    tone = sum(gain*np.sin(n*phase) for n, gain in enumerate(partials, 1))
    pcm[a:b] += (env*tone*level)[:, None]*pan


def write_wav(path, pcm):
    with wave.open(str(path), "wb") as out:
        out.setparams((2, 2, SR, 0, "NONE", ""))
        out.writeframes(np.rint(32767*pcm).astype("<i2").tobytes())


def at(notes, seconds):
    return {k for a, b, k in notes if a <= seconds*1000 < b}


class PolyphonicCarrier(unittest.TestCase):
    def test_extended_chord_offbeat_attacks_rests_and_unaccompanied_bass(self):
        pcm = np.zeros((3*SR, 2), "f4")
        for key, level, pan in [(45, .18, [.9, 1]), (60, .12, [1, .4]), (64, .09, [.3, 1]),
                                (67, .08, [1, .3]), (71, .046, [1, -1])]:
            play_note(pcm, key, .20, 1.15, level, pan, [1, .44, .19, .10])
        for t in (.34, .67, .91):
            play_note(pcm, 74, t, t+.105, .13, [.35, 1], [1, .3])
        play_note(pcm, 50, 1.55, 2.55, .13, [1, .7], [1, .66, .50, .35, .28, .19])
        notes = transcription(pcm, 3)
        chord = {45, 60, 64, 67, 71}
        for t in (.39, .71, .96):
            self.assertEqual(at(notes, t), chord | {74})
        for t in (.52, .84):
            self.assertEqual(at(notes, t), chord)
        self.assertEqual(at(notes, 1.38), set())
        for t in (1.75, 2.14):
            self.assertEqual(at(notes, t), {50})
        self.assertEqual(len([k for _, _, k in notes if k == 74]), 3)

    def test_recovery_fills_the_missing_chord_voice_only_while_it_sounds(self):
        pcm = np.zeros((2*SR, 2), "f4")
        play_note(pcm, 48, .1, 1.9, .18, [1, 1], [1, .4, .2])
        play_note(pcm, 64, .4, 1.2, .09, [.4, 1], [1, .25])
        notes = recover_chord_tones(pcm, 2, [[100, 1900, 48]])
        density = recover_chord_tones(pcm, 2, [[100, 1900, 48]], density_only=True)
        self.assertEqual(at(density, .8), {48, 64})
        self.assertEqual(at(density, 1.6), {48})
        self.assertEqual(at(notes, .8), {48, 64})
        self.assertEqual(at(notes, .2), {48})
        self.assertEqual(at(notes, 1.6), {48})
        self.assertTrue(all(key in (48, 64) for _, _, key in notes))

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/"linux-four"/"starred-arrangements/dynamite.json").is_file(),
                         "requires the installed recording chart")
    def test_audit_checks_the_matching_source_bar_on_each_side_of_a_splice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sources/studio-dynamite"
            source.mkdir(parents=True)
            (source / "master.info.json").write_text(json.dumps({"id": "VUjdiDeJ0xg"}))
            pcm = np.zeros((63 * SR, 2), "f4")
            for t, key in [(48.274, 48), (52.2731, 53), (56.2727, 60), (60.2726, 66)]:
                play_note(pcm, key, t, t + 1.9, .12, [1, 1], [1, .35])
            write_wav(source / "instrumental.wav", pcm)
            bed = root / "songs/studio-dynamite/main/backing.wav"
            bar = 240 / 126
            mask = source_confirmation(bed, (np.arange(4) + .5) * bar, 4 * bar)
            self.assertEqual([np.flatnonzero(row).tolist() for row in mask], [[48], [53], [60], [66]])

    def test_drum_only_replacement_invalidates_the_cached_backing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pcm = np.zeros((SR, 2), "f4")
            play_note(pcm, 60, .1, .9, .18, [1, 1], [1, .3])
            backing = root / "backing.wav"
            write_wav(backing, pcm)
            self.assertIn(60, at(prepare_score(backing)["notes"], .5))
            write_wav(root / "drums.wav", pcm)
            self.assertIsNone(read_score(backing))
            self.assertEqual(prepare_score(backing)["notes"], [])

    def test_orchestra_keeps_pitched_tracks_and_warps_repeated_refrain(self):
        import mido
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            midi = mido.MidiFile(ticks_per_beat=480)
            # At 120 BPM the note begins 100 ms into source bar 22, held for 500 ms.
            for channel, pitch in [(0, 45), (1, 64), (2, 71), (9, 36)]:
                track = midi.add_track()
                track.append(mido.Message("note_on", channel=channel, note=pitch, time=22*1920+96))
                track.append(mido.Message("note_off", channel=channel, note=pitch, time=480))
            midi.save(root / "orchestra.mid")
            cuts = [dict(id="full", audio="full", downbeats=list(range(0, 90, 2))),
                    dict(id="hook-1", audio="hook-1", bpm=160)]
            for cut in cuts:
                (root / cut["audio"]).mkdir()
                write_wav(root / cut["audio"] / "backing.wav", np.zeros((SR, 2), "f4"))
            with patch("stand_by_me.wave_length", return_value=48000*90):
                save_orchestra_scores(root, cuts)
            full = read_score(root / "full/backing.wav")
            cut = read_score(root / "hook-1/backing.wav")
            self.assertEqual(at(full["notes"], 44.3), {45, 64, 71})
            self.assertEqual(len(cut["notes"]), 12)
            for repeat in range(4):
                start = repeat*9*1.5
                self.assertEqual(at(cut["notes"], start+.05), set())
                self.assertEqual(at(cut["notes"], start+.2), {45, 64, 71})
                self.assertEqual(at(cut["notes"], start+.48), set())

    def test_scoreless_catalog_follows_the_active_cut_at_the_queued_audio_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, cut = "Example", "example/cut"
            songs = {}
            for key, pitches in [(base, [50, 57, 65]), (cut, [54, 61, 70])]:
                folder = root / "private/linux-four/playback" / key
                folder.mkdir(parents=True)
                bed = folder / "backing.wav"
                pcm = np.zeros((2*SR, 2), "f4")
                for pitch in pitches:
                    play_note(pcm, pitch, 0, 2, .12, [1, 1], [1, .15])
                write_wav(bed, pcm)
                songs[key] = dict(id=key, track=key, title=base, bpm=120, start=0, end=2,
                                  loop=True, loopFamily=base, beat_zero=0, rows={})
            with patch("playback.runtime_catalog", return_value=songs), patch("playback.DATA", root):
                songs = calibrated_catalog(catalog())
            desk = DeskAudio()
            self.addCleanup(desk.close)
            desk.import_scores(songs)
            base_id = songs[base]["backingScoreId"].encode()
            self.assertTrue(desk.call("has_carrier", base_id))
            first, second = ClipClock(songs[base]), ClipClock(songs[cut], .5)
            desk.prepare_track(np.zeros(2*96000, "i2"), None, first.region)
            desk.call("stage_clock", *desk.clock_args(first))
            desk.call("stage_lane", 1)
            desk.call("stage_carrier_backing", True)
            desk.call("launch_at", -1, 0)
            desk.call("pane_clock", 0, .5, *desk.clock_args(first), *desk.clock_args(second))
            desk.call("carrier_clock", 1, 0, 120)
            desk.call("config", 1, 1, base_id, 0, 0, (F*len(PARAMS))(*(p[3] for p in PARAMS)))
            desk.call("glide", 1, 0)
            desk.call("tempo", 120)
            desk.call("transport", 0, 1)
            left, right = np.zeros((2, 96000), "f4")
            desk.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
            for start, pitches, other in [(.25, (50, 57, 65), (54, 61, 70)),
                                          (1.25, (54, 61, 70), (50, 57, 65))]:
                samples = left[round(start*48000):round((start+.3)*48000)]
                def level(key):
                    t = np.arange(len(samples))/48000
                    return abs(np.mean(samples*np.hanning(len(t))*np.exp(-2j*np.pi*t*440*2**((key-69)/12))))
                self.assertGreater(min(map(level, pitches)), 4*max(map(level, other)))
