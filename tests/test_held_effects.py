"""Measure held carrier notes and harmonized vocal pitches in the desk's stereo output."""
import ctypes as C
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import FP, PARAMS, SP
from desk import CHORD_TUNE, TUNELET, TAL_VOCODER, VOCODER_HOLD
from desk_audio import DeskAudio


class HeldEffects(unittest.TestCase):
    rate = 48000

    def load_fixture(self, audio, key, score_notes, recorded):
        # D4/F4/A4, a long rest, then C4 (and another rest before wrap).
        notes = [(0.10, 0.23, 62), (0.10, 0.21, 65), (0.10, 0.20, 69),
                 (0.55, 0.68, 60)]
        if score_notes is not None:
            notes = score_notes
        axis = np.arange(self.rate) / self.rate
        side = np.zeros(self.rate)
        for start, end, pitch in notes:
            if recorded:
                hz = 440 * 2 ** ((pitch-69)/12)
                side += .12 * np.sin(2*np.pi*hz*axis) * ((axis >= start) & (axis < end))
            else:
                audio.call("note", key, start, end, pitch)
        stereo = np.repeat(np.rint(side * 32768).astype("i2"), 2)
        audio.call("score", key, 1, stereo.ctypes.data_as(SP) if recorded else None,
                   len(stereo) if recorded else 0)

    def render(self, voice, recorded=False, seek=0, rehearsal=False, controls=None, input_signal=None,
               score_notes=None, load_score=None, key=b"chords", stereo=False):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        song = dict(track="bed", start=0, end=1, loop=True, bpm=120)
        if load_score is None:
            self.load_fixture(audio, key, score_notes, recorded)
        else:
            load_score(audio)
        audio.prepare_track(np.zeros(2*self.rate, "i2"), None, song)
        audio.call("launch_at", -1, 0)
        values = (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS))
        audio.call("config", 0, voice, key, 0, 0, values)
        if controls:
            for lane, knobs in controls.items():
                audio.call("vocoder", lane, *knobs[:4])
                if len(knobs) > 4:
                    audio.call("vocoder_shape", lane, *knobs[4:7])
                if len(knobs) > 7:
                    audio.call("glide", lane, knobs[7])
        t = np.arange(int(1.5*self.rate)) / self.rate
        vocal = (.075*np.sin(2*np.pi*190*t) + .025*np.sin(2*np.pi*380*t)).astype("f4")
        if input_signal is not None:
            vocal = input_signal(t).astype("f4")
        inputs = (FP * 4)()
        if rehearsal:
            pcm = np.rint(vocal[:self.rate] * 32768).astype("i2")
            audio.test_source(0, pcm, song, song)
            audio.call("test_lanes", 1)
        else:
            inputs[0] = vocal.ctypes.data_as(FP)
            audio.call("live", 0, 1)
        audio.call("transport", 1, round(seek*self.rate))
        audio.call("transport", 0, 1)
        left, right = np.zeros((2, len(t)), "f4")
        audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(t))
        if stereo:
            return left, right
        np.testing.assert_array_equal(left, right)
        return left

    def level(self, pcm, start, end, hz):
        segment = pcm[round(start*self.rate):round(end*self.rate)]
        t = np.arange(len(segment))/self.rate
        return abs(np.dot(segment*np.hanning(len(segment)), np.exp(-2j*np.pi*hz*t)))/len(t)

    @staticmethod
    def sung_vowel(t):
        # A voiced mic with energy across TAL's bass, mid and vowel bands.
        return sum(.085 * (.8/np.sqrt(h)*np.exp(-.5*((h*187-570)/370)**2)
                           + .4*np.exp(-.5*((h*187-1620)/210)**2))
                   * np.sin(2*np.pi*h*187*t) for h in range(1, 18))

    def test_default_midi_glide_joins_short_gaps_and_leaves_phrase_rests(self):
        # An ordinary imported score, with a drone under two short melody notes.
        notes = [(0, .16, 48), (.25, .55, 48), (0, .18, 69), (.25, .5, 76), (.85, .97, 67)]
        def breath(t):
            return np.random.default_rng(11).uniform(-.2,.2,t.size)
        for mode in (1, VOCODER_HOLD):
            default = self.render(mode, score_notes=notes, input_signal=breath)
            clipped = self.render(mode, score_notes=notes, input_signal=breath,
                                  controls={0: [100,50,5,40,0,55,0,0]})
            # The slide crosses C5 before the E5 onset. The input has no pitched notes.
            self.assertGreater(self.level(default,.178,.214,523.25),
                               2*self.level(clipped,.178,.214,523.25))
            self.assertGreater(self.level(default,.28,.44,659.25), .001)
            # The repeated C3 below the sliding notes also rings through its gap.
            self.assertGreater(self.level(default,.18,.245,130.81), .0008)
            if mode==1:
                np.testing.assert_allclose(default[30000:37000], clipped[30000:37000], atol=1e-7)

    def test_tal_bends_the_scored_riff_over_a_continuous_bass_and_separate_lead(self):
        # A held A2 under A4 sliding to E5. C#5 is an independent lead entry.
        def load(audio):
            for start, end, pitch in [(0,.60,45),(0,.20,69),(.30,.60,76)]:
                audio.call("note", b"chords", start, end, pitch)
            audio.call("lead_note", b"chords", .38, .56, 73)
            audio.call("score", b"chords", 1, None, 0)

        for rehearsal in (False, True):
            bent = self.render(TAL_VOCODER, load_score=load, rehearsal=rehearsal, input_signal=self.sung_vowel)
            stepped = self.render(TAL_VOCODER, load_score=load, rehearsal=rehearsal, input_signal=self.sung_vowel,
                                  controls={0:[100,50,5,40,0,55,0,0]})
            self.assertGreater(self.level(bent,.224,.266,523.25),
                               2*self.level(stepped,.224,.266,523.25))
            self.assertGreater(self.level(bent,.206,.295,110), .003)
            for hz in (110, 554.37, 659.25):
                self.assertGreater(self.level(bent,.412,.544,hz), .001)
            self.assertLess(self.level(bent,.64,.79,554.37), .0008)

    def test_tal_follows_a_change_of_vowel_across_octave_shifts(self):
        # Move the vocal resonance from 660 to 1760 Hz without changing the
        # singer's vibrato or the scored pitch.
        def vowel(t):
            cycles = 220*t + .09*np.sin(2*np.pi*5*t)
            center = np.where(t < .5, 660, 1760)
            return sum(.10*(.13/h + np.exp(-.5*((220*h-center)/170)**2))
                       * np.sin(2*np.pi*h*cycles) for h in range(1, 25))

        def vowel_ratio(segment):
            power = abs(np.fft.rfft(segment*np.hanning(len(segment))))**2
            hz = np.fft.rfftfreq(len(segment), 1/self.rate)
            return power[(hz>1450) & (hz<2050)].sum() / power[(hz>400) & (hz<1000)].sum()

        for pitch in (45, 69):
            wet = self.render(TAL_VOCODER, score_notes=[(0, 1, pitch)], input_signal=vowel)
            self.assertGreater(vowel_ratio(wet[32000:46000]), 20*vowel_ratio(wet[8000:22000]))
            self.assertGreater(self.level(wet,.2,.44,440*2**((pitch-69)/12)), .001)

    def test_tal_level_scales_only_the_wet_chord_and_scored_bass(self):
        def score(audio):
            for pitch in (55, 62, 65):
                audio.call("note", b"chords", 0, 1, pitch)
            audio.call("bass_note", b"chords", 0, 1, 38)
            audio.call("score", b"chords", 1, None, 0)

        for rehearsal in (False, True):
            def render(mode, mix):
                pcm = self.render(mode, load_score=score, rehearsal=rehearsal,
                                  controls={0: (mix, 50, 5, 40)})
                return pcm[self.rate//5:self.rate*4//5]
            dry = render(0, 100)
            wet = render(TAL_VOCODER, 100)
            self.assertGreater(np.std(wet-dry), .025)
            np.testing.assert_array_equal(render(TAL_VOCODER, 0), 0)
            np.testing.assert_allclose(render(TAL_VOCODER, 50), wet/2, atol=1e-6)

    def test_tal_source_vocal_mixes_the_selected_singer_independently_of_the_stereo_return(self):
        def output(voice, source, wet, rehearsal, *, lane=0, score=True):
            def setup(audio):
                if score:
                    self.load_fixture(audio, b"chords", [(0, 1, 69)], False)
                audio.call("tal_source_level", lane, source)
                chorus = next(i for i, p in enumerate(audio.tal_parameters(0)) if p.name == b"chorus")
                audio.call("tal_set", 0, chorus, 1)
            pcm = self.render(voice, rehearsal=rehearsal, input_signal=self.sung_vowel,
                              load_score=setup, controls={0: (wet, 50, 5, 40)}, stereo=True)
            return np.asarray(pcm)[:, 9600:33600]

        for rehearsal in (False, True):
            with self.subTest(rehearsal=rehearsal):
                dry = output(0, 0, 100, rehearsal)
                wet = output(TAL_VOCODER, 0, 100, rehearsal)
                self.assertGreater(dry.std(), .03)
                self.assertGreater((wet[0] - wet[1]).std(), .002)
                both = output(TAL_VOCODER, 65, 40, rehearsal)
                np.testing.assert_allclose(both, .65 * dry + .4 * wet, atol=2e-6)
                np.testing.assert_allclose(output(TAL_VOCODER, 100, 0, rehearsal), dry, atol=1e-6)
                np.testing.assert_allclose(output(TAL_VOCODER, 100, 100, rehearsal, lane=1), wet, atol=1e-6)
                np.testing.assert_allclose(output(TAL_VOCODER, 65, 100, rehearsal, score=False), .65 * dry, atol=1e-6)

    def test_strict_vst_excludes_dry_pitch_and_sibilants_on_live_and_test_vocals(self):
        def mic(t):
            return sum(a*np.sin(2*np.pi*f*t) for a, f in ((.1,193),(.05,579),(.025,6301)))

        for rehearsal in (False, True):
            args = dict(score_notes=[(0,1,69)], rehearsal=rehearsal)
            wet = self.render(TAL_VOCODER, input_signal=mic, **args)
            inverse = self.render(TAL_VOCODER, input_signal=lambda t: -mic(t), **args)
            # Inverting the mic inverts any dry path. TAL's band envelopes are
            # polarity invariant, including while its filters are opening.
            np.testing.assert_allclose(wet, inverse, atol=2e-6)
            self.assertGreater(self.level(wet,.2,.8,440), .001)
            dry = self.render(0, input_signal=mic, **args)
            for hz in (193, 579, 6301):
                self.assertLess(self.level(wet,.2,.8,hz), .001*self.level(dry,.2,.8,hz))
            np.testing.assert_array_equal(self.render(TAL_VOCODER, input_signal=np.zeros_like, **args), 0)
            np.testing.assert_array_equal(self.render(TAL_VOCODER, input_signal=mic,
                                                     score_notes=[], rehearsal=rehearsal), 0)

    def test_switch_from_dry_clears_delay_reverb_and_hear_bypass(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        self.assertTrue(audio.call("tal_ready", 0))
        t = np.arange(self.rate)/self.rate
        mic = self.sung_vowel(t).astype("f4")
        inputs = (FP * 4)()
        inputs[0] = mic.ctypes.data_as(FP)
        values = (C.c_float * len(PARAMS))(*(p[3] for p in PARAMS))
        values[5], values[6], values[7] = 60, 70, 80

        def render():
            left, right = np.zeros((2, self.rate), "f4")
            audio.call("render", inputs, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
            return left

        audio.call("config", 0, 0, b"missing", 0, 3, values)
        audio.call("live", 0, 1)
        audio.call("reverb", 0, .6)
        self.assertGreater(render().std(), .01)
        # A character audition still pending when the strip enters strict mode
        # must not overwrite the muted return with its recorded sample.
        audio.call("sample", 1, mic.ctypes.data_as(FP), len(mic))
        self.assertTrue(audio.call("hear", 0, 1))
        audio.call("config", 0, TAL_VOCODER, b"missing", 0, 3, values)
        np.testing.assert_array_equal(render(), 0)

    def test_tal_native_controls_reach_the_audio_processor_on_one_mic(self):
        def render(params, lane=0, stereo=False):
            def setup(audio):
                self.load_fixture(audio, b"chords", [(0, .9, 69)], False)
                for i, p in enumerate(audio.tal_parameters(lane)):
                    if p.name in params:
                        audio.call("tal_set", lane, i, params[p.name])
            pcm = self.render(TAL_VOCODER, load_score=setup, input_signal=self.sung_vowel, stereo=stereo)
            return np.asarray(pcm)[..., 12000:30000]
        base = render({})
        self.assertGreater(base.std(), .02)
        np.testing.assert_array_equal(render({b"volume": 0}), 0)
        np.testing.assert_array_equal(render({b"Bypass": 1}), 0)
        np.testing.assert_allclose(render({b"volume": 0}, lane=2), base, atol=1e-6)
        self.assertGreater((render({b"vocoderband04": 0})-base).std(), .002)
        left, right = render({b"chorus": 1}, stereo=True)
        self.assertGreater((left-right).std(), .002)
        # Internal routing must receive held score MIDI even when a single note
        # starts in the same block as the routing edit.
        synth = render({b"inputmode": 0})
        self.assertGreater(synth.std(), .02)
        self.assertGreater((synth-base).std(), .03)

    def test_missing_tal_mutes_only_its_strip(self):
        # Load the desk in a fresh process, before the host opens its module.
        script = """
from test_held_effects import CHORD_TUNE, TAL_VOCODER, DeskAudio, HeldEffects, np
audio = DeskAudio()
assert not audio.call("tal_ready", 0)
audio.close()
h = HeldEffects()
try:
    assert np.max(abs(h.render(0))) > .05
    assert np.max(abs(h.render(TAL_VOCODER))) == 0
    assert np.max(abs(h.render(CHORD_TUNE))) > .01
finally:
    h.doCleanups()
"""
        subprocess.run([sys.executable, "-c", script], check=True,
                       env={**os.environ, "KARAOKE_TAL_VST3": "/missing/TAL-Vocoder-2.so"})

    def test_tal_tied_notes_do_not_restart_the_carrier(self):
        single = [(0,.8,52),(0,.8,59),(0,.8,64)]
        repeated = [(a,b,p) for a,b in [(0,.295),(.30,.8)] for p in (52,59,64)]
        sustained = self.render(TAL_VOCODER, score_notes=single)
        retriggered = self.render(TAL_VOCODER, score_notes=repeated)
        np.testing.assert_allclose(sustained[9600:30000], retriggered[9600:30000], atol=1e-6)

    def test_vocoder_hold_keeps_the_chord_and_switches_on_the_next_attack(self):
        hold = self.render(VOCODER_HOLD)
        clear = self.render(1)
        for hz in (293.665, 349.228, 440):
            self.assertGreater(self.level(hold, .33, .49, hz), .00025)
            self.assertLess(self.level(clear, .33, .49, hz), .00003)
        for start, end in ((.75,.98), (1.005,1.095)):
            self.assertGreater(self.level(hold, start, end, 261.626), .00025)
        self.assertLess(self.level(hold,.8,.98,440), .00006)

    def test_both_harmonizers_sing_the_scored_chord_and_preserve_a_single_sibilant(self):
        t = np.arange(round(self.rate*1.5))/self.rate
        noise = np.random.default_rng(2).normal(size=len(t))
        hiss = .11*signal.sosfilt(signal.butter(3, (4900, 9200), fs=self.rate,
                                               btype="bandpass", output="sos"), noise)
        syllable = self.sung_vowel(t)*(t < .65) + hiss*((t > .66) & (t < .84))
        chord = [(0, 1, key) for key in (62, 65, 69)]
        for mode, rehearsal in ((TUNELET,False), (TUNELET,True), (CHORD_TUNE,False), (CHORD_TUNE,True)):
            args = dict(score_notes=chord, input_signal=lambda _: syllable, rehearsal=rehearsal)
            wet = self.render(mode, **args)
            for hz in (293.665, 349.228, 440):
                self.assertGreater(self.level(wet, .2, .51, hz), .003 if mode==TUNELET else .001)
            self.assertLess(self.level(wet, .2, .51, 187), .0001)
            original = syllable[32160:38560]
            returned = wet[32160:41000]
            corr = signal.correlate(returned, original, "valid")
            lag = np.argmax(corr)
            similarity = corr[lag]/(np.linalg.norm(original)*np.linalg.norm(returned[lag:lag+len(original)]))
            self.assertGreater(similarity, .8)
            self.assertLess(np.max(abs(wet[45000:47500])), .00002)
            np.testing.assert_array_equal(self.render(mode, controls={0:(0,50,5,40)}, **args), 0)
            np.testing.assert_array_equal(self.render(mode, score_notes=[], rehearsal=rehearsal), 0)

    def test_chorale_releases_old_notes_through_a_fast_chord_progression(self):
        progression = [(0, .30, (52, 59, 64)), (.30, .56, (55, 62, 67)), (.56, .84, (57, 61, 69))]
        notes = [(start, end, key) for start, end, keys in progression for key in keys]
        wet = self.render(CHORD_TUNE, input_signal=self.sung_vowel, score_notes=notes)
        previous = ()
        for start, end, chord in progression:
            body = wet[round((start+.09)*self.rate):round((end-.01)*self.rate)]
            spectrum = abs(np.fft.rfft(body*np.hanning(len(body)), n=self.rate))/len(body)
            def fundamental(key):
                hz = round(440*2**((key-69)/12))
                return max(spectrum[hz-4:hz+5])
            quietest = min(map(fundamental, chord))
            self.assertGreater(quietest, .0006)
            for released in previous:
                self.assertLess(fundamental(released), quietest/6)
            previous = chord

    def test_missing_chorale_mutes_only_its_own_effect(self):
        script = """
from test_held_effects import CHORD_TUNE, TUNELET, HeldEffects, np
h = HeldEffects()
try:
    assert np.max(abs(h.render(CHORD_TUNE))) == 0
    assert np.max(abs(h.render(TUNELET))) > .01
finally:
    h.doCleanups()
"""
        subprocess.run([sys.executable, "-c", script], check=True,
                       env={**os.environ, "KARAOKE_CHORALE_VST3": "/missing/Chorale.so"})

    def test_tal_makes_all_three_pitches_from_one_voice(self):
        for rehearsal in (False, True):
            pcm = self.render(TAL_VOCODER, rehearsal=rehearsal, input_signal=self.sung_vowel)
            for hz in (293.665, 349.228, 440):
                self.assertGreater(self.level(pcm, .31, .51, hz), .005)
            self.assertLess(self.level(pcm,.31,.51,187), .001)
            self.assertGreater(self.level(pcm,.71,.94,261.626), .01)
            self.assertLess(self.level(pcm,.71,.94,440), .0005)

    def test_seeking_into_a_rest_recovers_notes_without_playing_the_attack(self):
        for mode in (TAL_VOCODER, VOCODER_HOLD):
            pcm = self.render(mode, seek=.3, input_signal=self.sung_vowel)
            for hz in (293.665, 349.228, 440):
                self.assertGreater(self.level(pcm,.06,.21,hz), .00025)

    def test_audio_only_carrier_supplies_simultaneous_pitches(self):
        pcm = self.render(TAL_VOCODER, recorded=True, input_signal=self.sung_vowel)
        for hz in (293.665, 349.228, 440):
            self.assertGreater(self.level(pcm,.14,.18,hz), .004)
        self.assertGreater(self.level(pcm,.76,.92,261.626), .01)

    def test_clip_supplies_only_notes_and_clear_observes_its_rests(self):
        # An off-grid pitch reveals clip playback. Reversing the reference polarity
        # preserves its note events, but reverses a vocoded recording.
        axis = np.arange(self.rate) / self.rate
        reference = .22*np.sin(2*np.pi*404*axis) * (axis < .42)
        reference += .22*np.sin(2*np.pi*521*axis) * (axis >= .67)

        def score(polarity):
            def load(audio):
                pcm = np.repeat(np.rint(polarity*reference*32767).astype("i2"), 2)
                audio.call("score", b"chords", 1, pcm.ctypes.data_as(SP), len(pcm))
            return load

        for mode in (1, VOCODER_HOLD):
            wet = self.render(mode, load_score=score(1))
            np.testing.assert_array_equal(wet, self.render(mode, load_score=score(-1)))
            # 404 Hz rounds to G#4. The synth locks to that note on both sides of wrap.
            for start, end in ((.10, .38), (1.10, 1.38)):
                self.assertGreater(self.level(wet, start, end, 415.305),
                                   8*self.level(wet, start, end, 404))
            self.assertGreater(self.level(wet, .74, .96, 523.251), .001)
            rest = self.level(wet, .49, .61, 415.305)
            if mode == 1:
                self.assertLess(rest, .00005)
                self.assertGreater(self.level(wet, .49, .61, 190), .01)
            else:
                self.assertGreater(rest, .001)
            lower = self.render(mode, load_score=score(1), controls={0:(100,50,5,40,0,55,-1)})
            self.assertGreater(self.level(lower, .10, .38, 207.652), .001)

    def test_catalog_prefers_the_bundled_midi_score_to_the_prepared_clip(self):
        key = b"Legacy keyboard"
        notes = [(0, .3, 60), (.6, .8, 69)]
        with tempfile.TemporaryDirectory() as tmp:
            prepared, bundled = (Path(tmp) / name for name in ("prepared", "bundled"))
            prepared.mkdir()
            bundled.mkdir()
            (bundled / "scores.json").write_text(json.dumps({key.decode(): {
                "duration": 1, "events": [[0, [60]], [.3, []], [.6, [69]], [.8, []]]}}))
            (prepared / "scores.json").write_text(json.dumps({key.decode(): {"audio": "keyboard.wav"}}))
            with wave.open(str(prepared / "keyboard.wav"), "wb") as clip:
                clip.setparams((1, 2, self.rate, 0, "NONE", ""))
                # Different pitches and a different loop length from the MIDI.
                axis = np.arange(self.rate // 2) / self.rate
                clip.writeframes(np.rint(15000 * np.sin(2*np.pi*350*axis)).astype("<i2").tobytes())
            for mode in (1, VOCODER_HOLD):
                wet = self.render(mode, load_score=lambda a: a.load_catalog_scores(prepared, bundled), key=key)
                expected = self.render(mode, score_notes=notes, key=key)
                np.testing.assert_array_equal(wet, expected)

            # A legacy installation with just the clip still gets playable synth notes.
            (bundled / "scores.json").unlink()
            wet = self.render(1, load_score=lambda a: a.load_catalog_scores(prepared, bundled), key=key)
            self.assertGreater(self.level(wet, .4, .8, 349.228), .001)

    def test_silent_reference_without_a_score_passes_the_microphone_through(self):
        def silence(audio):
            pcm = np.zeros(2*self.rate, "i2")
            audio.call("score", b"chords", 1, pcm.ctypes.data_as(SP), len(pcm))
            self.assertFalse(audio.call("has_carrier", b"chords"))

        dry = self.render(0)
        for mode in (1, VOCODER_HOLD):
            np.testing.assert_array_equal(self.render(mode, load_score=silence), dry)

    def test_spoken_shout_has_each_harmony_without_dry_sibilants_in_the_return(self):
        t = np.arange(int(self.rate*1.5))/self.rate
        noise = np.random.default_rng(505).normal(size=len(t))
        def band(lo, hi):
            return signal.sosfilt(signal.butter(2, (lo, hi), fs=self.rate,
                                                btype="bandpass", output="sos"), noise)
        # An irregular, breathy vowel, a separate sibilant and a silent rest.
        voice = (.23*band(250, 1700) + .025*np.sin(2*np.pi*(115*t+80*t*t))) * (t < .55)
        voice += .13*band(6200, 9500) * ((t >= .6) & (t < .74))
        notes = [(0, 1., key) for key in (62, 65, 69)]
        for rehearsal in (False, True):
            wet = self.render(TAL_VOCODER, rehearsal=rehearsal, input_signal=lambda _: voice, score_notes=notes)
            frequencies, power = signal.welch(wet[2400:25000], fs=self.rate, nperseg=6144)
            for hz in (293.665, 349.228, 440):
                peak = max(power[abs(frequencies-hz) < 9])
                nearby = power[(abs(frequencies-hz)>15) & (abs(frequencies-hz)<26)].mean()
                self.assertGreater(peak, 1.45*nearby, (rehearsal, hz))
            hiss = slice(round(.63*self.rate), round(.73*self.rate))
            self.assertLess(abs(np.corrcoef(voice[hiss], wet[hiss])[0,1]), .05)
            self.assertLess(abs(wet[round(.84*self.rate):round(.98*self.rate)]).max(), 2e-5)

    def test_get_low_spoken_and_shouted_words_follow_a_held_note(self):
        from audio import decode
        from library import catalog
        song = catalog().get("Get Low", {})
        guide = song.get("guide")
        if guide is None or not guide.is_file():
            self.skipTest("Get Low guide available on the provisioned desk")
        # Analyze the body of each audible syllable against the target period.
        # Each excerpt begins at a word attack in the installed lyric score.
        for start in (24.470, 29.180, 96.026, 101.696):
            src = decode(guide, 1, start=start, seconds=1.5, floating=True)
            src = .35*src/max(abs(src))
            wet = self.render(TAL_VOCODER, input_signal=lambda _: src, score_notes=[(0, 1, 64)])
            filt = signal.butter(3, (85, 1700), fs=self.rate, btype="bandpass", output="sos")
            dry_body, wet_body = [signal.sosfilt(filt, x) for x in (src, wet)]
            frames = np.lib.stride_tricks.sliding_window_view(dry_body, 1920)[::960]
            active = np.mean(frames**2, axis=1) > max(.001, np.percentile(abs(dry_body),75)*.18)**2
            def periodicity(body):
                frames = np.lib.stride_tricks.sliding_window_view(body, 1920)[::960][active]
                lag = round(self.rate/329.628)
                a, b = frames[:,lag:], frames[:,:-lag]
                return np.sum(a*b, axis=1)/np.sqrt(np.sum(a*a, axis=1)*np.sum(b*b, axis=1)+1e-12)
            self.assertGreater(active.sum(), 15)
            self.assertGreater(np.mean(periodicity(wet_body)>.5), .64, start)
            self.assertGreater(np.median(periodicity(wet_body))-np.median(periodicity(dry_body)), .3, start)

    def test_vocoder_controls_change_the_signal_on_the_addressed_strip(self):
        def syllables(t):
            tone = .05*np.sin(2*np.pi*190*t) + .025*np.sin(2*np.pi*3800*t)
            return tone * ((t < .3) | ((t >= .5) & (t < .8)))
        def output(knobs, lane=0):
            return self.render(VOCODER_HOLD, controls={lane: knobs}, input_signal=syllables)
        neutral = output((100,50,5,40))
        np.testing.assert_array_equal(neutral, output((0,0,80,300), lane=2))
        dry = output((0,0,2,16))
        np.testing.assert_allclose(dry[1000:], syllables(np.arange(len(dry))/self.rate)[1000:], atol=1e-6)
        wet = output((100,0,2,16))
        clearer = output((100,100,2,16))
        def unvoiced_power(pcm):
            start, end = round(.14*self.rate), round(.25*self.rate)
            power = abs(np.fft.rfft(pcm[start:end]*np.hanning(end-start), n=self.rate))**2
            hz = np.arange(3300, 4600)
            between = np.ones(len(hz), dtype=bool)
            for note in (293.665, 349.228, 440):
                between &= abs(hz/note-np.rint(hz/note))*note > 21
            return power[hz[between]].mean()
        self.assertGreater(unvoiced_power(clearer), 3*unvoiced_power(wet))
        # Adding consonants must not reintroduce the raw 3800 Hz test tone.
        self.assertLess(self.level(clearer,.14,.25,3800), .15*self.level(dry,.14,.25,3800))
        slow = output((100,0,80,16))
        attack = slice(24100,24800)
        self.assertGreater(wet[attack].std(), 1.4*slow[attack].std())
        tail = slice(16500,18500)
        self.assertGreater(output((100,0,2,250))[tail].std(), 2*wet[tail].std())


    def test_fully_wet_vocal_shapes_the_carrier_and_dry_pitch_stays_out(self):
        def vowels(t):
            center = np.where(t < .55, 630., 1430.)
            return sum(.07*np.sin(2*np.pi*(h*139)*t)*
                       (.8*np.exp(-((h*139-center)/155)**2)+.12/h) for h in range(1,32))
        def render_formant(shift, register=0):
            return self.render(VOCODER_HOLD, controls={0:(100,100,5,40,shift,55,register)},
                               score_notes=[(0,1,64)], input_signal=vowels)
        wet = render_formant(0)
        def spectrum(pcm, start):
            v = pcm[start:start+14400]
            return abs(np.fft.rfft(v*np.hanning(len(v)), n=self.rate))
        first, second = (spectrum(wet, start) for start in (9600, 33600))
        def mid_to_low(spec):
            return (spec[1100:1750]**2).sum()/(spec[425:850]**2).sum()
        self.assertGreater(mid_to_low(second), 8*mid_to_low(first))
        dry = spectrum(vowels(np.arange(len(wet))/self.rate), 9600)
        for hz in (139, 278, 556):
            self.assertLess(first[hz], .04*dry[hz])
        self.assertGreater(first[659], 5)
        shifted = spectrum(render_formant(10), 9600)
        self.assertGreater(mid_to_low(shifted), 4*mid_to_low(first))
        lower = spectrum(render_formant(0, register=-1), 9600)
        self.assertGreater(lower[165], 10*first[165])
        # Speech range transposes the entire high melody together, including
        # notes on opposite sides of the register boundary.
        notes = [(0, .3, 69), (.3, .6, 67), (.6, 1., 72)]
        def register(value):
            return self.render(VOCODER_HOLD, score_notes=notes, input_signal=vowels,
                               controls={0:(100,55,4,30,0,60,value)})
        np.testing.assert_array_equal(register(1), register(-2))
        silent = self.render(VOCODER_HOLD, input_signal=np.zeros_like)
        np.testing.assert_array_equal(silent, 0)

    def test_synth_brightness_adds_upper_harmonics_to_a_transcribed_bass(self):
        # Only the bass pitch is present in the reference. The oscillator supplies the partials.
        def vowel(t):
            return .07*np.sin(2*np.pi*645*t)+.05*np.sin(2*np.pi*1240*t)
        def sound(bite):
            return self.render(1, recorded=True, score_notes=[(0, 1, 42)],
                               controls={0:(100,0,5,40,0,bite,0)}, input_signal=vowel)
        plain, rich = sound(0), sound(95)
        # F#2's thirteenth harmonic falls near the second vowel resonance.
        hz = 13*440*2**((42-69)/12)
        self.assertGreater(self.level(rich,.4,.8,hz), 4*self.level(plain,.4,.8,hz))
        self.assertLess(self.level(rich,.4,.8,1240), .001)
