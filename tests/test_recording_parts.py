"""Check note intent and stereo through the same score and mixer paths as the desk."""
import ctypes as C
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from audio import F, FP, PARAMS
from carrier import SR, prepare_score, transcription
from desk_audio import DeskAudio
from recording_parts import PARTS, fingerprint
from test_polyphonic_carrier import play_note, write_wav, at
import wave


class RecordingParts(unittest.TestCase):
    def test_sliding_effect_drum_hits_and_guitar_partials_are_not_separate_keys(self):
        t = np.arange(SR*3) / SR
        sweep = .18 * signal.chirp(t, 1200, t[-1], 90, 'logarithmic')
        hits = np.zeros_like(t)
        rng = np.random.default_rng(21)
        for start in (.32, .89, 1.8):
            pos = (t>=start) & (t<start+.16)
            dt = t[pos]-start
            hits[pos] = (.4*rng.normal(size=len(dt))*np.exp(-45*dt)
                         + .3*signal.chirp(dt,140,.16,45)*np.exp(-20*dt))
        def transcribe(mono):
            return transcription(np.repeat(mono[:,None],2,axis=1).astype('f4'),3)
        self.assertEqual(transcribe(sweep+hits), [])
        guitar = .09 * sum(g*np.sin(2*np.pi*110*h*t)
                           for h,g in enumerate([0,.6,1,.8,.8,.6,.6,.55,.5]))
        notes = transcribe(guitar+hits)
        self.assertEqual({k for _,_,k in notes}, {45})
        self.assertGreater(sum(b-a for a,b,k in notes),2600)
        # A real high keyboard note sharing this range survives the sweep.
        notes = transcribe(sweep+hits+.15*np.sin(2*np.pi*1046.502*t))
        self.assertEqual(at(notes,2.5), {84})

    def test_landmarks_reject_another_recording_with_the_same_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            different = folder/"different.wav"
            recording = folder/"backing.wav"
            pcm = np.zeros((SR,2),'f4')
            play_note(pcm,60,.1,.9,.12,[1,.5],[1,.2])
            write_wav(recording,pcm)
            write_wav(different,-pcm)
            with wave.open(str(recording)) as wav:
                digest = fingerprint(wav)
            part = dict(arrangement="Keyboard", notes=[[140,810,60]],pan={"60":-.5})
            (folder/"keys.json").write_text(json.dumps(part))
            with patch("recording_parts.PARTS", folder), patch("recording_parts.editions",
                       return_value={str(SR):{digest:"keys.json"}}):
                self.assertEqual(prepare_score(recording)["notes"],part["notes"])
                self.assertNotIn("arrangement",prepare_score(different))

    def test_radio_mix_enters_the_library_only_with_both_playable_stems(self):
        from playback import catalog, playback_files
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            radio = root/"private/linux-four/playback/Bellissima - Radio Mix"
            radio.mkdir(parents=True)
            pcm = np.zeros((SR,2),'f4')
            write_wav(radio/"backing.wav",pcm)
            with patch("playback.DATA",root), patch("playback.runtime_catalog",side_effect=lambda:
                       {"Bellissima":dict(id="Bellissima",artist="DJ Quicksilver",bpm=136)}):
                self.assertNotIn("radioMix",catalog()["Bellissima"])
                write_wav(radio/"drums.wav",pcm)
                song = catalog()["Bellissima"]
                self.assertEqual(song["label"],"Bellissima · Radio Mix")
                self.assertEqual(playback_files(song),(radio/"backing.wav",radio/"drums.wav"))
                self.assertEqual(song["carrierBacking"],song["bed"])

    def test_radio_hook_verse_rests_and_isolated_bass_parts(self):
        def notes(name):
            return json.loads((PARTS/(name+".json")).read_text())["notes"]
        self.assertEqual({k for _,_,k in notes("a-real-hero")}, {34,36,38,41})
        self.assertEqual(at(notes("doin-it-right"),3.),{44,54,58})
        self.assertEqual(at(notes("doin-it-right"),92.),set())
        self.assertEqual(at(notes("closing-teaser"),.3),{43,50,59})
        self.assertEqual(at(notes("closing-teaser"),1.2),{50,57,66})
        self.assertEqual(at(notes("closing-teaser"),2.1),{45,52,60})
        self.assertEqual(at(notes("closing-teaser"),3.2),{48,55,64})
        bell=notes("bellissima-radio")
        self.assertEqual([k for _,_,k in bell if k>=63][:8],[67,67,67,63,63,63,68,68])
        self.assertFalse(any(a<40_000<b and k>=63 for a,b,k in bell))
        self.assertEqual(at(notes('blue'),2.35),{70})
        self.assertEqual(len(at(notes('2fly'),1.17)),1)

    def test_source_pan_reaches_the_monitor_clear_hold_and_tal_sidechain(self):
        audio = DeskAudio()
        self.addCleanup(audio.close)
        rate = 48000
        region = dict(id="keys",track="keys",bpm=120,start=0,end=2,loop=True)
        score = dict(source=[1],duration=2,notes=[[0,800,60],[1000,1800,67]],
                     pan={"60":-.7,"67":.7})
        ident = audio.load_backing_score("keys",region,"keys",score,False,"keys").encode()
        t = np.arange(2*rate)/rate
        mic = (.15*np.sin(2*np.pi*170*t)+.06*np.sin(2*np.pi*340*t)).astype('f4')
        inputs = (FP*4)()
        inputs[0] = mic.ctypes.data_as(FP)
        for mode in (0,1,-1,-2):
            for width in (100,0):
                with self.subTest(mode=mode,width=width):
                    audio.prepare_track(np.zeros(4*rate,'i2'),None,region)
                    audio.call("stage_lane",0)
                    audio.call("stage_carrier_backing",mode==0)
                    audio.call("launch_at",-1,0)
                    audio.call("config",0,mode,ident,0,0,(F*len(PARAMS))(*(p[3] for p in PARAMS)))
                    audio.call("live",0,mode!=0)
                    audio.call("glide",0,0)
                    audio.call("stereo_width",0,width)
                    audio.call("transport",1,0)
                    audio.call("transport",0,1)
                    pcm=np.zeros((2,2*rate),'f4')
                    audio.call("render",inputs,*(ch.ctypes.data_as(FP) for ch in pcm),2*rate)
                    for t,louder in ((.3,0),(1.3,1)):
                        excerpt=pcm[:,round(t*rate):round((t+.2)*rate)]
                        levels=np.sqrt((excerpt**2).mean(axis=1))
                        self.assertGreater(max(levels),.001)
                        if width:
                            self.assertGreater(levels[louder],levels[1-louder]*1.5)
                        else:
                            np.testing.assert_allclose(excerpt[0],excerpt[1],atol=2e-6)
