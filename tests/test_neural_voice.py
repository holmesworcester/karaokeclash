"""Exercise the provisioned offline model through the missing-stem desk path."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux-four"))
from neural_voice import installed_voice, word_phones
from test_singing import prepare_vocal


class SynthesizedLyrics(unittest.TestCase):
    def test_real_phrase_is_used_for_timed_words_and_overlapping_lanes(self):
        voice = installed_voice()
        self.assertIsNotNone(voice, "Run mise run test-desk to provision the neural voice")
        words = ["Count", "to", "21,", "then", "go!"]
        spans = [[100, 380], [380, 530], [530, 1150], [1450, 1690], [1690, 2300]]
        phones = word_phones(words)
        self.assertIn(" ", phones[2])  # both words in "twenty one" belong to one cue
        self.assertTrue(phones[2].endswith(","))
        self.assertTrue(phones[-1].endswith("!"))
        takes = voice.words(words)
        self.assertEqual(len(takes), 5)
        self.assertTrue(all(t.dtype == np.int16 and t.std() > 100 for t in takes))
        rows = {"lead": [[words, spans]]}
        with patch("test_singing.spoken_word", side_effect=AssertionError("unexpected Flite fallback")):
            dry = prepare_vocal(dict(rows=rows, score={"events": [[0, [61]]]}))
            reply = {"reply": [[["Hey!"], [[1500, 1900]]]]}
            together = prepare_vocal(dict(rows=rows | reply))
            reply_dry = prepare_vocal(dict(rows=reply))
        self.assertEqual(len(dry), 2300 * 48)
        self.assertFalse(dry[:100 * 48].any())
        self.assertFalse(dry[1150 * 48:1450 * 48].any())
        for a, b in spans:
            self.assertGreater(dry[a * 48:b * 48].std(), 100)
        expected = dry.astype(np.int32)
        expected[:len(reply_dry)] += reply_dry
        np.testing.assert_array_equal(together, np.clip(expected, -32768, 32767))

    def test_offline_install_without_weights_still_speaks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KARAOKE_TTS_MODEL_DIR=tmp):
            self.assertIsNone(installed_voice())
            pcm = prepare_vocal(dict(rows={"lead": [[["Ready"], [[75, 600]]]]}))
        self.assertFalse(pcm[:75 * 48].any())
        self.assertGreater(pcm[75 * 48:].std(), 100)

    def test_explicit_sources_never_substitute_the_other_voice(self):
        rows = {"lead": [[["Ready"], [[75, 600]]]]}
        with self.assertRaisesRegex(ValueError, "No extracted vocal"):
            prepare_vocal(dict(rows=rows), source="extracted")
        # A phrase recording takes precedence over a guide in Auto. Computer
        # must bypass both even when the recording has no usable lyric sheet.
        with patch("test_singing.phrase_recording", side_effect=AssertionError("used recording")):
            with self.assertRaisesRegex(ValueError, "No timed lyrics"):
                prepare_vocal(dict(phrase=True, guide=Path(__file__)), source="computer")
            instrumental = prepare_vocal(dict(cue=["Instrumental"], guide=Path(__file__)), source="computer")
            self.assertEqual(len(instrumental), 0)
