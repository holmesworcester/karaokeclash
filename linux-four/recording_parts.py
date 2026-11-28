"""Identify scored editions by PCM landmarks, independent of filename and WAV metadata."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import wave

PARTS = Path(__file__).with_name("recording-parts")


def fingerprint(wav):
    count = wav.getnframes()
    digest = hashlib.sha256()
    digest.update(str((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), count)).encode())
    for at in (0, count // 4, count // 2, 3 * count // 4, max(0, count - 4096)):
        wav.setpos(at)
        digest.update(wav.readframes(4096))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def editions():
    index = PARTS / "index.json"
    return json.loads(index.read_text()) if index.is_file() else {}


@lru_cache(maxsize=256)
def identify(backing, size, modified):
    with wave.open(str(backing)) as wav:
        candidates = editions().get(str(wav.getnframes()))
        if candidates:
            return candidates.get(fingerprint(wav))


def part_path(backing):
    st = backing.stat()
    name = identify(backing, st.st_size, st.st_mtime_ns)
    return PARTS / name if name else None
