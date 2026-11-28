"""Prepare a dry virtual microphone from the audience edition, off the audio thread."""

from functools import lru_cache
import bisect
from concurrent.futures import CancelledError
import ctypes as C
import os
import subprocess

import numpy as np

from audio import DATA, decode
from neural_voice import installed_voice


@lru_cache(maxsize=1024)
def spoken_word(word):
    # FFmpeg's local Flite voice needs no model download. Pass the literal lyric
    # as file data so quotes, punctuation and filter syntax stay out of the graph.
    with os.fdopen(os.memfd_create("karaoke-word"), "wb") as text:
        text.write(word.encode())
        text.flush()
        raw = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"flite=textfile=/proc/self/fd/{text.fileno()}:voice=slt",
                "-ar",
                "48000",
                "-f",
                "s16le",
                "-",
            ],
            pass_fds=(text.fileno(),),
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
    samples = np.frombuffer(raw, "<i2")
    active = np.flatnonzero(np.abs(samples.astype("i4")) > 110)
    if not len(active):
        return np.zeros(1, "i2")
    return samples[max(0, active[0] - 120) : active[-1] + 121]


def stretch(word, frames, pitch):
    # Sonic varies duration independently of pitch, retaining the consonants
    # when a word is held or squeezed into a short note.
    sonic = sonic_library()
    stream = sonic.sonicCreateStream(48000, 1)
    if not stream:
        raise MemoryError("Could not allocate a singing voice")
    try:
        sonic.sonicSetSpeed(stream, len(word) / frames)
        sonic.sonicSetPitch(stream, pitch)
        if not sonic.sonicWriteShortToStream(
            stream, word.ctypes.data, len(word)
        ) or not sonic.sonicFlushStream(stream):
            raise RuntimeError("Failed to synthesize a sung word")
        out = np.zeros(frames, np.int16)
        sonic.sonicReadShortFromStream(stream, out.ctypes.data, frames)
        return out
    finally:
        sonic.sonicDestroyStream(stream)


@lru_cache(maxsize=1)
def sonic_library():
    lib = C.CDLL("libsonic.so.0")
    ptr, integer = C.c_void_p, C.c_int
    for name, result, args in [
        ("CreateStream", ptr, [integer, integer]),
        ("DestroyStream", None, [ptr]),
        ("SetSpeed", None, [ptr, C.c_float]),
        ("SetPitch", None, [ptr, C.c_float]),
        ("WriteShortToStream", integer, [ptr, ptr, integer]),
        ("ReadShortFromStream", integer, [ptr, ptr, integer]),
        ("FlushStream", integer, [ptr]),
    ]:
        fn = getattr(lib, "sonic" + name)
        fn.restype, fn.argtypes = result, args
    return lib


def phrase_recording(song):
    return DATA / "chants-lbd/sources" / song["id"].replace(" ", "_") / "clean-vocal.wav"


def prepare_vocal(song, cancelled=None, *, source="auto"):
    rows = [
        row
        for lane in song.get("rows", {}).values()
        for row in lane
        if row[0] and row[1]
    ]
    if song.get("cuts") or song.get("performance"):
        rows = [
            row
            for row in rows
            if song["start"] * 1000 <= row[1][0][0] < song["end"] * 1000
        ]
    phrase = song.get("phrase")
    if phrase and source != "computer":
        original = phrase_recording(song)
        if original.is_file():
            tempo = song["bpm"] / phrase["bpm"]
            raw = subprocess.check_output(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-ss",
                    str(phrase["start"]),
                    "-i",
                    str(original),
                    "-af",
                    f"atempo={tempo}",
                    "-t",
                    str(16 * 60 / song["bpm"]),
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-f",
                    "s16le",
                    "-",
                ]
            )
            return np.frombuffer(raw, "<i2")
    if not rows and any(w.strip().casefold() == "instrumental" for w in song.get("cue") or []):
        return np.empty(0, "i2")
    if source != "computer" and song.get("guide") and song["guide"].is_file():
        try:
            guide = decode(song["guide"], 1)
        except (OSError, subprocess.CalledProcessError):
            if not rows or source == "extracted":
                raise
            guide = np.empty(0, "i2")
        if not rows:
            return guide
        # Isolate displayed phrases (including simultaneous call and response).
        # A guide can contain unused verses and pickups outside a selected cut.
        pcm = np.zeros_like(guide)
        intervals = [(spans[0][0], spans[-1][1]) for _, spans in rows]
        if window := song.get("teaserWindowMs"):
            # The reviewed rest on either side can include the attack or decay
            # of the real recording beyond the millisecond lyric highlight.
            intervals.append(window)
        for lo, hi in intervals:
            start = max(0, round(lo * 48))
            end = max(start, round(hi * 48))
            pcm[start:end] = guide[start:end]
        if source == "extracted" or np.any(pcm):
            return pcm
    if source == "extracted":
        raise ValueError("No extracted vocal for this song")
    if not rows:
        raise ValueError("No timed lyrics for the computer vocal")
    lines = [
        [(word.strip(), max(0, round(a * 48)), round(b * 48))
         for word, (a, b) in zip(tokens, spans, strict=True)
         if b > max(a, 0) and word.strip()]
        for tokens, spans in rows
    ]
    length = max((end for line in lines for _, _, end in line), default=0)
    pcm = np.zeros(length, "i2")
    if not length:
        return pcm
    events = song.get("score", {}).get("events", [])
    starts = [event[0] for event in events]
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()
    voice = installed_voice()
    for line in filter(None, lines):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError()
        sources = voice.words([word for word, _, _ in line]) if voice else [None] * len(line)
        for (word, start, end), source in zip(line, sources, strict=True):
            if cancelled is not None and cancelled.is_set():
                raise CancelledError()
            if source is None:
                source = spoken_word(word)
            if not len(source) or end <= start:
                continue
            i = bisect.bisect_right(starts, start / 48000) - 1
            notes = events[i][1] if i >= 0 else []
            # Keep the test voice in one octave when the score provides a carrier chord.
            pitch = 2 ** (((min(notes) - 57) % 12 - 4) / 12) if notes else 1.0
            take = stretch(source, end - start, pitch)
            # Cutting a connected utterance on the beat can land mid-wave.
            # Taper the splice over two milliseconds to avoid a click.
            fade = min(96, len(take) // 2)
            if fade:
                ramp = np.linspace(0, 1, fade)
                take[:fade] = take[:fade] * ramp
                take[-fade:] = take[-fade:] * ramp[::-1]
            pcm[start:end] = np.clip(pcm[start:end].astype("i4") + take, -32768, 32767)
    return pcm
