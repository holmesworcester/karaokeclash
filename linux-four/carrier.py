"""Offline polyphonic transcription of the actual desk backing, on its sample clock."""

from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np
from scipy import ndimage, signal

from recording_parts import part_path
from pitched_notes import confirmed_notes, pitch_evidence

SR, HOP = 22050, 256
FIRST_KEY = 21
VERSION = 8
DENSITY_GRIDS = ((587, 3968), (797, 4608))
MODEL = Path(__file__).with_name("basic-pitch") / "nmp.onnx"


@lru_cache(maxsize=1)
def network():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])


def decode(path):
    pcm = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", path,
                                   "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"])
    return np.frombuffer(pcm, "<f4").reshape(-1, 2)


def pitched_audio(backing, drums):
    pcm = decode(backing)
    if drums.is_file():
        beat = decode(drums)
        if len(pcm) != len(beat):
            raise ValueError(f"Backing and drum stem lengths differ: {backing}")
        # Some legacy drum stems were normalized separately or contain an alternate mix.
        # Remove only their in-phase contribution to each backing channel.
        sample, ref = pcm[::31], beat[::31]
        gain = np.sum(sample * ref, axis=0) / (np.sum(ref * ref, axis=0) + 1e-9)
        pcm = pcm - np.clip(gain, 0, 1) * beat
    return pcm


def activations(channel):
    # The network predicts 172 frames from 43844 samples. Discard its context at
    # either end and place the remaining frames back on the recording sample clock.
    count = (len(channel) + HOP - 1) // HOP
    padded = np.pad(channel, (30 * HOP, 43844))
    frames, onsets = [], []
    for i in range(0, len(channel), 112 * HOP):
        x = padded[None, i:i + 43844, None]
        frame, onset = network().run(["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
                                      {"serving_default_input_2:0": x})
        frames.append(frame[0, 30:142])
        onsets.append(onset[0, 30:142])
    return np.concatenate(frames)[:count], np.concatenate(onsets)[:count]


def spectral_peaks(channel):
    # Magnitude at each fundamental and contrast with its two flanks. Use
    # channel-wise evidence so a wide or polarity-inverted synth survives.
    bins = 440 * 2 ** ((np.arange(21, 109) - 69) / 12) * 8192 / SR
    center = np.rint(bins).astype(int)
    flank = np.maximum(3, np.rint(bins * .045).astype(int))
    count = (len(channel) + HOP - 1) // HOP
    padded = np.pad(channel, (2048, 4096))
    window = np.hanning(4096).astype("f4")
    peak = np.empty((count, 88), dtype="f4")
    contrast = np.empty_like(peak)
    resolved = np.zeros_like(peak, dtype=bool)
    for i in range(0, count, 128):
        blocks = np.lib.stride_tricks.sliding_window_view(padded[i * HOP:], 4096)[::HOP][:min(128, count - i)]
        spectrum = np.abs(np.fft.rfft(blocks * window, 8192))
        at = np.maximum.reduce([spectrum[:, center + off] for off in (-1, 0, 1)])
        floor = .5 * (spectrum[:, np.maximum(center - flank, 0)] + spectrum[:, center + flank])
        # A wide FFT peak can cover several adjacent bass keys. Assign its
        # interpolated center to one MIDI key before counting independent voices.
        middle = spectrum[:, 1:-1]
        rows, columns = np.nonzero((middle > spectrum[:, :-2]) & (middle >= spectrum[:, 2:])
                                   & (middle > .055 * at.max(axis=1, keepdims=True)) & (middle > .12))
        k = columns + 1
        a, b, c = (np.log(np.maximum(spectrum[rows, k + shift], 1e-12)) for shift in (-1, 0, 1))
        hz = (k + .5 * (a-c) / (a-2*b+c)) * SR / 8192
        key = np.rint(69 + 12 * np.log2(hz / 440)).astype(int) - FIRST_KEY
        valid = (key >= 0) & (key < 88)
        resolved[i + rows[valid], key[valid]] = True
        peak[i:i+len(at)] = at
        contrast[i:i+len(at)] = at / (floor + .015)
    return peak, contrast, resolved


def resolved_tones(pcm, hop=882, width=4096):
    # Estimate continuous frequencies directly from local FFT maxima. Exclude
    # peaks near a lower peak's harmonics, where a mixed recording is ambiguous.
    fft = 8192
    times = np.arange(0, len(pcm), hop)
    padded = np.pad(pcm, ((width//2, width//2), (0, 0)))
    window = np.hanning(width)[:, None]
    reference = np.zeros((len(times), 128), bool)
    for i, t in enumerate(times):
        spectrum = abs(np.fft.rfft(padded[t:t+width]*window, fft, axis=0)).max(axis=1)
        top = spectrum[10:1500].max()
        peaks, _ = signal.find_peaks(spectrum[:1500], height=max(.65, top*.13),
                              prominence=max(.35, top*.065))
        lows = []
        for k in peaks:
            if k < 10:
                continue
            a, b, c = np.log(np.maximum(spectrum[k-1:k+2], 1e-12))
            hz = (k + .5*(a-c)/(a-2*b+c)) * SR/fft
            midi = 69 + 12*np.log2(hz/440)
            ratio = hz / np.asarray(lows)
            lows.append(hz)
            if abs(midi-round(midi)) > .32 or np.any((np.rint(ratio) >= 2)
                    & (abs(ratio/np.rint(ratio)-1) < .018)):
                continue
            reference[i, round(midi)] = True
    # An isolated attack can be a drum transient or a spectral side lobe.
    reference = ndimage.binary_opening(reference, structure=np.ones((3, 1), bool))
    return times / SR, reference


def stable_fundamentals(pcm, first, second):
    """Fundamentals that remain resolved across two time and frequency grids."""
    hop, width = first
    times, reference = resolved_tones(pcm, hop, width)
    other_hop, other_width = second
    others, confirmation = resolved_tones(pcm, other_hop, other_width)
    indices = np.minimum(len(others) - 1, np.rint(times * SR / other_hop).astype(int))
    reference &= confirmation[indices]
    return times, ndimage.minimum_filter1d(reference, 3, axis=0, mode="constant")


def note_roll(times, notes, tolerance):
    roll = np.zeros((len(times), 128), bool)
    for start, end, key in notes:
        a, b = np.searchsorted(times, [start * .001 - tolerance, end * .001 + tolerance])
        roll[a:b, key] = True
    return roll


def pitch_classes(roll):
    return np.stack([roll[:, key::12].any(axis=1) for key in range(12)], axis=1)


def recover_chord_tones(pcm, duration, notes, *, density_only=False):
    # Re-examine chord passages on a finer grid. Only add a voice when two
    # measurements resolve a stable fundamental and its pitch class is missing.
    # Extend the repair only within the measured note, preserving short attacks.
    grids = DENSITY_GRIDS if density_only else ((617, 3840), (769, 4864))
    times, reference = stable_fundamentals(pcm, *grids)
    observed = pitch_classes(reference).sum(axis=1)
    chords = observed >= 2
    present = pitch_classes(note_roll(times, notes, .035))
    if density_only:
        chords &= observed > present.sum(axis=1)
    half_frame = grids[0][0] / (2 * SR)
    extra = []
    for key in range(FIRST_KEY, FIRST_KEY + 88):
        absent = chords & reference[:, key] & ~present[:, key % 12]
        for a, b in np.flatnonzero(np.diff(np.r_[False, absent, False])).reshape(-1, 2):
            start = max(0., times[a] - half_frame)
            end = min(duration, times[b - 1] + half_frame)
            extra.append([round(start * 1000), round(end * 1000), key])
    return sorted([*notes, *extra])


def note_candidates(pcm, duration):
    frame, onset = activations(pcm[:, 0])
    left, contrast, resolved = spectral_peaks(pcm[:, 0])
    right = left
    if not np.array_equal(pcm[:, 0], pcm[:, 1]):
        rframe, ronset = activations(pcm[:, 1])
        right, rcontrast, rresolved = spectral_peaks(pcm[:, 1])
        frame, onset = np.maximum(frame, rframe), np.maximum(onset, ronset)
        contrast, resolved = np.maximum(contrast, rcontrast), resolved | rresolved
    peak = np.maximum(left, right)
    relative = peak / (.1 + peak.max(axis=1, keepdims=True))
    clear = resolved & (relative > .065) & (contrast > 1.35)
    candidate = (peak > .16) & ((resolved & (frame > .25)) | (clear & (frame > .155)))
    active = np.zeros_like(candidate)
    allowed_l, allowed_r = np.zeros_like(left), np.zeros_like(right)
    pan = left / (left + right + .0001)
    # Discount partials of already sounding lower notes. Work in stereo so
    # an independently panned upper voice can exceed the harmonic envelope
    # even when the same pitch is also a partial in the other channel.
    harmonics = [(round(12 * np.log2(n)), .94 / np.sqrt(n)) for n in range(2, 11)
                 if abs(12 * np.log2(n) - round(12 * np.log2(n))) < .40]
    for k in range(88):
        excess = np.maximum(left[:, k] - allowed_l[:, k], right[:, k] - allowed_r[:, k])
        # Strong model support and independently changing amplitude also admit
        # doubled octaves in dense mixes with less predictable instrument timbres.
        independent = excess > .08 * peak[:, k]
        pan_lo, pan_hi = np.ones(len(peak)), np.zeros(len(peak))
        for delta, _ in harmonics:
            if delta > k:
                continue
            contributes = active[:, k-delta] & (peak[:, k-delta] > .15 * peak[:, k])
            other_pan = pan[:, k-delta]
            pan_lo = np.where(contributes, np.minimum(pan_lo, other_pan), pan_lo)
            pan_hi = np.where(contributes, np.maximum(pan_hi, other_pan), pan_hi)
            lower = peak[:, k-delta]
            ratio = peak[:, k] / (.1 + lower)
            recent = ndimage.minimum_filter1d(ratio, 19, origin=9)
            independent |= ((frame[:, k] > .39) & (ratio > .25) & (ratio > recent * 1.8)
                            & active[:, k-delta] & (onset[:, k] > .40))
        independent |= (frame[:, k] > .22) & clear[:, k] & (
            (pan[:, k] < pan_lo - .13) | (pan[:, k] > pan_hi + .13))
        active[:, k] = candidate[:, k] & independent
        for delta, limit in harmonics:
            if k+delta < 88:
                allowed_l[:, k+delta] += left[:, k] * limit * active[:, k]
                allowed_r[:, k+delta] += right[:, k] * limit * active[:, k]
    # Bridge a two-frame trough within a sustain without eroding the first or
    # last frame of a cut. Keep short notes and distinct reattacks of one key.
    active = ndimage.binary_closing(np.pad(active, ((2, 2), (0, 0)), mode="edge"),
                                    structure=np.ones((3, 1), bool))[2:-2]
    # A steady quiet chord tone can fall below the network threshold. A second
    # detector resolves fundamentals from continuous peak frequencies and rejects
    # harmonic multiples before contributing independently sustained tones.
    frame_times = np.arange(len(active)) * HOP / SR
    def sample_tones(hop, width, offset=0):
        times, tonal = resolved_tones(pcm[offset:], hop, width)
        nearest = np.clip(np.rint((frame_times * SR-offset)/hop).astype(int), 0, len(times)-1)
        return tonal[nearest, FIRST_KEY:FIRST_KEY+88]

    active |= sample_tones(882, 4096)
    # The shorter window resolves faster chord changes. Require agreement with
    # a longer window on a different time grid before adding unmodeled pitches.
    active |= sample_tones(661, 3584, min(307, len(pcm)-1)) & sample_tones(701, 5120, min(509, len(pcm)-1))
    # Centered FFT windows extend past hard cuts. Gate their estimates with
    # energy on the much shorter analysis hop to keep count-ins and rests silent.
    padded = np.pad(pcm, ((0, (-len(pcm)) % HOP), (0, 0)))
    rms = np.sqrt(np.mean(padded.reshape(-1, HOP, 2)**2, axis=(1, 2)))
    active &= (rms > .0002)[:, None]
    notes = []
    for col, held in enumerate(active.T):
        edges = np.flatnonzero(np.diff(np.r_[False, held, False]))
        attacks, _ = signal.find_peaks(onset[:, col], height=.45, prominence=.20, distance=5)
        floor = ndimage.minimum_filter1d(peak[:, col], 9, origin=4)
        # A network onset over a decaying tail alone is not a second attack.
        attacks = attacks[peak[attacks, col] > 1.22 * floor[attacks]]
        for a, b in edges.reshape(-1, 2):
            if b - a < 4:
                continue
            starts = [a, *attacks[(attacks >= a + 5) & (attacks <= b - 4)]]
            for start, end in zip(starts, [*starts[1:], b]):
                notes.append([round(start * HOP * 1000 / SR), min(round(duration*1000), round(end * HOP * 1000 / SR)),
                              int(col + FIRST_KEY)])
    notes = recover_chord_tones(pcm, duration, notes)
    return recover_chord_tones(pcm, duration, notes, density_only=True)


def transcription(pcm, duration):
    notes, _ = confirmed_notes(note_candidates(pcm, duration), pitch_evidence(pcm, SR))
    return notes


def signature(backing):
    stamps = []
    for path in (backing, backing.with_name("drums.wav")):
        if path.is_file():
            st = path.stat()
            stamps.append([st.st_size, st.st_mtime_ns])
        else:
            stamps.append(None)
    part = part_path(backing)
    if part:
        stamps.append(hashlib.sha256(part.read_bytes()).hexdigest())
    return [VERSION, *stamps]


def score_path(backing):
    return backing.with_suffix(".carrier.json.gz")


def read_score(backing, version=VERSION):
    try:
        with gzip.open(score_path(backing), "rt") as stream:
            score = json.load(stream)
    except (FileNotFoundError, gzip.BadGzipFile, EOFError, json.JSONDecodeError):
        return None
    stamp = signature(backing)
    if score.get("rendered") and score["source"][1:] == stamp[1:]:
        return score
    if score.get("source") == [version, *stamp[1:]]:
        return score
    return None


def prepare_score(backing, *, archive=False):
    backing = Path(backing)
    cached = read_score(backing)
    if cached is not None:
        if archive and not (cached.get("rendered") or cached.get("arrangement") or "earlier" in cached):
            pcm = pitched_audio(backing, backing.with_name("drums.wav"))
            earlier = note_candidates(pcm, cached["duration"])
            if cached["source"] != signature(backing):
                raise ValueError(f"Stems changed during transcription: {backing}")
            return save_score(backing, dict(cached, earlier=earlier))
        return cached
    stamp = signature(backing)
    with wave.open(str(backing)) as wav:
        duration = wav.getnframes() / wav.getframerate()
    part = part_path(backing)
    if part:
        arrangement = json.loads(part.read_text())
        score = dict(source=stamp, duration=duration, **arrangement)
        if stamp != signature(backing):
            raise ValueError(f"Recording changed while loading its part: {backing}")
        return save_score(backing, score)
    pcm = pitched_audio(backing, backing.with_name("drums.wav"))
    previous = read_score(backing, version=7)
    if previous is not None:
        candidates = previous["notes"]
    else:
        candidates = note_candidates(pcm, duration)
    notes, pan = confirmed_notes(candidates, pitch_evidence(pcm, SR))
    score = dict(source=stamp, duration=duration, notes=notes, pan=pan, earlier=candidates)
    if stamp != signature(backing):
        raise ValueError(f"Stems changed during transcription: {backing}")
    return save_score(backing, score)


def save_score(backing, score):
    with tempfile.NamedTemporaryFile(dir=backing.parent, delete=False) as tmp:
        try:
            with gzip.GzipFile(fileobj=tmp, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(score, separators=(",", ":")).encode())
            tmp.close()
            os.replace(tmp.name, score_path(backing))
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return score


def save_rendered_score(backing, duration, notes):
    """Use the note-on/off data that actually drove this instrumental render."""
    spans = sorted([round(a*1000), round(b*1000), int(key)] for a, b, key in notes
                   if round(b*1000) > round(a*1000))
    return save_score(backing, dict(source=signature(backing), duration=duration, notes=spans,
                                    rendered=True))


def carrier(path, downbeats):
    """Keep the importer's event format for test singing, plus the timed carrier sidecar."""
    score = prepare_score(Path(path))
    changes = {}
    for start, end, key in score["notes"]:
        changes.setdefault(start, []).append((key, 1))
        changes.setdefault(end, []).append((key, -1))
    held = np.zeros(128, dtype=int)
    events = []
    for ms, edges in sorted(changes.items()):
        for key, delta in edges:
            held[key] += delta
        notes = np.flatnonzero(held).tolist()
        if not events or events[-1][1] != notes:
            events.append([ms / 1000, notes])
    return {"duration": score["duration"], "events": events, "transcription": VERSION}
