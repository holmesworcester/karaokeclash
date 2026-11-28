"""Score a single singer-driven bass voice on the source clock before stretching cuts."""

import itertools
import subprocess

import numpy as np
from scipy import ndimage, signal

RATE = 4000
HOP = 40  # 10 ms, independent of the beat grid
WINDOW = 512
FFT = 2048


def bass_audio(path):
    pcm = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-ar",
            str(RATE),
            "-ac",
            "2",
            "-f",
            "f32le",
            "-",
        ]
    )
    return np.frombuffer(pcm, "<f4").reshape(-1, 2)


def bass_pitch_frames(pcm, bpm, isolated):
    """Track a resolved low fundamental in either stereo channel."""
    highest = 55 if isolated else 47
    keys = np.arange(23, highest + 1)
    bins = 440 * 2 ** ((keys - 69) / 12) * FFT / RATE
    centers = np.rint(bins).astype(int)
    # Interpolate the position of a local spectral maximum. At bass frequencies
    # the skirt of one peak can cover several adjacent MIDI notes.
    offsets = np.arange(-2, 3)
    window = np.hanning(WINDOW)[:, None]
    padded = np.pad(pcm, ((WINDOW // 2, WINDOW // 2), (0, 0)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW, axis=0)[::HOP]
    result = np.full((len(pcm) + HOP - 1) // HOP, -1, dtype=int)
    levels = np.zeros_like(result, dtype=float)
    for first in range(0, len(result), 256):
        chunk = frames[first : first + min(256, len(result) - first)].transpose(0, 2, 1)
        mag = np.abs(np.fft.rfft(chunk * window, FFT, axis=1)).max(axis=2)
        selected = np.argmax(mag[:, centers[:, None] + offsets], axis=2)
        peak = centers + offsets[selected]
        rows = np.arange(len(chunk))[:, None]
        a, b, c = (mag[rows, peak + j] for j in (-1, 0, 1))
        shift = (a - c) / np.minimum(-1e-12, 2 * (a - 2 * b + c))
        fitted = peak + np.clip(shift, -0.5, 0.5)
        tuned = np.abs(12 * np.log2(fitted / bins)) < 0.40
        credible = (b >= a) & (b > c) & tuned & (b > 0.002 * WINDOW / 4)
        # Discount low-frequency leakage from an upper line or a loud kick.
        credible &= b > 0.045 * mag.max(axis=1, keepdims=True)
        strength = (
            b
            + 0.55 * mag[rows, np.rint(2 * fitted).astype(int)]
            + 0.25 * mag[rows, np.rint(3 * fitted).astype(int)]
        )
        strength = np.where(credible, strength, 0)
        best = strength.argmax(axis=1)
        indices = np.arange(len(best))
        result[first : first + len(best)] = np.where(strength[indices, best] > 0, keys[best], -1)
        levels[first : first + len(best)] = b[indices, best]

    # A centered transform can see into the next offbeat. Close the note in a
    # real amplitude valley, including between two attacks on the same pitch.
    low = signal.sosfilt(
        signal.butter(4, 440 * 2 ** ((highest + 2 - 69) / 12), fs=RATE, output="sos"), pcm, axis=0
    )
    low = np.pad(low, ((0, (-len(low)) % HOP), (0, 0)))
    energy = np.sqrt(np.mean(low.reshape(-1, HOP, 2) ** 2, axis=(1, 2)))
    result[energy < np.maximum(0.0008, 0.14 * ndimage.maximum_filter1d(energy, 21))] = -1
    # A keyboard can hold the same pitch under a pulsing bass in the full mix.
    # Compare with a nearby beat so that quieter sections retain their own dynamics.
    neighborhood = 1 + 2 * round(0.68 * 60 / bpm * RATE / HOP)
    floor = 0.12 if isolated else 0.34
    result[levels < floor * ndimage.maximum_filter1d(levels, neighborhood)] = -1
    return ndimage.median_filter(result, size=3, mode="nearest")


def chord_bass(score):
    """When a recording has no low line, use sustained unambiguous major/minor chords."""
    events = score["events"]
    merged = []
    for i, (start, pitches) in enumerate(events):
        classes = {k % 12 for k in pitches}
        roots = [
            k
            for k in classes
            if classes == {k, (k + 7) % 12, (k + 3) % 12}
            or classes == {k, (k + 7) % 12, (k + 4) % 12}
        ]
        if len(roots) != 1:
            continue
        root = 30 + (roots[0] - 30) % 12
        stop = events[i + 1][0] if i + 1 < len(events) else score["duration"]
        a, b = round(1000 * start), round(1000 * stop)
        if merged and merged[-1][1:] == [a, root]:
            merged[-1][1] = b
        else:
            merged.append([a, b, root])
    return [note for note in merged if note[1] - note[0] >= 200]


def aligned_bass(path, seconds):
    if path is None or path.is_symlink() or not path.is_file():
        return None
    try:
        pcm = bass_audio(path)
    except subprocess.CalledProcessError:
        return None
    return pcm if len(pcm) and abs(len(pcm) / RATE - seconds) <= 0.15 else None


def recording_bass(backing, drums, isolated, score, bpm):
    """Return [start_ms, end_ms, MIDI pitch] for the full, unwarped recording."""
    pcm = aligned_bass(isolated, score["duration"])
    has_stem = pcm is not None
    if not has_stem:
        pcm = bass_audio(backing)
        percussion = bass_audio(drums)
        # The import normalizes these two stems onto the same sample clock.
        gain = (pcm * percussion).sum(axis=0, dtype=float) / (
            (percussion**2).sum(axis=0, dtype=float) + 1e-12
        )
        pcm = pcm - np.clip(gain, 0, 1.5).astype("f4") * percussion
    if not len(pcm) or np.max(np.abs(pcm)) < 0.001:
        return []
    frames = bass_pitch_frames(pcm, bpm, has_stem)
    edges = np.r_[0, np.flatnonzero(np.diff(frames)) + 1, len(frames)]
    duration_ms = round(score["duration"] * 1000)
    notes = [
        [a * 10, min(b * 10, duration_ms), int(frames[a])]
        for a, b in itertools.pairwise(edges)
        if b - a >= 7 and frames[a] >= 0 and a * 10 < duration_ms
    ]
    if not notes:
        return [] if has_stem else chord_bass(score)
    # Use one octave for the performance to preserve the contour across chord
    # changes. Already-deep bass stays where it is. Fold rare sub-30 Hz outliers up.
    # Weight by sounding time and exclude rejected short transients.
    pitches = np.repeat([k for a, b, k in notes], [(b - a + 9) // 10 for a, b, k in notes])
    shift = 12 if np.percentile(pitches, 10) >= 37 and np.median(pitches) >= 40 else 0
    return [[int(a), int(b), k - shift if k - shift >= 24 else k - shift + 12] for a, b, k in notes]


def cut_bass(notes, grid, bars, bar_seconds):
    """Project attacks and releases, including tied notes crossing a bar or cut boundary."""
    positions = np.arange(len(grid))
    output = []
    for first in range(0, bars, len(grid) - 1):
        count = min(bars - first, len(grid) - 1)
        for start, end, pitch in notes:
            a, b = max(start / 1000, grid[0]), min(end / 1000, grid[count])
            if a < b:
                times = np.interp([a, b], grid, positions)
                lo, hi = np.rint((first + times) * bar_seconds * 1000).astype(int)
                if lo < hi:
                    output.append([int(lo), int(hi), pitch])
    return output
