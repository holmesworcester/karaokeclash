"""Require a pitch plateau and measure its stereo position before admitting a MIDI key."""
import numpy as np
from scipy import ndimage


def pitch_evidence(pcm, rate):
    hop, width, fft = round(.016 * rate), 3072, 8192
    count = (len(pcm) + hop - 1) // hop
    hz = 440 * 2 ** ((np.arange(21, 109) - 69) / 12)
    center = (hz * fft / rate).round().astype(int)
    # Nearest spectral maximum, including detuned guitar and low bass bins.
    offsets = np.arange(-2, 3)
    padded = np.pad(pcm, ((width//2, width), (0, 0)))
    win = np.hanning(width)[:, None]
    cents = np.full((count, 88), np.nan, dtype='f4')
    pans = np.zeros_like(cents)
    levels = np.zeros_like(cents)
    for first in range(0, count, 96):
        size = min(96, count-first)
        frames = np.lib.stride_tricks.sliding_window_view(padded[first*hop:], width, axis=0)[::hop][:size]
        stereo = abs(np.fft.rfft(frames.transpose(0, 2, 1)*win, fft, axis=1))
        mag = stereo.max(axis=2)
        peak = center + offsets[np.argmax(mag[:, center[:, None]+offsets], axis=2)]
        row = np.arange(size)[:, None]
        a, b, c = (mag[row, peak+i] for i in (-1, 0, 1))
        logs = [np.log(np.maximum(x, 1e-8)) for x in (a, b, c)]
        shift = (.5*(logs[0]-logs[2])) / np.minimum(-1e-8, logs[0]-2*logs[1]+logs[2])
        fitted = np.log2((peak+np.clip(shift,-.5,.5))*rate/fft/hz)*1200
        flank = np.maximum(4, np.rint(peak*.052).astype(int))
        floor = .5*(mag[row,np.maximum(0,peak-flank)] + mag[row,peak+flank])
        credible = (b>=a) & (b>c) & (b>1.5*floor+.09) & (b>.027*mag.max(axis=1)[:, None])
        sl = slice(first,first+size)
        levels[sl] = b
        cents[sl] = np.where(credible & (abs(fitted)<38), fitted, np.nan)
        left, right = stereo[row,peak].transpose(2,0,1)
        pans[sl] = (right-left)/(right+left+.001)
    # A glissando crosses the center of a semitone, but does not settle there.
    # Three consecutive measurements must stay within a fifth of a semitone.
    high = ndimage.maximum_filter1d(np.nan_to_num(cents,nan=999.),3,axis=0)
    low = ndimage.minimum_filter1d(np.nan_to_num(cents,nan=-999.),3,axis=0)
    stable = high-low < 18
    # At a hard file edge a centered window sees only one side of a sweep.
    stable[:2] = False
    stable[-2:] = False
    # An electric guitar can have a louder second or fifth partial than its
    # fundamental. Check whether their stereo position and amplitude ratio
    # move together throughout the note. A separately attacked octave or a
    # differently panned instrument can still occupy the same frequency.
    partial = np.zeros_like(stable)
    for harmonic in range(2, 11):
        semitones = 12*np.log2(harmonic)
        delta = round(semitones)
        if abs(semitones-delta) > .35:
            continue
        upper, lower = levels[:,delta:], levels[:,:-delta]
        ratio = upper/(lower+.001)
        lo = ndimage.minimum_filter1d(ratio, 9, axis=0)
        hi = ndimage.maximum_filter1d(ratio, 9, axis=0)
        tuned = abs(cents[:,delta:]-cents[:,:-delta] - 100*(semitones-delta)) < 10
        partial[:,delta:] |= (stable[:,:-delta] & tuned & (lower > .16*upper)
                              & (abs(pans[:,delta:]-pans[:,:-delta]) < .095)
                              & (lo > .83*hi))
    # A kick can briefly mask the lower fundamental of a ringing guitar
    # string. Its overtone before and after the hit belongs to the same note.
    partial |= ndimage.binary_closing(partial, structure=np.ones((19, 1), bool))
    return np.arange(count)*hop/rate, stable, pans, partial


def confirmed_notes(notes, evidence):
    times, stable, pans, partial = evidence
    kept, positions = [], {}
    for start, end, key in notes:
        a,b = np.searchsorted(times, [start*.001,end*.001])
        reliable = stable[a:b,key-21]
        if end-start < 55 or len(reliable)==0 or reliable.sum()<max(3,.30*len(reliable)):
            continue
        if np.count_nonzero(reliable & partial[a:b,key-21]) >= .62*reliable.sum():
            continue
        kept.append([start,end,key])
        positions.setdefault(key,[]).append(float(np.median(pans[a:b,key-21][reliable])))
    return kept, {str(k):round(float(np.median(v)),3) for k,v in positions.items()}
