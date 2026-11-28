"""Measure chord-tone coverage on the backing's clock for the installed carrier library."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from pitched_notes import confirmed_notes, pitch_evidence

from carrier import DENSITY_GRIDS, SR, note_roll, pitch_classes, pitched_audio, stable_fundamentals


def source_confirmation(path, times, duration):
    """Check spliced and tempo-warped excerpts on their original recording clock."""
    from import_job import BAR
    from carrier import resolved_tones

    edition = path.parent.parent
    slug = edition.name
    manifest = Path(__file__).with_name("starred-arrangements") / (slug.removeprefix("studio-") + ".json")
    if not manifest.is_file():
        return None
    plan = json.loads(manifest.read_text())
    source = edition.parent.parent / "sources" / slug
    info = source / "master.info.json"
    if (slug != plan["slug"] or not info.is_file() or not (source / "instrumental.wav").is_file()
            or json.loads(info.read_text()).get("id") != plan["recording"]):
        return None
    cut = next((c for c in plan["cuts"] if c["id"] == path.parent.name), None)
    if not cut:
        return None
    grids = [np.asarray(s["grid"]) for s in cut["sections"]]
    if abs(sum(len(g) - 1 for g in grids) * BAR - duration) > .002:
        return None
    original = pitched_audio(source / "instrumental.wav", source / "drums.wav")
    reference = np.zeros((len(times), 128), bool)
    start = 0.
    for grid in grids:
        local = np.arange(len(grid)) * BAR
        rows = (times >= start) & (times < start + local[-1])
        target = np.interp(times[rows] - start, local, grid)
        # Context on either side avoids inventing silence at splice boundaries.
        a, b = max(0, round((grid[0] - .25) * SR)), round((grid[-1] + .25) * SR)
        t, tones = resolved_tones(original[a:b], hop=320, width=3584)
        tones = ndimage.maximum_filter1d(tones, 7, axis=0, mode="constant")
        frames = np.clip(np.rint((target * SR - a) / 320).astype(int), 0, len(t) - 1)
        reference[rows] = tones[frames]
        start += local[-1]
    return reference


def score_metrics(path, score, extra=()):
    pcm = pitched_audio(path, path.with_name("drums.wav"))
    # Use a fresh grid between the extraction frames. Score stable fundamentals
    # for which a second, higher-resolution frequency estimate agrees. Count a
    # chord as covered only when every observed pitch class is present together.
    sounding = [*score["notes"], *extra]
    grid, measured = stable_fundamentals(pcm, *DENSITY_GRIDS)
    evidence = pitch_evidence(pcm, SR)
    checked, _ = confirmed_notes(score["notes"], evidence)
    def musical_reference(times, reference):
        idx = np.clip(np.searchsorted(evidence[0], times), 0, len(evidence[0])-1)
        reference[:,21:109] &= evidence[1][idx] & ~evidence[3][idx]
        return reference
    observed_count = pitch_classes(musical_reference(grid, measured)).sum(axis=1)
    scored_count = pitch_classes(note_roll(grid, sounding, .035)).sum(axis=1)
    thin = (observed_count >= 2) & (scored_count < observed_count)
    times, reference = stable_fundamentals(pcm, (643, 4032), (739, 4480))
    reference = musical_reference(times, reference)
    source = source_confirmation(path, times, score["duration"])
    if source is not None:
        reference &= source
    # A single intersection of the analysis grids during a sweep does not
    # establish a chord tone. Require three consecutive qualified observations.
    reference = ndimage.binary_opening(reference, structure=np.ones((3,1),bool))
    predicted = note_roll(times, sounding, .035)
    chord = pitch_classes(reference)
    heard = pitch_classes(predicted)
    poly = chord.sum(axis=1) >= 2
    ref = chord[poly]
    samples = int(ref.sum())
    matches = int((ref & heard[poly]).sum())
    enough_voices = (heard[poly].sum(axis=1) >= ref.sum(axis=1)).sum()
    return dict(notes=len(sounding), extractedNotes=len(score["notes"]), confirmedNotes=len(checked), referenceChordTones=samples, matchedChordTones=matches,
                densityFrames=int((observed_count >= 2).sum()), underVoicedFrames=int(thin.sum()),
                equalOrRicherFrames=int(enough_voices),
                observedFrames=int(poly.sum()), completeFrames=int((~(ref & ~heard[poly]).any(axis=1)).sum()),
                referenceKeys=int(reference.sum()), matchedKeys=int((reference & predicted).sum()),
                rendered=bool(score.get("rendered")), checkedSourceRecording=source is not None)


def check_regions(songs):
    """Play the real carrier monitor through every selectable recording and cut."""
    from audio import BUILD, MEDIA, F, FP, PARAMS
    from backing_loops import backing_loops
    from carrier import read_score
    from clip_clock import ClipClock
    from desk_audio import DeskAudio

    desk = DeskAudio()
    checked = 0
    try:
        desk.load_catalog_scores(BUILD / "sounds", MEDIA / "sounds")
        desk.import_scores(songs)
        params = (F * len(PARAMS))(*(p[3] for p in PARAMS))
        for family, choices in backing_loops(songs).items():
            for choice in choices.values():
                song = songs[choice.source]
                score = read_score(song["carrierBacking"])
                region = choice.region(song, score["duration"])
                lo, hi = region["start"], region["end"]
                notes = score["notes"] + song.get("carrierNotesMs", [])
                intervals = [(max(lo, a / 1000), min(hi, b / 1000))
                             for a, b, _ in notes if b / 1000 > lo and a / 1000 < hi]
                sounding = [span for span in intervals if span[1] - span[0] > .08]
                if intervals:
                    candidates = sounding or intervals
                    start, end = candidates[len(candidates) // 2]
                else:
                    # Confirm an empty region against the recording before accepting silence.
                    path = song["carrierBacking"]
                    pcm = pitched_audio(path, path.with_name("drums.wav"))[round(lo*SR):round(hi*SR)]
                    _, observed = stable_fundamentals(pcm, (587, 3968), (797, 4608))
                    if observed.any():
                        raise ValueError(f"Empty carrier over pitched backing for {family}, {choice.label}")
                    start, end = (lo + hi) / 2, hi
                # The silent bed makes any measured signal come from the scored synth.
                bed = np.zeros(2 * round(hi * 48000), "i2")
                if not desk.prepare_track(bed, None, region):
                    raise ValueError(f"Could not stage {choice.source}")
                desk.call("stage_clock", *desk.clock_args(ClipClock(region)))
                desk.call("stage_lane", 0)
                desk.call("stage_carrier_backing", True)
                desk.call("launch_at", -1, 0)
                desk.call("config", 0, 1, song["backingScoreId"].encode(), 0, 0, params)
                desk.call("carrier_clock", 0, 0, song["bpm"])
                desk.call("tempo", song["bpm"])
                desk.call("glide", 0, 0)
                desk.call("transport", 1, round((start - lo) * 48000))
                desk.call("transport", 0, 1)
                left, right = np.zeros((2, max(1, round(min(.08, end - start) * 48000))), "f4")
                desk.call("render", None, left.ctypes.data_as(FP), right.ctypes.data_as(FP), len(left))
                if not np.isfinite(left).all() or (intervals and np.std(left) < .001):
                    raise ValueError(f"Silent carrier for {family}, {choice.label}, at {start:.3f} s")
                desk.call("collect")
                checked += 1
    finally:
        desk.close()
    return checked
