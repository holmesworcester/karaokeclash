"""Prepare or audit the carrier for each playable backing, including previously unscored songs."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from backing_loops import calibrated_catalog
from carrier import prepare_score
from playback import catalog, playback_files


def prepare_one(path, audit, extra):
    score = prepare_score(path, archive=True)
    if audit:
        from carrier_audit import score_metrics
        if score.get("arrangement"):
            from recording_parts import part_path
            expected = json.loads(part_path(path).read_text())
            if score["notes"] != expected["notes"] or not score["notes"]:
                raise ValueError("Playback differs from the recording arrangement")
            return {"notes": len(score["notes"]), "arrangement": score["arrangement"]}
        return score_metrics(path, score, extra)
    return {"notes": len(score["notes"])}


def run(workers, audit=False, report=None):
    songs = calibrated_catalog(catalog())
    paths = {}
    for name, song in songs.items():
        backing, _ = playback_files(song)
        paths.setdefault(backing, []).append(name)
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {}
        for path, entries in paths.items():
            # Count written corrections in file-level measurements only when
            # every entry sharing that recording uses them.
            overlays = [set(map(tuple, songs[name].get("carrierNotesMs", []))) for name in entries]
            jobs[path] = pool.submit(prepare_one, path, audit, sorted(set.intersection(*overlays)))
        for path, job in jobs.items():
            try:
                result = job.result()
                results[str(path)] = dict(entries=paths[path], **result)
                summary = f'{result["notes"]} notes'
                if result.get("arrangement"):
                    summary += ", " + result["arrangement"]
                elif audit:
                    summary += f', {result["matchedChordTones"]}/{result["referenceChordTones"]} measured chord tones'
                    if not result["rendered"] and result["confirmedNotes"] < .98 * result["extractedNotes"]:
                        raise ValueError("Over 2% of carrier notes lack a stable independent pitch")
                    # Mixed recordings can contain ambiguous partials. A score
                    # missing a substantial share of resolved tones needs review.
                    if (not result["rendered"] and result["matchedChordTones"] <
                            .90 * result["referenceChordTones"]):
                        raise ValueError(f"Carrier misses over 10% of measured chord tones: {summary}")
                    if result["referenceKeys"] and not result["notes"]:
                        raise ValueError("pitched backing has an empty carrier")
                print(f'{", ".join(paths[path])}  {summary}', flush=True)
            except Exception as error:
                failures.append(path)
                print(f'{path}: {error}', flush=True)
    if report is not None:
        Path(report).write_text(json.dumps(results, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} backings failed")
    print(f"{len(songs)} entries across {len(paths)} backing files")
    if audit:
        from carrier_audit import check_regions
        print(f"{check_regions(songs)} selectable regions rendered through the native carrier")
