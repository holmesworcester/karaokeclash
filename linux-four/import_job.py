"""Prepare an import privately, validate the musical grid, then publish to the local desk."""

import argparse
import bisect
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave

import numpy as np

from library import IMPORTS
from audio import DATA
from recording import local_path, fetch
from carrier import carrier
from import_bass import recording_bass, cut_bass as project_notes
from import_queue import enqueue, job_root, read, save, sync_dir

RATE = 48000
BPM = 126
BAR = 4 * 60 / BPM
LENGTHS = (4, 8, 16, 32)


def stage(message):
    print(message, flush=True)


def duration(path):
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
    )


def analyze(job, recording, model, bars, notes, original_name):
    installed_skill = Path.home() / ".codex/skills/karaoke-is-all-you-need/SKILL.md"
    bundled_skill = Path(__file__).resolve().parents[1] / ".agents/skills/karaoke-is-all-you-need/SKILL.md"
    skill = Path(
        os.environ.get(
            "KARAOKE_SKILL",
            installed_skill if installed_skill.is_file() else bundled_skill,
        )
    )
    tools = Path(
        os.environ.get(
            "KARAOKE_SKILL_TOOLS", "/mnt/storage/holmes-wt/karaoke-is-all-you-need"
        )
    )
    drums_model = Path(os.environ.get("KARAOKE_DRUM_MODEL", tools / "build/models/drums.onnx"))
    if not drums_model.is_file() and "KARAOKE_DRUM_MODEL" not in os.environ:
        drums_model = DATA / "chants-lbd/sources/drums.onnx"
    if recording is None:
        source_instructions = f"""Song requested: {original_name!r}. Find this edition in the local audio collection at {DATA}/chants-lbd/sources
or among other local original recordings identified by the operator. Copy or decode the matching full recording to recording.wav in this job.
If there is no matching local recording, write source-request.json with a "source" string containing the exact public HTTP(S) source URL
or an artist, song title and edition search query. Then stop so the desk can fetch it locally. Do not synthesize or substitute a song."""
    else:
        source_instructions = f"Use {recording.name!r} from this folder. Requested song or original filename: {original_name!r}. Check that the edition matches the request before preparing it."
    prompt = f"""Prepare a recording for a private live karaoke desk. {source_instructions}
Use the attached karaoke-is-all-you-need skill and the local scripts at {tools}.
Work only in this song directory. The recording and any stems stay local. Cloud reasoning may use text and musical measurements.
Use CPU only. Keep the local Radeon available for performance. Do not publish, upload, or modify other directories.
Identify the actual edition. Obtain or separate a clean vocal and a matching instrumental, use the skill's local alignment tools,
the provisioned Python interpreter at {tools}/.venv/bin/python and model files (separator.onnx, align.onnx, vocab.json) at {tools}/build/models.
On an interrupted request, inspect the existing stems, alignment and measurements in this directory and finish from valid work.
Check the timing at several entrances and repeats, and identify the chorus or principal repeated refrain. No artwork or video is needed.
Save backing.wav (full vocal-removed accompaniment), drums.wav (isolated percussion only), and guide.wav (isolated vocal) against the original recording clock, including the intro.
When an isolated bass stem is available for this exact edition, save it as bass.wav on that same clock, including rests.
The desk derives the vocoder bass from that stem or from the instrumental with percussion removed.
When the recording has pitched backing voices, isolate them from the lead and check their notes against the vocal stem.
Write the confirmed chant, response and harmony notes in analysis.json as backingVocalNotesMs, an array of
[start_ms, end_ms, MIDI pitch] on the recording clock. Leave rests, spoken interjections and effects silent.
For percussion separation, the HT-Demucs FT drums specialist is expected at {drums_model}.
Use {tools}/scripts/separate_drums.py with backing.wav, the model path, and drums.wav.
Save analysis.json with title, artist, key, bpm (quarter notes/minute), meter (4), downbeats (measured bar-start seconds
including the ending boundary of the final complete bar), optional fullStartMs (the measured start on the original clock if the operator wants to skip an intro), rows (array of [array of sung words, matching array of
[start_ms,end_ms] pairs] for the whole song), teaser (one object) and hooks (an ordered array).
Each object has label, start_bar (zero-based downbeat index), bars, and voice_ms (two source-clock
milliseconds in the rests bracketing the intended words). Choose Teaser and Main independently.
For Teaser, use the identifiable opening line or short opening clause, including its first consonant
and held last word. If that phrase spans multiple subtitle rows, provide its exact token list in words. Follow it with accompaniment alone for a clear gap before it repeats, preferably
a full bar. For the first hook (Main), use the complete chorus or principal refrain. Add another
distinct hook if appropriate. Keep both accompaniment seams on actual downbeats in the same
groove. Include a pickup bar when needed. An extra whole bar to finish the chorus is fine.
Check the first and last words, the vocal-free gap, and two consecutive passes with the guide
against both backing and percussion. Describe the selected words and bar evidence in review.
The requested Main length is {bars} bars. It can repeat whole choruses to approach that length.
Inspect for gradual tempo drift and grid errors. Use actual downbeats across the recording. Do not invent timings or stems.
If the real meter is incompatible, report that instead of passing it off as 4/4. Write a short source and verification
summary in the JSON field review. Operator's song notes: {notes}

{skill.read_text()}"""
    if (job / "error.json").is_file():
        prompt += "\nThe last attempt failed this check. Correct the deliverables: " + read(job / "error.json")
    cmd = [
        os.environ.get("KARAOKE_CODEX", "codex"),
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        str(job),
    ]
    if model.strip():
        cmd += ["-m", model.strip()]
    cmd += ["-"]
    with (job / "workflow.log").open("a") as log:
        result = subprocess.run(
            cmd,
            pass_fds=tuple(map(int, os.environ.get("KARAOKE_IMPORT_LEASE", "").split())),
            input=prompt,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1",
                "XDG_CACHE_HOME": str(job / "cache"),
                "HF_HOME": str(job / "cache/hf"),
                "PIP_CACHE_DIR": str(job / "cache/pip"),
                "TORCH_HOME": str(job / "cache/torch"),
            },
        )
    if result.returncode < 0 or result.returncode in (128 + signal.SIGKILL, 128 + signal.SIGTERM):
        raise InterruptedError("Codex was interrupted")
    if result.returncode:
        raise RuntimeError(f"Codex exited with status {result.returncode}. See {job / 'workflow.log'}")


def validate(meta, seconds):
    if not all(
        isinstance(meta.get(k), str) and meta[k].strip()
        for k in ("title", "artist", "key", "review")
    ):
        raise ValueError("Missing title, artist, key or timing review")
    if meta["meter"] != 4 or not 35 <= meta["bpm"] <= 300:
        raise ValueError("Expected a measured 4/4 grid and BPM in range")
    grid = meta["downbeats"]
    if len(grid) < 5 or not all(
        math.isfinite(x) and 0 <= x <= seconds + 0.02 for x in grid
    ):
        raise ValueError("Downbeats lie outside the recording")
    if any(not 0.7 <= b - a <= 8 for a, b in zip(grid, grid[1:])):
        raise ValueError("Duplicate, missing or unordered downbeats")
    if abs(np.median(np.diff(grid)) * meta["bpm"] / 240 - 1) > 0.15:
        raise ValueError("Tempo disagrees with bar grid")
    voices = meta.get("backingVocalNotesMs", [])
    if not isinstance(voices, list) or any(
        not isinstance(note, list) or len(note) != 3 or any(type(x) is not int for x in note)
        or not 0 <= note[0] < note[1] <= round(seconds * 1000) or not 0 <= note[2] <= 127
        for note in voices
    ):
        raise ValueError("Invalid backing vocal MIDI notes")
    rows = meta["rows"]
    if not rows:
        raise ValueError("No timed lyrics")
    for words, spans in rows:
        if not words or len(words) != len(spans):
            raise ValueError("Word and span counts differ")
        for w, (a, b) in zip(words, spans):
            if (
                not isinstance(w, str)
                or not w.strip()
                or not 0 <= a < b <= seconds * 1000 + 20
            ):
                raise ValueError("Invalid lyric interval")
        if any(a[0] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("Words out of order")
    if not meta["hooks"]:
        raise ValueError("No singable hook")
    if not meta.get("teaser"):
        raise ValueError("Choose an opening line and a vocal-free gap for Teaser")
    teaser = meta["teaser"]
    for h in [teaser, *meta["hooks"]]:
        i, n = h["start_bar"], h["bars"]
        if (type(i) != int or type(n) != int or not 0 <= i < i + n < len(grid)
                or not h["label"].strip()):
            raise ValueError("Loop is outside the measured grid")
        lo, hi = h["voice_ms"]
        if not grid[i] * 1000 <= lo < hi <= grid[i + n] * 1000:
            raise ValueError("Vocal window is outside the loop")
        kept = selected_rows(rows, lo, hi)
        if not kept:
            raise ValueError("Loop contains no sung words")
        if h is teaser:
            if len(kept) > 1:
                # A musical line can span adjacent subtitle rows. In that case the
                # author must identify the exact sung phrase across those rows.
                words = [w for tokens, _ in kept for w in tokens]
                if h.get("words") != words:
                    raise ValueError("Identify the single teaser phrase across lyric rows")
            if grid[i + n] - hi / 1000 < (grid[i + n] - grid[i + n - 1]) / 4:
                raise ValueError("Leave a vocal-free beat after the teaser phrase")
        elif any(spans[0][0] < edge < max(b for _, b in spans)
                 for _, spans in rows for edge in (lo, hi)):
            raise ValueError("Main cuts through a lyric line")


def selected_rows(rows, lo, hi):
    selected = []
    for words, spans in rows:
        pairs = []
        for word, (a, b) in zip(words, spans):
            if b <= lo or a >= hi:
                continue
            if a < lo or b > hi:
                raise ValueError(f"Vocal window cuts through {word!r}")
            pairs.append((word, [a, b]))
        if pairs:
            selected.append(tuple(zip(*pairs)))
    return selected


def normalize(path, out, frames):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-ar",
            str(RATE),
            "-ac",
            "2",
            "-af",
            f"aresample={RATE},apad,atrim=end_sample={frames}",
            "-c:a",
            "pcm_s16le",
            out,
        ],
        check=True,
    )


def check_stems(backing, guide):
    energies = np.zeros(3)
    with wave.open(str(backing)) as bed, wave.open(str(guide)) as voice:
        while True:
            a = np.frombuffer(bed.readframes(RATE), "<i2").astype(float)
            b = np.frombuffer(voice.readframes(RATE), "<i2").astype(float)
            n = min(len(a), len(b))
            if not n:
                break
            a, b = a[:n] / 32768, b[:n] / 32768
            d = a - b
            energies += [a @ a, b @ b, d @ d]
    if min(energies[:2]) <= 1e-6 or energies[2] < 0.000001 * max(energies[:2]):
        raise ValueError("Expected distinct, audible backing and guide stems")


def warped_stem(source, dest, grid, bars, voice_ms=None):
    # Tempo-correct each measured bar. Give atempo neighboring audio so its
    # overlap-add windows do not repeatedly fade the first consonant of a bar.
    with wave.open(str(source)) as src, wave.open(str(dest), "wb") as out:
        out.setparams((2, 2, RATE, 0, "NONE", ""))
        overlap = round(0.003 * RATE)
        ramp = np.linspace(0, 1, overlap)[:, None]
        cache = {}
        order = []
        for j in range(bars):
            i = j % (len(grid) - 1)
            n = round((j + 1) * BAR * RATE) - round(j * BAR * RATE)
            order.append((i, n))
            if (i, n) in cache:
                continue
            a, b = grid[i : i + 2]
            ratio = (b - a) / BAR
            # Keep each FFmpeg atempo stage in its pitch-preserving overlap range.
            stages = max(1, math.ceil(abs(math.log2(ratio))))
            tempo = ",".join([f"atempo={ratio ** (1 / stages)}"] * stages)
            start = max(0, round((a - 0.15) * RATE))
            src.setpos(start)
            count = min(src.getnframes() - start, round((b + 0.15) * RATE) - start)
            pcm = src.readframes(count)
            if voice_ms is not None:
                # Silence neighboring lyrics before stretching. This also protects
                # joins inside repeated hooks, where the desk gate never wraps.
                data = np.frombuffer(pcm, "<i2").reshape(-1, 2).copy()
                lo, hi = (round(ms * RATE / 1000) - start for ms in voice_ms)
                data[:max(0, lo)] = 0
                data[max(0, hi):] = 0
                pcm = data.tobytes()
            cut = round((a - start / RATE) * RATE / ratio)
            out_pcm = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "s16le",
                    "-ar",
                    str(RATE),
                    "-ac",
                    "2",
                    "-i",
                    "-",
                    "-af",
                    f"{tempo},apad,atrim=start_sample={cut}:end_sample={cut + n + overlap}",
                    "-f",
                    "s16le",
                    "-",
                ],
                input=pcm,
            )
            if len(out_pcm) != (n + overlap) * 4:
                raise ValueError("Incomplete normalized bar")
            cache[i, n] = np.frombuffer(out_pcm, "<i2").reshape(-1, 2)
        # Crossfade previous stretched audio over each join, including the final
        # bar into the opening bar. Overlap never advances the bar downbeat.
        prev = cache[order[-1]][-overlap:]
        for key in order:
            clip = cache[key]
            join = prev * (1 - ramp) + clip[:overlap] * ramp
            out.writeframesraw(np.rint(join).astype("<i2").tobytes())
            out.writeframesraw(clip[overlap:-overlap].tobytes())
            prev = clip[-overlap:]


def warped_rows(rows, grid, bars):
    n = len(grid) - 1
    out = []
    for cycle in range(0, bars, n):
        count = min(n, bars - cycle)

        def project(ms):
            sec = min(grid[count], max(grid[0], ms / 1000))
            i = min(count - 1, max(0, bisect.bisect_right(grid, sec) - 1))
            return round(
                (cycle + i + (sec - grid[i]) / (grid[i + 1] - grid[i])) * BAR * 1000
            )

        for words, spans in rows:
            inside = [
                (w, [project(a), project(b)])
                for w, (a, b) in zip(words, spans)
                if a < grid[count] * 1000 and b > grid[0] * 1000
            ]
            inside = [(w, t) for w, t in inside if t[0] < t[1]]
            if inside:
                out.append([[w for w, t in inside], [t for w, t in inside]])
    return sorted(out, key=lambda row: row[1][0][0])


@contextmanager
def publication_folder(imports, job):
    if job is None:
        with tempfile.TemporaryDirectory(prefix=".publish-", dir=imports) as folder:
            yield Path(folder)
        return
    # Only the leased job can own this staging folder. Reclaim partial renders
    # left by a hard stop before spending space on a new attempt.
    folder = imports / (".publish-" + job.name)
    if folder.is_dir():
        shutil.rmtree(folder)
    folder.mkdir(mode=0o700)
    try:
        yield folder
    finally:
        shutil.rmtree(folder)


def publish(job, recording, imports, bars, sections=False, *, request=None):
    meta = json.loads((job / "analysis.json").read_text())
    seconds = duration(recording)
    validate(meta, seconds)
    imports.mkdir(parents=True, exist_ok=True)
    # The scanner reads only */song.json. Stage one level deeper on the target filesystem.
    with publication_folder(imports, job if request else None) as staging:
        out = staging / "song"
        out.mkdir()
        for name in ("backing", "drums", "guide"):
            path = job / (name + ".wav")
            if path.is_symlink() or abs(duration(path) - seconds) > 0.15:
                raise ValueError(f"{name} must match the source clock")
            normalize(path, out / (name + ".wav"), round(seconds * RATE))
        check_stems(out / "backing.wav", out / "guide.wav")
        full = {
            "id": "full",
            "label": "Full song",
            "start": 0,
            "end": seconds,
            "loop": False,
            "bpm": meta["bpm"],
            "bars": "full",
            "rows": {"lead": meta["rows"]},
            "audio": ".",
        }
        if "fullStartMs" in meta:
            start = meta["fullStartMs"]
            if type(start) is not int or not 0 <= start < seconds * 1000:
                raise ValueError("Full start lies outside the recording")
            full["fullStartMs"] = start
        voices = meta.get("backingVocalNotesMs", [])
        if voices:
            full.update(carrierNotesMs=voices, carrierNotesLabel="Backing vocals")
        full["downbeats"] = meta["downbeats"]
        full["score"] = carrier(out / "backing.wav", meta["downbeats"])
        stage("Scoring the vocoder bass on the original recording clock")
        bass = recording_bass(out / "backing.wav", out / "drums.wav",
                              job / "bass.wav", full["score"], meta["bpm"])
        full["score"]["bassNotesMs"] = bass
        cuts = [full]
        # A chorus with a pickup can occupy e.g. 17 bars. Repeat only whole
        # passages. An arbitrary split or a partial last repeat severs the line.
        phrases = [(f"hook-{i}", h, max(1, round(bars / h["bars"])) * h["bars"])
                   for i, h in enumerate(meta["hooks"], 1)]
        teaser = meta["teaser"]
        phrases.append(("teaser", teaser, teaser["bars"]))
        if sections:
            total = len(meta["downbeats"]) - 1
            for start in range(0, total, bars):
                n = min(bars, total - start)
                h = dict(label=f"Song bars {start + 1}–{start + n}", start_bar=start, bars=n)
                phrases.append((f"section-{start // bars + 1}", h, bars))
        for ident, phrase, length in phrases:
            label = phrase["label"]
            start = phrase["start_bar"]
            grid = meta["downbeats"][start : start + phrase["bars"] + 1]
            part = out / ident
            part.mkdir()
            stage(f"Fitting {label} to {length} bars at {BPM} BPM")
            gate = phrase.get("voice_ms")
            rows = selected_rows(meta["rows"], *gate) if gate else meta["rows"]
            for name in ("backing.wav", "drums.wav", "guide.wav"):
                warped_stem(out / name, part / name, grid, length, gate if name == "guide.wav" else None)
            cut = {
                "id": ident, "label": f"{label} · {length} bars", "audio": ident,
                "start": 0, "end": round(length * BAR * RATE) / RATE,
                "bpm": BPM, "bars": length, "loop": True,
                "rows": {"lead": warped_rows(rows, grid, length)},
                "score": carrier(part / "backing.wav", np.arange(length + 1) * BAR),
            }
            cut["score"]["bassNotesMs"] = project_notes(bass, grid, length, BAR)
            vocal_part = voices if not gate else [
                [max(a, gate[0]), min(b, gate[1]), pitch]
                for a, b, pitch in voices if a < gate[1] and b > gate[0]
            ]
            if notes := project_notes(vocal_part, grid, length, BAR):
                cut.update(carrierNotesMs=notes, carrierNotesLabel="Backing vocals")
            if ident == "teaser":
                cut["loopRole"] = "teaser"
                cut["teaserWindowMs"] = [round(np.interp(ms / 1000, grid, np.arange(len(grid)))
                                              * BAR * 1000) for ms in gate]
            cuts.append(cut)
        record = {k: meta[k] for k in ("title", "artist", "key", "review")}
        if request:
            record["importRequest"] = {k: request[k] for k in ("source", "notes")}
        record["cuts"] = cuts
        (out / "song.json").write_text(json.dumps(record, ensure_ascii=False))
        target = imports / (job.name if request else uuid.uuid4().hex)
        # Persist the completed media and manifests before exposing the catalog row.
        for directory, _, files in os.walk(out, topdown=False):
            for filename in files:
                with (Path(directory) / filename).open("rb") as stream:
                    os.fsync(stream.fileno())
            sync_dir(directory)
        if request and (job / "cancel").exists():
            raise InterruptedError("Import cancelled")
        if request and target.exists() and not (target / "song.json").is_file():
            # Retire the incomplete publication only after its replacement is durable.
            target.rename(staging / "incomplete")
        out.rename(target)
        sync_dir(imports)
        return target


def resolve_recording(job, source, ask, progress):
    checkpoint = job / "recording.json"
    if checkpoint.is_file():
        name = read(checkpoint)
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Expected a job-local recording checkpoint")
        recording = job / name
        if recording.is_symlink():
            raise ValueError("Expected a job-local recording checkpoint")
        if recording.is_file():
            return recording

    if source.startswith(("http://", "https://")):
        progress("Fetching recording")
        recording = fetch(source, job)
    elif (path := local_path(source)) is not None:
        if not path.is_file():
            raise ValueError("Recording file not found.")
        recording = job / ("recording" + path.suffix)
        shutil.copyfile(path, recording)
    else:
        progress("Codex is identifying the requested recording")
        ask(None)
        recording = job / "recording.wav"
        if recording.is_symlink():
            raise ValueError("Expected a job-local recording.")
        if not recording.is_file():
            choice = job / "source-request.json"
            if not choice.is_file():
                raise ValueError("Could not identify the recording. Try adding the artist or a link.")
            source = read(choice)["source"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError("No usable source found.")
            progress("Fetching recording")
            recording = fetch(source.strip(), job)
        elif all((job / name).is_file() for name in ("analysis.json", "guide.wav", "backing.wav", "drums.wav")):
            save(job / "prepared.json", True)

    save(checkpoint, recording.name)
    return recording


def prepare(job):
    if job.parent.resolve() != job_root(IMPORTS):
        raise ValueError("Request belongs to a different library")
    request = read(job / "request.json")
    target = IMPORTS / job.name
    if (target / "song.json").is_file():
        return target

    def ask(recording):
        analyze(job, recording, request["model"], request["bars"], request["notes"], request["source"])

    recording = resolve_recording(job, request["source"], ask, stage)
    prepared = job / "prepared.json"
    if not prepared.is_file():
        stage("Codex is preparing stems, lyric timing and the bar grid")
        ask(recording)
        save(prepared, True)
    stage("Checking the grid and preparing playable cuts")
    try:
        return publish(job, recording, IMPORTS, request["bars"], request["sections"], request=request)
    except Exception:
        # A retry gives Codex the validation error and the same working files.
        prepared.unlink(missing_ok=True)
        raise


def run(source, model, bars, notes, sections=False):
    source = str(source).strip()
    if not source:
        raise ValueError("Enter a song, link or recording path.")
    job = enqueue(IMPORTS, source, notes, model, bars, sections)
    stage(f"Job folder {job}")
    from import_queue import supervise, status
    supervise(job, wait=True)
    target = IMPORTS / job.name
    if not (target / "song.json").is_file():
        raise ValueError(status(job).get("message") or "Import interrupted")
    stage("Added " + str(target))
    return target


def guard_preparation(job, parent):
    # The preparer owns its session. If its supervisor disappears, a stalled
    # Codex process must not keep the job and CPU leases forever.
    while os.getppid() == parent and not (job / "cancel").exists():
        time.sleep(.3)
    os.killpg(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--job"]:
        job = Path(sys.argv[2])
        owner = os.environ.get("KARAOKE_IMPORT_PARENT")
        if owner and os.getsid(0) == os.getpid():
            threading.Thread(target=guard_preparation, args=(job, int(owner)), daemon=True).start()
        try:
            prepare(job)
        except InterruptedError:
            sys.exit(75)
        except Exception as exc:
            save(job / "error.json", str(exc))
            sys.exit(str(exc))
        sys.exit(0)
    p = argparse.ArgumentParser()
    p.add_argument(
        "recording", help="Song and artist, HTTP(S) link or local recording path"
    )
    p.add_argument("--model", default="")
    p.add_argument("--bars", type=int, choices=LENGTHS, default=32)
    p.add_argument("--notes", default="")
    p.add_argument("--sections", action="store_true")
    a = p.parse_args()
    try:
        run(a.recording, a.model, a.bars, a.notes, a.sections)
    except Exception as e:
        sys.exit(str(e))
