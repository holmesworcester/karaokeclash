"""Voicings for Céline Dion's 4:20 Tony Moran mix (Wr460YHRtVE), on the recording clock."""

BPM = 131
DOWNBEAT = 0.257
DURATION = 260.24
# The piano chart supplies the minor sevenths, added seconds and suspensions.
# The remix bass and upper parts determine the inversions, cadences and form.
# https://www.partiture.it/wordpress/wp-content/uploads/2020/12/My-heart-will-go-on-piano-partiture.it_.pdf
VOICES = {
    "Emaj9": (40, 56, 59, 63, 66),
    "Eadd9": (40, 56, 59, 64, 66),
    "C#m7": (37, 56, 59, 61, 64),
    "C#m9": (37, 52, 59, 63, 68),
    "Badd9": (35, 54, 59, 61, 63),
    "Bsus4": (35, 54, 59, 64),
    "Amaj9": (45, 52, 56, 59, 61),
    "A6/9": (45, 52, 54, 59, 61),
    "Aadd9/B": (35, 52, 57, 59, 61),
    "Emaj7/G#": (44, 52, 59, 63),
    "F#m11": (42, 52, 57, 59, 61),
    "G#7sus4": (44, 54, 56, 61, 63),
    "G#7": (44, 54, 60, 63),
    "Fm7": (41, 51, 56, 60, 65),
    "Fm9": (41, 51, 56, 60, 67),
    "Ebadd9": (39, 55, 58, 63, 65),
    "Ebsus4": (39, 56, 58, 63),
    "Dbmaj9": (37, 53, 56, 60, 63),
    "Db6/9": (37, 53, 58, 63, 68),
    "Abmaj7/C": (48, 56, 63, 67),
    "Bbm11": (46, 53, 56, 61, 63),
    "Abmaj9": (44, 55, 58, 60, 63),
    "Eb/Ab": (44, 55, 58, 63),
    "Dbadd9/Ab": (44, 53, 56, 61, 63),
    "Ebmaj7/Ab": (44, 51, 55, 58, 62),
}
VERSE = ["Eadd9", "Badd9", "A6/9", "Amaj9"]
REFRAIN = [
    "C#m9",
    "Badd9",
    "Amaj9",
    "Badd9",
    "C#m7",
    "Badd9",
    "Amaj9",
    "Emaj7/G#",
    "C#m9",
    "Badd9",
    "A6/9",
    "Badd9",
    "C#m7",
    "Badd9",
    "Amaj9",
    "Bsus4",
]
LAST = [
    "Fm7",
    "Ebadd9",
    "Dbmaj9",
    "Ebadd9",
    "Fm9",
    "Ebadd9",
    "Dbmaj9",
    "Abmaj7/C",
    "Fm9",
    "Ebadd9",
    "Db6/9",
    "Ebadd9",
    "Fm7",
    "Ebadd9",
    "Dbmaj9",
    "Ebsus4",
]
FORM = (
    ["Emaj9", "C#m7", "Badd9", "Amaj9", "Badd9", "C#m7", "Badd9", "Aadd9/B", "Bsus4"]
    + VERSE * 6
    + REFRAIN
    + VERSE * 6
    + REFRAIN
    + VERSE * 2
    + ["C#m7", "Badd9", "Amaj9", "Badd9", "C#m9", "Badd9", "Amaj9", "G#7sus4"]
    + LAST
    + ["Abmaj9", "Ebadd9", "Db6/9", "Dbmaj9"] * 2
    + ["Abmaj9", "Eb/Ab", "Dbadd9/Ab", "Dbadd9/Ab"] * 2
    + ["Ebmaj7/Ab", "Ebmaj7/Ab", "Abmaj9", "Abmaj9", "Abmaj9"]
)
# Additional bass changes within the written bar. Bar numbers are zero-based.
CHANGES = {
    7: [(3, "Bsus4")],
    40: [(2, "F#m11")],
    56: [(2, "Bsus4")],
    60: [(2, "Bsus4")],
    80: [(2, "F#m11")],
    96: [(2, "Bsus4")],
    103: [(3, "G#7sus4")],
    104: [(2, "G#7")],
    112: [(2, "Bbm11")],
    132: [(2, "Eb/Ab")],
}


def score():
    events = []
    for bar, chord in enumerate(FORM):
        for beat, symbol in [(0, chord), *CHANGES.get(bar, [])]:
            time = DOWNBEAT + (4 * bar + beat) * 60 / BPM if bar or beat else 0
            notes = list(VOICES[symbol])
            if time < DURATION and (not events or notes != events[-1][1]):
                events.append([round(time, 6), notes])
    return {"duration": DURATION, "events": events}


def write_midi(path):
    import mido

    mid = mido.MidiFile(type=0, ticks_per_beat=960)
    track = mid.add_track("My Heart Will Go On - Tony Moran mix")
    tempo = mido.bpm2tempo(BPM)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    track.append(mido.Message("program_change", program=48))
    prev, held = 0, set()
    for time, notes in [*score()["events"], (DURATION, [])]:
        tick = round(mido.second2tick(time, mid.ticks_per_beat, tempo))
        keys = set(notes)
        for kind, pitches in [("note_off", held - keys), ("note_on", keys - held)]:
            for note in sorted(pitches):
                track.append(mido.Message(kind, note=note, velocity=66, time=tick - prev))
                prev = tick
        held = keys
    mid.save(path)
