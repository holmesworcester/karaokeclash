"""Bass parts for the desk's short MIDI and recorded carrier excerpts, in source beats."""

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class Pattern:
    bpm: float
    beats: int
    notes: tuple  # (attack, release, MIDI pitch)
    offset: float = 0  # excerpt start relative to the pattern downbeat, in source beats

    def excerpt(self, duration):
        beat = 60 / self.bpm
        first = floor(self.offset / self.beats)
        last = floor((self.offset + duration / beat) / self.beats)
        for cycle in range(first, last + 1):
            anchor = cycle * self.beats - self.offset
            for a, b, pitch in self.notes:
                start = max(0, (anchor + a) * beat)
                end = min(duration, (anchor + b) * beat)
                if end > start:
                    yield start, end, pitch


def held(*spans):
    notes = []
    beat = 0
    for length, pitch in spans:
        notes.append((beat, beat + length, pitch))
        beat += length
    return tuple(notes)


# Pesenka's turnaround lands on E. Blue's MIDI bass (GM synth bass 2)
# enters on the first full hook, 2.281 s into its prepared keyboard excerpt.
PESENKA = held((2, 33), (2, 28), (2, 29), (1, 36), (1, 40))
BLUE = held((2, 34), (2, 29), (4, 31), (2, 26), (2, 29), (4, 31))
# Fergalicious, "Bs Sin" track, beats 4–12. Preserve the syncopated rests.
FERGIE = (
    (0, 0.625, 34),
    (1.5, 2.125, 34),
    (4, 4.625, 34),
    (5.5, 5.875, 34),
    (6.5, 6.75, 34),
    (7.25, 8, 32),
)
# The two chant banks share the G -> D revoicing at beat six.
DUET = held((6, 31), (10, 38))
CHANTS = {
    "Fergalicious",
    "2Fly - Rap",
    "O Superman - Ah",
    "Walking on a Dream - Chant Loop",
}

# The eight-second audio carriers start inside the source riff. The chant
# extracts start on a downbeat and were stretched to 106 BPM before slicing.
FIREFLIES = held((2, 34), (2, 39), (4, 32))
DREAM = held((1, 37), (4, 41), (2, 39), (1, 36))
HELLO = held((6, 39), (2, 34), (6, 39), (2, 34))
COWGIRL = ((0, 1.45, 31), (1.5, 2.90, 31), (3, 3.90, 38))
PARTS = {
    "Pesenka": Pattern(126, 8, PESENKA),
    "Blue (Da Ba Dee)": Pattern(127.99, 16, BLUE, -2.281 * 127.99 / 60),
    "Fergalicious": Pattern(106, 8, FERGIE),
    "Sunglasses at Night": Pattern(
        127.29, 16, held((7, 34), (1, 32), (6, 30), (2, 32))
    ),
    "2Fly": Pattern(90, 8, FIREFLIES, (6 - 0.073) * 90 / 60),
    "2Fly - Rap": Pattern(106, 8, FIREFLIES),
    "The Way Life Goes": Pattern(
        82, 8, held((2, 42), (2, 39), (2, 37), (2, 35)), (46 - 0.16) * 82 / 60 - 6
    ),
    "Walking on a Dream": Pattern(127, 8, DREAM, (46 - 31.377) * 127 / 60),
    "Walking on a Dream - Chant Loop": Pattern(106, 8, DREAM),
    "Hello": Pattern(
        128,
        32,
        HELLO + ((16, 22, 39), (22, 24, 34), (24, 32, 44)),
        (95 - 90.14) * 128 / 60,
    ),
    "Hello - Chant Loop": Pattern(106, 16, HELLO),
    "Amor Eléctrico": Pattern(131, 8, held((4, 41), (4, 44)), (47 + 0.011) * 131 / 60),
    "studio-cowgirl/main": Pattern(126.92, 4, COWGIRL),
    "studio-cowgirl/teaser": Pattern(126.92, 4, COWGIRL),
    "studio-cowgirl/full": Pattern(126.92, 4, COWGIRL),
}
for anchor, bpm in (("Lights Burn Dimmer", 106), ("My Heart Will Go On", 131)):
    for chant in CHANTS:
        PARTS[f"{anchor} / {chant}"] = Pattern(bpm, 16, DUET)
