# Carrier parts

Each JSON file contains recording-clock note spans in milliseconds and the median stereo balance at the fundamental of each key. Negative pan values move left. The index matches sample count and PCM landmarks in the prepared stereo backing. A different master or cut gets its own transcription. The recordings and separated stems live in the private desk library.

| Library entry | Instrument and source |
| --- | --- |
| A Real Hero | The four-key bass part from the isolated soundtrack stem, preserved in the runtime’s album MIDI. |
| 2 Times | One bass fundamental at a time from Ann Lee’s isolated bass. |
| 2Fly | The exposed Fireflies arpeggiator transcribed in the runtime’s `tools/simple_carrier.py`, on the 90 BPM mashup clock shared by both vocal lanes. |
| Blue (Da Ba Dee) | The Ice Pop recording transcription in `songs/eiffel-blue/source-riff.mid`, with the verse rests and octave changes. |
| Around the World | The ATC riff in `carrier-loops.json`, including its written bass and the final key change. Playback uses its native gains, glide and optional lead. |
| Doin’ It Right | The original held album MIDI in `songs/doin-it-right/live.mid`. The backing-vocal row keeps the syllable-gated version in `doin-it-right-vocoder.json`. |
| Technologic | The original two-bar saw part in `assets/chants/Technologic.mid`, fitted to this desk edition’s tempo. |
| Constant Headache | Rhythm guitar and solo from the three-guitar GP5 on the measured album bar map. The four-bar chant uses the same guitar notes at 126 BPM. The final ringing strings have separate releases. |
| Closing Time | A three-voice guitar reduction of the verse, chorus, bridge and outro. The practice cuts use the section and splice maps in `starred-arrangements/closing-time.json`. |
| Amor Eléctrico | Pitched keyboard parts measured on the stereo stem after vocal, drums and bass separation. |
| Bellissima | The DJ Quicksilver Radio Mix. The lead follows the recurring two-bar figure, its breaks and its shifted re-entry. The score also includes the lower keyboard and bass. [DJ Quicksilver recording](https://www.youtube.com/watch?v=tJho0YeqUmM) |

To provision the radio edition, decode that recording to stereo 48 kHz 16-bit PCM as `backing.wav`. Separate the percussion from the same master, then resample it to the backing’s frame count as `drums.wav`. Install the pair in `private/linux-four/playback/Bellissima - Radio Mix` under the desk’s data directory. This edition has 11,074,175 frames and runs 230.712 seconds. The desk makes it available in the existing Bellissima row when both stems are present.

The **All parts** edition adds independently supported instruments to the focused score. Each file in `versions` carries the extra note spans, their pan, and the complete earlier extraction for that recording. ATC and the Daft Punk tracks already have their full written parts. The operator can select a superseded catalog loop on recordings that have one.

To rebuild these local files, install NumPy, SciPy, ONNX Runtime and PyGuitarPro, then run `scripts/build-midi-versions RUNTIME DATA --workers 4` against the prepared recordings, separated bass stems and guitar tab. Use `--output DIRECTORY` to compare a rebuild before installing it.
