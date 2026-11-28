"""Playback stems on the clock of each prepared catalog entry."""
from audio import DATA
from library import catalog as runtime_catalog

# These alternate lyric parts share their parent arrangement and clock.
SHARED = {"2Fly - Rap": "2Fly", "Doin' It Right - Vocoder": "Doin' It Right",
          "O Superman - Ah": "O Superman"}


def playback_files(song):
    if song.get("cuts"):
        return song["bed"], song["bed"].with_name("drums.wav")
    if song.get("radioMix"):
        folder = DATA / "private/linux-four/playback/Bellissima - Radio Mix"
        return folder / "backing.wav", folder / "drums.wav"
    name = "Get Low" if song.get("performance") else SHARED.get(song["id"], song["id"])
    folder = DATA / "private/linux-four/playback" / name
    return folder / "backing.wav", folder / "drums.wav"


def catalog():
    songs = runtime_catalog()
    radio = DATA / "private/linux-four/playback/Bellissima - Radio Mix"
    if "Bellissima" in songs and all((radio / f).is_file() for f in ("backing.wav", "drums.wav")):
        songs["Bellissima"].update(radioMix=True, artist="DJ Quicksilver", part="Radio Mix",
                                   label="Bellissima · Radio Mix", bpm=135.98, beat_zero=.069,
                                   bed=radio / "backing.wav", guide=None, teaserBar=0)
    for song in songs.values():
        song["carrierBacking"] = playback_files(song)[0]
    return songs
