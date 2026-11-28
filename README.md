# Karaoke Playground 4 for Linux

The current Linux app has one shared song library, four microphone strips, five selectable audio sources and a separate 2×2 singer display. This checkout supplies the operator and audience interfaces and the native loop transport.

GitHub carries the desk source and karaoke preparation skill. The runtime and recording-specific data are provisioned separately. On a prepared machine, `scripts/desk-data store` saves the local charts, model and reference fixtures into the user's desk data directory. On a new checkout, run `scripts/desk-data link /path/to/desk-assets` (or set `KARAOKE_DESK_ASSETS`) to use that library. The installer fetches the pinned Basic Pitch model when needed. The source-only CI task builds the native desk and exercises the controls with synthetic recordings. The provisioned Framework runs the full library suite, using the same Mise tasks as a workstation.

## Install and open

Install the runtime dependencies and build tools on Ubuntu 26.04 x86-64:

```sh
sudo apt install curl g++ rustc patch libasound2t64 libfreetype6 libjack-jackd2-dev pipewire-jack pipewire-bin python3-gi-cairo gir1.2-gtk-3.0 python3-numpy python3-mido python3-scipy python3-onnxruntime ffmpeg
```

Extract the [pinned Linux runtime](https://github.com/holmesworcester/karaokeclash/releases/tag/four-way-linux-clean), then run the installer from this checkout:

```sh
scripts/install-linux-four /path/to/KaraokePlayground-4-linux-x86_64
karaoke-playground-4
```

The desktop application and command use the same checkout. Its mixer builds on first launch. The runtime supplies audio preparation, worker support and the pitch processor. The old eight-list control desk is deprecated. On the Framework, keep the provisioned runtime and media in place. A clean runtime starts with an empty library; use **Add song** or set `KARAOKE_IMPORTS` to existing prepared songs. Automated preparation requires a signed-in Codex CLI and the local karaoke tools. Character voices require their local models.

## Operator desk

On one display the operator desk opens first. Choose **Audience** to open the singer window. With a second display connected, the singer window opens fullscreen there. **F11** toggles fullscreen.

Each launch starts with empty song targets and keeps your mic and audio settings.
The library lists each song once, with its artist. Drag a song from the left library onto the microphone icon to set a mic’s vocal anchor,
or onto the video icon to set its audience lyrics.
You can also select a song and click its destination. Each mic exposes all
dials for pulse, delay, compression, reverb and the level of a scored vocal lead.

Each mic has one voice selection. **Vocoder Clear** plays the assigned notes and rests on the desk’s own synthesizer.
With playback stopped, **Live** and **Hear** preview a newly selected score from its first note. Pressing **Play** puts the effect back on the song clock. For **Sweet Dreams**, the carrier follows the right-hand synth arpeggio from the recording’s MIDI, including the instrumental breaks.
**Homer Simpson** and the other characters apply character conversion. **Technologic Vanilla**
applies its pitch shift. **Vocoder Hold** sustains the carrier until the next note or chord.
**Chord Tune (Chorale VST3)** uses the free Linux x86-64 Chorale 1.3.0 plug-in to sing up to four simultaneous notes from the score. The desk sends MIDI notes, sets the plug-in fully wet and uses its 2,048-sample Studio mode (about 43 ms at 48 kHz) for low voices. **Harmony…** sets the level, register and stereo width. Install the bundle on an existing desk with `scripts/install-chorale-vst3` and reopen the desk. [Chorale release](https://github.com/rithulkamesh/chorale/releases/tag/v1.3.0)

**Chord Tune (Tunelet)** keeps the original harmony engine, including its glide control. Previous Chord Tune selections reopen on Tunelet. Both engines follow the live mic, **Hear**, and **Test vocal**, with the unshifted lead muted. TAL has its own vocoder entry.

**TAL Vocoder** runs the live mic, **Hear** sample or **Test vocal** through TAL‑Vocoder‑2 3.0.4 VST3. The mic shapes the sound and the scored chord drives the plug-in’s sidechain. On each TAL pane, **Source vocal** adds the selected vocal directly to the mix. It defaults to 0%. **Wet level** sets the level of the vocoded voice and scored bass together. The source stays audible at its chosen level during score gaps and when the plug-in is unavailable. Building the Linux desk requires rustc and patch for the pinned Tunelet v0.2.0 source. The Linux installer provisions the vendor’s x64 plug-in. Existing installations can run `scripts/install-tal-vst3` and reopen the desk. [TAL‑Vocoder‑2](https://tal-software.com/products/tal-vocoder)

Select **TAL Vocoder** on a mic to see its rotary controls directly on the pane. Use the **Vocoder**, **Oscillators**, **EQ** and **MIDI** tabs. **Input mode** chooses the carrier and modulator. Both synth-carrier modes receive the selected MIDI arrangement. Bypass mutes the wet return. **Reset** restores the plug-in defaults for that pane.

Each pane starts on **MIDI → All parts**. The selector lists the focused score, additional instruments, original score and earlier extraction when the recording has them. Switch during playback to hear the new arrangement on the same beat. The MIDI monitor, harmony and vocoder follow that pane’s choice through tempo changes and practice cuts.

Use **Presets… → Whole pane** to name and recall a sound on any mic. For a local preset, choose an effect or individual control in the same menu, or right-click its dial. **Save preset** creates a name and **Update preset** replaces that name for the selected controls. Recalling a preset preserves song assignments, mic routing and the transport. Pulse, Delay and Comp retain their settings while bypassed.

**Glide** connects the moving riff, and **Lead** blends in a separately scored vocal melody. The desk matches prepared recordings and practice cuts to their instrumental parts by PCM landmarks. The ATC carrier plays its written arrangement. “A Real Hero” and Ann Lee’s “2 Times” follow the isolated bass. Prepared parts also cover the Joyce Manor guitars, Semisonic chord changes, La Prohibida keyboards, Owl City arpeggio, Eiffel 65 riff, Daft Punk scores and the DJ Quicksilver Radio Mix.

For a backing without a score, the desk subtracts the drum stem, checks pitch and onset estimates against sustained spectral peaks, and rejects moving sweeps and overtones that track a lower note. It caches the result beside the backing. **Stereo width** in the vocoder settings scales each key’s measured left/right placement in the MIDI monitor and the sung effect. Zero centers the carrier. **Carrier harmonics** controls brightness. [Spotify Basic Pitch](https://github.com/spotify/basic-pitch)

Run `karaoke-playground-4 --prepare-carriers` to prepare an installation ahead of a session. The scheduled **Installed song carriers** workflow and `mise run audit-carriers` use the provisioned Framework desk, its runtime configuration and private playback WAVs. The audit checks inferred notes for stable pitch evidence, measures coverage of sustained chord tones on another time grid, and compares prepared arrangements against their recording-specific scores. Spliced excerpts with a mapped original recording also use the source passage before tempo adjustment. Per-backing measurements go to `build/evidence/carriers.json`.

Selecting **Vocoder Clear** or **Vocoder Hold** opens the preset, synth register, stereo width and sound-shaping dials directly on that mic’s pane. **Glide** defaults to 100 ms. The upper MIDI line slides into the next pitch across short gaps while held chord tones ring. Set it to 0 ms for the score’s original articulation. The **Lead** dial on the mic sets the level of a separately scored vocal melody when available.
Choose **Tight speech** for rap, **Smooth vowels** for longer notes or **Vowel stack** for a brighter carrier.
At **100% Dry / Wet**, the singer’s changing vowel shape controls the sound of the carrier. **Consonants**
adds speech-shaped high-frequency noise, **Formant shift** moves the vowel resonances, and
**Carrier harmonics** blends a rounded sawtooth for sustained vowels with a brighter sawtooth and pulse for speech. **Synth register** can lower an entire chord into a speech-friendly range. **Vocal attack** and **Release** set how quickly
the sound opens and closes around each syllable. The desk saves these settings per mic.
**Dry** passes the voice through. Assigning songs and following
another mic preserve each mic’s voice selection. The same choices apply to **Test vocal**
and **Live**. Pulse, delay, compression, and reverb are independent controls.

Each singer sees up to four closely spaced lines. The highlight sweeps across the
active words. Read down the four lines, then return to the top. New lyrics replace
finished phrases in place, so a line stays where you first saw it. During an
intro or instrumental break, a countdown at the top shows the time to that
singer’s next entrance. It clears on the pickup.

Choose **Alternate bars** on the mic that should sing first, then select its partner.
Both audience panes use the first mic’s lyric selection, falling back to its partner’s
selection or the backing. Assign lyrics to either pane to change the shared song or cut.
**Sing** marks the current singer. **Next bar** cues the partner. The microphones stay
live with separate inputs, voice effects and levels. Choose **Unbind panes** from either
strip to return to individual lyric selections.

The first assigned song becomes the audio source. Press the **♫** button at the upper right of a slot’s controls to select its
backing loop. The selected note stays pressed. Each mic plays its vocal anchor song when assigned and otherwise
uses its lyric song. **Follow Red** borrows Red’s vocal score and keeps the selected pane’s own backing. A following pane with no song supplies silent backing. **Space** toggles playback on the desk or audience display.
Each pane has its own backing setting. **Full** plays the complete vocal-removed backing when its note is selected. **Stripped** plays percussion only. **MIDI carrier** plays the pane’s vocoder synth directly. Select the pane’s ♫ button and press **Play** to hear the notes with the microphone off. The status line names the score, including a score borrowed with **Follow**. You can switch back to **Full** at the same position for comparison. **Mute backing** silences the accompaniment. The transport keeps running, and live or test vocals continue through that mic’s effects. **Mute all** silences the room output.
Set **BPM** beside **Launch on** to change the tempo for the whole room. It starts at 126 and remembers your setting. Backings and rehearsal vocals stretch to that tempo while keeping their pitch. All four panes share the same bar clock. A Teaser or Main launched halfway through a bar enters halfway through its opening bar. Arrow taps move one bar inside that pane’s loop on the selected launch interval. Changing tempo keeps queued switches on their musical boundary.

Each imported cut carries `backing.wav`, `drums.wav`, and its separate `guide.wav`
in the cut’s audio directory. Legacy catalog mixes live under
`$KARAOKE_DATA/private/linux-four/playback/<song>/`.

Toggle **Test vocal** on an assigned mic to hear its vocal through that strip. The menu beside the button selects the source. **Extracted** plays the recorded vocal. **Computer** sings the timed lyrics even when a recording exists. **Auto** prefers the recording and falls back to the computer voice. You can switch during playback, and each mic remembers its own choice.

Test uses the lyric edition on that singer’s screen, or the mic’s Hear song when only Hear is assigned. The optional English Kokoro voice speaks each line in context and fits its words to the song’s cues and score. Synthesis runs on the CPU, leaving the GPU available for character effects. Each strip applies its own pitch, voice, pulse, delay, compression and reverb settings. Toggle **Test vocal** off to restore that pane’s physical input.

Homer converts the physical microphone, extracted guides and computer-sung test lyrics through the same live worker. The Framework GPU runs one character strip at a time. A second character strip passes its direct vocal, shows that the GPU is in use and takes over when the first strip releases it.

Homer plays the character model’s waveform. The stable mic setting uses the previous 400 ms of the singer’s voice and 10 ms of actual right context to guide each conversion and reduces random changes between short windows. If the GPU misses a packet, the desk briefly filters the current mic with the last measured tone. Homer’s software monitor delay is 45 ms fast and 90 ms stable. **Hear** plays an example captured through the stable live setting.

### Install the neural test voice

Install the CPU runtime and fetch the pinned model and US English Echo voice once (326 MB). For an extracted release, use the default below. For a checkout runtime, set `KARAOKE_DATA` to its media directory. On the Framework this is `/mnt/storage/holmes-karaoke`.

```sh
sudo apt install python3-onnxruntime espeak-ng
scripts/install-speech-model "${KARAOKE_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}/karaoke-playground-4}/private/linux-four/kokoro-v1"
```

Reopen the desk, choose **Computer** on a song with timed lyrics, and enable **Test vocal**. Playback works offline. For a custom model location, pass that directory to the installer and set `KARAOKE_TTS_MODEL_DIR` to the same directory when launching. Installations without the neural voice use the built-in Flite fallback.

## Song requests

**Add song** saves the song, prompt and import options in the private library before starting Codex with the karaoke-is-all-you-need skill. The library row shows progress and keeps the prompt searchable after the track is ready. Closing the desk leaves the worker running. Reopening it reconnects to live work and resumes interrupted requests from their saved recordings. Failed or cancelled requests keep their prompt and offer **Retry**.

The GTK handoff, worker crash, queue and retry tests run in `mise run test-desk` with a local Codex fixture. On Linux with a systemd user manager, run `mise run test-import-scope` to stop a real desk service during preparation and verify the track still arrives. This uses the same GTK, Xvfb and audio runtime prerequisites as the desk suite and runs on the Framework runner weekly and on manual dispatch.

## Loop launches

Each of the four mic panes has its own **◀ T M F ▶** controls. Tap **T**, **M**, or **F** directly. **Teaser** plays one recognizable opening line, leaves a gap with only the accompaniment, then repeats. **Main** repeats the complete chorus or recurring refrain. **Full** repeats the recording. Instrumental editions use a short riff and a longer passage. Prepared hooks and additional named cuts are available alongside those three. The pane’s selected mix applies to its loops.

Add Song places the teaser and chorus independently on the recording’s measured downbeats. The guide and lyrics share a vocal window so pickups and held notes can cross a bar line. Imported hooks repeat in whole phrases at 126 BPM. The requested bar count is a target when a complete chorus includes an extra pickup or release bar.

**Next loop** launches at the end of the playing cycle. Choose a bar or fractional-bar interval under **Launch on** to use the next matching global grid line. Tap another loop before the boundary to replace the queued choice. The left and right arrows jump one bar backward or forward inside that pane’s loop at the same global boundary, even while another pane supplies the backing. Repeated arrow taps add up and opposite taps cancel. The operator can see the playing and queued choices while the singer display previews the entrance. Paired panes keep their **Sing** and **Next bar** cues during a transition.

## Verify the desk

For routing, tempo and voice changes, run `mise run test-routing`. It uses the installed runtime when available, builds the native engine if needed, and checks the rendered stereo output, catalog loop grids and per-mic controls with short fixtures. The Linux desk pre-commit hook runs this focused suite. To enable it in a fresh checkout with no existing pre-commit hook, run `ln -sr .githooks/pre-commit "$(git rev-parse --git-path hooks/pre-commit)"`. The full desk suite includes these checks on every push and pull request.

On Linux with the app and neural voice dependencies above, python3-scipy, python3-mido, Mise, curl, Xvfb and xauth, run `mise run test-desk`. It fetches the pinned runtime and speech model and exercises the GTK operator desk, singer display and native playback. Set `KARAOKE_RUNTIME` to test against an installed runtime. Set `KARAOKE_DRUM_MODEL` to the local HT-Demucs FT drums ONNX model when importing new songs.


The Framework listening check uses a provisioned Radeon 8060S runner with the local ROCm voice lab, Homer checkpoint, idle-slot service, and a whisper.cpp executable and small English-capable model. Set `KARAOKE_RUNTIME`, `KARAOKE_WHISPER_CLI` and `KARAOKE_WHISPER_MODEL` in the runner environment. Run `mise run test-homer-speech` to build the engine, provision Kokoro, press the operator’s Test vocal button on a missing-stem song, run the mixer at its live callback rate and transcribe the converted packets. The same task replays recorded speech and a held note through the live mixer and checks pitch and packet deadlines on both monitor settings. The scheduled and manually dispatched **Framework Homer speech** workflow targets the `karaoke-framework` self-hosted runner label and retains the audio, transcript, timing and log.

Run `mise run test-homer-identity` on the same Framework runner to check character likeness and distortion. The test replays identical microphone inputs at their real playback rate through the shipping stable monitor. It compares them with the earlier RVC voice using the voice lab’s WeSpeaker verifier, two reserved Homer references and the pinned Microsoft DNSMOS P.835 speech-signal model. The lyric reference is the earlier demo the listener preferred. The spoken check and the isolated Falcon singing clip run through the original live converter for their baseline. The verifier must also rank the listener-preferred lyric demo above the rejected version. A material per-clip loss fails the check even when the average improves. The runner also needs the voice lab CPU interpreter (PyTorch, torchaudio and ONNX Runtime), its verifier and reference files. The workflow saves the scores, packet timing and a level-matched blind listening page in `build/evidence/homer-identity-check`. Human listeners can rate likeness and audible artifacts independently before revealing A and B. DNSMOS predicts speech quality and is less informative for singing.
