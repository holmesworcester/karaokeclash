#!/usr/bin/env bash
set -euo pipefail
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cases=(discover -v -s tests -p 'test*.py')
if [[ ${1:-} == --source ]]; then
    export KARAOKE_DESK_ASSETS="$work/empty-desk-assets"
    cases=(-v test_voice_state test_midi_versions test_held_effects test_audio_routing
           test_polyphonic_carrier test_screens test_routing test_mic_effects test_vocal_toggle
           test_import_bass test_song_import
           test_tempo.RoomTempo.test_global_tempo_edit_keeps_a_pending_fractional_handoff_and_pitch
           test_tempo.RoomTempo.test_native_transport_warps_each_measured_bar_and_preserves_wrap_phase
           test_tempo.RoomTempo.test_harmonizer_follows_leaders_note_changes_on_the_global_beat
           test_tempo.RoomTempo.test_legacy_prepared_main_discards_partial_bars_at_both_ends
           test_recording_parts.RecordingParts.test_sliding_effect_drum_hits_and_guitar_partials_are_not_separate_keys
           test_recording_parts.RecordingParts.test_source_pan_reaches_the_monitor_clear_hold_and_tal_sidechain)
elif [[ ${1:-} == --routing ]]; then
    cases=(-v test_audio_routing test_tempo test_held_effects test_atc_carrier test_vocoder_bass test_voice_state)
elif [[ ${1:-} == --import-scope ]]; then
    cases=(-v test_song_import.SongRequests.test_desk_service_stop_leaves_codex_in_its_own_scope)
    export KARAOKE_TEST_IMPORT_SCOPE=1
fi
if [[ $# -gt 0 ]]; then
    config=${XDG_CONFIG_HOME:-"$HOME/.config"}/karaokeclash/runtime
    if [[ -z ${KARAOKE_RUNTIME:-} && -f $config ]]; then
        IFS= read -r KARAOKE_RUNTIME < "$config"
    fi
fi
if [[ -z ${KARAOKE_RUNTIME:-} ]]; then
    archive=KaraokePlayground-4-linux-x86_64.tar.gz
    curl --fail --silent --show-error --location --retry 3 \
        "https://github.com/holmesworcester/karaokeclash/releases/download/four-way-linux-clean/$archive" \
        --output "$work/$archive"
    (cd "$work" && echo "1328377b3cf47723d053606252f416dbcd226b2556586056a2d41b4620884048  $archive" | sha256sum -c -)
    tar -xzf "$work/$archive" -C "$work"
    KARAOKE_RUNTIME="$work/KaraokePlayground-4-linux-x86_64"
fi
if [[ $# == 0 ]]; then
    export KARAOKE_TTS_MODEL_DIR="${KARAOKE_TTS_MODEL_DIR:-$work/kokoro-v1}"
    scripts/install-speech-model "$KARAOKE_TTS_MODEL_DIR"
fi
if [[ ${1:-} != --source ]]; then scripts/desk-data link; fi
scripts/install-pitch-model
scripts/install-tal-vst3
scripts/install-chorale-vst3
KARAOKE_DESK_ENGINE=$(scripts/build-engine.sh "${KARAOKE_LINUX_BUILD:-$KARAOKE_RUNTIME/build/linux-four}")
export KARAOKE_STATE="$work/session.json"
# Keep installed recordings in catalog tests, with a private request queue.
installed_songs=$(PYTHONPATH="$KARAOKE_RUNTIME/linux-four" /usr/bin/python3 -c 'from library import IMPORTS; print(IMPORTS.resolve())')
export KARAOKE_IMPORTS="$work/songs"
mkdir "$KARAOKE_IMPORTS"
for manifest in "$installed_songs"/*/song.json; do
    [[ -f $manifest ]] || continue
    folder=${manifest%/*}
    ln -s "$folder" "$KARAOKE_IMPORTS/${folder##*/}"
done
export KARAOKE_DESK_ENGINE
export PYTHONPATH="$PWD/tests:$KARAOKE_RUNTIME/linux-four"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
xvfb-run -a /usr/bin/python3 -m unittest "${cases[@]}"
