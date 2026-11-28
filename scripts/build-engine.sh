#!/usr/bin/env bash
# Link the current transport to the installed runtime's three singer_* DSP exports.
set -euo pipefail
repo=$(dirname "$(dirname "$(realpath "$0")")")
runtime=$(realpath "${1:?Pass the runtime build/linux-four directory}")
workers="${KARAOKE_RUNTIME:-$(dirname "$(dirname "$runtime")")}/linux-four"
key=$(echo "$runtime" | sha256sum | cut -c1-16)
build=${KARAOKE_ENGINE_BUILD:-$repo/build/engine-$key}
mkdir -p "$build"
engine="$build/transport.so"
exec 9>"$build/build.lock"
flock 9
harmony=$("$repo/scripts/build-tunelet-harmony")
if [[ "$harmony" -nt "$engine" || ! -f "$engine" || "$0" -nt "$engine" || "$runtime/desk.so" -nt "$engine" || "$workers/character.h" -nt "$engine" ]] ||
    [[ -n $(find "$repo/native" -type f -newer "$engine" -print -quit 2>/dev/null) ]]; then
    sdk=$("$repo/scripts/install-tal-vst3" --headers)
    tal="$repo/build/tal-vst3/TAL-Vocoder-2.vst3/Contents/x86_64-linux/TAL-Vocoder-2.so"
    chorale="$repo/build/chorale-v1.3.0/Chorale.vst3/Contents/x86_64-linux/Chorale.so"
    output=$(mktemp "$build/transport.XXXXXXXX.so")
    trap 'rm -f "$output"' EXIT
    multi=$(gcc -print-multiarch)
    libdir="${KARAOKE_JACK_LIB:-$runtime/pkg/root/usr/lib/$multi/pipewire-0.3/jack}"
    includedir="${KARAOKE_JACK_INCLUDE:-$runtime/pkg/root/usr/include}"
    flags=()
    if [[ -d "$includedir" ]]; then flags+=("-I$includedir"); fi
    if [[ -d "$libdir" ]]; then flags+=("-L$libdir" "-Wl,-rpath,$libdir"); fi
    g++ -std=c++17 -O3 -fPIC -shared -Wl,-z,defs -Wl,-Bsymbolic-functions \
        "${flags[@]}" "-I$workers" "-I$repo/native" -I"$sdk" "-DKARAOKE_TAL_PATH=\"$tal\"" \
        "-DKARAOKE_CHORALE_PATH=\"$chorale\"" \
        "$repo/native/engine.cpp" "$repo/native/ScoredVst.cpp" "$harmony" "$runtime/desk.so" -ldl -ljack -o "$output"
    mv "$output" "$engine"
fi
realpath "$engine"
