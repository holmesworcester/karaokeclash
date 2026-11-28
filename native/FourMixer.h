#pragma once
#include "LiveDynamics.h"
#include "RoomReverb.h"
#include "ScoredVoiceEffect.h"
#include "Singer.h"
#include "VocalDelay.h"
#include <array>
#include <map>
#include <memory>
#include <string>

namespace karaoke_four {
constexpr unsigned quantum = ScoredVst::blockSize, lanes = 4;
using Block = std::array<float, quantum>;
struct Sample {
    double rate = 0;
    std::vector<float> data;
};
struct Lane {
    std::atomic<bool> live{false}, tuned{false}, pulse{false}, delayOn{false};
    std::atomic<int> voice{0}, token{0}, sampleID{-1};
    int renderedToken = -1;
    std::atomic<unsigned> mask{0xfff};
    std::atomic<float> peak{0};
    std::array<std::atomic<float>, 8> knobs{};
    std::atomic<const Score *> score{nullptr};
    std::atomic<double> scoreBpm{0};
    std::atomic<int> follows{-1}, midiVersion{0};
    double samplePos = 0;
    std::unique_ptr<Singer, decltype(&singer_free)> singer{nullptr, singer_free};
    TechnologicVoice shifter;
    bool shifting = false, strict = false;
    ScoredVoiceEffect effect;
    LiveDynamics compressor;
    DynamicsControls dynamics;
    VocalDelay delay, echo, sideDelay, sideEcho;
    Block signal{}, wet{};
    RoomReverb room, sideRoom;
    std::atomic<float> roomMix{0};
    float filtered = 0, sideFiltered = 0;
    void prepare(double rate) {
        singer.reset(singer_new(rate));
        shifter.prepare(rate);
        effect.prepare(rate);
        compressor.prepare(rate);
        delay.prepare(rate);
        echo.prepare(rate);
        room.prepare(rate);
        sideDelay.prepare(rate);
        sideEcho.prepare(rate);
        sideRoom.prepare(rate);
        filtered = 0;
    }
};
struct FourMixer {
    Lane lane[lanes];
    double sourceTime[lanes] = {-1, -1, -1, -1};
    double scoreBar[lanes] = {NAN, NAN, NAN, NAN};
    double albumBar[lanes] = {};
    const Score *clipScore[lanes] = {};
    double clipTime[lanes] = {}, clipSpeed[lanes] = {};
    double roomBar = NAN;
    double rate = 48000, time = 0;
    std::atomic<double> bpm{101}, sync{0};
    std::atomic<unsigned> revision{0};
    unsigned lastRevision = 0;
    Sample samples[10];
    std::map<std::string, Score> scores;
    LiveDynamics limiter;
    DynamicsControls master;
    float captured[lanes][quantum]{};
    void prepare(double hz) {
        rate = hz;
        limiter.prepare(hz);
        for (auto &l : lane)
            l.prepare(hz);
    }
    template <class Transform>
    void renderCaptured(float *left, float *right, unsigned n, Transform transform,
                        unsigned testing = 0, int carrierLane = -1) {
        auto rev = revision.load();
        if (rev != lastRevision) {
            time = sync.load();
            lastRevision = rev;
        }
        double beat = 60 / bpm.load();
        std::fill_n(left, n, 0);
        std::fill_n(right, n, 0);
        Block audition, rawCarrier, carrierSide;
        for (unsigned j = 0; j < lanes; j++) {
            auto &l = lane[j];
            Block side{};
            const int voice = l.voice.load();
            if (fullyWet(voice) && !l.strict) {
                // Drop dry samples already stored by the previous effect.
                l.delay.clear();
                l.echo.clear();
                l.sideEcho.clear();
                l.sideDelay.clear();
                l.sideRoom.process(nullptr,0,0);
                l.sideFiltered=0;
                l.room.process(nullptr,0,0);
                l.filtered=0;
            }
            l.strict = fullyWet(voice);
            int token = l.token.load();
            if (l.renderedToken != token) {
                l.samplePos = 0;
                l.renderedToken = token;
                if (!std::isfinite(roomBar)) l.effect.deselect();
            }
            bool testInput = testing & (1u << j);
            int id = testInput ? -1 : l.sampleID.load();
            bool live = testInput || l.live.load();
            unsigned auditionFrames = 0;
            float peak = 0;
            const Score *score = l.score.load();
            int leader = l.follows.load();
            unsigned source = leader >= 0 ? unsigned(leader) : j;
            const Score *clip = clipScore[source];
            const bool onClip = score && clip && !score->backingFamily.empty()
                && score->backingFamily == clip->backingFamily;
            if (onClip) score = clip;
            if (score) score = score->version(l.midiVersion.load());
            const bool monitor = int(j) == carrierLane && score;
            if (!monitor && !live && id < 0 && !l.delayOn && !l.pulse && l.roomMix.load() == 0) {
                l.room.process(nullptr, 0, 0);
                l.sideRoom.process(nullptr, 0, 0);
                l.effect.deselect();
                l.peak = 0;
                continue;
            }
            for (unsigned i = 0; i < n; i++) {
                float x = live ? captured[j][i] : 0;
                float meter = x;
                if (id >= 0) {
                    const auto &s = samples[id];
                    size_t at = l.samplePos;
                    if (at < s.data.size()) {
                        if (id == 0) {
                            x += s.data[at];
                            meter = x;
                        } else {
                            // Keep the mic feeding its worker while the example plays.
                            audition[i] = meter = s.data[at];
                            ++auditionFrames;
                        }
                        l.samplePos += s.rate / rate;
                    } else if (l.token.load() == token) {
                        int expected = id;
                        l.sampleID.compare_exchange_strong(expected, -1);
                    }
                }
                l.signal[i] = x;
                peak = std::max(peak, std::abs(meter));
            }
            l.peak = std::max(peak, l.peak.load() * .95f);
            if (l.tuned && !tunesChord(voice) && l.singer) {
                singer_process(l.singer.get(), l.mask, l.signal.data(), l.wet.data(), n);
                std::copy_n(l.wet.data(), n, l.signal.data());
            }
            if (live && voice >= 3)
                transform(j, l.signal.data(), n);
            bool shift = id <= 0 && voice == 2;
            if (shift) {
                if (!l.shifting)
                    l.shifter.reset();
                for (unsigned i = 0; i < n; i++)
                    l.signal[i] = limitVoice(l.shifter.process(l.signal[i]));
            }
            l.shifting = shift;
            const bool preview = !std::isfinite(roomBar);
            const bool vocode = (id <= 0 || l.strict) && usesCarrier(voice);
            if ((monitor || vocode) && score && (!preview || live || id == 0)) {
                double tempo = l.scoreBpm.load();
                double start = sourceTime[source] >= 0 ? sourceTime[source] : time;
                double speed = 1;
                if (onClip && !preview) {
                    start = clipTime[source];
                    speed = clipSpeed[source];
                } else if (tempo > 0 && std::isfinite(scoreBar[source])) {
                    if (score->recordingGrid.bpm > 0) {
                        start = score->recordingGrid.seconds(albumBar[source]);
                        speed = score->recordingGrid.secondsPerBar(start) * bpm.load()/240.;
                    } else {
                        start = scoreBar[source]*240./tempo;
                        speed = bpm.load()/tempo;
                    }
                }
                if (preview && tempo > 0) speed = bpm.load()/tempo;
                l.effect.process(l.signal.data(), n, start, *score, vocode ? voice : 0, speed,
                                 preview, monitor ? rawCarrier.data() : nullptr,
                                 monitor ? carrierSide.data() : nullptr, side.data());
                if (monitor)
                    for (unsigned i = 0; i < n; ++i) {
                        left[i] += .5f * (rawCarrier[i]-carrierSide[i]);
                        right[i] += .5f * (rawCarrier[i]+carrierSide[i]);
                    }
            } else {
                l.effect.deselect();
                if (l.strict) std::fill_n(l.signal.data(),n,0.f);
            }
            // Recorded character examples already contain the voice conversion.
            if (!l.strict) {
                std::copy_n(audition.begin(), auditionFrames, l.signal.begin());
                std::fill_n(side.begin(), auditionFrames, 0.f);
            }
            if (l.pulse) {
                const float depth = l.knobs[0] * .01f, width = std::max(.02f, l.knobs[1] * .01f);
                const float blend = l.knobs[2] * .01f;
                const float coeff = 1 - std::exp(-2 * M_PI * (550 + 65 * l.knobs[3]) / rate);
                for (unsigned i = 0; i < n; i++) {
                    float dry = l.signal[i];
                    l.filtered += coeff * (dry - l.filtered);
                    double beats = std::isfinite(roomBar) ? 4 * roomBar : time / beat;
                    double phase = beats + i/(rate*beat);
                    phase -= std::floor(phase);
                    float env = phase < width ? std::sin(M_PI * phase / width) : 0;
                    float wetGain=blend * (1 - depth + depth * env);
                    l.signal[i] = dry * (1 - blend) + wetGain * l.filtered;
                    l.sideFiltered += coeff * (side[i] - l.sideFiltered);
                    side[i] = side[i] * (1 - blend) + wetGain * l.sideFiltered;
                }
            }
            constexpr double division[] = {.125, .25, .5, .75, 1, 2};
            l.echo.process(l.signal.data(), n, l.pulse ? l.knobs[7] * .01f : 0, beat, .28f);
            l.sideEcho.process(side.data(), n, l.pulse ? l.knobs[7] * .01f : 0, beat, .28f);
            l.compressor.processSinger(l.signal.data(), n, l.dynamics, side.data());
            l.delay.process(l.signal.data(), n, l.delayOn ? l.knobs[6] * .01f : 0,
                            beat * division[std::clamp(int(l.knobs[4]), 0, 5)], l.knobs[5] * .01f);
            l.sideDelay.process(side.data(), n, l.delayOn ? l.knobs[6] * .01f : 0,
                            beat * division[std::clamp(int(l.knobs[4]), 0, 5)], l.knobs[5] * .01f);
            l.room.process(l.signal.data(), n, l.roomMix.load());
            l.sideRoom.process(side.data(), n, l.roomMix.load());
            for (unsigned i = 0; i < n; i++) {
                left[i] += l.signal[i] - side[i];
                right[i] += l.signal[i] + side[i];
            }
        }
        time += n / rate;
    }
};

} // namespace karaoke_four
