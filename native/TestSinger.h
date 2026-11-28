// A prepared mono vocal on the same recording clock as library.lyric_position.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
struct SingerClock {
    double start, end, bpm, origin, cycle;
    int loop;
    std::vector<double> bars;
    double position(double elapsed) const {
        double length = end - start;
        if (length <= 0)
            return elapsed;
        return loop ? elapsed - length * std::floor(elapsed / length) : std::clamp(elapsed, 0., length);
    }
    double bar(double t) const {
        if (bars.size() < 2)
            return (t - origin) * bpm / 240;
        auto i = std::clamp<long>(std::upper_bound(bars.begin(), bars.end(), t) - bars.begin() - 1, 0,
                                  bars.size() - 2);
        return i + (t - bars[i]) / (bars[i + 1] - bars[i]);
    }
    double secondsPerBar(double t) const {
        if (bars.size() < 2) return 240 / bpm;
        auto i = std::clamp<long>(std::upper_bound(bars.begin(), bars.end(), t) - bars.begin() - 1, 0, bars.size()-2);
        return bars[i+1]-bars[i];
    }
    double seconds(double bar) const {
        if (bars.size() < 2)
            return origin + bar * 240 / bpm;
        auto i = std::clamp<long>(long(bar), 0, bars.size() - 2);
        return bars[i] + (bar - i) * (bars[i + 1] - bars[i]);
    }
};
struct TestSinger {
    SingerClock lyrics, backing;
    bool same;
    std::vector<int16_t> pcm;
    double position(double t, const SingerClock &backing, bool same) const {
        if (same) {
            if (!lyrics.loop)
                return backing.start + backing.position(t);
            t = backing.position(t) + backing.start - lyrics.start;
        } else {
            double clock = (backing.loop || backing.bars.size() >= 2) ? backing.start + backing.position(t) : t;
            t = lyrics.seconds(backing.bar(clock));
        }
        if (lyrics.cycle > 0)
            return t - lyrics.cycle * std::floor(t / lyrics.cycle);
        return lyrics.start + lyrics.position(t);
    }
    float sample(double seconds, const SingerClock &clock, bool sameTrack) const {
        double offset = position(seconds, clock, sameTrack) * 48000;
        if (offset < 0 || offset >= pcm.size())
            return 0;
        size_t i = size_t(offset);
        double frac = offset - i;
        return ((1 - frac) * pcm[i] + frac * (i + 1 < pcm.size() ? pcm[i + 1] : 0)) / 32768.;
    }
};
