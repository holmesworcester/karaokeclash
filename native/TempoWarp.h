// Random-access overlap-add for decoded PCM. Grain positions follow the beat
// grid, and both stereo channels share a splice so the original pitch and
// stereo image survive a tempo change. All buffers belong to the audio thread.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

struct TempoWarp {
    static constexpr int hop = 1024, search = 720;
    const int16_t *audio = nullptr;
    int64_t older = 0, newer = 0;
    int phase = 0;
    double expected = -1;

    template<int channels>
    void read(const int16_t *pcm, int64_t first, int64_t end, double frame,
              double speed, float *out) {
        int64_t at = std::llround(frame);
        auto sample = [&](int64_t i, int ch) -> float {
            return i >= first && i < end ? pcm[i*channels+ch] : 0;
        };
        if (std::abs(speed - 1.) < .00002) {
            for (int c=0; c<channels; ++c) out[c] = sample(frame < end ? std::min(at, end-1) : at, c);
            audio = nullptr;
            return;
        }
        if (pcm != audio || std::abs(frame - expected) > 2) {
            audio = pcm;
            older = newer = at;
            phase = 0;
        } else if (phase == hop) {
            older = newer + hop;
            // Match the audible overlap to the next grain near the musical
            // cursor. On silence or equally good matches prefer the cursor.
            newer = at;
            double best = -2;
            for (int offset = -search; offset <= search; offset += 4) {
                int64_t candidate = at + offset;
                if (candidate < first || candidate >= end) continue;
                double xy = 0, aa = 0, bb = 0;
                for (int k=0; k<512; k+=12) {
                    float a = sample(older+k, 0), b = sample(candidate+k, 0);
                    xy += a*b; aa += a*a; bb += b*b;
                }
                double quality = xy / std::sqrt(aa*bb+1.) - .015*std::abs(offset)/search;
                if (quality > best) { best = quality; newer = candidate; }
            }
            phase = 0;
        }
        float blend = float(phase)/hop;
        for (int c=0; c<channels; ++c)
            out[c] = (1-blend)*sample(older+phase, c) + blend*sample(newer+phase, c);
        ++phase;
        expected = frame + speed;
    }
};
