#pragma once
#include <array>
#include <cmath>
#include <vector>

// Four damped comb filters followed by two all-pass diffusers. The direct signal
// stays sample-aligned. Storage is allocated before the audio device opens.
class RoomReverb {
    struct Tap {
        std::vector<float> buffer;
        unsigned pos = 0, valid = 0;
        float lowpass = 0;
        void prepare(double rate, double seconds) { buffer.assign(std::lround(rate * seconds), 0); }
        float delayed() const { return valid == buffer.size() ? buffer[pos] : 0.f; }
        void put(float v) {
            buffer[pos] = v;
            if (++pos == buffer.size())
                pos = 0;
            if (valid < buffer.size())
                ++valid;
        }
        void clear() {
            valid = 0;
            lowpass = 0;
        }
    };
    std::array<Tap, 4> comb;
    std::array<Tap, 2> diffuser;
    bool active = false;

  public:
    void prepare(double rate) {
        constexpr double times[] = {.0297, .0371, .0411, .0437, .0051, .0017};
        for (unsigned i = 0; i < comb.size(); ++i)
            comb[i].prepare(rate, times[i]);
        for (unsigned i = 0; i < diffuser.size(); ++i)
            diffuser[i].prepare(rate, times[4 + i]);
    }
    void process(float *x, unsigned n, float mix) {
        if (mix <= 0) {
            if (active) {
                for (auto &t : comb)
                    t.clear();
                for (auto &t : diffuser)
                    t.clear();
                active = false;
            }
            return;
        }
        active = true;
        for (unsigned i = 0; i < n; ++i) {
            float wet = 0;
            for (auto &t : comb) {
                const float tail = t.delayed();
                t.lowpass += .43f * (tail - t.lowpass);
                t.put(x[i] + .72f * t.lowpass);
                wet += .25f * tail;
            }
            for (auto &t : diffuser) {
                const float delayed = t.delayed(), next = delayed - .5f * wet;
                t.put(wet + .5f * next);
                wet = next;
            }
            x[i] += mix * wet;
        }
    }
};
