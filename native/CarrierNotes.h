// Transcribe legacy audio clips once at load time, including changes into silence.
#pragma once
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

struct CarrierChord {
    double start;
    std::bitset<128> notes;
};

inline std::vector<CarrierChord> pitchChanges(const int16_t *stereo, int frames, double rate) {
    constexpr int size = 1024, window = 512, stride = 8, hop = 120;
    using Complex = std::complex<float>;
    std::array<Complex, size / 2> roots;
    std::array<float, window> hann;
    for (int i = 0; i < size / 2; ++i)
        roots[i] = std::polar(1.f, float(-6.28318530718 * i / size));
    for (int i = 0; i < window; ++i)
        hann[i] = .5f - .5f * std::cos(6.28318530718 * i / (window - 1));

    // Average each stereo channel over the decimation interval and use the
    // louder channel to avoid losing a carrier to stereo phase cancellation.
    std::vector<float> samples(frames / stride);
    for (size_t i = 0; i < samples.size(); ++i) {
        int l = 0, r = 0;
        for (int j = 0; j < stride; ++j) {
            l += stereo[2 * (i * stride + j)];
            r += stereo[2 * (i * stride + j) + 1];
        }
        samples[i] = float(std::abs(l) >= std::abs(r) ? l : r) / (32768 * stride);
    }
    std::vector<CarrierChord> chords;
    std::bitset<128> previous;
    for (int center = 0; center < int(samples.size()); center += hop) {
        std::array<Complex, size> bins{};
        for (int i = 0, rev = 0; i < size; ++i) {
            int pos = center + i - window / 2;
            if (i < window && pos >= 0 && pos < int(samples.size()))
                bins[rev] = samples[pos] * hann[i];
            int bit = size / 2;
            while (bit && (rev & bit)) { rev ^= bit; bit >>= 1; }
            rev ^= bit;
        }
        for (int width = 2; width <= size; width *= 2)
            for (int base = 0; base < size; base += width)
                for (int j = 0; j < width / 2; ++j) {
                    auto a = bins[base+j], b = bins[base+j+width/2] * roots[j*size/width];
                    bins[base+j] = a+b;
                    bins[base+j+width/2] = a-b;
                }
        std::array<float, size / 2> power;
        for (int k = 0; k < size / 2; ++k) power[k] = std::norm(bins[k]);
        const float floor = std::max(.03f, .055f * *std::max_element(power.begin(), power.end()));
        std::bitset<128> notes;
        std::vector<double> fundamentals;
        for (int k = 4; k < size / 2 - 1; ++k) {
            float a = power[k-1], b = power[k], c = power[k+1];
            if (b < floor || b <= a || b <= c) continue;
            double hz = (k + .5 * (a-c)/(a-2*b+c)) * rate / (stride*size);
            // Suppress overtones of an already detected lower fundamental.
            bool overtone = std::any_of(fundamentals.begin(), fundamentals.end(), [hz](double low) {
                double n = std::round(hz/low);
                return n >= 2 && std::abs(hz/(n*low)-1) < .018;
            });
            if (!overtone) {
                int key = std::lround(69 + 12 * std::log2(hz/440));
                if (key >= 0 && key < 128) notes.set(key);
                fundamentals.push_back(hz);
            }
        }
        if (notes != previous)
            chords.push_back({center * stride / rate, notes});
        previous = notes;
    }
    if (previous.any()) chords.push_back({frames / rate, {}});
    return chords;
}
