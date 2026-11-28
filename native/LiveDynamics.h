#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>

// The desk writes these values. Each render callback reads them once per block.
struct DynamicsControls {
    std::atomic<bool> compressor{true}, muted{false};
    std::atomic<float> thresholdDB{-20}, ratio{6}, attackMs{8}, releaseMs{100};
    std::atomic<float> maxLiftDB{15}, floorDB{-48}, ceilingDB{-6};
};

class LiveDynamics {
    float rate = 48000;
    float rmsStep = 0, bypassStep = 0, limiterRelease = 0;
    float power = 0, vocalGain = 1, targetGain = 1, limitGain = 1, unmuteGain = 1;
    unsigned controlCountdown = 0;
    float attackMs = 0, releaseMs = 0, attackStep = 0, releaseStep = 0;
    float ceilingDB = 0, ceiling = 1;

    float step(float ms) const { return -std::expm1(-1000.f / (ms * rate)); }
    static float amplitude(float db) { return std::pow(10.f, db / 20.f); }

public:
    void prepare(float hz) {
        rate = hz;
        rmsStep = step(12);
        bypassStep = step(5);
        limiterRelease = step(120);
        power = 0;
        controlCountdown = 0;
        vocalGain = targetGain = limitGain = unmuteGain = 1;
        attackMs = releaseMs = ceilingDB = 0;
        ceiling = 1;
    }

    // Compress above the RMS threshold and lift softer phrases towards it.
    // Below the room floor the upward part of the curve returns to unity.
    void processSinger(float *samples, unsigned frames, const DynamicsControls &controls, float *side=nullptr) {
        const bool enabled = controls.compressor.load();
        const float threshold = controls.thresholdDB.load(), floor = controls.floorDB.load();
        const float slope = 1.f - 1.f / controls.ratio.load(), maxLift = controls.maxLiftDB.load();
        const float attack = controls.attackMs.load(), release = controls.releaseMs.load();
        if (attack != attackMs) { attackMs = attack; attackStep = step(attack); }
        if (release != releaseMs) { releaseMs = release; releaseStep = step(release); }
        for (unsigned i = 0; i < frames; ++i) {
            const float sample = samples[i];
            power += rmsStep * (sample * sample + (side ? side[i]*side[i] : 0) - power);
            // Evaluate the dB curve every 16 samples independently of callback size.
            if (controlCountdown == 0) {
                controlCountdown = 16;
                const float db = 10.f * std::log10(std::max(power, 1.e-12f));
                float change = std::min(maxLift, (threshold - db) * slope);
                if (change > 0) change *= std::clamp((db - floor) / 6.f, 0.f, 1.f);
                targetGain = amplitude(change);
            }
            --controlCountdown;
            const float target = enabled ? targetGain : 1.f;
            const float speed = enabled ? (target < vocalGain ? attackStep : releaseStep) : bypassStep;
            vocalGain += speed * (target - vocalGain);
            if (std::abs(target - vocalGain) < .0001f) vocalGain = target;
            samples[i] = sample * vocalGain;
            if (side) side[i] *= vocalGain;
        }
    }

    // One gain for the complete stereo mic return, including both delay tails.
    // Attack is immediate. A new peak also interrupts any release in progress.
    void finishMix(float *left, float *right, unsigned frames, const DynamicsControls &controls) {
        const float db = controls.ceilingDB.load();
        if (db != ceilingDB) { ceilingDB = db; ceiling = amplitude(db); }
        const bool muted = controls.muted.load();
        for (unsigned i = 0; i < frames; ++i) {
            const float peak = std::max(std::abs(left[i]), std::abs(right[i]));
            const float allowed = peak > ceiling ? ceiling / peak : 1.f;
            limitGain = std::min(allowed, limitGain + limiterRelease * (1.f - limitGain));
            if (allowed == 1.f && limitGain > .9998f) limitGain = 1.f;
            if (muted) unmuteGain = 0;
            else unmuteGain = std::min(1.f, unmuteGain + 1.f / (.005f * rate));
            const float gain = limitGain * unmuteGain;
            left[i] *= gain;
            right[i] *= gain;
        }
    }
    float singerPower() const { return power; }
    float singerGain() const { return vocalGain; }
    float limiterGain() const { return limitGain; }
};
