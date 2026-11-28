#pragma once
#include <bitset>
#include <cstdint>
#include <memory>

struct TalParameter {
    uint32_t id;
    int steps, flags;
    float initial;
    char name[128];
};

enum class VstKind { Tal, Chorale };

// A per-mic processor; instances share only their loaded VST3 module.
class ScoredVst {
    VstKind kind;
    struct Processor;
    std::unique_ptr<Processor> processor;

public:
    static constexpr unsigned blockSize = 512;
    explicit ScoredVst(VstKind kind);
    ~ScoredVst();
    void prepare(double rate);
    bool ready() const;
    int parameterCount() const;
    int parameterIndex(const char *name) const;
    unsigned latency() const;
    const TalParameter *parameter(int index) const;
    const char *display(int index, float value) const;
    void setParameter(int index, float value);
    void notes(const std::bitset<128> &keys, unsigned offset);
    bool render(float *mic, float *carrier, float *wet, unsigned frames, float *side = nullptr);
};
