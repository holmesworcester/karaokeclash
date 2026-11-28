#pragma once
#include "ScoredVst.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <string>

struct Diction;
extern "C" {
Diction *score_diction_new(unsigned rate, unsigned latency);
void score_diction_free(Diction *);
void score_diction_reset(Diction *);
void score_diction_process(Diction *, const float *mic, float *mid, float *side, size_t frames);
}

// Chorale uses one mono input, assigning held MIDI keys to voices from low to high.
class ChoraleHarmony {
    ScoredVst plugin{VstKind::Chorale};
    std::unique_ptr<Diction, decltype(&score_diction_free)> diction{nullptr,score_diction_free};
    std::array<int,4> gain{}, pan{};
    unsigned warmup=0;
    bool attempted=false;
public:
    bool ready() const { return bool(diction); }
    void prepare(unsigned rate) {
        if (attempted) return;
        attempted=true;
        plugin.prepare(rate);
        if (!plugin.ready()) return;
        const int mix=plugin.parameterIndex("Mix");
        const int humanize=plugin.parameterIndex("Humanize");
        if (mix<0 || humanize<0) return;
        plugin.setParameter(mix,1.f);
        plugin.setParameter(humanize,0.f);
        // Keep Studio's larger pitch-analysis window for low rap vocals.
        for (unsigned i=0;i<gain.size();++i) {
            std::string prefix="Voice "+std::to_string(i+1)+" ";
            int mode=plugin.parameterIndex((prefix+"Mode").c_str());
            gain[i]=plugin.parameterIndex((prefix+"Gain").c_str());
            pan[i]=plugin.parameterIndex((prefix+"Pan").c_str());
            if (mode<0 || gain[i]<0 || pan[i]<0) return;
            plugin.setParameter(mode,1.f); // Off / Scale / Note / MIDI
        }
        diction.reset(score_diction_new(rate,plugin.latency()));
    }
    void reset() {
        if (!ready()) return;
        plugin.notes({},0);
        score_diction_reset(diction.get());
        warmup=plugin.latency();
    }
    void render(float *mic,float *side,unsigned n,const std::array<int,4> &keys,
                const std::array<float,4> &gains,const std::array<float,4> &pans,unsigned used) {
        if (!ready()) {
            std::fill_n(mic,n,0.f);
            std::fill_n(side,n,0.f);
            return;
        }
        std::array<unsigned,4> order{0,1,2,3};
        std::sort(order.begin(),order.begin()+used,[&](unsigned a,unsigned b){return keys[a]<keys[b];});
        std::bitset<128> held;
        const float normalization=1.f/std::sqrt(float(std::max(used,1u)));
        for (unsigned i=0;i<used;++i) {
            unsigned key=order[i];
            held.set(keys[key]);
            plugin.setParameter(gain[i],gains[key]*normalization);
            plugin.setParameter(pan[i],.5f*(1+pans[key]));
        }
        plugin.notes(held,0);
        std::array<float,ScoredVst::blockSize> wet{};
        bool rendered=plugin.render(mic,nullptr,wet.data(),n,side);
        if (rendered) {
            score_diction_process(diction.get(),mic,wet.data(),side,n);
            std::copy_n(wet.begin(),n,mic);
        }
        const unsigned priming=std::min(n,warmup);
        warmup-=priming;
        const unsigned muted=rendered && used ? priming : n;
        std::fill_n(mic,muted,0.f);
        std::fill_n(side,muted,0.f);
    }
};
