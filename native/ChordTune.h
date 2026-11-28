#pragma once
#include "Score.h"
#include "ChoraleHarmony.h"
#include <memory>

struct ScoreHarmony;
extern "C" {
ScoreHarmony *score_harmony_new(unsigned rate);
void score_harmony_free(ScoreHarmony *);
void score_harmony_reset(ScoreHarmony *);
void score_harmony_render(ScoreHarmony *, const int notes[4], const float levels[4][2],
                          float glideMs, float *mid, float *side, size_t frames);
}

class ChordTune {
    std::unique_ptr<ScoreHarmony, decltype(&score_harmony_free)> dsp{nullptr, score_harmony_free};
    std::array<int,4> assigned{};
    ChoraleHarmony chorale;
    size_t leadIndex=0, bassIndex=0;

    static int sounding(const std::vector<Note> &line, double time, size_t &index) {
        const Note *n=Score::noteBefore(line,time,index);
        return n && time<n->end ? n->pitch : -1;
    }
public:
    void prepare(double rate) { dsp.reset(score_harmony_new(unsigned(rate))); }
    void prepareChorale(unsigned rate) { chorale.prepare(rate); }
    bool choraleReady() const { return chorale.ready(); }
    void reset(bool vst) {
        if (vst) chorale.reset();
        else if (dsp) score_harmony_reset(dsp.get());
        assigned.fill(-1);
        leadIndex=bassIndex=0;
    }

    void render(float *mic, float *side, unsigned frames, const Score &score, double time,
                double step, int octave, float lead, float width, float glideMs, bool vst) {
        // Limit note-control quantization to 1.3 ms even for long offline blocks.
        for (unsigned pos=0; pos<frames; pos+=64) {
            const unsigned count=std::min(64u,frames-pos);
            const double at=time+pos*step;
            auto held=score.heldAt(at);
            std::array<int,4> keys{-1,-1,-1,-1};
            std::array<float,4> gains{}, pans{};
            unsigned used=0;
            auto add=[&](int key, float gain) {
                if (key<0 || gain<=0 || key+12*octave<0 || key+12*octave>127) return;
                for (unsigned k=0;k<used;++k) if (keys[k]==key) {
                    gains[k]=std::max(gains[k],gain);
                    return;
                }
                if (used<4) {
                    keys[used]=key;
                    gains[used]=gain;
                    pans[used]=score.pan[key]*width;
                    ++used;
                }
            };
            add(sounding(score.lead,at,leadIndex),lead);
            if (held) {
                // Preserve the outer voices when a dense score exceeds four singers.
                for (int k=127;k>=0;--k) if (held->notes[k]) { add(k,score.gains[k]); break; }
                for (int k=0;k<128;++k) if (held->notes[k]) { add(k,score.gains[k]); break; }
                add(sounding(score.bass,score.localTime(at),bassIndex),.65f);
                for (int k=0;k<128 && used<4;++k) if (held->notes[k]) add(k,score.gains[k]);
            }
            std::array<int,4> next{-1,-1,-1,-1};
            float levels[4][2]{};
            float scale=.85f/std::sqrt(float(std::max(1u,used)));
            // Keep tied notes on the same Tunelet reader when another part moves.
            for (unsigned k=0;k<used;++k) keys[k]+=12*octave;
            if (vst) {
                chorale.render(mic+pos,side+pos,count,keys,gains,pans,used);
                continue;
            }
            for (unsigned k=0;k<used;++k) {
                auto old=std::find(assigned.begin(),assigned.end(),keys[k]);
                if (old!=assigned.end()) next[old-assigned.begin()]=keys[k];
            }
            for (unsigned k=0;k<used;++k) {
                auto slot=std::find(next.begin(),next.end(),keys[k]);
                if (slot==next.end()) slot=std::find(next.begin(),next.end(),-1);
                *slot=keys[k];
                auto &pair=levels[slot-next.begin()];
                pair[0]=scale*gains[k]*(1-pans[k]);
                pair[1]=scale*gains[k]*(1+pans[k]);
            }
            assigned=next;
            if (dsp) score_harmony_render(dsp.get(),assigned.data(),levels,glideMs,mic+pos,side+pos,count);
            if (!dsp || !used) {
                std::fill_n(mic+pos,count,0.f);
                std::fill_n(side+pos,count,0.f);
            }
        }
    }
};
