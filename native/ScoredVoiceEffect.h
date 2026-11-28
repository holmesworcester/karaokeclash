#pragma once
#include "ClearVoice.h"
#include "ChordTune.h"
#include "CarrierStereo.h"
#include "ScoredVst.h"
#include <atomic>
#include "Score.h"
#include "VocalBass.h"

constexpr int vocoderHold = -1, tunelet = -2, talVocoder = -3, chorale = -4;
inline bool tunesChord(int voice) { return voice == tunelet || voice == chorale; }
inline bool fullyWet(int voice) { return tunesChord(voice) || voice == talVocoder; }
inline bool usesCarrier(int voice) { return voice == 1 || voice == vocoderHold || fullyWet(voice); }

class ScoredVoiceEffect {
    ScoredVst tal{VstKind::Tal};
    ChordTune harmony;
    CarrierStereo talStereo;
    int lastMode=0;
    unsigned quietFrames=0, quietHold=0;
    ClearVoice vocoder;
    VocalBass bass;
    ClearChord chord;
    const Pitches *previous=nullptr;
    const Score *lastScore=nullptr;
    bool previewing=false;
    size_t riffIndex=0, leadIndex=0, bassIndex=0;
    int previousRiff=-1, previousRegister=0;
    float leadLevel=.7f;
    double rate=0, nextTime=0;
    float gate=0, mix=1.f, clarity=.5f, harmonics=.55f, lastAttack=5, lastRelease=40, lastFormant=0;
public:
    std::atomic<float> stereoWidth{1.f};
    float width=1.f;
    std::atomic<int> synthRegister{0};
    std::atomic<float> glideMs{100}, targetLead{.7f};
    std::atomic<float> targetMix{1.f}, targetClarity{.5f}, attackMs{5}, releaseMs{40}, formant{0}, targetHarmonics{.55f};
private:
    void updateEnvelope() {
        float semitones=formant.load();
        if (semitones!=lastFormant) {
            vocoder.formants(semitones); lastFormant=semitones;
        }
        const float attack=attackMs.load(), release=releaseMs.load();
        if (attack!=lastAttack || release!=lastRelease) {
            vocoder.attack=1-std::exp(-1000/(attack*rate));
            vocoder.release=1-std::exp(-1000/(release*rate));
            lastAttack=attack; lastRelease=release;
        }
    }
public:
    ScoredVst &talPlugin() { return tal; }
    void prepareChorale() { harmony.prepareChorale(unsigned(rate)); }
    bool choraleReady() const { return harmony.choraleReady(); }
    bool talReady() const { return tal.ready(); }

    void prepare(double hz) {
        rate=hz; quietHold=unsigned(.012*hz); lastScore=nullptr; previous=nullptr; gate=0;
        vocoder.prepare(hz); chord.prepare(hz); tal.prepare(hz); harmony.prepare(hz); talStereo.prepare(hz); bass.prepare(hz);
    }

    void process(float *signal, unsigned n, double time, const Score &score, int mode, double speed,
                 bool preview, float *carrierOut, float *carrierSide, float *voiceSide) {
        bool changed=lastScore!=&score || mode!=lastMode || preview!=previewing;
        // With the transport stopped there is no backing clock to follow.
        // Let the mic or Hear sample audition the new score from its first attack.
        if (preview) time=changed ? score.chords.front().start : nextTime;
        previewing=preview;
        if (changed || (mode!=vocoderHold && std::abs(time-nextTime)>.02)) {
            previous=nullptr; previousRiff=-1; gate=0; chord.reset(); leadLevel=targetLead.load();
            vocoder.reset(); bass.reset(); talStereo.reset();
            if (tunesChord(mode)) harmony.reset(mode==chorale);
            if (fullyWet(mode)) { mix=targetMix.load(); quietFrames=quietHold; }
        }
        updateEnvelope();
        int reg=synthRegister.load();
        int octave=reg==1 ? score.speechOctave : reg;
        if (octave!=previousRegister) { previous=nullptr; previousRiff=-1; }
        previousRegister=octave;
        chord.setRegister(octave);

        const bool hold=mode==vocoderHold || fullyWet(mode);
        const float blend=targetMix.load(), detail=targetClarity.load(), bite=targetHarmonics.load();
        const double glide=glideMs.load()*.001*speed;
        const float leadTarget=targetLead.load();
        lastMode=mode;
        lastScore=&score; nextTime=time+n*speed/rate;
        std::array<float,ScoredVst::blockSize> sidechain, sideDifference, wet, wetGain, wetBass;
        const float targetWidth=stereoWidth.load();
        if (changed) width=targetWidth;
        if (tunesChord(mode)) {
            harmony.render(signal,voiceSide,n,score,time,speed/rate,octave,leadTarget,targetWidth,glideMs.load(),mode==chorale);
            for (unsigned i=0;i<n;++i) {
                mix+=std::clamp(blend-mix,-.005f,.005f);
                signal[i]*=mix;
                voiceSide[i]*=mix;
            }
            if (!carrierOut) return;
        }
        for (unsigned i=0;i<n;++i) {
            const float input=signal[i];
            mix+=std::clamp(blend-mix,-.005f,.005f);
            clarity+=std::clamp(detail-clarity,-.005f,.005f);
            harmonics+=std::clamp(bite-harmonics,-.005f,.005f);
            const double at=time+i*speed/rate;
            const Pitches *notes=nullptr;
            if (hold) {
                if (auto held=score.heldAt(at)) notes=&held->notes;
            } else if (auto frame=score.at(at,true,glide>0)) {
                notes=glide>0 ? &frame->legato : &frame->held;
            }
            Score::Slide slide;
            if (glide>0 && at>=0)
                slide=Score::slideAt(score.upper,score.localTime(at),glide,score.loop,riffIndex,hold);
            if (notes!=previous || previousRiff!=slide.key) {
                unsigned char keys[128]; unsigned count=0;
                if (notes) for (unsigned k=0;k<128;++k) if ((*notes)[k] && int(k)!=slide.key) keys[count++]=k;
                chord.setNotes(keys,count,score.gains.data());
                previous=notes; previousRiff=slide.key;
            }
            leadLevel+=std::clamp(leadTarget-leadLevel,-.005f,.005f);
            Score::Slide melody;
            if (leadLevel>0) melody=Score::slideAt(score.lead,at,glide,0,leadIndex);
            float leadPitch=melody.key<0 ? -1 : melody.pitch;
            const float riffPitch=slide.key<0 ? -1 : slide.pitch;
            const float riffGain=slide.key<0 ? 0 : score.gains[slide.key];
            width+=std::clamp(targetWidth-width,-.005f,.005f);
            float side=0;
            float carrier=chord.next(harmonics,score.unison,riffPitch,riffGain,leadPitch,leadLevel,
                                     score.stereo ? score.pan.data() : nullptr,&side);
            side*=width;
            if (carrierSide) carrierSide[i]=side;
            if (carrierOut) carrierOut[i]=carrier;
            if (tunesChord(mode) || !usesCarrier(mode)) continue;  // Audition the synth while leaving the mic in its own mode.
            float lowVoice=0;
            if (!score.bass.empty()) {
                const double local=score.localTime(at);
                const auto *note=at>=0 ? Score::noteBefore(score.bass,local,bassIndex) : nullptr;
                lowVoice=bass.process(input,note && local<note->end ? note->pitch : -1);
            }
            bool active=(notes && notes->any()) || slide.key>=0 || leadPitch>=0;
            if (mode==talVocoder) {
                // TAL's analysis filters can ring after the singer stops. The
                // short hold bridges zero crossings and closes on a silent mic.
                quietFrames=std::abs(input)>1.e-5f ? 0 : std::min(quietFrames+1,quietHold);
                float target=quietFrames<quietHold ? 1.f : 0.f;
                gate+=std::clamp(target-gate,-float(1/(.002*rate)),float(1/(.002*rate)));
                Pitches keys=notes ? *notes : Pitches{};
                if (melody.key>=0) keys.set(melody.key);
                tal.notes(octave<0 ? keys >> (-12*octave) : keys << (12*octave), i);
                sidechain[i]=carrier;
                sideDifference[i]=side;
                wetGain[i]=active ? mix*gate : 0;
                wetBass[i]=mix*gate*lowVoice;
                continue;
            }
            float target=active ? 1 : 0;
            gate+=std::clamp(target-gate,-float(1/(.002*rate)),float(1/(.002*rate)));
            float shapedSide=0;
            const float voiced=vocoder.process(input,carrier,gate>0,mix,clarity,side,
                                                score.stereo ? &shapedSide : nullptr);
            voiceSide[i]=limitVoice(gate*mix*shapedSide);
            signal[i]=limitVoice(input+gate*(voiced-input)+mix*lowVoice);
        }
        if (mode==talVocoder) {
            if (tal.render(signal,sidechain.data(),wet.data(),n,voiceSide)) {
                for (unsigned i=0;i<n;++i) {
                    signal[i]=limitVoice(wetGain[i]*wet[i]+wetBass[i]);
                    if (score.stereo)
                        voiceSide[i]+=talStereo.shape(wet[i],sidechain[i],sideDifference[i]);
                    voiceSide[i]*=wetGain[i];
                }
            } else {
                std::fill_n(signal,n,0.f);
            }
        }
    }
    void deselect() { lastScore=nullptr; }
};
