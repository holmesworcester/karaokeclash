#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

inline float limitVoice(float x) {
    float excess=std::abs(x)-.8f;
    return excess>0 ? std::copysign(.8f+excess/(1+5*excess),x) : x;
}

// Also compiled for the four-mic AudioWorklet. All storage belongs to a single
// microphone. Only setup calls libm; the sample path allocates no memory.
struct ClearVoice {
    static constexpr unsigned bands = 24;
    struct Filter {
        float z1=0, z2=0;
        float run(float x, float b, float a1, float a2) {
            const float y = b*x + z1;
            z1 = z2 - a1*y;
            z2 = -b*x - a2*y;
            return y;
        }
    };
    struct Band {
        float b=0, a1=0, a2=0, envelope=0, level=0, unvoiced=0, fraction=0;
        unsigned lower=0, upper=0;
        Filter speech, synth, shaped, sideSynth, sideShaped;
    } bank[bands];
    float attack, release, carrierRelease, carrierAttack, rate;
    float carrierPeak=0;
    uint32_t noiseState = 1;
    static float absf(float x) { return x < 0 ? -x : x; }
    static float minimum(float x,float y) { return x < y ? x : y; }

    void reset() {
        carrierPeak=0; noiseState=1;
        for (auto &f:bank) { f.speech={}; f.synth={}; f.shaped={}; f.sideSynth={}; f.sideShaped={}; f.level=f.envelope=0; }
    }
    void formants(float semitones) {
        const float bandsPerSemitone = (bands-1)/(12*std::log2(105.f));
        for (unsigned i=0; i<bands; ++i) {
            float source=std::clamp(float(i)-semitones*bandsPerSemitone,0.f,float(bands-1));
            auto &f=bank[i];
            f.lower=unsigned(source);
            f.upper=std::min(f.lower+1,bands-1);
            f.fraction=source-f.lower;
        }
    }
    void prepare(float hz) {
        rate=hz;
        attack=1-__builtin_exp(-1/(.005*hz));
        release=1-__builtin_exp(-1/(.040*hz));
        carrierAttack=1-__builtin_exp(-1/(.0015*hz));
        carrierRelease=1-__builtin_exp(-1/(.035*hz));
        for (unsigned i=0; i<bands; ++i) {
            auto &f=bank[i];
            f={};
            const double hzCenter=80*__builtin_pow(105.,double(i)/(bands-1));
            const double angle=6.28318530718*minimum(hzCenter,.43f*hz)/hz;
            const double alpha=__builtin_sin(angle)/(2*3.3);
            f.b=alpha/(1+alpha); f.a1=-2*__builtin_cos(angle)/(1+alpha);
            f.a2=(1-alpha)/(1+alpha);
            f.unvoiced=std::clamp(float((hzCenter-1700)/2200),0.f,1.f);
        }
        formants(0);
        reset();
    }

    float process(float speech, float carrier, bool active, float mix, float consonants,
                  float carrierSide=0, float *wetSide=nullptr) {
        // The modulator contributes only band envelopes to the wet bus.
        // One shared noise excitation supplies high-band consonants for carriers
        // with little treble. It passes through the same speech-shaped filters.
        float high=0, total=0;
        for (auto &f:bank) {
            float mod=absf(f.speech.run(speech,f.b,f.a1,f.a2));
            f.envelope += (mod>f.envelope ? attack : release)*(mod-f.envelope);
            total += f.envelope;
            high += f.unvoiced*f.envelope;
        }
        float magnitude=absf(carrier);
        carrierPeak+=(magnitude>carrierPeak ? carrierAttack : carrierRelease)*(magnitude-carrierPeak);
        float excitation=carrier/(.025f+carrierPeak);
        noiseState=noiseState*1664525u+1013904223u;
        float noise=float(noiseState >> 8)*(2.f/16777216.f)-1.f;
        float hiss=consonants*minimum(1.f,5*high/(.0001f+total));
        float wet=0;
        for (auto &f:bank) {
            float blend=hiss*f.unvoiced;
            float band=f.synth.run(excitation*(1-blend)+noise*blend,f.b,f.a1,f.a2);
            band=f.shaped.run(band,f.b,f.a1,f.a2);
            f.level+=carrierRelease*(absf(band)-f.level);
            float env=bank[f.lower].envelope+f.fraction*(bank[f.upper].envelope-bank[f.lower].envelope);
            // Normalize partials within their own bands. A floor keeps leakage
            // from a distant fundamental from flattening the vowel spectrum.
            float gain=env/(.018f+f.level);
            wet+=gain*band;
            if (wetSide) {
                float side=f.sideSynth.run(carrierSide/(.025f+carrierPeak)*(1-blend),f.b,f.a1,f.a2);
                *wetSide+=gain*f.sideShaped.run(side,f.b,f.a1,f.a2);
            }
        }
        return active ? speech+mix*(wet-speech) : speech;
    }
};

// A dual-read-head pitch shifter. Read at twice the write rate, and move back
// by one locally matching period before catching up. This raises formants as
// well as pitch while keeping the length and articulation of the utterance.
class TechnologicVoice {
    static constexpr unsigned length=8192, mask=length-1;
    float ring[length]{}, low1=0, low2=0, lowCoef=0;
    unsigned write=0, delay=0, oldDelay=0, fade=0, fadeLength=0;
    unsigned minPeriod=0,maxPeriod=0,lookback=0,step=0;
    float at(unsigned lag) const { return ring[(write-lag)&mask]; }
    unsigned alignedPeriod() const {
        unsigned best=minPeriod;
        float bestMatch=-2;
        for (unsigned lag=minPeriod;lag<=maxPeriod;lag+=step) {
            float xy=0,xx=1e-8f,yy=1e-8f;
            for (unsigned j=0;j<lookback;j+=step) {
                float x=at(delay+j),y=at(delay+lag+j);
                xy+=x*y; xx+=x*x; yy+=y*y;
            }
            float score=xy*__builtin_fabsf(xy)/(xx*yy);
            // Prefer the first good period over nearly identical multiples.
            if (score>bestMatch+.012f) { bestMatch=score; best=lag; }
        }
        return best;
    }
public:
    void reset() {
        for (float &x:ring) x=0;
        low1=low2=0; write=fade=oldDelay=0; delay=fadeLength+maxPeriod+2;
    }
    void prepare(float rate) {
        low1=low2=0;
        lowCoef=1-__builtin_exp(-6.2831853*.19);
        fadeLength=unsigned(.002f*rate)+1;
        minPeriod=unsigned(rate/390);
        maxPeriod=unsigned(rate/65);
        lookback=unsigned(.005f*rate);
        step=unsigned(rate/12000)+1;
        reset();
    }
    float process(float x) {
        low1+=lowCoef*(x-low1); low2+=lowCoef*(low1-low2);
        ring[write&mask]=low2;
        if (!fade && delay<=fadeLength+2) {
            oldDelay=delay;
            delay+=alignedPeriod();
            fade=fadeLength;
        }
        float y=at(delay);
        if (fade) {
            float mix=float(fade)/fadeLength;
            y+=mix*(at(oldDelay)-y);
            --fade; --oldDelay;
        }
        ++write; --delay;
        return 1.22f*y;
    }
};

class ClearChord {
    struct Oscillator { float phase=0, second=.37f, mellow=0; };
    Oscillator keys[128]{}, riff{}, lead{};
    float step[128]{}, power=0, scale=0;
    float gains[128]{};
    unsigned notes[128]{}, count=0;
    float pitchRatio=1, body=0, sideBody=0, bodyCoef=0;
    // Polynomial correction at the discontinuity keeps high notes from aliasing.
    static float saw(float phase,float dt) {
        float wave=2*phase-1;
        if (phase<dt) { float t=phase/dt; wave-=(2-t)*t-1; }
        if (phase>1-dt) { float t=(phase-1)/dt; wave-=(t+2)*t+1; }
        return wave;
    }
    float wave(Oscillator &osc,float increment,float brightness,float unison) {
        float ramp=saw(osc.phase,increment);
        float shifted=osc.phase+.18f;
        if (shifted>=1) shifted-=1;
        float bright=ramp-.8f*saw(shifted,increment);
        float cutoff=6.28319f*increment;
        osc.mellow+=cutoff/(1+cutoff)*(ramp-osc.mellow);
        osc.phase+=increment; if (osc.phase>=1) osc.phase-=1;
        float result=osc.mellow+brightness*(bright-osc.mellow);
        if (unison>0) {
            // A second saw four cents sharp fills gaps in the pulse spectrum.
            float detuned=increment*1.002313f;
            result+=unison*saw(osc.second,detuned);
            osc.second+=detuned; if (osc.second>=1) osc.second-=1;
        }
        return result;
    }
public:
    void reset() {
        count=0; power=0; scale=0; riff={}; lead={}; body=sideBody=0;
        for (unsigned k=0;k<128;++k) {
            keys[k]={};
            keys[k].phase=((k*13)%37)/37.f;
            keys[k].second=((k*7)%29)/29.f;
        }
    }
    void prepare(float rate) {
        bodyCoef=1-std::exp(-6.283185f*1150/rate);
        for (unsigned k=0;k<128;++k)
            step[k]=ClearVoice::minimum(.49f,440*__builtin_pow(2.,(int(k)-69)/12.)/rate);
        reset();
    }
    void setRegister(int octave) {
        pitchRatio=octave==-2 ? .25f : octave==-1 ? .5f : 1.f;
    }
    void setNotes(const unsigned char *keys,unsigned n,const float *levels=nullptr) {
        count=n<128 ? n : 128;
        power=0;
        for (unsigned i=0;i<count;++i) {
            notes[i]=keys[i]&127;
            gains[i]=levels ? levels[notes[i]] : 1.f;
            power+=gains[i]*gains[i];
        }
        scale=.65f/__builtin_sqrtf(std::max(1.f,power));
    }
    float incrementAt(float pitch) const {
        int k=int(pitch);
        return (step[k]+(step[std::min(127,k+1)]-step[k])*(pitch-k))*pitchRatio;
    }
    float next(float brightness,float unison=0,float riffPitch=-1,float riffGain=0,
               float leadPitch=-1,float leadGain=0, const float *pan=nullptr, float *sideOut=nullptr) {
        float sum=0, side=0;
        for (unsigned j=0;j<count;++j) {
            const auto k=notes[j];
            float voice=gains[j]*wave(keys[k],step[k]*pitchRatio,brightness,unison);
            sum+=voice;
            if (pan) side+=voice*pan[k];
        }
        float normalization=scale;
        if (riffPitch>=0) {
            float voice=riffGain*wave(riff,incrementAt(riffPitch),brightness,unison);
            sum+=voice;
            if (pan) side+=voice*pan[std::clamp(int(riffPitch+.5f),0,127)];
        }
        if (leadPitch>=0)
            sum+=leadGain*wave(lead,incrementAt(leadPitch),brightness,unison);
        if (riffPitch>=0 || leadPitch>=0)
            normalization=.65f/__builtin_sqrtf(std::max(1.f,power+riffGain*riffGain+leadGain*leadGain));
        sum*=normalization;
        side*=normalization;
        if (unison>0) {
            // Emphasize upper partials before the speech envelopes shape them.
            body+=bodyCoef*(sum-body);
            sum+=5*brightness*unison*(sum-body);
            sideBody+=bodyCoef*(side-sideBody);
            side+=5*brightness*unison*(side-sideBody);
        }
        if (sideOut) *sideOut=side;
        return sum;
    }
};
