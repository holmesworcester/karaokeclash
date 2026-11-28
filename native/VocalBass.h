#pragma once
#include <algorithm>
#include <array>
#include <cmath>

// A scored bass voice on the wet bus. A singer can open it with a vowel
// even when their microphone has very little energy at the written fundamental.
class VocalBass {
    std::array<float,1025> wave{};
    std::array<float,128> steps{};
    float phase=0, step=0, envelope=0, low=0, mid=0, body=0;
    float lowCoef=0, midCoef=0, attack=0, release=0;
public:
    void reset() { phase=step=envelope=low=mid=body=0; }
    void prepare(double rate) {
        for (unsigned i=0;i<wave.size();++i) {
            double a=2*M_PI*i/(wave.size()-1);
            wave[i]=std::sin(a)+.38*std::sin(2*a)+.16*std::sin(3*a);
        }
        for (unsigned k=0;k<steps.size();++k)
            steps[k]=440*std::exp2((int(k)-69)/12.)/rate;
        lowCoef=1-std::exp(-2*M_PI*95/rate);
        midCoef=1-std::exp(-2*M_PI*1300/rate);
        attack=1-std::exp(-1/(.009*rate));
        release=1-std::exp(-1/(.035*rate));
        reset();
    }
    float process(float mic, int pitch) {
        low+=lowCoef*(mic-low);
        mid+=midCoef*(mic-low-mid);
        body+=midCoef*(mid-body);
        float target=pitch<0 ? 0 : std::max(0.f,std::abs(body)-.0015f);
        envelope+=(target>envelope ? attack : release)*(target-envelope);
        if (pitch>=0) step=steps[pitch];
        phase+=step;
        if (phase>=1) phase-=1;
        float at=phase*(wave.size()-1);
        unsigned i=unsigned(at);
        return envelope*(wave[i]+(at-i)*(wave[i+1]-wave[i]));
    }
};
