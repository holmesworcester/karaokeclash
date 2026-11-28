#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

class VocalDelay {
    std::vector<float> ring;
    size_t write=0, valid=0;
    double hz=0;
    float filtered=0, coefficient=0;
public:
    void clear() { valid=0; filtered=0; }
    void prepare(double sampleRate) {
        hz=sampleRate;
        ring.assign(size_t(std::ceil(hz*2))+1,0);
        write=valid=0; filtered=0;
        coefficient=1-std::exp(-2*3.141592653589793*4200/hz);
    }
    void process(float *voice,unsigned frames,float mix,double seconds,float feedback) {
        if (mix<=0) { clear(); return; }
        size_t lag=std::clamp<long>(std::lround(hz*seconds),1,ring.size()-1);
        feedback=std::clamp(feedback,0.f,.85f);
        for (unsigned i=0;i<frames;++i) {
            float tap=valid>=lag ? ring[(write+ring.size()-lag)%ring.size()] : 0.f;
            filtered+=coefficient*(tap-filtered);
            ring[write]=voice[i]+feedback*filtered;
            voice[i]+=mix*filtered;
            if (++write==ring.size()) write=0;
            if (valid<ring.size()) ++valid;
        }
    }
};
