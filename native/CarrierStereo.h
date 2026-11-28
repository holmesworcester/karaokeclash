#pragma once
#include <algorithm>
#include <array>
#include <cmath>

// TAL sums the external carrier internally. Recover its recorded placement on
// the wet return from the actual oscillators, including during a glide.
class CarrierStereo {
    struct Crossover {
        float low=0, high=0;
        std::array<float,3> split(float x, float lowStep, float highStep) {
            low += lowStep*(x-low);
            high += highStep*(x-high);
            return {low,high-low,x-high};
        }
    };
    Crossover mid, side, wet;
    std::array<float,3> energy{}, covariance{};
    float lowStep=0, highStep=0, smooth=0;
public:
    void prepare(double rate) {
        lowStep=1-std::exp(-2*3.141592653589793*250/rate);
        highStep=1-std::exp(-2*3.141592653589793*1200/rate);
        smooth=1-std::exp(-1/(.025*rate));
        reset();
    }
    void reset() {
        mid={}; side={}; wet={}; energy={}; covariance={};
    }
    float shape(float voice, float carrier, float difference) {
        auto m=mid.split(carrier,lowStep,highStep);
        auto s=side.split(difference,lowStep,highStep);
        auto v=wet.split(voice,lowStep,highStep);
        float result=0;
        for (unsigned i=0;i<3;++i) {
            energy[i]+=smooth*(m[i]*m[i]-energy[i]);
            covariance[i]+=smooth*(m[i]*s[i]-covariance[i]);
            result+=v[i]*std::clamp(covariance[i]/(energy[i]+1.e-9f),-.95f,.95f);
        }
        return result;
    }
};
