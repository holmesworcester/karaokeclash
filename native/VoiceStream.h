#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>

// One writer and one reader. The producer never blocks the device callback.
template <typename T, unsigned N> class VoiceQueue {
    std::array<T,N> data{};
    std::atomic<unsigned> head{0}, tail{0};
public:
    bool push(const T &item) {
        auto h = head.load(std::memory_order_relaxed);
        if (h-tail.load(std::memory_order_acquire) == N) return false;
        data[h%N] = item;
        head.store(h+1, std::memory_order_release);
        return true;
    }
    bool pop(T &item) {
        auto t = tail.load(std::memory_order_relaxed);
        if (t == head.load(std::memory_order_acquire)) return false;
        item = data[t%N];
        tail.store(t+1, std::memory_order_release);
        return true;
    }
};

// Symmetric Blackman-windowed sinc at 7 kHz, 48 kHz device rate. Each
// direction contributes 15 device samples. Zero-stuffing is normalized by 3.
class VoiceRateFilter {
    std::array<float,31> taps{}, history{};
    unsigned index=0;
public:
    VoiceRateFilter() {
        constexpr double pi=3.141592653589793;
        double total=0;
        for (int i=0;i<31;++i) {
            const double x=i-15, f=7000./48000;
            taps[i]=(x==0 ? 2*f : std::sin(2*pi*f*x)/(pi*x)) *
                (.42-.5*std::cos(2*pi*i/30)+.08*std::cos(4*pi*i/30));
            total+=taps[i];
        }
        for (auto &t:taps) t/=total;
    }
    void clear() { history.fill(0); index=0; }
    float tick(float v) {
        history[index]=v;
        float out=0;
        unsigned j=index;
        for (float tap:taps) { out+=tap*history[j]; j=j ? j-1 : 30; }
        if (++index==31) index=0;
        return out;
    }
};

struct VoiceRequest {
    uint64_t generation=0, sequence=0;
    int voice=0;
    std::array<float,240> audio{};
};
struct VoiceResponse {
    uint64_t generation=0, sequence=0;
    std::array<float,208> pcm{};
};

class VoiceStream {
public:
    static constexpr unsigned hop=624, defaultPlayout=1152, delay=defaultPlayout+30;
    // Each profile waits for a hop and 1 ms of right context before dispatch.
    const unsigned block, deviceHop, playout, dryDelay;
    explicit VoiceStream(unsigned wait=defaultPlayout, unsigned modelBlock=208)
        : block(modelBlock==64 ? 64 : 208), deviceHop(block*3),
          playout(std::clamp(wait,deviceHop+48,2018u)), dryDelay(playout+30) {}
    VoiceQueue<VoiceRequest,4> requests;
    VoiceQueue<VoiceResponse,4> results;
    std::atomic<uint64_t> generation{0};
    std::atomic<int> selected{0};
    std::atomic<unsigned> missed{0}, dropped{0};
private:
    VoiceRateFilter down, up;
    VoiceRequest building;
    VoiceResponse pending, playing;
    bool hasPending=false, wet=false;
    unsigned used=16, dryIndex=0;
    uint64_t frame=0;
    std::array<float,2048> dry{};
    float mix=0;

    void select(int voice) {
        building={};
        building.voice=voice;
        selected.store(voice);
        building.generation=generation.fetch_add(1)+1;
        // Remove completed jobs from the previous generation before queueing
        // the first fresh frame. A concurrent late result is tagged and discarded in receive().
        while (results.pop(pending)) {}
        hasPending=wet=false;
        down.clear(); up.clear(); dry.fill(0);
        frame=0; dryIndex=0; used=16; mix=0;
        missed.store(0); dropped.store(0);
    }

    void receive(uint64_t sequence) {
        wet=false;
        while (hasPending || results.pop(pending)) {
            hasPending=false;
            if (pending.generation!=building.generation || pending.sequence<sequence) continue;
            if (pending.sequence>sequence) { hasPending=true; break; }
            playing=pending; wet=true; break;
        }
        if (!wet) ++missed;
    }
public:
    // Device-thread only. Zero selects the app's existing tuned singer.
    bool process(int voice, const float *input, float *output, unsigned frames) {
        if (voice!=building.voice) select(voice);
        if (!voice) return false;
        for (unsigned i=0;i<frames;++i,++frame) {
            const float sample=input[i], fallback=dry[dryIndex];
            dry[dryIndex]=sample;
            if (++dryIndex==dryDelay) dryIndex=0;
            const float filtered=down.tick(sample);
            if (frame%3==0) {
                building.audio[used++]=filtered;
                if (used==block+32) {
                    if (!requests.push(building)) ++dropped;
                    ++building.sequence;
                    std::copy_n(building.audio.begin()+block,32,building.audio.begin());
                    used=32;
                }
            }
            unsigned phase=0;
            if (frame>=playout) {
                phase=(frame-playout)%deviceHop;
                if (phase==0) receive((frame-playout)/deviceHop);
            }
            float excitation=wet && phase%3==0 ? 3*playing.pcm[phase/3] : 0;
            float converted=up.tick(excitation);
            mix=std::clamp(mix+(wet ? 1.f : -1.f)/48,0.f,1.f);
            output[i]=fallback+(converted-fallback)*mix;
        }
        return true;
    }
};
