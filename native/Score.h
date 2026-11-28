#pragma once
#include "TestSinger.h"
#include "CarrierNotes.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <atomic>
#include <cmath>
#include <vector>
#include <string>

struct Note { double start, end; int pitch; };
using Pitches = std::bitset<128>;

struct Score {
    static constexpr double phraseGap=.16;  // maximum scored rest to connect, in seconds
    struct Frame {
        double start;
        Pitches held, attacks, legato;
    };
    // The device thread can select an edition while the library publishes another cut.
    std::array<std::atomic<const Score *>, 4> versions{};
    const Score *version(int slot) const {
        if (slot > 0 && slot <= int(versions.size()))
            if (auto alternative = versions[slot-1].load()) return alternative;
        return playable() ? this : nullptr;
    }
    std::string backingFamily;
    std::vector<Note> notes;
    std::vector<Note> upper, lead, bass;
    float unison=0;
    std::array<float, 128> pan{};  // -1 left, +1 right, measured at the note fundamental
    bool stereo = false;
    std::array<float, 128> gains;
    Score() { gains.fill(1.f); }
    SingerClock recordingGrid{};
    bool playable() const { return !chords.empty(); }
    std::vector<Frame> frames;
    std::vector<CarrierChord> chords;
    Pitches loopTies;  // keys sustained across the join of an extracted MIDI loop
    double previewStart=0;
    double cycle = 0;        // optional break fill, in seconds
    double loop = 0, downbeat = 0; // repeating chant, on its backing clock

    int speechOctave = 0;
    void findSpeechRegister() {
        // Choose one transposition for the whole melody, using the median
        // lowest sounding chord note. Keep intervals across high/low phrases intact.
        std::array<size_t, 128> counts{};
        size_t total = 0;
        for (const auto &chord : chords) {
            for (unsigned k=0; k<128; ++k) if (chord.notes[k]) {
                ++counts[k]; ++total; break;
            }
        }
        speechOctave = 0;
        size_t sum = 0;
        for (unsigned k=0; k<128 && total; ++k) {
            sum += counts[k];
            if (sum*2 >= total) {
                speechOctave = k >= 68 ? -2 : k >= 56 ? -1 : 0;
                break;
            }
        }
    }

    // Compile overlaps once when loading the file. The device thread only reads.
    void compile(const int16_t *reference = nullptr, int stereoFrames = 0) {
        struct Edge { double time; int pitch, change; };
        std::vector<Edge> edges;
        // An isolated bass can be the only pitched instrument recovered for a cut.
        const auto &carrierNotes = notes.empty() && !reference ? bass : notes;
        for (const auto &n : carrierNotes) {
            edges.push_back({n.start,n.pitch,1});
            edges.push_back({n.end,n.pitch,-1});
        }
        std::sort(notes.begin(),notes.end(),[](auto &a,auto &b){return a.start<b.start;});
        std::sort(edges.begin(),edges.end(),[](auto &a,auto &b){return a.time<b.time;});
        std::array<int,128> count{};
        Pitches active;
        frames.clear();
        for (size_t i=0; i<edges.size();) {
            double time=edges[i].time;
            Pitches attacks;
            do {
                auto &e=edges[i++];
                if (e.change>0) attacks.set(e.pitch);
                count[e.pitch]+=e.change;
                active.set(e.pitch,count[e.pitch]>0);
            } while (i<edges.size() && edges[i].time-time<1e-8);
            frames.push_back({time,active,attacks});
        }
        if (notes.empty() && reference && stereoFrames > 0) {
            for (const auto &change : pitchChanges(reference, stereoFrames, 48000)) {
                frames.push_back({change.start, change.notes, change.notes & ~active});
                active = change.notes;
            }
        }
        compileLegato();
        // Follow the highest attacked part. Releasing it must not turn a
        // sustained bass note into the next pitch of a slide.
        upper.clear();
        for (const auto &frame : frames) {
            if (!upper.empty() && !frame.held[upper.back().pitch])
                upper.back().end=std::min(upper.back().end,frame.start);
            if (frame.attacks.none()) continue;
            int top=127;
            while (top>0 && !frame.attacks[top]) --top;
            // A new lower accompaniment note does not displace a sounding melody.
            if (!upper.empty() && upper.back().end>frame.start && top<upper.back().pitch) continue;
            if (!upper.empty()) upper.back().end=std::min(upper.back().end,frame.start);
            upper.push_back({frame.start,frames.back().start,top});
        }
        std::sort(lead.begin(),lead.end(),[](auto &a,auto &b){return a.start<b.start;});
        chords.clear();
        for (const auto &frame : frames)
            if (frame.attacks.any()) chords.push_back({frame.start, frame.held});
        if (!frames.empty() && frames.front().start==0)
            frames.front().attacks &= ~loopTies;
    }

    // Tie pitches common to adjacent chord attacks, including across a loop.
    // Longer rests still close the carrier when Vocoder Clear is selected.
    void compileLegato() {
        for (auto &frame:frames) frame.legato=frame.held;
        for (size_t i=1;i<frames.size();++i) {
            auto released=frames[i-1].held & ~frames[i].held;
            if (released.none()) continue;
            for (size_t j=i;j<i+frames.size();++j) {
                bool wrapped=j>=frames.size();
                if (wrapped && loop<=0) break;
                auto &next=frames[j%frames.size()];
                if (next.start+(wrapped ? loop : 0)-frames[i].start>phraseGap) break;
                if (next.attacks.none()) continue;
                auto common=released & next.attacks;
                for (size_t k=i;common.any() && k<j;++k) frames[k%frames.size()].legato|=common;
                break;
            }
        }
    }

    // Project a written part onto this backing. Full-album parts can have a
    // measured tempo grid, while older chant/bass excerpts repeat every few bars.
    std::vector<Note> project(const std::vector<Note> &part, const Score &source,
                              double duration, double offsetBars, bool repeat) const {
        std::vector<Note> result;
        if (part.empty()) return result;
        const auto &grid = source.recordingGrid.bpm > 0 ? source.recordingGrid : recordingGrid;
        auto destination = [&](double t) { return recordingGrid.seconds(grid.bar(t) - offsetBars); };
        double stop = grid.seconds(recordingGrid.bar(duration) + offsetBars);
        double period = repeat ? source.loop : 0;
        double shift = period > 0 ? period * std::floor(
            grid.seconds(recordingGrid.bar(0) + offsetBars) / period) : 0.;
        do {
            for (const Note &n : part) {
                double a = std::max(0., destination(n.start + shift));
                double b = std::min(duration, destination(n.end + shift));
                if (b > a) result.push_back({a, b, n.pitch});
            }
            shift += period;
        } while (period > 0 && shift < stop);
        return result;
    }

    void excerpt(const Score &source, double start, double end, double ratio) {
        notes.clear();
        gains=source.gains;
        loopTies.reset();
        for (const Note &n : source.notes) {
            if (n.end<=start || n.start>=end) continue;
            notes.push_back({(std::max(n.start,start)-start)*ratio,
                             (std::min(n.end,end)-start)*ratio, n.pitch});
            if (n.end>end) loopTies.set(n.pitch);
        }
        compile();
    }

    // Copy the polyphonic note events onto a second backing's beat grid.
    void fitChant(const Score &source,double timeRatio,double beatZero) {
        notes.clear();
        gains=source.gains;
        loopTies=source.loopTies;
        for (const Note &note:source.notes)
            notes.push_back({note.start*timeRatio,note.end*timeRatio,note.pitch});
        loop=source.loop*timeRatio;
        compile();
        downbeat=beatZero;
    }

    // The common case stays on the same note for an entire audio block.
    // A seek or loop wrap repositions directly on the score clock.
    static const Note *noteBefore(const std::vector<Note> &line, double t, size_t &index) {
        if (line.empty() || !std::isfinite(t) || t<line.front().start) return nullptr;
        if (index>=line.size() || t<line[index].start || (index+1<line.size() && t>=line[index+1].start)) {
            auto it=std::upper_bound(line.begin(),line.end(),t,
                [](double time,const Note &note){ return time<note.start; });
            index=size_t(it-line.begin()-1);
        }
        return &line[index];
    }

    double localTime(double t) const {
        if (loop<=0) return t;
        t=std::fmod(t-downbeat,loop);
        return t<0 ? t+loop : t;
    }

    struct Slide { int key=-1; float pitch=0; };
    static Slide slideAt(const std::vector<Note> &line,double t,double seconds,double period,size_t &index,
                         bool hold=false) {
        if (period>0 && !line.empty() && t<line.front().start) t+=period;
        const Note *note=noteBefore(line,t,index);
        if (!note) return {};
        const Note *next=index+1<line.size() ? &line[index+1] : nullptr;
        double offset=0;
        if (!next && period>0) { next=&line.front(); offset=period; }
        if (seconds>0 && next && next->start+offset-note->end<=phraseGap) {
            double arrival=next->start+offset;
            double span=std::min(seconds,arrival-note->start);
            if (span>0) {
                double amount=std::clamp(1-(arrival-t)/span,0.,1.);
                return {note->pitch,float(note->pitch+(next->pitch-note->pitch)*amount)};
            }
        }
        return hold || t<note->end ? Slide{note->pitch,float(note->pitch)} : Slide{};
    }

    const Frame *lookup(double t) const {
        auto i=std::upper_bound(frames.begin(),frames.end(),t,
            [](double time,const Frame &f){return time<f.start;});
        return i==frames.begin() ? nullptr : &*std::prev(i);
    }

    const CarrierChord *heldAt(double t) const {
        if (chords.empty() || !std::isfinite(t)) return nullptr;
        t=localTime(t);
        auto next = std::upper_bound(chords.begin(), chords.end(), t,
            [](double time, const CarrierChord &c) { return time < c.start; });
        if (next != chords.begin()) return &*std::prev(next);
        return loop > 0 ? &chords.back() : &chords.front();
    }

    const Frame *at(double t, bool fill, bool legato=false) const {
        if (!std::isfinite(t) || t<0 || frames.empty()) return nullptr;
        t=localTime(t);
        if (t>=frames.back().start || t<frames.front().start)
            return legato && loop>0 && frames.back().legato.any() ? &frames.back() : nullptr;
        auto f=lookup(t);
        if (!f || (legato ? f->legato : f->held).any() || !fill || cycle<=0) return f;
        return lookup(frames.front().start+std::fmod(t-frames.front().start,cycle));
    }
};
