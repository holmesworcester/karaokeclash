// The PipeWire/JACK graph supplies planar 48 kHz audio. This callback never
// enters Python, allocates, decodes a file, takes a mutex or waits for the GPU.
#include "FourMixer.h"
#include "TempoWarp.h"
#include "character.h"
#include "TestSinger.h"
#include <fstream>
#include <jack/jack.h>
#include <vector>
using namespace karaoke_four;
struct Recording {
    std::vector<int16_t> bed, guide;
};
// A pane can follow the shared beat while reading a different recording.
struct SourceClock {
    SingerClock clip{};
    const Score *backingScore = nullptr;
    double origin = 0, first = 0, bars = 0, length = 0;
    double voiceStart = 0, voiceEnd = 0, albumBarOffset = 0;
    bool same = false;
    void set(const double *p, const double *grid, int n) {
        clip = {p[0], p[1], p[2], p[3], 0, 1, {}};
        if (n) clip.bars.assign(grid, grid + n);
        origin = p[4];
        voiceStart = p[5];
        voiceEnd = p[6];
        same = p[7] != 0;
        albumBarOffset = p[8];
        first = clip.bar(clip.start);
        bars = clip.bar(clip.end) - first;
        if (std::round(bars) > 0 && std::abs(bars - std::round(bars)) < .00005)
            bars = std::round(bars);
        length = clip.end - clip.start;
    }
    double global(double t) const {
        double laps = std::floor(t / length);
        return origin + laps * bars + clip.bar(clip.start + (t - laps * length)) - first;
    }
    double local(double beat) const {
        double phase = beat - origin;
        return clip.seconds(first + (phase - bars * std::floor(phase / bars))) - clip.start;
    }
};
struct PanePlan {
    SourceClock current, next;
    double boundary;
};
struct Track {
    std::shared_ptr<const Recording> audio;
    int64_t start = 0, end = 0;
    bool loop = false;
    double bpm = 120;
    int64_t voiceStart = 0, voiceEnd = INT64_MAX;
    SourceClock clock;
    std::atomic<int> lane{-1};
    std::atomic<bool> backingEnabled{true};
    bool carrierBacking = false;
    int id = 0;
    // Internal time is measured in 120 BPM frames so a queued bar boundary
    // survives live tempo edits and switches between recordings.
    double ticks(double sourceFrame) const {
        if (!clock.length) return sourceFrame * bpm / 120.;
        return (clock.global(sourceFrame/48000.) - clock.origin) * 96000.;
    }
    double launchTicks(int64_t frame) const {
        double exact = ticks(frame);
        if (!clock.length) return exact;
        // The control API addresses 48 kHz recording samples. Recover a
        // musical subdivision when that sample is its nearest representation.
        double origin = clock.origin*96000.;
        double subdivision = std::round((origin+exact)/6000.)*6000. - origin;
        return std::abs(sourceFrame(subdivision)-frame) <= .501 ? subdivision : exact;
    }
    double sourceFrame(double ticks) const {
        if (!clock.length) return ticks * 120. / bpm;
        double bar = ticks / 96000.;
        double laps = std::floor(bar / clock.bars);
        return 48000. * (laps*clock.length +
            clock.clip.seconds(clock.first + bar - laps*clock.bars) - clock.clip.start);
    }
    bool vocalAt(int64_t pos) const { return pos >= voiceStart && pos < voiceEnd; }
};
struct Timeline {
    std::shared_ptr<Track> current, next;
    double origin = 0, boundary = INFINITY, offset = 0;
    bool launched(double frame) const { return next && frame + 1e-7 >= boundary; }
};
struct Engine {
    FourMixer mix;
    CharacterLane characters[4];
    jack_client_t *client = nullptr;
    jack_port_t *in[4]{}, *out[2]{};
    std::atomic<unsigned> xruns{0}, period{0};
    std::atomic<bool> offline{false}, playing{false}, guide{false};
    // Only the control thread owns or retires plans. The callback reads an
    // immutable plan and splits a render block at its exact launch frame.
    std::atomic<Timeline *> timeline{nullptr};
    std::shared_ptr<Track> staged;
    int serial = 0;
    std::atomic<unsigned> readers{0}, revision{0};
    std::atomic<double> seek{0}, elapsed{0}, roomBpm{0};
    TempoWarp backingWarp, guideWarp, singerWarp[lanes];
    std::vector<std::unique_ptr<Timeline>> plans;
    void publish(Timeline plan) {
        plans.push_back(std::make_unique<Timeline>(std::move(plan)));
        timeline.store(plans.back().get());
    }
    double frame() const { return lastRevision.load() == revision.load() ? elapsed.load() : seek.load(); }
    Timeline currentTimeline() const {
        auto *p = timeline.load();
        if (!p)
            return {};
        if (p->launched(frame()))
            return {p->next, nullptr, p->boundary - p->offset};
        return {p->current, nullptr, p->origin};
    }
    std::atomic<unsigned> lastRevision{0};
    std::atomic<unsigned> testing{0}, mutedBackings{0};
    std::atomic<TestSinger *> singers[4]{};
    std::atomic<PanePlan *> pane[4]{};
    std::array<std::unique_ptr<PanePlan>, lanes> paneOwners;
    std::vector<std::unique_ptr<PanePlan>> retiredPanes;
    std::array<std::unique_ptr<TestSinger>, lanes> current;
    std::vector<std::unique_ptr<TestSinger>> retired;
    double clock = 0;
    const Score *carrier(const char *id) const {
        if (!id || !*id) return nullptr;
        auto it = mix.scores.find(id);
        if (it == mix.scores.end()) return nullptr;
        if (it->second.playable()) return &it->second;
        for (const auto &variant : it->second.versions)
            if (variant.load()) return &it->second;
        return nullptr;
    }
    Engine() { mix.prepare(48000); }
    ~Engine() {
        if (client)
            jack_client_close(client);
    }
    void render(const float *const *input, float *left, float *right, unsigned frames) {
        readers.fetch_add(1);
        auto *plan = timeline.load();
        unsigned rev = revision.load();
        if (rev != lastRevision.load())
            clock = seek.load();
        for (unsigned at = 0; at < frames;) {
            bool launched = plan && plan->launched(clock);
            auto *t = plan ? (launched ? plan->next : plan->current).get() : nullptr;
            double ticks = plan ? std::max(0., clock - (launched ? plan->boundary-plan->offset : plan->origin)) : 0;
            bool run = t && playing.load();
            const double globalBpm = roomBpm.load();
            double tempo = globalBpm > 0 ? globalBpm : t ? t->bpm : 120.;
            double step = tempo / 120.;
            auto size = std::min(quantum, frames - at);
            auto until = [&](double distance) {
                return unsigned(std::clamp(std::ceil(distance/step - 1e-7), 1., double(size)));
            };
            if (run && !launched && plan->next)
                size = until(plan->boundary - clock);
            double local = 0, pos = 0, speed = 1;
            double bar = ticks/96000. + (t ? t->clock.origin : 0);
            if (t) {
                double length = t->end - t->start;
                local = t->sourceFrame(ticks);
                if (std::abs(local-std::round(local)) < 1e-6) local = std::round(local);
                pos = t->start + (t->loop ? std::fmod(local, length) : std::min(local, length));
                if (run && pos >= t->end) { playing = false; run = false; }
                if (run) {
                    // Split at a loop join and each measured downbeat. Inside
                    // a measured bar the recording-to-room map is linear.
                    double next = t->ticks((std::floor(local/length) + 1)*length);
                    if (t->clock.length && !t->clock.clip.bars.empty()) {
                        double phase = t->clock.clip.bar(pos/48000.);
                        double edge = t->clock.clip.seconds(std::floor(phase+1e-8)+1)*48000.;
                        next = std::min(next, t->ticks(local + edge - pos));
                    }
                    size = until(next - ticks);
                    speed = (t->sourceFrame(ticks + step*size) - local)/size;
                    mix.bpm = tempo;
                    mix.time = pos / 48000.;
                }
            }
            if (run && t->clock.length) {
                for (auto &lane : pane) {
                    auto *schedule = lane.load();
                    if (!schedule) continue;
                    // Effects following another pane must see its launch and
                    // loop wrap on the same sample as that pane's test singer.
                    if (bar < schedule->boundary && std::isfinite(schedule->boundary))
                        size = until((schedule->boundary-bar)*96000.);
                    auto &source = bar >= schedule->boundary ? schedule->next : schedule->current;
                    double laps = std::floor((bar-source.origin)/source.bars + 1e-10);
                    size = until((source.origin + (laps+1)*source.bars-bar)*96000.);
                }
            }
            mix.roomBar = run ? bar : NAN;
            unsigned test = testing.load();
            for (unsigned j = 0; j < lanes; ++j) {
                auto *schedule = pane[j].load();
                bool independent = schedule && t && t->clock.length > 0;
                auto phaseAt = [&](unsigned k, const SourceClock *&source) {
                    double beat = bar + k * step / 96000.;
                    source = beat >= schedule->boundary ? &schedule->next : &schedule->current;
                    return source->local(beat);
                };
                const SourceClock *source = nullptr;
                double phase = independent ? phaseAt(0, source) : 0;
                mix.sourceTime[j] = independent ? source->clip.start + phase : -1;
                mix.scoreBar[j] = run && t->clock.length ?
                    (independent ? source->clip.bar(source->clip.start+phase) : t->clock.clip.bar(pos/48000.)) : NAN;
                mix.albumBar[j] = mix.scoreBar[j] + (independent ? source->albumBarOffset : t ? t->clock.albumBarOffset : 0);
                const SourceClock *clip = independent ? source : t ? &t->clock : nullptr;
                mix.clipScore[j] = clip ? clip->backingScore : nullptr;
                mix.clipTime[j] = independent ? source->clip.start + phase : pos/48000.;
                mix.clipSpeed[j] = independent ? source->clip.secondsPerBar(mix.clipTime[j])*tempo/240. : speed;
                if (test & (1u << j)) {
                    auto *singer = singers[j].load();
                    for (unsigned k = 0; k < size; ++k) {
                        float sample = 0;
                        if (run && singer) {
                            const SourceClock *part = nullptr;
                            double phase = independent ? phaseAt(k, part) : (local+k*speed)/48000.;
                            bool audible = independent ? (part->voiceStart <= part->clip.start+phase &&
                                part->clip.start+phase < part->voiceEnd) : t->vocalAt(std::llround(pos+k*speed));
                            if (audible) {
                                const auto &clip = independent ? part->clip : singer->backing;
                                bool same = independent ? part->same : singer->same;
                                double secs = singer->position(phase, clip, same);
                                double rate = singer->lyrics.secondsPerBar(secs) * tempo / 240.;
                                singerWarp[j].read<1>(singer->pcm.data(), 0, singer->pcm.size(),
                                                      secs*48000., rate, &sample);
                                sample /= 32768.f;
                            } else singerWarp[j].audio = nullptr;
                        }
                        mix.captured[j][k] = sample;
                    }
                } else if (input && input[j])
                    std::copy_n(input[j] + at, size, mix.captured[j]);
                else
                    std::fill_n(mix.captured[j], size, 0);
            }
            const int owner = run ? t->lane.load() : -1;
            const bool audibleBed = run && t->backingEnabled.load() &&
                (owner < 0 || !(mutedBackings.load() & (1u << owner)));
            mix.renderCaptured(
                left + at, right + at, size,
                [this](unsigned i, float *p, unsigned n) { characters[i].process(p, n); }, test,
                audibleBed && t->carrierBacking ? owner : -1);
            if (audibleBed && !t->carrierBacking) {
                bool vocal = guide.load();
                for (unsigned i = 0; i < size; ++i) {
                    double cursor = pos + i*speed;
                    float stereo[2];
                    backingWarp.read<2>(t->audio->bed.data(), t->start, t->end, cursor, speed, stereo);
                    if (vocal && t->vocalAt(std::llround(cursor)) && !t->audio->guide.empty()) {
                        float guide[2];
                        guideWarp.read<2>(t->audio->guide.data(), std::max(t->start, t->voiceStart),
                                          std::min<int64_t>(t->voiceEnd, t->audio->guide.size()/2),
                                          cursor, speed, guide);
                        for (int c=0; c<2; ++c) stereo[c] += .8f*guide[c];
                    }
                    left[at+i] += stereo[0]/32768.f;
                    right[at+i] += stereo[1]/32768.f;
                }
            }
            if (run) clock += size * step;
            mix.limiter.finishMix(left + at, right + at, size, mix.master);
            at += size;
        }
        elapsed.store(clock);
        lastRevision.store(rev);
        readers.fetch_sub(1);
    }
};
static int callback(jack_nframes_t n, void *arg) {
    auto &e = *static_cast<Engine *>(arg);
    const float *inputs[4];
    for (unsigned i = 0; i < 4; i++)
        inputs[i] = static_cast<float *>(jack_port_get_buffer(e.in[i], n));
    e.render(inputs, static_cast<float *>(jack_port_get_buffer(e.out[0], n)),
             static_cast<float *>(jack_port_get_buffer(e.out[1], n)), n);
    return 0;
}
extern "C" {
Engine *desk_new() { return new Engine(); }
void desk_free(Engine *e) { delete e; }
const char *desk_open(Engine *e) {
    if (e->client)
        return e->offline ? "Audio server disconnected. Reopen the desk." : nullptr;
    jack_status_t status{};
    auto *c = jack_client_open("Karaoke Playground 4", JackNoStartServer, &status);
    if (!c)
        return "Cannot open the PipeWire JACK graph.";
    if (jack_get_sample_rate(c) != 48000) {
        jack_client_close(c);
        return "Set the audio graph to 48000 Hz.";
    }
    for (int i = 0; i < 4; i++)
        e->in[i] = jack_port_register(c, ("mic_" + std::to_string(i + 1)).c_str(),
                                      JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    for (int i = 0; i < 2; i++)
        e->out[i] =
            jack_port_register(c, i ? "right" : "left", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    if (std::count(e->in, e->in + 4, nullptr) || !e->out[0] || !e->out[1]) {
        jack_client_close(c);
        return "Could not register the six audio ports.";
    }
    jack_set_process_callback(c, callback, e);
    jack_set_xrun_callback(
        c,
        [](void *arg) {
            ++static_cast<Engine *>(arg)->xruns;
            return 0;
        },
        e);
    jack_set_buffer_size_callback(
        c,
        [](jack_nframes_t n, void *arg) {
            static_cast<Engine *>(arg)->period = n;
            return 0;
        },
        e);
    jack_on_shutdown(c, [](void *arg) { static_cast<Engine *>(arg)->offline = true; }, e);
    e->period = jack_get_buffer_size(c);
    if (jack_activate(c)) {
        jack_client_close(c);
        return "Could not activate audio.";
    }
    e->client = c;
    return nullptr;
}
const char **desk_ports(Engine *e, int capture) {
    return jack_get_ports(e->client, nullptr, JACK_DEFAULT_AUDIO_TYPE,
                          capture ? JackPortIsOutput : JackPortIsInput);
}
void desk_ports_free(const char **p) {
    if (p)
        jack_free(p);
}
int desk_route(Engine *e, int capture, int i, const char *name) {
    auto port = capture ? e->in[i] : e->out[i];
    auto self = jack_port_name(port);
    if (*name && jack_port_connected_to(port, name))
        return 0;
    jack_port_disconnect(e->client, port);
    if (!*name)
        return 0;
    return capture ? jack_connect(e->client, name, self) : jack_connect(e->client, self, name);
}
int desk_status(Engine *e, int which) {
    return which == 0 ? e->period.load() : which == 1 ? e->xruns.load() : e->offline.load();
}
void desk_note(Engine *e, const char *id, double a, double b, int key) {
    e->mix.scores[id].notes.push_back({a, b, key});
}
void desk_bass_note(Engine *e, const char *id, double a, double b, int key) {
    e->mix.scores[id].bass.push_back({a,b,key});
}
void desk_lead_note(Engine *e, const char *id, double a, double b, int key) {
    e->mix.scores[id].lead.push_back({a,b,key});
}
void desk_score_unison(Engine *e, const char *id, float amount) {
    e->mix.scores[id].unison=std::clamp(amount,0.f,1.f);
}
void desk_score_gains(Engine *e, const char *id, const float *gains) {
    auto &levels = e->mix.scores[id].gains;
    std::copy_n(gains, levels.size(), levels.begin());
}
void desk_score(Engine *e, const char *id, double loop, const int16_t *pcm, int n) {
    auto &s = e->mix.scores[id];
    s.loop = loop;
    s.compile(pcm, n / 2);
    s.findSpeechRegister();
}
void desk_backing_score(Engine *e, const char *id, const char *written, const char *family,
                       const int *notes, int count, double duration, double bpm, double origin,
                       const double *bars, int n, double offsetBars, int keepWritten, int repeat, const float *pan) {
    auto [entry, inserted] = e->mix.scores.try_emplace(id);
    if (!inserted) return;
    Score &s = entry->second;
    s.backingFamily = family;
    if (pan) {
        std::copy_n(pan, s.pan.size(), s.pan.begin());
        s.stereo = std::any_of(s.pan.begin(), s.pan.end(), [](float p) { return p != 0; });
    }
    s.recordingGrid.bpm = bpm;
    s.recordingGrid.origin = origin;
    if (n) s.recordingGrid.bars.assign(bars, bars+n);
    for (int i=0; i<count; ++i) {
        int start = notes[i*3], end = notes[i*3+1], pitch = notes[i*3+2];
        if (start >= 0 && end > start && pitch >= 0 && pitch < 128)
            s.notes.push_back({start*.001, end*.001, pitch});
    }
    std::vector<Note> riff;
    if (const Score *source = e->carrier(written)) {
        auto fit = [&](const std::vector<Note> &part, bool cycle) {
            return s.project(part, *source, duration, offsetBars, cycle);
        };
        s.bass = fit(source->bass, repeat);
        if (keepWritten) {
            s.unison = source->unison;
            s.gains = source->gains;
            s.lead = fit(source->lead, false);
            auto accompaniment = fit(source->notes, repeat);
            s.notes.insert(s.notes.end(), accompaniment.begin(), accompaniment.end());
            riff = fit(source->upper, repeat);
        }
    }
    s.compile();
    if (!riff.empty()) s.upper.swap(riff);
    s.findSpeechRegister();
}
void desk_score_variant(Engine *e, const char *primary, int slot, const char *alternate) {
    auto it = e->mix.scores.find(primary);
    if (it != e->mix.scores.end() && slot > 0 && slot <= int(it->second.versions.size()))
        it->second.versions[slot-1] = e->carrier(alternate);
}
void desk_midi_version(Engine *e, int lane, int slot) {
    e->mix.lane[lane].midiVersion = slot;
}
void desk_sample(Engine *e, int id, const float *pcm, int n) {
    auto &s = e->mix.samples[id];
    s.rate = 48000;
    s.data.assign(pcm, pcm + n);
}
int desk_tal_count(Engine *e,int lane) { return e->mix.lane[lane].effect.talPlugin().parameterCount(); }
const TalParameter *desk_tal_info(Engine *e,int lane,int index) {
    return e->mix.lane[lane].effect.talPlugin().parameter(index);
}
const char *desk_tal_display(Engine *e,int lane,int index,float value) {
    return e->mix.lane[lane].effect.talPlugin().display(index,value);
}
void desk_tal_set(Engine *e,int lane,int index,float value) {
    e->mix.lane[lane].effect.talPlugin().setParameter(index,value);
}
int desk_chorale_ready(Engine *e, int i) { return e->mix.lane[i].effect.choraleReady(); }
int desk_tal_ready(Engine *e, int i) { return e->mix.lane[i].effect.talReady(); }
int desk_has_carrier(Engine *e, const char *anchor) { return e->carrier(anchor) != nullptr; }
void desk_config(Engine *e, int i, int voice, const char *anchor, int mask, int flags, const float *v) {
    auto &l = e->mix.lane[i];
    l.score = e->carrier(anchor);
    if (voice==chorale) l.effect.prepareChorale(); // Construct the VST on the control thread.
    l.voice = voice;
    l.tuned = mask != 0;
    l.mask = mask;
    l.pulse = flags & 1;
    l.delayOn = flags & 2;
    l.dynamics.compressor = flags & 4;
    for (int j = 0; j < 8; j++)
        l.knobs[j] = v[j];
    auto &d = l.dynamics;
    d.thresholdDB = v[8];
    d.ratio = v[9];
    d.attackMs = v[10];
    d.releaseMs = v[11];
    d.maxLiftDB = v[12];
    d.floorDB = v[13];
}
void desk_vocoder(Engine *e, int i, float mix, float clarity, float attack, float release) {
    auto &effect=e->mix.lane[i].effect;
    effect.targetMix=std::clamp(mix*.01f,0.f,1.f);
    effect.targetClarity=std::clamp(clarity*.01f,0.f,1.f);
    effect.attackMs=std::clamp(attack,1.f,80.f);
    effect.releaseMs=std::clamp(release,5.f,300.f);
}
void desk_vocoder_shape(Engine *e, int i, float formant, float harmonics, int registerMode) {
    auto &effect=e->mix.lane[i].effect;
    effect.synthRegister=std::clamp(registerMode,-2,1);
    effect.formant=std::clamp(formant,-12.f,12.f);
    effect.targetHarmonics=std::clamp(harmonics*.01f,0.f,1.f);
}
void desk_stereo_width(Engine *e, int lane, float percent) {
    e->mix.lane[lane].effect.stereoWidth = std::clamp(percent * .01f, 0.f, 1.f);
}
void desk_glide(Engine *e,int lane,float ms) {
    e->mix.lane[lane].effect.glideMs=std::clamp(ms,0.f,240.f);
}
void desk_lead_level(Engine *e,int lane,float percent) {
    e->mix.lane[lane].effect.targetLead=std::clamp(percent*.01f,0.f,1.f);
}
void desk_reverb(Engine *e, int i, float mix) { e->mix.lane[i].roomMix = std::clamp(mix, 0.f, .65f); }
void desk_live(Engine *e, int i, int on) { e->mix.lane[i].live = on; }
int desk_hear(Engine *e, int i, int id) {
    auto &l = e->mix.lane[i];
    if (id >= 0 && e->mix.samples[id].data.empty())
        return 0;
    l.sampleID = id;
    ++l.token;
    return 1;
}
int desk_hearing(Engine *e, int i) { return e->mix.lane[i].sampleID >= 0; }
float desk_peak(Engine *e, int i) { return e->mix.lane[i].peak.load(); }
void desk_master(Engine *e, float ceiling, int mute) {
    e->mix.master.ceilingDB = ceiling;
    e->mix.master.muted = mute;
}
void desk_buffer(Engine *e, int k) {
    for (auto &lane : e->characters)
        lane.hops = (k == 2 ? 2 : 4);
}
// Bake the upstream pitch stage into pre-rendered character vocals on a worker thread.
void desk_tune_recording(int16_t *pcm, int frames, unsigned mask) {
    std::unique_ptr<Singer, decltype(&singer_free)> singer(singer_new(48000), singer_free);
    Block in{}, out{};
    for (int pos = 0; pos < frames;) {
        int n = std::min<int>(quantum, frames - pos);
        for (int i = 0; i < n; ++i) in[i] = pcm[pos+i] / 32768.f;
        singer_process(singer.get(), mask, in.data(), out.data(), n);
        for (int i = 0; i < n; ++i) pcm[pos+i] = std::clamp(std::lround(out[i]*32768.f), -32768L, 32767L);
        pos += n;
    }
}
void desk_test_lanes(Engine *e, unsigned mask) { e->testing = mask; }
void desk_backing_enabled(Engine *e, int enabled) {
    auto plan = e->timeline.load();
    if (!plan) return;
    if (plan->current) plan->current->backingEnabled = enabled;
    if (plan->next) plan->next->backingEnabled = enabled;
}
void desk_stage_backing_enabled(Engine *e, int enabled) {
    if (e->staged) e->staged->backingEnabled = enabled;
}
void desk_stage_carrier_backing(Engine *e, int enabled) {
    if (e->staged) e->staged->carrierBacking = enabled;
}
void desk_mute_lanes(Engine *e, unsigned mask) { e->mutedBackings = mask; }
// An aligned handoff of the same loop changes ownership without restarting its clock.
void desk_active_lane(Engine *e, int lane) {
    auto track = e->currentTimeline().current;
    if (track) track->lane = lane;
}
void desk_stage_lane(Engine *e, int lane) { if (e->staged) e->staged->lane = lane; }
void desk_test_source(Engine *e, int lane, const int16_t *pcm, int frames, int same,
                      const double *spec, const double *lyricBars, int nl,
                      const double *backingBars, int nb) {
    std::unique_ptr<TestSinger> source;
    if (frames > 0) {
        auto clock = [](const double *p, const double *bars, int n) {
            SingerClock c{p[0],p[1],p[2],p[3],p[4],int(p[5]),{}};
            if (n) c.bars.assign(bars,bars+n);
            return c;
        };
        source = std::make_unique<TestSinger>();
        source->lyrics = clock(spec, lyricBars, nl);
        source->backing = clock(spec+6, backingBars, nb);
        source->same = same;
        source->pcm.assign(pcm, pcm+frames);
    }
    e->singers[lane].store(source.get());
    if (e->current[lane]) e->retired.push_back(std::move(e->current[lane]));
    e->current[lane] = std::move(source);
}
void desk_stage_clock(Engine *e, const double *p, const double *grid, int n, const char *score) {
    if (e->staged) {
        e->staged->clock.set(p, grid, n);
        e->staged->clock.backingScore = e->carrier(score);
    }
}
void desk_pane_clock(Engine *e, int lane, double boundary,
                     const double *p, const double *grid, int n, const char *score,
                     const double *q, const double *nextGrid, int m, const char *nextScore) {
    std::unique_ptr<PanePlan> plan;
    if (p) {
        plan = std::make_unique<PanePlan>();
        plan->boundary = boundary;
        plan->current.set(p, grid, n);
        plan->current.backingScore = e->carrier(score);
        if (q) {
            plan->next.set(q, nextGrid, m);
            plan->next.backingScore = e->carrier(nextScore);
        }
    }
    e->pane[lane].store(plan.get());
    if (e->paneOwners[lane]) e->retiredPanes.push_back(std::move(e->paneOwners[lane]));
    e->paneOwners[lane] = std::move(plan);
}
void desk_collect(Engine *e) {
    if (!e->readers.load()) {
        e->retired.clear();
        e->retiredPanes.clear();
    }
    if (e->plans.size() > 1 && !e->readers.load())
        e->plans.erase(e->plans.begin(), e->plans.end() - 1);
}
void desk_character(Engine *e, int i, unsigned epoch) { e->characters[i].epoch = epoch; }
int desk_take(Engine *e, int i, CharacterHop *h) { return e->characters[i].requests.pop(*h); }
int desk_put(Engine *e, int i, const CharacterHop *h) { return e->characters[i].results.push(*h); }
int desk_wet(Engine *e, int i) { return e->characters[i].converted.load(); }
int desk_misses(Engine *e, int i) { return e->characters[i].misses.load(); }
void desk_render(Engine *e, const float **input, float *l, float *r, unsigned n) {
    e->render(input, l, r, n);
}
void desk_tempo(Engine *e, double bpm) {
    if (std::isfinite(bpm) && bpm > 0) e->roomBpm = bpm;
}
void desk_score_grid(Engine *e, const char *id, double bpm, double zero, const double *bars, int n) {
    auto &grid = e->mix.scores.at(id).recordingGrid;
    grid.bpm = bpm;
    grid.origin = zero;
    if (n) grid.bars.assign(bars, bars+n);
}
void desk_carrier_clock(Engine *e, int lane, int leader, double bpm) {
    e->mix.lane[lane].scoreBpm = bpm;
    e->mix.lane[lane].follows = leader;
}
void desk_transport(Engine *e, int op, int64_t arg) {
    switch (op) {
    case 0:
        e->playing = arg;
        break;
    case 1:
        if (auto plan = e->currentTimeline(); plan.current) e->seek = plan.origin + plan.current->ticks(arg);
        ++e->revision;
        break;
    case 2:
        e->guide = arg;
        break;
    }
}
int64_t desk_clock(Engine *e) {
    auto frame = e->frame();
    auto *p = e->timeline.load();
    if (!p) return 0;
    bool launched = p->launched(frame);
    auto *track = (launched ? p->next : p->current).get();
    return track ? std::llround(track->sourceFrame(frame - (launched ? p->boundary-p->offset : p->origin))) : 0;
}
int desk_playing(Engine *e) { return e->playing; }
int desk_active(Engine *e) {
    auto track = e->currentTimeline().current;
    return track ? track->id : 0;
}
void desk_cancel(Engine *e) { e->publish(e->currentTimeline()); }
int desk_launch_at(Engine *e, int64_t boundary, int64_t offset) {
    if (!e->staged)
        return 0;
    auto plan = e->currentTimeline();
    if (boundary >= 0 && plan.current) {
        if (e->frame() >= plan.origin + plan.current->launchTicks(boundary))
            return 0;
        plan.offset = e->staged->ticks(offset);
        plan.next = e->staged;
        plan.boundary = plan.origin + plan.current->launchTicks(boundary);
    } else {
        e->playing = false;
        e->seek = e->staged->ticks(offset);
        ++e->revision;
        plan = {e->staged, nullptr, 0};
    }
    int id = e->staged->id;
    e->publish(std::move(plan));
    e->staged.reset();
    return id;
}
int desk_launch(Engine *e, int64_t boundary) { return desk_launch_at(e, boundary, 0); }
int64_t desk_prepare(Engine *e, const int16_t *bed, int frames, const int16_t *guide, int guideFrames,
                     int start, int end, int loop, double bpm, int voiceStart, int voiceEnd) {
    auto t = std::make_shared<Track>();
    if (frames <= 0 || start < 0 || start >= std::min(frames, end))
        return 0;
    if (bed) {
        auto audio = std::make_shared<Recording>();
        audio->bed.assign(bed, bed + 2ull * frames);
        if (guideFrames)
            audio->guide.assign(guide, guide + 2ull * guideFrames);
        t->audio = std::move(audio);
    } else {
        auto current = e->currentTimeline().current;
        if (!current)
            return 0;
        t->audio = current->audio;
        frames = t->audio->bed.size() / 2;
        if (start >= std::min(frames, end))
            return 0;
    }
    t->start = start;
    t->end = std::min(frames, end);
    t->loop = loop;
    t->bpm = bpm;
    t->voiceStart = voiceStart;
    t->voiceEnd = voiceEnd;
    t->id = ++e->serial;
    e->staged = t;
    return t->end - start;
}
int64_t desk_track(Engine *e, const int16_t *bed, int frames, const int16_t *guide, int guideFrames,
                   int start, int end, int loop, double bpm) {
    auto len = desk_prepare(e, bed, frames, guide, guideFrames, start, end, loop, bpm, 0, INT32_MAX);
    if (len)
        desk_launch(e, -1);
    return len;
}
}
