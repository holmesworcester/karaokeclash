#include "ScoredVst.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/ivstevents.h"
#include "pluginterfaces/vst/ivstmessage.h"
#include <atomic>
#include <cmath>
#include <codecvt>
#include <locale>
#include <vector>
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivsthostapplication.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {
using namespace Steinberg;
using namespace Steinberg::Vst;

// The module owns the host, and the processor reads the preset synchronously.
template <class Interface> class Borrowed : public Interface {
public:
    uint32 PLUGIN_API addRef() override { return 1; }
    uint32 PLUGIN_API release() override { return 1; }
    tresult identify(const TUID id, const TUID own, void **out) {
        if (!std::memcmp(id, own, 16) || !std::memcmp(id, FUnknown_iid, 16)) {
            *out = static_cast<Interface *>(this);
            return kResultOk;
        }
        *out = nullptr;
        return kNoInterface;
    }
};

class Host : public Borrowed<IHostApplication> {
    tresult PLUGIN_API queryInterface(const TUID id, void **out) override {
        return identify(id, IHostApplication_iid, out);
    }
    tresult PLUGIN_API getName(String128 name) override {
        constexpr char16_t title[] = u"Karaoke Playground";
        std::copy(std::begin(title), std::end(title), name);
        return kResultOk;
    }
    tresult PLUGIN_API createInstance(TUID, TUID, void **out) override {
        *out = nullptr;
        return kNoInterface;
    }
};

std::string utf8(const TChar *text) {
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>().to_bytes(text);
}

class Setting : public Borrowed<IParamValueQueue> {
public:
    TalParameter meta{};
    std::atomic<float> target{0};
    float applied = 0;
    explicit Setting(const ParameterInfo &info) {
        meta.id = info.id;
        meta.steps = info.stepCount;
        meta.flags = info.flags;
        std::snprintf(meta.name,sizeof(meta.name),"%s",utf8(info.title).c_str());
        applied = info.defaultNormalizedValue;
        meta.initial = applied;
        if (!std::strcmp(meta.name,"inputmode")) meta.initial = 2.f/3; // external carrier
        if (!std::strcmp(meta.name,"esserintensity") || !std::strcmp(meta.name,"enveloperelease"))
            meta.initial = 0;
        target = meta.initial;
    }
    tresult PLUGIN_API queryInterface(const TUID iid, void **out) override {
        return identify(iid, IParamValueQueue_iid, out);
    }
    ParamID PLUGIN_API getParameterId() override { return meta.id; }
    int32 PLUGIN_API getPointCount() override { return 1; }
    tresult PLUGIN_API getPoint(int32 index, int32 &offset, ParamValue &result) override {
        if (index != 0) return kInvalidArgument;
        offset = 0;
        result = applied;
        return kResultOk;
    }
    tresult PLUGIN_API addPoint(int32, ParamValue, int32 &) override { return kNotImplemented; }
};

class Parameters : public Borrowed<IParameterChanges> {
    std::vector<IParamValueQueue *> changed;
    std::atomic<unsigned> revision{1};
    unsigned sent = 0;
public:
    std::vector<std::unique_ptr<Setting>> list;
    Setting *bypass = nullptr, *inputMode = nullptr;
    bool load(IEditController *controller) {
        for (int i=0;i<controller->getParameterCount();++i) {
            ParameterInfo info{};
            if (controller->getParameterInfo(i,info)!=kResultOk) return false;
            list.push_back(std::make_unique<Setting>(info));
            if (info.flags & ParameterInfo::kIsBypass) bypass=list.back().get();
            if (!std::strcmp(list.back()->meta.name,"inputmode")) inputMode=list.back().get();
        }
        changed.reserve(list.size());
        return !list.empty();
    }
    Setting *at(int i) const { return i>=0 && size_t(i)<list.size() ? list[i].get() : nullptr; }
    void set(int i,float value) {
        auto *param=at(i);
        if (!param || !std::isfinite(value) || (param->meta.flags & ParameterInfo::kIsReadOnly)) return;
        value=std::clamp(value,0.f,1.f);
        if (param->meta.steps) value=std::round(value*param->meta.steps)/param->meta.steps;
        if (param->target.exchange(value)!=value) ++revision;
    }
    // Control-thread edits are collected by the audio thread at the next block.
    IParameterChanges *collect() {
        unsigned current=revision.load();
        if (current==sent) return nullptr;
        sent=current;
        changed.clear();
        for (auto &param:list) {
            float value=param->target.load();
            if (value!=param->applied) {
                param->applied=value;
                changed.push_back(param.get());
            }
        }
        return changed.empty() ? nullptr : this;
    }
    tresult PLUGIN_API queryInterface(const TUID iid, void **out) override {
        return identify(iid, IParameterChanges_iid, out);
    }
    int32 PLUGIN_API getParameterCount() override { return changed.size(); }
    IParamValueQueue *PLUGIN_API getParameterData(int32 i) override {
        return i>=0 && size_t(i)<changed.size() ? changed[i] : nullptr;
    }
    IParamValueQueue *PLUGIN_API addParameterData(const ParamID &, int32 &) override { return nullptr; }
};

class Notes : public Borrowed<IEventList> {
    std::array<Event,512> events{};
    std::bitset<128> held;
public:
    int32 count=0;
    void update(const std::bitset<128> &keys, unsigned offset) {
        if (held==keys) return;
        auto change=held^keys;
        for (int k=0;k<128;++k) if (change[k] && size_t(count)<events.size()) {
            Event &e=events[count++];
            e={};
            e.sampleOffset=offset;
            if (keys[k]) {
                e.type=Event::kNoteOnEvent; e.noteOn.pitch=k; e.noteOn.velocity=.8f; e.noteOn.noteId=k;
            } else {
                e.type=Event::kNoteOffEvent; e.noteOff.pitch=k; e.noteOff.noteId=k;
            }
            held[k]=keys[k];
        }
    }
    void retrigger() {
        auto active=held;
        held.reset();
        update(active,0);
    }
    tresult PLUGIN_API queryInterface(const TUID iid, void **out) override {
        return identify(iid,IEventList_iid,out);
    }
    int32 PLUGIN_API getEventCount() override { return count; }
    tresult PLUGIN_API getEvent(int32 i,Event &e) override {
        if (i<0 || i>=count) return kInvalidArgument;
        e=events[i]; return kResultOk;
    }
    tresult PLUGIN_API addEvent(Event &) override { return kNotImplemented; }
};

class Module {
    void *library = nullptr;
    bool (*finish)() = nullptr;
    IPluginFactory *factory = nullptr;
    PClassInfo info{};

public:
    Host host;
    explicit Module(VstKind kind) {
        bool tal=kind==VstKind::Tal;
        const char *name=tal ? "TAL-Vocoder-2" : "Chorale";
        const char *path=std::getenv(tal ? "KARAOKE_TAL_VST3" : "KARAOKE_CHORALE_VST3");
        if (!path) path=tal ? KARAOKE_TAL_PATH : KARAOKE_CHORALE_PATH;
        library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            std::fprintf(stderr, "%s: %s\n", name, dlerror());
            return;
        }
        auto entry = reinterpret_cast<bool (*)(void *)>(dlsym(library, "ModuleEntry"));
        auto exitModule = reinterpret_cast<bool (*)()>(dlsym(library, "ModuleExit"));
        auto get = reinterpret_cast<IPluginFactory *(*)()>(dlsym(library, "GetPluginFactory"));
        if (!get || !entry || !exitModule || !entry(library)) return;
        finish = exitModule;
        factory = get();
        if (!factory) return;
        for (int32 index = 0; index < factory->countClasses(); ++index) {
            PClassInfo candidate{};
            if (factory->getClassInfo(index, &candidate) == kResultOk &&
                !std::strcmp(candidate.category, kVstAudioEffectClass) &&
                !std::strcmp(candidate.name, name)) {
                info = candidate;
                break;
            }
        }
    }
    ~Module() {
        if (factory) factory->release();
        if (finish) finish();
        if (library) dlclose(library);
    }
    IEditController *controller(IComponent *component) {
        TUID cid;
        IEditController *result=nullptr;
        if (component->getControllerClassId(cid)==kResultOk)
            factory->createInstance(cid,IEditController_iid,reinterpret_cast<void **>(&result));
        return result;
    }
    IComponent *create() {
        IComponent *result = nullptr;
        if (info.name[0])
            factory->createInstance(info.cid, IComponent_iid, reinterpret_cast<void **>(&result));
        return result;
    }
};

std::shared_ptr<Module> module(VstKind kind) {
    if (kind==VstKind::Tal) {
        static const auto tal=std::make_shared<Module>(VstKind::Tal);
        return tal;
    }
    static const auto chorale=std::make_shared<Module>(VstKind::Chorale);
    return chorale;
}
} // namespace

struct ScoredVst::Processor {
    VstKind kind;
    const std::shared_ptr<Module> plugin;
    explicit Processor(VstKind kind) : kind(kind), plugin(module(kind)) {}
    IComponent *component = nullptr;
    IAudioProcessor *audio = nullptr;
    IEditController *controller = nullptr;
    IConnectionPoint *componentPoint = nullptr, *controllerPoint = nullptr;
    bool controllerStarted = false;
    Parameters params;
    Notes notes;
    bool initialized = false, active = false, processing = false;
    std::array<float, blockSize> right{};
    ~Processor() {
        if (processing) audio->setProcessing(false);
        if (active) component->setActive(false);
        if (componentPoint && controllerPoint) {
            componentPoint->disconnect(controllerPoint);
            controllerPoint->disconnect(componentPoint);
        }
        if (controllerPoint) controllerPoint->release();
        if (componentPoint) componentPoint->release();
        if (controllerStarted) controller->terminate();
        if (controller) controller->release();
        if (audio) audio->release();
        if (initialized) component->terminate();
        if (component) component->release();
    }
    bool start(double rate) {
        component = plugin->create();
        if (!component || component->initialize(&plugin->host) != kResultOk) return false;
        initialized = true;
        if (component->queryInterface(IAudioProcessor_iid, reinterpret_cast<void **>(&audio)) != kResultOk)
            return false;
        bool tal=kind==VstKind::Tal;
        SpeakerArrangement inputs[] = {tal ? SpeakerArr::kStereo : SpeakerArr::kMono, SpeakerArr::kStereo};
        // Chorale also advertises optional stem buses. Only the main stereo mix is connected.
        std::vector<SpeakerArrangement> outputs(component->getBusCount(kAudio,kOutput),SpeakerArr::kEmpty);
        if (outputs.empty()) return false;
        outputs.front()=SpeakerArr::kStereo;
        if (audio->setBusArrangements(inputs,tal ? 2 : 1,outputs.data(),outputs.size())!=kResultOk)
            return false;
        ProcessSetup setup{kRealtime, kSample32, blockSize, rate};
        if (audio->setupProcessing(setup) != kResultOk) return false;
        if (component->activateBus(kAudio, kInput, 0, true) != kResultOk ||
            (tal && component->activateBus(kAudio, kInput, 1, true) != kResultOk) ||
            component->activateBus(kAudio, kOutput, 0, true) != kResultOk)
            return false;
        if (component->getBusCount(kEvent,kInput)>0) component->activateBus(kEvent,kInput,0,true);
        active = component->setActive(true) == kResultOk;
        processing = active && audio->setProcessing(true) == kResultOk;
        if (!processing) return false;
        controller=plugin->controller(component);
        if (!controller || controller->initialize(&plugin->host)!=kResultOk) return false;
        controllerStarted=true;
        // JUCE publishes the parameter list only once the two VST3 components connect.
        component->queryInterface(IConnectionPoint_iid,reinterpret_cast<void **>(&componentPoint));
        controller->queryInterface(IConnectionPoint_iid,reinterpret_cast<void **>(&controllerPoint));
        if (!componentPoint || !controllerPoint) return false;
        componentPoint->connect(controllerPoint);
        controllerPoint->connect(componentPoint);
        if (!params.load(controller)) return false;
        std::array<float,64> mic{}, carrier{}, wet{};
        return render(mic.data(),carrier.data(),wet.data(),mic.size());
    }

    bool render(float *mic, float *carrier, float *wet, unsigned frames) {
        float *main[] = {mic, mic}, *side[] = {carrier, carrier}, *stereo[] = {wet, right.data()};
        AudioBusBuffers in[2], out;
        in[0].numChannels = in[1].numChannels = out.numChannels = 2;
        in[0].channelBuffers32 = main;
        in[1].channelBuffers32 = side;
        out.channelBuffers32 = stereo;
        ProcessData data;
        data.numSamples = frames;
        bool tal=kind==VstKind::Tal;
        in[0].numChannels=tal ? 2 : 1;
        data.numInputs=tal ? 2 : 1;
        data.numOutputs = 1;
        data.inputs = in;
        data.outputs = &out;
        const float previousMode=params.inputMode ? params.inputMode->applied : 0;
        data.inputParameterChanges = params.collect();
        data.inputEvents = &notes;
        bool ok=audio->process(data)==kResultOk;
        notes.count=0;
        // Changing input routing clears TAL's internal synth after it reads this block's
        // MIDI. Re-assert held keys on the next block, including a single sustained note.
        if (params.inputMode && params.inputMode->applied!=previousMode) notes.retrigger();
        // A host bypass returns the dry mic. This strip is a wet-only return.
        if (params.bypass && params.bypass->applied>=.5f) return false;
        return ok;
    }
};

ScoredVst::ScoredVst(VstKind kind) : kind(kind) {}
ScoredVst::~ScoredVst() = default;
bool ScoredVst::ready() const {
    return processor != nullptr;
}
void ScoredVst::prepare(double rate) {
    auto next = std::make_unique<Processor>(kind);
    processor = next->start(rate) ? std::move(next) : nullptr;
}
bool ScoredVst::render(float *mic, float *carrier, float *wet, unsigned frames, float *side) {
    if (!processor || frames>blockSize || !processor->render(mic,carrier,wet,frames)) return false;
    if (side) for (unsigned i=0;i<frames;++i) {
        side[i]=.5f*(processor->right[i]-wet[i]);
        wet[i]+=side[i];
    }
    return true;
}

int ScoredVst::parameterCount() const { return processor ? processor->params.list.size() : 0; }
const TalParameter *ScoredVst::parameter(int i) const {
    auto *p=processor ? processor->params.at(i) : nullptr;
    return p ? &p->meta : nullptr;
}
const char *ScoredVst::display(int i,float value) const {
    auto *p=parameter(i);
    String128 label{};
    if (!p || processor->controller->getParamStringByValue(p->id,value,label)!=kResultOk) return "";
    thread_local std::string result;
    result=utf8(label);
    return result.c_str();
}
void ScoredVst::setParameter(int i,float value) { if (processor) processor->params.set(i,value); }
void ScoredVst::notes(const std::bitset<128> &keys,unsigned offset) {
    if (processor) processor->notes.update(keys,offset);
}

int ScoredVst::parameterIndex(const char *name) const {
    for (int i=0;i<parameterCount();++i)
        if (!std::strcmp(parameter(i)->name,name)) return i;
    return -1;
}
unsigned ScoredVst::latency() const {
    return processor ? processor->audio->getLatencySamples() : 0;
}
