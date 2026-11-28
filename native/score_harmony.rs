// One Tunelet tracker and vocal-tract model per mic, shared by four scored readers.
use tunelet_harmony::{
    ControlEvent, MidiEvent, MultiVoiceMode, PrepareConfig, Processor, ProcessorConfig, SampleRate,
    TimedEvent,
};

// A fricative is one acoustic event even when several scored singers share a vowel.
// Align it with the formant stage, and admit only aperiodic high-band energy.
pub struct Diction {
    delay: Vec<f32>,
    index: usize,
    highpass: [f32; 5],
    filter: [f32; 2],
    prev: [f32; 2],
    moments: [f32; 4],
    amount: f32,
    smooth: f32,
    dc: [f32; 2],
    dc_step: f32,
    envelope: f32,
}

impl Diction {
    fn new(hz: f32, latency: usize) -> Self {
        let w = 2. * std::f32::consts::PI * 3200. / hz;
        let c = w.cos();
        let a = w.sin() / std::f32::consts::SQRT_2;
        let b = (1. + c) / (2. * (1. + a));
        Self {
            delay: vec![0.; latency],
            index: 0,
            highpass: [b, -2. * b, b, -2. * c / (1. + a), (1. - a) / (1. + a)],
            filter: [0.; 2],
            prev: [0.; 2],
            moments: [0.; 4],
            amount: 0.,
            smooth: 1. - (-1. / (0.003 * hz)).exp(),
            dc: [0.; 2],
            dc_step: 1. - (-2. * std::f32::consts::PI * 25. / hz).exp(),
            envelope: 0.,
        }
    }

    fn reset(&mut self) {
        self.delay.fill(0.);
        self.filter.fill(0.);
        self.prev.fill(0.);
        self.moments.fill(0.);
        self.dc.fill(0.);
        self.amount = 0.;
        self.envelope = 0.;
    }

    fn process(&mut self, input: f32, mut wet: [f32; 2]) -> [f32; 2] {
        let x = std::mem::replace(&mut self.delay[self.index], input);
        self.index = (self.index + 1) % self.delay.len();
        let [b0, b1, b2, a1, a2] = self.highpass;
        let high = b0 * x + self.filter[0];
        self.filter = [b1 * x - a1 * high + self.filter[1], b2 * x - a2 * high];
        for (avg, sample) in self.moments.iter_mut().zip([
            x * x,
            high * high,
            high * self.prev[0],
            high * self.prev[1],
        ]) {
            *avg += self.smooth * (sample - *avg);
        }
        self.prev = [high, self.prev[0]];
        let [total, power, lag1, lag2] = self.moments;
        let r1 = lag1 / (power + 1e-10);
        let r2 = lag2 / (power + 1e-10);
        // A pure tone is predictable from two samples, even high above the singer's range.
        let residual = 1. - r1 * r1 - (r2 - r1 * r1).powi(2) / (1. - r1 * r1).max(0.01);
        let hiss = ((power / (total + 1e-10) - 0.3) * 3.).clamp(0., 1.)
            * ((residual - 0.02) * 14.).clamp(0., 1.);
        self.amount += self.smooth * (hiss - self.amount);
        self.envelope = x.abs().max(self.envelope * (1. - self.smooth));
        for (ch, y) in wet.iter_mut().enumerate() {
            // Remove the formant model's DC offset and fade its silent tail.
            self.dc[ch] += self.dc_step * (*y - self.dc[ch]);
            *y = ((*y - self.dc[ch]) * (1. - self.amount) + high * self.amount)
                * (self.envelope * 100000.).min(1.);
        }
        wet
    }
}

pub struct Harmony {
    diction: Diction,
    dsp: Processor,
    notes: [i32; 4],
}

#[no_mangle]
pub extern "C" fn score_harmony_new(hz: u32) -> *mut Harmony {
    let rate = match hz {
        48000 => SampleRate::Hz48000,
        44100 => SampleRate::Hz44100,
        _ => return std::ptr::null_mut(),
    };
    let mut cfg = ProcessorConfig::default();
    let voice = &mut cfg.resynthesis;
    voice.mode = MultiVoiceMode::Harmony;
    voice.stereo_output = true;
    voice.formant.enabled = true;
    voice.correction.enabled = false; // MIDI harmony already sets each reader's absolute pitch.
    voice.dry_mix = 0.;
    let dsp = Processor::new(PrepareConfig::new(rate, 512).unwrap(), cfg).unwrap();
    Box::into_raw(Box::new(Harmony {
        diction: Diction::new(hz as f32, dsp.latency_frames()),
        dsp,
        notes: [-1; 4],
    }))
}

#[no_mangle]
pub unsafe extern "C" fn score_harmony_free(ptr: *mut Harmony) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn score_harmony_reset(h: &mut Harmony) {
    h.dsp.reset();
    h.diction.reset();
    h.notes.fill(-1);
}

#[no_mangle]
pub unsafe extern "C" fn score_harmony_render(
    h: &mut Harmony,
    notes: &[i32; 4],
    levels: &[[f32; 2]; 4],
    glide_ms: f32,
    input_mid: *mut f32,
    side: *mut f32,
    frames: usize,
) {
    let mid = std::slice::from_raw_parts_mut(input_mid, frames);
    let side = std::slice::from_raw_parts_mut(side, frames);
    let mut cfg = h.dsp.config();
    if cfg.resynthesis.harmony.portamento_ms != glide_ms {
        cfg.resynthesis.harmony.portamento_ms = glide_ms;
        h.dsp.set_config(cfg).unwrap();
    }
    h.dsp.set_score_mix(*levels);
    let mut events = [TimedEvent {
        offset: 0,
        event: ControlEvent::Midi(MidiEvent::AllNotesOff),
    }; 4];
    let mut count = 0;
    for (i, &key) in notes.iter().enumerate() {
        if h.notes[i] == key {
            continue;
        }
        let channel = i as u8;
        let message = if key >= 0 {
            MidiEvent::NoteOn {
                channel,
                note: key as u8,
                velocity: 85,
            }
        } else {
            MidiEvent::NoteOff {
                channel,
                note: h.notes[i] as u8,
            }
        };
        events[count].event = ControlEvent::Midi(message);
        count += 1;
        h.notes[i] = key;
    }
    let (mut pcm, mut left, mut right) = ([0i16; 512], [0i16; 512], [0i16; 512]);
    for (sample, &x) in pcm.iter_mut().zip(mid.iter()) {
        *sample = (32767. * x.clamp(-1., 1.)).round() as i16;
    }
    h.dsp
        .process_planar(
            &pcm[..frames],
            &mut left[..frames],
            &mut right[..frames],
            &events[..count],
        )
        .unwrap();
    for i in 0..frames {
        let [l, r] = h
            .diction
            .process(mid[i], [left[i] as f32 / 32768., right[i] as f32 / 32768.]);
        mid[i] = (l + r) * 0.5;
        side[i] = (r - l) * 0.5;
    }
}

// Chorale mutes its MIDI readers on unvoiced audio. Share the desk's aligned
// consonant detector without sending any voiced samples through Tunelet.
#[no_mangle]
pub extern "C" fn score_diction_new(hz: u32, latency: u32) -> *mut Diction {
    Box::into_raw(Box::new(Diction::new(hz as f32, latency.max(1) as usize)))
}

#[no_mangle]
pub unsafe extern "C" fn score_diction_free(ptr: *mut Diction) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn score_diction_reset(diction: &mut Diction) {
    diction.reset();
}

#[no_mangle]
pub unsafe extern "C" fn score_diction_process(
    diction: &mut Diction,
    mic: *const f32,
    mid: *mut f32,
    side: *mut f32,
    frames: usize,
) {
    for i in 0..frames {
        let m = *mid.add(i);
        let s = *side.add(i);
        let [l, r] = diction.process(*mic.add(i), [m - s, m + s]);
        *mid.add(i) = (l + r) * 0.5;
        *side.add(i) = (r - l) * 0.5;
    }
}
