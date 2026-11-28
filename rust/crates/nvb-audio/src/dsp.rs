//! Pure-Rust real-time Neural Audio AI (RNNoise via `nnnoiseless`) + STFT DSP engine.
//!
//! Pipeline (48 kHz mono, zero heap allocations per 10 ms frame on the hot path):
//!   1. 2nd-order Butterworth high-pass rumble filter (cuts laptop fan / AC sub-bass)
//!   2. Real neural Noise Removal via pure-Rust RNNoise GRU (`nnnoiseless`, 480 samples)
//!      with neural VAD speech-probability tracking and soft non-speech floor suppression
//!   3. STFT Room Echo Removal (and spectral Wiener fallback) via `realfft` (512/256 OLA)
//!   4. Studio Voice 2-band Butterworth crossover tilt EQ (`warm`/`bright`/`full`/`default`)
//!   5. Smooth attack/release Noise Gate (neural-VAD-aware when RNNoise is active)
//!   6. Neural-VAD-gated Voice Auto-Gain (AGC) + soft-knee `tanh` limiter
//!      (boosts speech cleanly without pumping silence or fan noise)

use std::collections::VecDeque;
use std::f64::consts::{PI, SQRT_2};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;

use nnnoiseless::DenoiseState;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{compute_rms_db, f32_to_pcm16, pcm16_to_f32, AudioMeterReading, RNNOISE_FRAME_SAMPLES, SAMPLE_RATE};

const STFT_FRAME_SIZE: usize = 512;
const STFT_HOP_SIZE: usize = STFT_FRAME_SIZE / 2;
const STFT_BINS: usize = STFT_FRAME_SIZE / 2 + 1;

/// Flat configuration snapshot consumed by [`AudioProcessor::process_pcm16`] and
/// [`AudioProcessor::process_samples`]. Can be built directly or parsed from
/// Python's `microphone` / `speaker` JSON config dictionary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioProcessConfig {
    pub highpass_enabled: bool,
    pub highpass_hz: f64,
    pub noise_removal_enabled: bool,
    pub noise_removal_strength: f64,
    pub room_echo_removal_enabled: bool,
    pub room_echo_removal_strength: f64,
    pub studio_voice_enabled: bool,
    pub studio_voice_profile: String,
    pub noise_gate_enabled: bool,
    pub noise_gate_db: f64,
    pub auto_gain_enabled: bool,
    pub auto_gain_target_db: f64,
}

impl Default for AudioProcessConfig {
    fn default() -> Self {
        Self {
            highpass_enabled: true,
            highpass_hz: 85.0,
            noise_removal_enabled: true,
            noise_removal_strength: 0.85,
            room_echo_removal_enabled: false,
            room_echo_removal_strength: 0.5,
            studio_voice_enabled: false,
            studio_voice_profile: "default".to_string(),
            noise_gate_enabled: false,
            noise_gate_db: -46.0,
            auto_gain_enabled: false,
            auto_gain_target_db: -16.0,
        }
    }
}

impl AudioProcessConfig {
    /// Parse from the exact JSON shape used by Python `AudioProcessor.process_pcm16(pcm, cfg)`.
    pub fn from_json_value(val: &Value) -> Self {
        let mut cfg = Self::default();
        let Some(obj) = val.as_object() else {
            return cfg;
        };

        if let Some(v) = obj.get("highpass_enabled").and_then(Value::as_bool) {
            cfg.highpass_enabled = v;
        }
        if let Some(v) = obj.get("highpass_hz").and_then(Value::as_f64) {
            cfg.highpass_hz = v.clamp(20.0, 300.0);
        }
        if let Some(v) = obj.get("noise_gate_enabled").and_then(Value::as_bool) {
            cfg.noise_gate_enabled = v;
        }
        if let Some(v) = obj.get("noise_gate_db").and_then(Value::as_f64) {
            cfg.noise_gate_db = v.clamp(-80.0, -10.0);
        }
        if let Some(v) = obj.get("auto_gain_enabled").and_then(Value::as_bool) {
            cfg.auto_gain_enabled = v;
        }
        if let Some(v) = obj.get("auto_gain_target_db").and_then(Value::as_f64) {
            cfg.auto_gain_target_db = v.clamp(-30.0, -3.0);
        }

        cfg.noise_removal_enabled = false;
        if let Some(effects) = obj.get("effects").and_then(Value::as_object) {
            if let Some(nr) = effects
                .get("noise_removal")
                .or_else(|| effects.get("speaker_noise_removal"))
                .and_then(Value::as_object)
            {
                cfg.noise_removal_enabled = nr.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                cfg.noise_removal_strength = nr
                    .get("strength")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.85)
                    .clamp(0.0, 1.0);
            }
            if let Some(echo) = effects
                .get("room_echo_removal")
                .or_else(|| effects.get("speaker_echo_removal"))
                .and_then(Value::as_object)
            {
                cfg.room_echo_removal_enabled = echo.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                cfg.room_echo_removal_strength = echo
                    .get("strength")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
            }
            if let Some(sv) = effects.get("studio_voice").and_then(Value::as_object) {
                cfg.studio_voice_enabled = sv.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(p) = sv.get("micProfile").and_then(Value::as_str) {
                    cfg.studio_voice_profile = p.to_string();
                }
            }
        }
        cfg
    }
}

/// 2nd-order Butterworth high-pass biquad (Direct Form II Transposed), matching
/// `scipy.signal.butter(2, fc / (0.5 * fs), btype="highpass", output="sos")`.
#[derive(Debug, Clone)]
struct BiquadHighPass {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z0: f64,
    z1: f64,
    cutoff_cached: f64,
    initialized: bool,
}

impl BiquadHighPass {
    fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z0: 0.0,
            z1: 0.0,
            cutoff_cached: -1.0,
            initialized: false,
        }
    }

    fn reset(&mut self) {
        self.z0 = 0.0;
        self.z1 = 0.0;
        self.initialized = false;
    }

    fn configure(&mut self, cutoff_hz: f64, sample_rate: u32, first_sample: f64) {
        if (self.cutoff_cached - cutoff_hz).abs() > 0.5 || !self.initialized {
            let k = (PI * cutoff_hz / f64::from(sample_rate)).tan();
            let k2 = k * k;
            let norm = 1.0 / (1.0 + SQRT_2 * k + k2);
            self.b0 = norm;
            self.b1 = -2.0 * norm;
            self.b2 = norm;
            self.a1 = 2.0 * (k2 - 1.0) * norm;
            self.a2 = (1.0 - SQRT_2 * k + k2) * norm;
            // Steady-state DF-II transposed state for DC step `first_sample` (y_dc = 0)
            self.z0 = -self.b0 * first_sample;
            self.z1 = self.b2 * first_sample;
            self.cutoff_cached = cutoff_hz;
            self.initialized = true;
        }
    }

    fn process_in_place(&mut self, samples: &mut [f32], cutoff_hz: f64, sample_rate: u32) {
        if samples.is_empty() {
            return;
        }
        self.configure(cutoff_hz, sample_rate, f64::from(samples[0]));
        let (b0, b1, b2, a1, a2) = (self.b0, self.b1, self.b2, self.a1, self.a2);
        let (mut z0, mut z1) = (self.z0, self.z1);
        for s in samples.iter_mut() {
            let x = f64::from(*s);
            let y = b0 * x + z0;
            z0 = b1 * x - a1 * y + z1;
            z1 = b2 * x - a2 * y;
            *s = y as f32;
        }
        self.z0 = z0;
        self.z1 = z1;
    }
}

/// 1st-order Butterworth low-pass crossover at 1100 Hz for Studio Voice 2-band tilt EQ.
#[derive(Debug, Clone)]
struct StudioVoiceEq {
    b0: f64,
    b1: f64,
    a1: f64,
    z0: f64,
    initialized: bool,
}

impl StudioVoiceEq {
    fn new(sample_rate: u32) -> Self {
        let k = (PI * 1100.0 / f64::from(sample_rate)).tan();
        let b0 = k / (1.0 + k);
        let b1 = b0;
        let a1 = (k - 1.0) / (1.0 + k);
        Self {
            b0,
            b1,
            a1,
            z0: 0.0,
            initialized: false,
        }
    }

    fn reset(&mut self) {
        self.z0 = 0.0;
        self.initialized = false;
    }

    fn process_in_place(&mut self, samples: &mut [f32], profile: &str) {
        if samples.is_empty() {
            return;
        }
        let (low_g, high_g): (f64, f64) = match profile {
            "warm" => (1.30, 0.92),
            "bright" => (0.88, 1.35),
            "full" => (1.24, 1.22),
            _ => (1.12, 1.15),
        };
        if !self.initialized {
            // Steady-state DF-II state for DC step `samples[0]` (y_dc = x)
            self.z0 = (1.0 - self.b0) * f64::from(samples[0]);
            self.initialized = true;
        }
        let (b0, b1, a1) = (self.b0, self.b1, self.a1);
        let mut z0 = self.z0;
        for s in samples.iter_mut() {
            let x = f64::from(*s);
            let low = b0 * x + z0;
            z0 = b1 * x - a1 * low;
            let high = x - low;
            *s = (low * low_g + high * high_g).clamp(-1.0, 1.0) as f32;
        }
        self.z0 = z0;
    }
}

/// Real-time streaming neural audio processor (`nnnoiseless` RNNoise + `realfft` STFT DSP).
pub struct AudioProcessor {
    pub sample_rate: u32,
    pub backend_name: &'static str,
    pub last_meter: AudioMeterReading,

    // Neural RNNoise state & ring buffers (pre-allocated)
    rnnoise: Option<Box<DenoiseState<'static>>>,
    rn_in_buf: VecDeque<f32>,
    rn_out_buf: VecDeque<f32>,
    rn_frame_in: [f32; RNNOISE_FRAME_SAMPLES],
    rn_frame_out: [f32; RNNOISE_FRAME_SAMPLES],
    last_speech_prob: f64,
    vad_hold_frames: u32,

    // STFT Room Echo Removal & spectral fallback (pre-allocated)
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    r2c_scratch: Vec<Complex<f32>>,
    c2r_scratch: Vec<Complex<f32>>,
    fft_time_buf: Vec<f32>,
    fft_spec_buf: Vec<Complex<f32>>,
    hann_window: [f32; STFT_FRAME_SIZE],
    stft_in_buf: VecDeque<f32>,
    stft_out_buf: VecDeque<f32>,
    ola_tail: [f32; STFT_HOP_SIZE],
    noise_psd: [f64; STFT_BINS],
    reverb_env: [f64; STFT_BINS],
    frames_seen: u64,

    // Filters & dynamics
    highpass: BiquadHighPass,
    studio_eq: StudioVoiceEq,
    gate_gain: f64,
    agc_gain: f64,
    work_buf: Vec<f32>,
}

impl AudioProcessor {
    /// Create a new streaming processor. Enables pure-Rust `nnnoiseless` RNNoise at 48 kHz.
    pub fn new(sample_rate: u32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(STFT_FRAME_SIZE);
        let c2r = planner.plan_fft_inverse(STFT_FRAME_SIZE);
        let r2c_scratch = r2c.make_scratch_vec();
        let c2r_scratch = c2r.make_scratch_vec();
        let fft_time_buf = vec![0.0_f32; STFT_FRAME_SIZE];
        let fft_spec_buf = vec![Complex::new(0.0_f32, 0.0_f32); STFT_BINS];

        let mut hann_window = [0.0_f32; STFT_FRAME_SIZE];
        for (i, slot) in hann_window.iter_mut().enumerate() {
            *slot = (0.5 * (1.0 - (2.0 * PI * i as f64 / STFT_FRAME_SIZE as f64).cos())) as f32;
        }

        let (rnnoise, backend_name) = if sample_rate == SAMPLE_RATE {
            (Some(DenoiseState::new()), "nnnoiseless")
        } else {
            (None, "dsp")
        };

        let mut stft_in_buf = VecDeque::with_capacity(STFT_FRAME_SIZE * 4);
        stft_in_buf.resize(STFT_HOP_SIZE, 0.0);

        Self {
            sample_rate,
            backend_name,
            last_meter: AudioMeterReading {
                backend: backend_name.to_string(),
                ..AudioMeterReading::default()
            },
            rnnoise,
            rn_in_buf: VecDeque::with_capacity(RNNOISE_FRAME_SAMPLES * 4),
            rn_out_buf: VecDeque::with_capacity(RNNOISE_FRAME_SAMPLES * 4),
            rn_frame_in: [0.0; RNNOISE_FRAME_SAMPLES],
            rn_frame_out: [0.0; RNNOISE_FRAME_SAMPLES],
            last_speech_prob: 0.0,
            vad_hold_frames: 0,
            r2c,
            c2r,
            r2c_scratch,
            c2r_scratch,
            fft_time_buf,
            fft_spec_buf,
            hann_window,
            stft_in_buf,
            stft_out_buf: VecDeque::with_capacity(STFT_FRAME_SIZE * 4),
            ola_tail: [0.0; STFT_HOP_SIZE],
            noise_psd: [1e-6; STFT_BINS],
            reverb_env: [0.0; STFT_BINS],
            frames_seen: 0,
            highpass: BiquadHighPass::new(),
            studio_eq: StudioVoiceEq::new(sample_rate),
            gate_gain: 1.0,
            agc_gain: 1.0,
            work_buf: Vec::with_capacity(RNNOISE_FRAME_SAMPLES * 2),
        }
    }

    pub fn reset(&mut self) {
        if self.rnnoise.is_some() {
            self.rnnoise = Some(DenoiseState::new());
        }
        self.rn_in_buf.clear();
        self.rn_out_buf.clear();
        self.last_speech_prob = 0.0;
        self.vad_hold_frames = 0;
        self.stft_in_buf.clear();
        self.stft_in_buf.resize(STFT_HOP_SIZE, 0.0);
        self.stft_out_buf.clear();
        self.ola_tail.fill(0.0);
        self.noise_psd.fill(1e-6);
        self.reverb_env.fill(0.0);
        self.frames_seen = 0;
        self.highpass.reset();
        self.studio_eq.reset();
        self.gate_gain = 1.0;
        self.agc_gain = 1.0;
    }

    /// Process signed 16-bit mono PCM bytes into signed 16-bit mono PCM bytes.
    pub fn process_pcm16(&mut self, pcm_bytes: &[u8], cfg: &AudioProcessConfig) -> Vec<u8> {
        let usable = (pcm_bytes.len() / 2) * 2;
        if usable == 0 {
            return Vec::new();
        }
        self.work_buf.clear();
        let inv = 1.0_f64 / 32768.0_f64;
        for &chunk in pcm_bytes[..usable].as_chunks::<2>().0 {
            let s = i16::from_le_bytes(chunk);
            self.work_buf.push((f64::from(s) * inv) as f32);
        }
        let mut local = std::mem::take(&mut self.work_buf);
        self.process_samples_in_place(&mut local, cfg);
        let out = f32_to_pcm16(&local);
        self.work_buf = local;
        out
    }

    /// Process a slice of normalized float samples in `[-1.0, 1.0]` into a newly allocated Vec.
    pub fn process_samples(&mut self, samples: &[f32], cfg: &AudioProcessConfig) -> Vec<f32> {
        let mut out = samples.to_vec();
        self.process_samples_in_place(&mut out, cfg);
        out
    }

    /// Process normalized float samples in-place (zero allocation when length fits pre-allocated buffers).
    pub fn process_samples_in_place(&mut self, work: &mut [f32], cfg: &AudioProcessConfig) {
        if work.is_empty() {
            return;
        }
        let (in_rms_db, in_peak_db) = compute_rms_db(work);

        let mut nr_on = cfg.noise_removal_enabled;
        let mut nr_strength = cfg.noise_removal_strength.clamp(0.0, 1.0);
        let mut echo_on = cfg.room_echo_removal_enabled;
        let mut echo_strength = cfg.room_echo_removal_strength.clamp(0.0, 1.0);
        let sv_on = cfg.studio_voice_enabled;

        if sv_on {
            nr_on = true;
            nr_strength = nr_strength.max(0.85);
            echo_on = true;
            echo_strength = echo_strength.max(0.35);
        }

        // 1. High-pass rumble filter
        if cfg.highpass_enabled {
            let cutoff = cfg.highpass_hz.clamp(20.0, 300.0);
            self.highpass.process_in_place(work, cutoff, self.sample_rate);
        }

        // 2. Neural Noise Removal (pure-Rust RNNoise) + optional STFT Room Echo Removal
        let speech_prob: f64;
        if nr_on && self.rnnoise.is_some() {
            speech_prob = self.run_rnnoise_in_place(work, nr_strength);
            if echo_on {
                self.process_spectral_in_place(work, false, 0.0, true, echo_strength);
            }
        } else if nr_on || echo_on {
            self.process_spectral_in_place(work, nr_on, nr_strength, echo_on, echo_strength);
            let (cur_rms, _) = compute_rms_db(work);
            speech_prob = if cur_rms > -35.0 { 1.0 } else { 0.1 };
        } else {
            speech_prob = 0.0;
        }

        // 3. Studio Voice broadcast parametric EQ
        if sv_on {
            self.studio_eq
                .process_in_place(work, &cfg.studio_voice_profile);
        }

        // 4. Smooth Attack/Release Noise Gate
        let mut gate_open = true;
        if cfg.noise_gate_enabled {
            let gate_thresh_db = cfg.noise_gate_db;
            let (chunk_rms_db, _) = compute_rms_db(work);
            gate_open = if self.rnnoise.is_some() && nr_on {
                self.vad_hold_frames > 0 || chunk_rms_db >= gate_thresh_db + 6.0
            } else {
                chunk_rms_db >= gate_thresh_db
            };
            let target_gate = if gate_open { 1.0 } else { 0.04 };
            let coeff = if target_gate > self.gate_gain { 0.45 } else { 0.08 };
            self.gate_gain += coeff * (target_gate - self.gate_gain);
            let g = self.gate_gain as f32;
            for s in work.iter_mut() {
                *s *= g;
            }
        }

        // 5. Speech-gated Voice Auto-Gain (ONLY boosts real speech, NEVER amplifies room noise!)
        if cfg.auto_gain_enabled || sv_on {
            let target_db = cfg.auto_gain_target_db;
            let (chunk_rms_db, _) = compute_rms_db(work);
            let is_speech = if self.rnnoise.is_some() && nr_on {
                self.vad_hold_frames > 0 && chunk_rms_db > -40.0
            } else {
                chunk_rms_db > -34.0
            };
            if is_speech {
                let desired_db = (target_db - chunk_rms_db).clamp(-6.0, 12.0);
                let desired_lin = 10.0_f64.powf(desired_db / 20.0);
                self.agc_gain = 0.85 * self.agc_gain + 0.15 * desired_lin;
            } else {
                self.agc_gain = 0.96 * self.agc_gain + 0.04 * 1.0;
            }
            let gain = self.agc_gain;
            for s in work.iter_mut() {
                let mut v = f64::from(*s) * gain;
                let a = v.abs();
                if a > 0.85 {
                    v = (0.85 + 0.15 * ((a - 0.85) / 0.15).tanh()).copysign(v);
                }
                *s = v.clamp(-1.0, 1.0) as f32;
            }
        }

        let (out_rms_db, out_peak_db) = compute_rms_db(work);
        self.last_meter = AudioMeterReading {
            input_rms_db: (in_rms_db * 10.0).round() / 10.0,
            output_rms_db: (out_rms_db * 10.0).round() / 10.0,
            input_peak_db: (in_peak_db * 10.0).round() / 10.0,
            output_peak_db: (out_peak_db * 10.0).round() / 10.0,
            noise_reduction_db: ((in_rms_db - out_rms_db).max(0.0) * 10.0).round() / 10.0,
            speech_prob: (speech_prob * 100.0).round() / 100.0,
            gate_open,
            backend: self.backend_name.to_string(),
        };
    }

    fn run_rnnoise_in_place(&mut self, work: &mut [f32], strength: f64) -> f64 {
        let Some(rnnoise) = self.rnnoise.as_mut() else {
            return 0.0;
        };

        // Fast zero-copy path when input is exactly 1 frame (480 samples) and buffers are empty
        let s = strength.clamp(0.0, 1.0) as f32;
        if work.len() == RNNOISE_FRAME_SAMPLES
            && self.rn_in_buf.is_empty()
            && self.rn_out_buf.is_empty()
        {
            for (dst, &src) in self.rn_frame_in.iter_mut().zip(work.iter()) {
                *dst = (src * 32767.0).clamp(-32768.0, 32767.0);
            }
            let p = f64::from(rnnoise.process_frame(&mut self.rn_frame_out, &self.rn_frame_in));
            self.last_speech_prob = p;
            if p > 0.35 {
                self.vad_hold_frames = 18;
            } else if self.vad_hold_frames > 0 {
                self.vad_hold_frames -= 1;
            }
            let atten = if self.vad_hold_frames == 0 && strength > 0.45 {
                (1.0 - 0.85 * strength * (1.0 - p)).max(0.08) as f32
            } else {
                1.0_f32
            };
            let inv = 1.0_f32 / 32768.0_f32;
            let dry = 1.0_f32 - s;
            for (w, &den) in work.iter_mut().zip(self.rn_frame_out.iter()) {
                let d = den * inv * atten;
                *w = (*w) * dry + d * s;
            }
            return p;
        }

        // General arbitrary chunk length path
        for &sample in work.iter() {
            self.rn_in_buf
                .push_back((sample * 32767.0).clamp(-32768.0, 32767.0));
        }

        let mut prob_sum = 0.0_f64;
        let mut prob_cnt = 0u32;
        let inv = 1.0_f32 / 32768.0_f32;

        while self.rn_in_buf.len() >= RNNOISE_FRAME_SAMPLES {
            for slot in &mut self.rn_frame_in {
                *slot = self.rn_in_buf.pop_front().unwrap_or(0.0);
            }
            let p = f64::from(rnnoise.process_frame(&mut self.rn_frame_out, &self.rn_frame_in));
            prob_sum += p;
            prob_cnt += 1;
            if p > 0.35 {
                self.vad_hold_frames = 18;
            } else if self.vad_hold_frames > 0 {
                self.vad_hold_frames -= 1;
            }
            let atten = if self.vad_hold_frames == 0 && strength > 0.45 {
                (1.0 - 0.85 * strength * (1.0 - p)).max(0.08) as f32
            } else {
                1.0_f32
            };
            for &den in &self.rn_frame_out {
                self.rn_out_buf.push_back(den * inv * atten);
            }
        }

        if prob_cnt > 0 {
            self.last_speech_prob = prob_sum / f64::from(prob_cnt);
        }

        let n = work.len();
        let dry = 1.0_f32 - s;
        if self.rn_out_buf.len() >= n {
            for w in work.iter_mut() {
                let d = self.rn_out_buf.pop_front().unwrap_or(0.0);
                *w = (*w) * dry + d * s;
            }
        } else if !self.rn_out_buf.is_empty() {
            let pad = n - self.rn_out_buf.len();
            for (i, w) in work.iter_mut().enumerate() {
                let d = if i < pad {
                    0.0
                } else {
                    self.rn_out_buf.pop_front().unwrap_or(0.0)
                };
                *w = (*w) * dry + d * s;
            }
        }
        self.last_speech_prob
    }

    fn process_spectral_in_place(
        &mut self,
        work: &mut [f32],
        nr_on: bool,
        nr_strength: f64,
        echo_on: bool,
        echo_strength: f64,
    ) {
        self.stft_in_buf.extend(work.iter().copied());

        while self.stft_in_buf.len() >= STFT_FRAME_SIZE {
            for (i, dst) in self.fft_time_buf.iter_mut().enumerate() {
                *dst = self.stft_in_buf[i] * self.hann_window[i];
            }
            for _ in 0..STFT_HOP_SIZE {
                self.stft_in_buf.pop_front();
            }

            self.frames_seen += 1;
            let _ = self.r2c.process_with_scratch(
                &mut self.fft_time_buf,
                &mut self.fft_spec_buf,
                &mut self.r2c_scratch,
            );

            for k in 0..STFT_BINS {
                let c = self.fft_spec_buf[k];
                let mag_sq = f64::from(c.re) * f64::from(c.re) + f64::from(c.im) * f64::from(c.im);

                let mut npsd = self.noise_psd[k];
                if self.frames_seen <= 8 {
                    npsd = 0.65 * npsd + 0.35 * mag_sq;
                } else if mag_sq < npsd {
                    npsd = 0.88 * npsd + 0.12 * mag_sq;
                } else if mag_sq < npsd * 4.5 {
                    npsd = 0.93 * npsd + 0.07 * mag_sq;
                }
                self.noise_psd[k] = npsd;

                let mut gain = 1.0_f64;
                if nr_on {
                    let oversub = 1.0 + 3.2 * nr_strength;
                    let floor = (0.25 * (1.0 - 0.88 * nr_strength)).max(0.02);
                    let clean_sq = (mag_sq - oversub * npsd).max(0.0);
                    let wiener = clean_sq / (mag_sq + 1e-12);
                    gain *= wiener.max(floor);
                }
                if echo_on {
                    let renv = 0.88 * self.reverb_env[k] + 0.12 * mag_sq;
                    self.reverb_env[k] = renv;
                    let late = echo_strength * 0.72 * renv;
                    let derev = (mag_sq - late).max(0.0) / (mag_sq + 1e-12);
                    gain *= derev.max(0.10);
                }

                let g = gain as f32;
                self.fft_spec_buf[k] = Complex::new(c.re * g, c.im * g);
            }
            // realfft requires DC and Nyquist imaginary components to be strictly 0.0
            self.fft_spec_buf[0].im = 0.0;
            self.fft_spec_buf[STFT_BINS - 1].im = 0.0;

            let _ = self.c2r.process_with_scratch(
                &mut self.fft_spec_buf,
                &mut self.fft_time_buf,
                &mut self.c2r_scratch,
            );

            let inv_n = 1.0_f32 / STFT_FRAME_SIZE as f32;
            for i in 0..STFT_HOP_SIZE {
                let s = self.fft_time_buf[i] * inv_n;
                self.stft_out_buf.push_back(self.ola_tail[i] + s);
            }
            for i in 0..STFT_HOP_SIZE {
                self.ola_tail[i] = self.fft_time_buf[STFT_HOP_SIZE + i] * inv_n;
            }
        }

        let need = work.len();
        if self.stft_out_buf.len() >= need {
            for w in work.iter_mut() {
                *w = self.stft_out_buf.pop_front().unwrap_or(0.0);
            }
        } else {
            let pad = need - self.stft_out_buf.len();
            for (i, w) in work.iter_mut().enumerate() {
                *w = if i < pad {
                    0.0
                } else {
                    self.stft_out_buf.pop_front().unwrap_or(0.0)
                };
            }
        }
    }
}

/// Build an FFmpeg `-af` filter graph mirroring Python's `build_ffmpeg_audio_filter`.
pub fn build_ffmpeg_audio_filter(cfg: &AudioProcessConfig) -> String {
    let mut filters: Vec<String> = Vec::new();
    if cfg.highpass_enabled {
        let hz = cfg.highpass_hz.round() as i64;
        filters.push(format!("highpass=f={hz}:poles=2"));
    }
    if cfg.noise_removal_enabled {
        let nr_db = 6.0 + 34.0 * cfg.noise_removal_strength.clamp(0.0, 1.0);
        filters.push(format!("afftdn=nr={nr_db:.1}:nf=-35:tn=1"));
    }
    if cfg.room_echo_removal_enabled {
        let thresh = -48.0 + 14.0 * cfg.room_echo_removal_strength.clamp(0.0, 1.0);
        filters.push(format!("agate=threshold={thresh:.1}dB:ratio=3:attack=8:release=140"));
    }
    if cfg.studio_voice_enabled {
        let eq = match cfg.studio_voice_profile.as_str() {
            "warm" => "equalizer=f=160:t=q:w=1.0:g=3.5,equalizer=f=3200:t=q:w=1.2:g=1.5",
            "bright" => "equalizer=f=4500:t=q:w=1.1:g=4.0,equalizer=f=200:t=q:w=1.0:g=-1.5",
            "full" => "equalizer=f=140:t=q:w=1.0:g=3.0,equalizer=f=3800:t=q:w=1.2:g=3.0",
            _ => "equalizer=f=180:t=q:w=1.0:g=2.0,equalizer=f=3500:t=q:w=1.2:g=2.0",
        };
        filters.push(eq.to_string());
        filters.push("acompressor=threshold=-20dB:ratio=3:attack=6:release=80:makeup=3".to_string());
    }
    if cfg.noise_gate_enabled && !cfg.room_echo_removal_enabled {
        filters.push(format!(
            "agate=threshold={:.1}dB:ratio=2.5:attack=6:release=120",
            cfg.noise_gate_db
        ));
    }
    if cfg.auto_gain_enabled {
        filters.push("dynaudnorm=f=150:g=11:p=0.9:m=6.0".to_string());
    }
    if filters.is_empty() {
        "anull".to_string()
    } else {
        filters.join(",")
    }
}

/// Compute normalized `[0..1]` amplitude bars for UI waveform previews.
pub fn compute_waveform_peaks(pcm_bytes: &[u8], buckets: usize) -> Vec<f64> {
    if pcm_bytes.is_empty() || buckets == 0 {
        return vec![0.0; buckets.max(1)];
    }
    let floats = pcm16_to_f32(pcm_bytes);
    if floats.is_empty() {
        return vec![0.0; buckets];
    }
    let n = floats.len();
    let mut peaks = Vec::with_capacity(buckets);
    for i in 0..buckets {
        let start = i * n / buckets;
        let end = ((i + 1) * n / buckets).max(start + 1).min(n);
        let p = floats[start..end]
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(f64::from(v.abs())));
        peaks.push(p);
    }
    let mx = peaks.iter().copied().fold(0.0_f64, f64::max);
    if mx > 1e-4 {
        let scale = (0.95 / mx).min(3.0);
        for p in &mut peaks {
            *p = (((*p * scale).min(1.0)) * 10_000.0).round() / 10_000.0;
        }
    }
    peaks
}

/// Read a 16-bit PCM WAV file into `(mono_pcm16_bytes, sample_rate)`.
pub fn read_wav_mono16(path: &Path) -> io::Result<(Vec<u8>, u32)> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid WAV header"));
    }

    let mut pos = 12usize;
    let mut channels = 1u16;
    let mut sample_rate = SAMPLE_RATE;
    let mut bits_per_sample = 16u16;
    let mut pcm_slice: Option<&[u8]> = None;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        pos += 8;
        let end = (pos + chunk_size).min(data.len());
        if chunk_id == b"fmt " && end >= pos + 16 {
            channels = u16::from_le_bytes([data[pos + 2], data[pos + 3]]);
            sample_rate = u32::from_le_bytes([
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            bits_per_sample = u16::from_le_bytes([data[pos + 14], data[pos + 15]]);
        } else if chunk_id == b"data" {
            pcm_slice = Some(&data[pos..end]);
            break;
        }
        pos = end + (chunk_size & 1);
    }

    if bits_per_sample != 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected 16-bit PCM WAV, got {bits_per_sample}-bit"),
        ));
    }
    let raw = pcm_slice
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "WAV missing data chunk"))?;

    if channels <= 1 {
        return Ok((raw.to_vec(), sample_rate));
    }

    let ch = usize::from(channels);
    let frame_bytes = ch * 2;
    let mut mono = Vec::with_capacity((raw.len() / frame_bytes) * 2);
    for frame in raw.chunks_exact(frame_bytes) {
        let mut acc = 0i32;
        for c in 0..ch {
            acc += i32::from(i16::from_le_bytes([frame[c * 2], frame[c * 2 + 1]]));
        }
        let avg = (acc / ch as i32).clamp(-32768, 32767) as i16;
        mono.extend_from_slice(&avg.to_le_bytes());
    }
    Ok((mono, sample_rate))
}

/// Write signed 16-bit mono PCM bytes to a standard RIFF WAV file.
pub fn write_wav_mono16(path: &Path, pcm_bytes: &[u8], sample_rate: u32) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let data_len = ((pcm_bytes.len() / 2) * 2) as u32;
    let riff_len = 36u32.saturating_add(data_len);
    let byte_rate = sample_rate * 2;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(&pcm_bytes[..data_len as usize]);

    let mut file = File::create(path)?;
    file.write_all(&out)?;
    Ok(())
}

/// Deterministic sample clip generator for standalone Rust audio latency benchmarks & demos.
pub fn generate_sample_clip(kind: &str, sample_rate: u32, duration_sec: f64) -> Vec<u8> {
    let n = (f64::from(sample_rate) * duration_sec).round() as usize;
    let mut voice = vec![0.0_f64; n];
    let mut noise = vec![0.0_f64; n];
    let mut phase = 0.0_f64;
    let mut lcg: u32 = 42_123_456;

    let mut next_rand = || -> f64 {
        lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((lcg & 0x7FFF_FFFF) as f64 / 1_073_741_823.5) - 1.0
    };

    for i in 0..n {
        let t = i as f64 / f64::from(sample_rate);
        let raw_env = (2.0 * PI * 0.45 * t).sin();
        let env = if raw_env > 0.12 {
            raw_env.powf(0.7) * (0.65 + 0.35 * (2.0 * PI * 3.8 * t).sin())
        } else {
            0.0
        };
        let f0 = 155.0 + 28.0 * (2.0 * PI * 0.9 * t).sin();
        phase += 2.0 * PI * f0 / f64::from(sample_rate);
        voice[i] = env
            * (0.22 * phase.sin()
                + 0.14 * (2.0 * phase).sin()
                + 0.09 * (3.0 * phase).sin()
                + 0.05 * (5.0 * phase).sin());

        if matches!(kind, "fan" | "fan_echo" | "pure_noise") {
            let hum = 0.04 * (2.0 * PI * 120.0 * t).sin() + 0.03 * (2.0 * PI * 240.0 * t).sin();
            noise[i] += next_rand();
            // Store hum separately after 12-tap fan lowpass below
            let _ = hum;
        }
    }

    if matches!(kind, "fan" | "fan_echo" | "pure_noise") {
        // 12-tap moving average low-passed fan roar + 120/240 Hz AC hum (matches Python corpus)
        let raw_white = noise.clone();
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += raw_white[i];
            if i >= 12 {
                acc -= raw_white[i - 12];
            }
            let t = i as f64 / f64::from(sample_rate);
            let fan = (acc / 12.0) * 0.12;
            let hum = 0.04 * (2.0 * PI * 120.0 * t).sin() + 0.03 * (2.0 * PI * 240.0 * t).sin();
            noise[i] = fan + hum;
        }
    }

    if matches!(kind, "echo" | "fan_echo") {
        let mut reverb = vec![0.0_f64; n];
        for (delay_ms, gain) in [(45usize, 0.42), (95, 0.28), (160, 0.16)] {
            let d = (usize::try_from(sample_rate).unwrap_or(48_000) * delay_ms) / 1000;
            for i in d..n {
                reverb[i] += voice[i - d] * gain;
            }
        }
        for i in 0..n {
            voice[i] += reverb[i];
        }
    }

    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let s = if kind == "pure_noise" {
                noise[i]
            } else {
                voice[i] + noise[i]
            };
            s.clamp(-0.95, 0.95) as f32
        })
        .collect();
    f32_to_pcm16(&samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn nnnoiseless_suppresses_pure_noise_without_agc_pumping() {
        let mut proc = AudioProcessor::new(SAMPLE_RATE);
        assert_eq!(proc.backend_name, "nnnoiseless");

        let cfg = AudioProcessConfig {
            highpass_enabled: true,
            highpass_hz: 85.0,
            noise_removal_enabled: true,
            noise_removal_strength: 0.85,
            room_echo_removal_enabled: false,
            room_echo_removal_strength: 0.5,
            studio_voice_enabled: false,
            studio_voice_profile: "default".to_string(),
            noise_gate_enabled: true,
            noise_gate_db: -46.0,
            auto_gain_enabled: true,
            auto_gain_target_db: -16.0,
        };

        let raw = generate_sample_clip("pure_noise", SAMPLE_RATE, 2.0);
        let mut processed = Vec::with_capacity(raw.len());
        for chunk in raw.chunks(RNNOISE_FRAME_SAMPLES * 2) {
            processed.extend_from_slice(&proc.process_pcm16(chunk, &cfg));
        }

        let (in_rms, _) = compute_rms_db(&pcm16_to_f32(&raw));
        let (out_rms, _) = compute_rms_db(&pcm16_to_f32(&processed));
        assert!(
            out_rms < in_rms - 6.0,
            "expected strong pure-noise suppression without AGC boost: in={in_rms:.2} dB, out={out_rms:.2} dB"
        );
    }

    #[test]
    fn studio_voice_and_room_echo_handle_edge_frames_without_panics() {
        let mut proc = AudioProcessor::new(SAMPLE_RATE);
        let cfg = AudioProcessConfig {
            studio_voice_enabled: true,
            studio_voice_profile: "warm".to_string(),
            auto_gain_enabled: true,
            ..AudioProcessConfig::default()
        };

        // Empty, odd byte, short chunk, and normal 480-sample chunks
        assert!(proc.process_pcm16(&[], &cfg).is_empty());
        assert!(proc.process_pcm16(&[0x01], &cfg).is_empty());
        let short = vec![0u8; 64];
        assert_eq!(proc.process_pcm16(&short, &cfg).len(), 64);

        let clip = generate_sample_clip("fan_echo", SAMPLE_RATE, 1.5);
        let mut out = Vec::new();
        for chunk in clip.chunks(RNNOISE_FRAME_SAMPLES * 2) {
            out.extend_from_slice(&proc.process_pcm16(chunk, &cfg));
        }
        assert_eq!(out.len(), clip.len());
        assert!(proc.last_meter.output_rms_db.is_finite());
    }

    #[test]
    fn wav_roundtrip_and_ffmpeg_filter_match_contract() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nvb_audio_wav_{nonce}.wav"));
        let pcm = generate_sample_clip("fan", SAMPLE_RATE, 0.25);
        write_wav_mono16(&path, &pcm, SAMPLE_RATE).unwrap();
        let (read_back, sr) = read_wav_mono16(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(sr, SAMPLE_RATE);
        assert_eq!(read_back, pcm);

        let cfg = AudioProcessConfig {
            studio_voice_enabled: true,
            studio_voice_profile: "bright".to_string(),
            auto_gain_enabled: true,
            ..AudioProcessConfig::default()
        };
        let graph = build_ffmpeg_audio_filter(&cfg);
        assert!(graph.contains("highpass=f=85:poles=2"));
        assert!(graph.contains("afftdn="));
        assert!(graph.contains("acompressor="));
        assert!(graph.contains("dynaudnorm="));
    }
}
