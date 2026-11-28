//! Configuration model, clamping, persistence, and mutual-exclusion rules
//! matching `nv_broadcast/config.py` (NVIDIA Broadcast v2.3.0 schema).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

pub const CONFIG_VERSION: u32 = 2;

pub const AUDIO_BACKENDS: &[&str] = &[
    "auto",
    "pyrnnoise",
    "deepfilternet",
    "rnnoise_ladspa",
    "webrtc_aec",
    "ffmpeg_afftdn",
    "dsp",
];

pub const SEGMENTATION_BACKENDS: &[&str] = &[
    "auto",
    "rvm_cuda",
    "onnx_cuda",
    "mediapipe",
    "rvm_cpu",
    "onnx_cpu",
    "fast_portrait",
];

pub const STUDIO_VOICE_PROFILES: &[&str] = &["default", "bright", "full", "warm"];
pub const BLUR_MODES: &[&str] = &["performance", "quality"];
pub const BG_REPLACE_PRESETS: &[&str] = &[
    "studio_dark",
    "office_blur",
    "neon_green",
    "soft_daylight",
];

/// Canonical mutual-exclusion groups matching NVIDIA Broadcast v2.3.0.
pub const INCOMPATIBLE_GROUPS: &[&[&str]] = &[
    &["background_blur", "background_replace", "background_remove"],
    &["studio_voice", "noise_removal"],
    &["studio_voice", "room_echo_removal"],
];

pub const ALL_EFFECT_IDS: &[&str] = &[
    "noise_removal",
    "room_echo_removal",
    "studio_voice",
    "speaker_noise_removal",
    "speaker_echo_removal",
    "background_blur",
    "background_replace",
    "background_remove",
    "auto_frame",
    "eye_contact",
    "video_noise_removal",
    "virtual_key_light",
    "vignette",
];

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiConfig {
    pub show_gpu_indicator: bool,
    pub start_minimized: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            show_gpu_indicator: true,
            start_minimized: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrengthEffect {
    pub enabled: bool,
    pub strength: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudioVoiceEffect {
    pub enabled: bool,
    #[serde(rename = "micProfile")]
    pub mic_profile: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackgroundBlurEffect {
    pub enabled: bool,
    pub strength: f64,
    pub mode: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackgroundReplaceEffect {
    pub enabled: bool,
    #[serde(rename = "imagePath")]
    pub image_path: String,
    pub preset: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackgroundRemoveEffect {
    pub enabled: bool,
    #[serde(rename = "chromaColor")]
    pub chroma_color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoFrameEffect {
    pub enabled: bool,
    pub zoom: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EyeContactEffect {
    pub enabled: bool,
    pub strength: f64,
    #[serde(rename = "lookAway")]
    pub look_away: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VirtualKeyLightEffect {
    pub enabled: bool,
    pub warmth: f64,
    pub intensity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VignetteEffect {
    pub enabled: bool,
    pub strength: f64,
    #[serde(rename = "faceTracking")]
    pub face_tracking: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicrophoneEffects {
    pub noise_removal: StrengthEffect,
    pub room_echo_removal: StrengthEffect,
    pub studio_voice: StudioVoiceEffect,
}

impl Default for MicrophoneEffects {
    fn default() -> Self {
        Self {
            noise_removal: StrengthEffect {
                enabled: true,
                strength: 0.85,
            },
            room_echo_removal: StrengthEffect {
                enabled: false,
                strength: 0.5,
            },
            studio_voice: StudioVoiceEffect {
                enabled: false,
                mic_profile: "default".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEffects {
    pub speaker_noise_removal: StrengthEffect,
    pub speaker_echo_removal: StrengthEffect,
}

impl Default for SpeakerEffects {
    fn default() -> Self {
        Self {
            speaker_noise_removal: StrengthEffect {
                enabled: false,
                strength: 0.75,
            },
            speaker_echo_removal: StrengthEffect {
                enabled: false,
                strength: 0.5,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraEffects {
    pub background_blur: BackgroundBlurEffect,
    pub background_replace: BackgroundReplaceEffect,
    pub background_remove: BackgroundRemoveEffect,
    pub auto_frame: AutoFrameEffect,
    pub eye_contact: EyeContactEffect,
    pub video_noise_removal: StrengthEffect,
    pub virtual_key_light: VirtualKeyLightEffect,
    pub vignette: VignetteEffect,
}

impl Default for CameraEffects {
    fn default() -> Self {
        Self {
            background_blur: BackgroundBlurEffect {
                enabled: false,
                strength: 0.7,
                mode: "quality".to_string(),
            },
            background_replace: BackgroundReplaceEffect {
                enabled: false,
                image_path: String::new(),
                preset: "studio_dark".to_string(),
            },
            background_remove: BackgroundRemoveEffect {
                enabled: false,
                chroma_color: "#00ff00".to_string(),
            },
            auto_frame: AutoFrameEffect {
                enabled: false,
                zoom: 0.35,
            },
            eye_contact: EyeContactEffect {
                enabled: false,
                strength: 0.7,
                look_away: true,
            },
            video_noise_removal: StrengthEffect {
                enabled: false,
                strength: 0.6,
            },
            virtual_key_light: VirtualKeyLightEffect {
                enabled: false,
                warmth: 0.5,
                intensity: 0.55,
            },
            vignette: VignetteEffect {
                enabled: false,
                strength: 0.4,
                face_tracking: true,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicrophoneConfig {
    pub device: String,
    pub virtual_name: String,
    pub sample_rate: u32,
    pub highpass_enabled: bool,
    pub highpass_hz: i64,
    pub noise_gate_enabled: bool,
    pub noise_gate_db: f64,
    pub auto_gain_enabled: bool,
    pub auto_gain_target_db: f64,
    pub effects: MicrophoneEffects,
}

impl Default for MicrophoneConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            virtual_name: "NVIDIA Broadcast (Microphone)".to_string(),
            sample_rate: 48_000,
            highpass_enabled: true,
            highpass_hz: 85,
            noise_gate_enabled: true,
            noise_gate_db: -46.0,
            auto_gain_enabled: true,
            auto_gain_target_db: -16.0,
            effects: MicrophoneEffects::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerConfig {
    pub device: String,
    pub virtual_name: String,
    pub sample_rate: u32,
    pub effects: SpeakerEffects,
}

impl Default for SpeakerConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            virtual_name: "NVIDIA Broadcast (Speakers)".to_string(),
            sample_rate: 48_000,
            effects: SpeakerEffects::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraConfig {
    pub device: String,
    pub output_device: String,
    pub virtual_name: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub prefer_mjpg: bool,
    pub mirror: bool,
    pub split_preview: bool,
    pub split_position: f64,
    pub on_demand: bool,
    pub idle_timeout_sec: f64,
    pub v4l2_controls: BTreeMap<String, Value>,
    pub effects: CameraEffects,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            output_device: "auto".to_string(),
            virtual_name: "NVIDIA Broadcast Camera".to_string(),
            width: 1280,
            height: 720,
            fps: 30,
            prefer_mjpg: true,
            mirror: false,
            split_preview: false,
            split_position: 0.5,
            on_demand: true,
            idle_timeout_sec: 3.0,
            v4l2_controls: BTreeMap::new(),
            effects: CameraEffects::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BroadcastConfig {
    pub version: u32,
    pub audio_backend: String,
    pub segmentation_backend: String,
    pub ui: UiConfig,
    pub microphone: MicrophoneConfig,
    pub speaker: SpeakerConfig,
    pub camera: CameraConfig,
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            audio_backend: "auto".to_string(),
            segmentation_backend: "auto".to_string(),
            ui: UiConfig::default(),
            microphone: MicrophoneConfig::default(),
            speaker: SpeakerConfig::default(),
            camera: CameraConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BroadcastState {
    pub microphone: MicrophoneEffects,
    pub speaker: SpeakerEffects,
    pub camera: CameraEffects,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectApplyResult {
    pub applied: Vec<String>,
    pub disabled_conflicts: BTreeMap<String, Vec<String>>,
    pub errors: Vec<String>,
}

pub fn clamp_float(value: f64, lo: f64, hi: f64, default: f64) -> f64 {
    if !value.is_finite() {
        return default;
    }
    value.clamp(lo, hi)
}

fn value_to_f64(val: &Value, default: f64) -> f64 {
    match val {
        Value::Number(n) => n.as_f64().unwrap_or(default),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::String(s) => s.trim().parse::<f64>().unwrap_or(default),
        _ => default,
    }
}

fn value_to_bool(val: &Value, default: bool) -> bool {
    match val {
        Value::Bool(b) => *b,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                default
            }
        }
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
        Value::Null => false,
    }
}

fn value_to_string(val: &Value, default: &str) -> String {
    match val {
        Value::Null => default.to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn deep_merge(base: &Value, override_val: &Value) -> Value {
    match (base, override_val) {
        (Value::Object(base_map), Value::Object(ov_map)) => {
            let mut out = base_map.clone();
            for (k, v) in ov_map {
                if let Some(existing) = out.get(k) {
                    if existing.is_object() && v.is_object() {
                        out.insert(k.clone(), deep_merge(existing, v));
                        continue;
                    }
                }
                out.insert(k.clone(), v.clone());
            }
            Value::Object(out)
        }
        (_, other) => other.clone(),
    }
}

impl BroadcastConfig {
    /// Merge an arbitrary JSON override onto defaults and apply Python-equivalent sanitization.
    pub fn from_json_override(override_val: &Value) -> Self {
        let default_cfg = Self::default();
        let default_json =
            serde_json::to_value(&default_cfg).expect("default config serializes cleanly");
        let merged = if override_val.is_object() {
            deep_merge(&default_json, override_val)
        } else {
            default_json
        };

        let mut cfg = Self::default();
        if let Some(obj) = merged.as_object() {
            if let Some(v) = obj.get("version").and_then(Value::as_u64) {
                cfg.version = v as u32;
            }
            if let Some(s) = obj.get("audio_backend").and_then(Value::as_str) {
                cfg.audio_backend = s.to_string();
            }
            if let Some(s) = obj.get("segmentation_backend").and_then(Value::as_str) {
                cfg.segmentation_backend = s.to_string();
            }
            if let Some(ui) = obj.get("ui").and_then(Value::as_object) {
                if let Some(v) = ui.get("show_gpu_indicator") {
                    cfg.ui.show_gpu_indicator = value_to_bool(v, true);
                }
                if let Some(v) = ui.get("start_minimized") {
                    cfg.ui.start_minimized = value_to_bool(v, false);
                }
            }
            if let Some(mic) = obj.get("microphone").and_then(Value::as_object) {
                if let Some(s) = mic.get("device").and_then(Value::as_str) {
                    cfg.microphone.device = s.to_string();
                }
                if let Some(s) = mic.get("virtual_name").and_then(Value::as_str) {
                    cfg.microphone.virtual_name = s.to_string();
                }
                if let Some(sr) = mic.get("sample_rate").and_then(Value::as_u64) {
                    cfg.microphone.sample_rate = sr as u32;
                }
                if let Some(v) = mic.get("highpass_enabled") {
                    cfg.microphone.highpass_enabled = value_to_bool(v, true);
                }
                if let Some(v) = mic.get("highpass_hz") {
                    cfg.microphone.highpass_hz =
                        clamp_float(value_to_f64(v, 85.0), 20.0, 300.0, 85.0) as i64;
                }
                if let Some(v) = mic.get("noise_gate_enabled") {
                    cfg.microphone.noise_gate_enabled = value_to_bool(v, true);
                }
                if let Some(v) = mic.get("noise_gate_db") {
                    cfg.microphone.noise_gate_db =
                        clamp_float(value_to_f64(v, -46.0), -80.0, -10.0, -46.0);
                }
                if let Some(v) = mic.get("auto_gain_enabled") {
                    cfg.microphone.auto_gain_enabled = value_to_bool(v, true);
                }
                if let Some(v) = mic.get("auto_gain_target_db") {
                    cfg.microphone.auto_gain_target_db =
                        clamp_float(value_to_f64(v, -16.0), -30.0, -3.0, -16.0);
                }
                if let Some(effs) = mic.get("effects").and_then(Value::as_object) {
                    Self::merge_mic_effects(&mut cfg.microphone.effects, effs);
                }
            }
            if let Some(spk) = obj.get("speaker").and_then(Value::as_object) {
                if let Some(s) = spk.get("device").and_then(Value::as_str) {
                    cfg.speaker.device = s.to_string();
                }
                if let Some(s) = spk.get("virtual_name").and_then(Value::as_str) {
                    cfg.speaker.virtual_name = s.to_string();
                }
                if let Some(sr) = spk.get("sample_rate").and_then(Value::as_u64) {
                    cfg.speaker.sample_rate = sr as u32;
                }
                if let Some(effs) = spk.get("effects").and_then(Value::as_object) {
                    Self::merge_speaker_effects(&mut cfg.speaker.effects, effs);
                }
            }
            if let Some(cam) = obj.get("camera").and_then(Value::as_object) {
                if let Some(s) = cam.get("device").and_then(Value::as_str) {
                    cfg.camera.device = s.to_string();
                }
                if let Some(s) = cam.get("output_device").and_then(Value::as_str) {
                    cfg.camera.output_device = s.to_string();
                }
                if let Some(s) = cam.get("virtual_name").and_then(Value::as_str) {
                    cfg.camera.virtual_name = s.to_string();
                }
                if let Some(w) = cam.get("width").and_then(Value::as_u64) {
                    cfg.camera.width = w as u32;
                }
                if let Some(h) = cam.get("height").and_then(Value::as_u64) {
                    cfg.camera.height = h as u32;
                }
                if let Some(f) = cam.get("fps").and_then(Value::as_u64) {
                    cfg.camera.fps = f as u32;
                }
                if let Some(v) = cam.get("prefer_mjpg") {
                    cfg.camera.prefer_mjpg = value_to_bool(v, true);
                }
                if let Some(v) = cam.get("mirror") {
                    cfg.camera.mirror = value_to_bool(v, false);
                }
                if let Some(v) = cam.get("split_preview") {
                    cfg.camera.split_preview = value_to_bool(v, false);
                }
                if let Some(v) = cam.get("split_position") {
                    cfg.camera.split_position =
                        clamp_float(value_to_f64(v, 0.5), 0.05, 0.95, 0.5);
                }
                if let Some(v) = cam.get("on_demand") {
                    cfg.camera.on_demand = if v.is_null() {
                        true
                    } else {
                        value_to_bool(v, true)
                    };
                }
                if let Some(v) = cam.get("idle_timeout_sec") {
                    cfg.camera.idle_timeout_sec =
                        clamp_float(value_to_f64(v, 3.0), 0.5, 60.0, 3.0);
                }
                if let Some(ctrls) = cam.get("v4l2_controls").and_then(Value::as_object) {
                    cfg.camera.v4l2_controls = ctrls
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                }
                if let Some(effs) = cam.get("effects").and_then(Value::as_object) {
                    Self::merge_camera_effects(&mut cfg.camera.effects, effs);
                }
            }
        }
        cfg.sanitize();
        cfg
    }

    fn merge_mic_effects(dst: &mut MicrophoneEffects, src: &Map<String, Value>) {
        if let Some(e) = src.get("noise_removal").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.noise_removal.enabled = value_to_bool(v, true);
            }
            if let Some(v) = e.get("strength") {
                dst.noise_removal.strength = clamp_float(value_to_f64(v, 0.85), 0.0, 1.0, 0.85);
            }
        }
        if let Some(e) = src.get("room_echo_removal").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.room_echo_removal.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("strength") {
                dst.room_echo_removal.strength = clamp_float(value_to_f64(v, 0.5), 0.0, 1.0, 0.5);
            }
        }
        if let Some(e) = src.get("studio_voice").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.studio_voice.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("micProfile") {
                dst.studio_voice.mic_profile = value_to_string(v, "default");
            }
        }
    }

    fn merge_speaker_effects(dst: &mut SpeakerEffects, src: &Map<String, Value>) {
        if let Some(e) = src.get("speaker_noise_removal").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.speaker_noise_removal.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("strength") {
                dst.speaker_noise_removal.strength =
                    clamp_float(value_to_f64(v, 0.75), 0.0, 1.0, 0.75);
            }
        }
        if let Some(e) = src.get("speaker_echo_removal").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.speaker_echo_removal.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("strength") {
                dst.speaker_echo_removal.strength =
                    clamp_float(value_to_f64(v, 0.5), 0.0, 1.0, 0.5);
            }
        }
    }

    fn merge_camera_effects(dst: &mut CameraEffects, src: &Map<String, Value>) {
        if let Some(e) = src.get("background_blur").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.background_blur.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("strength") {
                dst.background_blur.strength = clamp_float(value_to_f64(v, 0.7), 0.0, 1.0, 0.7);
            }
            if let Some(v) = e.get("mode") {
                dst.background_blur.mode = value_to_string(v, "quality");
            }
        }
        if let Some(e) = src.get("background_replace").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.background_replace.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("imagePath") {
                dst.background_replace.image_path = value_to_string(v, "");
            }
            if let Some(v) = e.get("preset") {
                dst.background_replace.preset = value_to_string(v, "studio_dark");
            }
        }
        if let Some(e) = src.get("background_remove").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.background_remove.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("chromaColor") {
                dst.background_remove.chroma_color = value_to_string(v, "#00ff00");
            }
        }
        if let Some(e) = src.get("auto_frame").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.auto_frame.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("zoom") {
                dst.auto_frame.zoom = clamp_float(value_to_f64(v, 0.35), 0.0, 1.0, 0.35);
            }
        }
        if let Some(e) = src.get("eye_contact").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.eye_contact.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("strength") {
                dst.eye_contact.strength = clamp_float(value_to_f64(v, 0.7), 0.0, 1.0, 0.7);
            }
            if let Some(v) = e.get("lookAway") {
                dst.eye_contact.look_away = value_to_bool(v, true);
            }
        }
        if let Some(e) = src.get("video_noise_removal").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.video_noise_removal.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("strength") {
                dst.video_noise_removal.strength = clamp_float(value_to_f64(v, 0.6), 0.0, 1.0, 0.6);
            }
        }
        if let Some(e) = src.get("virtual_key_light").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.virtual_key_light.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("warmth") {
                dst.virtual_key_light.warmth = clamp_float(value_to_f64(v, 0.5), 0.0, 1.0, 0.5);
            }
            if let Some(v) = e.get("intensity") {
                dst.virtual_key_light.intensity =
                    clamp_float(value_to_f64(v, 0.55), 0.0, 1.0, 0.55);
            }
        }
        if let Some(e) = src.get("vignette").and_then(Value::as_object) {
            if let Some(v) = e.get("enabled") {
                dst.vignette.enabled = value_to_bool(v, false);
            }
            if let Some(v) = e.get("strength") {
                dst.vignette.strength = clamp_float(value_to_f64(v, 0.4), 0.0, 1.0, 0.4);
            }
            if let Some(v) = e.get("faceTracking") {
                dst.vignette.face_tracking = value_to_bool(v, true);
            }
        }
    }

    pub fn load_from_path(path: &Path) -> Self {
        if let Ok(raw) = fs::read_to_string(path) {
            if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                return Self::from_json_override(&val);
            }
        }
        let mut cfg = Self::default();
        cfg.sanitize();
        cfg
    }

    pub fn save_to_path(&mut self, path: &Path) -> Result<(), ConfigError> {
        self.sanitize();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        let serialized = format!("{}\n", serde_json::to_string_pretty(self)?);
        fs::write(&tmp, serialized)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn sanitize(&mut self) {
        if !AUDIO_BACKENDS.contains(&self.audio_backend.as_str()) {
            self.audio_backend = "auto".to_string();
        }
        if !SEGMENTATION_BACKENDS.contains(&self.segmentation_backend.as_str()) {
            self.segmentation_backend = "auto".to_string();
        }

        self.microphone.highpass_hz =
            clamp_float(self.microphone.highpass_hz as f64, 20.0, 300.0, 85.0) as i64;
        self.microphone.noise_gate_db =
            clamp_float(self.microphone.noise_gate_db, -80.0, -10.0, -46.0);
        self.microphone.auto_gain_target_db =
            clamp_float(self.microphone.auto_gain_target_db, -30.0, -3.0, -16.0);

        self.camera.split_position = clamp_float(self.camera.split_position, 0.05, 0.95, 0.5);
        self.camera.idle_timeout_sec = clamp_float(self.camera.idle_timeout_sec, 0.5, 60.0, 3.0);

        self.microphone.effects.noise_removal.strength = clamp_float(
            self.microphone.effects.noise_removal.strength,
            0.0,
            1.0,
            0.85,
        );
        self.microphone.effects.room_echo_removal.strength = clamp_float(
            self.microphone.effects.room_echo_removal.strength,
            0.0,
            1.0,
            0.5,
        );
        if !STUDIO_VOICE_PROFILES.contains(&self.microphone.effects.studio_voice.mic_profile.as_str())
        {
            self.microphone.effects.studio_voice.mic_profile = "default".to_string();
        }

        self.speaker.effects.speaker_noise_removal.strength = clamp_float(
            self.speaker.effects.speaker_noise_removal.strength,
            0.0,
            1.0,
            0.75,
        );
        self.speaker.effects.speaker_echo_removal.strength = clamp_float(
            self.speaker.effects.speaker_echo_removal.strength,
            0.0,
            1.0,
            0.5,
        );

        self.camera.effects.background_blur.strength = clamp_float(
            self.camera.effects.background_blur.strength,
            0.0,
            1.0,
            0.7,
        );
        if !BLUR_MODES.contains(&self.camera.effects.background_blur.mode.as_str()) {
            self.camera.effects.background_blur.mode = "quality".to_string();
        }
        if !BG_REPLACE_PRESETS.contains(&self.camera.effects.background_replace.preset.as_str()) {
            self.camera.effects.background_replace.preset = "studio_dark".to_string();
        }
        self.camera.effects.auto_frame.zoom =
            clamp_float(self.camera.effects.auto_frame.zoom, 0.0, 1.0, 0.35);
        self.camera.effects.eye_contact.strength =
            clamp_float(self.camera.effects.eye_contact.strength, 0.0, 1.0, 0.7);
        self.camera.effects.video_noise_removal.strength = clamp_float(
            self.camera.effects.video_noise_removal.strength,
            0.0,
            1.0,
            0.6,
        );
        self.camera.effects.virtual_key_light.warmth = clamp_float(
            self.camera.effects.virtual_key_light.warmth,
            0.0,
            1.0,
            0.5,
        );
        self.camera.effects.virtual_key_light.intensity = clamp_float(
            self.camera.effects.virtual_key_light.intensity,
            0.0,
            1.0,
            0.55,
        );
        self.camera.effects.vignette.strength =
            clamp_float(self.camera.effects.vignette.strength, 0.0, 1.0, 0.4);
    }

    fn effect_enabled_mut(&mut self, effect_id: &str) -> Option<&mut bool> {
        match effect_id {
            "noise_removal" => Some(&mut self.microphone.effects.noise_removal.enabled),
            "room_echo_removal" => Some(&mut self.microphone.effects.room_echo_removal.enabled),
            "studio_voice" => Some(&mut self.microphone.effects.studio_voice.enabled),
            "speaker_noise_removal" => Some(&mut self.speaker.effects.speaker_noise_removal.enabled),
            "speaker_echo_removal" => Some(&mut self.speaker.effects.speaker_echo_removal.enabled),
            "background_blur" => Some(&mut self.camera.effects.background_blur.enabled),
            "background_replace" => Some(&mut self.camera.effects.background_replace.enabled),
            "background_remove" => Some(&mut self.camera.effects.background_remove.enabled),
            "auto_frame" => Some(&mut self.camera.effects.auto_frame.enabled),
            "eye_contact" => Some(&mut self.camera.effects.eye_contact.enabled),
            "video_noise_removal" => Some(&mut self.camera.effects.video_noise_removal.enabled),
            "virtual_key_light" => Some(&mut self.camera.effects.virtual_key_light.enabled),
            "vignette" => Some(&mut self.camera.effects.vignette.enabled),
            _ => None,
        }
    }

    /// Update an effect and enforce NVIDIA Broadcast v2.3.0 mutual-exclusion rules.
    pub fn set_effect(
        &mut self,
        effect_id: &str,
        enabled: Option<bool>,
        params: Option<&Map<String, Value>>,
        resolve_conflicts: bool,
    ) -> EffectApplyResult {
        if !ALL_EFFECT_IDS.contains(&effect_id) {
            return EffectApplyResult {
                applied: Vec::new(),
                disabled_conflicts: BTreeMap::new(),
                errors: vec![format!("Unknown effectId: {effect_id}")],
            };
        }

        if let Some(p) = params {
            match effect_id {
                "noise_removal" => {
                    if let Some(v) = p.get("strength") {
                        self.microphone.effects.noise_removal.strength =
                            clamp_float(value_to_f64(v, 0.85), 0.0, 1.0, 0.85);
                    }
                }
                "room_echo_removal" => {
                    if let Some(v) = p.get("strength") {
                        self.microphone.effects.room_echo_removal.strength =
                            clamp_float(value_to_f64(v, 0.5), 0.0, 1.0, 0.5);
                    }
                }
                "studio_voice" => {
                    if let Some(s) = p.get("micProfile").and_then(Value::as_str) {
                        if STUDIO_VOICE_PROFILES.contains(&s) {
                            self.microphone.effects.studio_voice.mic_profile = s.to_string();
                        }
                    }
                }
                "speaker_noise_removal" => {
                    if let Some(v) = p.get("strength") {
                        self.speaker.effects.speaker_noise_removal.strength =
                            clamp_float(value_to_f64(v, 0.75), 0.0, 1.0, 0.75);
                    }
                }
                "speaker_echo_removal" => {
                    if let Some(v) = p.get("strength") {
                        self.speaker.effects.speaker_echo_removal.strength =
                            clamp_float(value_to_f64(v, 0.5), 0.0, 1.0, 0.5);
                    }
                }
                "background_blur" => {
                    if let Some(v) = p.get("strength") {
                        self.camera.effects.background_blur.strength =
                            clamp_float(value_to_f64(v, 0.7), 0.0, 1.0, 0.7);
                    }
                    if let Some(s) = p.get("mode").and_then(Value::as_str) {
                        if BLUR_MODES.contains(&s) {
                            self.camera.effects.background_blur.mode = s.to_string();
                        }
                    }
                }
                "background_replace" => {
                    if let Some(v) = p.get("imagePath") {
                        self.camera.effects.background_replace.image_path =
                            value_to_string(v, "");
                    }
                    if let Some(s) = p.get("preset").and_then(Value::as_str) {
                        if BG_REPLACE_PRESETS.contains(&s) {
                            self.camera.effects.background_replace.preset = s.to_string();
                        }
                    }
                }
                "background_remove" => {
                    if let Some(v) = p.get("chromaColor") {
                        self.camera.effects.background_remove.chroma_color =
                            value_to_string(v, "#00ff00");
                    }
                }
                "auto_frame" => {
                    if let Some(v) = p.get("zoom") {
                        self.camera.effects.auto_frame.zoom =
                            clamp_float(value_to_f64(v, 0.35), 0.0, 1.0, 0.35);
                    }
                }
                "eye_contact" => {
                    if let Some(v) = p.get("strength") {
                        self.camera.effects.eye_contact.strength =
                            clamp_float(value_to_f64(v, 0.7), 0.0, 1.0, 0.7);
                    }
                    if let Some(v) = p.get("lookAway") {
                        self.camera.effects.eye_contact.look_away = value_to_bool(v, true);
                    }
                }
                "video_noise_removal" => {
                    if let Some(v) = p.get("strength") {
                        self.camera.effects.video_noise_removal.strength =
                            clamp_float(value_to_f64(v, 0.6), 0.0, 1.0, 0.6);
                    }
                }
                "virtual_key_light" => {
                    if let Some(v) = p.get("warmth") {
                        self.camera.effects.virtual_key_light.warmth =
                            clamp_float(value_to_f64(v, 0.5), 0.0, 1.0, 0.5);
                    }
                    if let Some(v) = p.get("intensity") {
                        self.camera.effects.virtual_key_light.intensity =
                            clamp_float(value_to_f64(v, 0.55), 0.0, 1.0, 0.55);
                    }
                }
                "vignette" => {
                    if let Some(v) = p.get("strength") {
                        self.camera.effects.vignette.strength =
                            clamp_float(value_to_f64(v, 0.4), 0.0, 1.0, 0.4);
                    }
                    if let Some(v) = p.get("faceTracking") {
                        self.camera.effects.vignette.face_tracking = value_to_bool(v, true);
                    }
                }
                _ => {}
            }
        }

        if let Some(en) = enabled {
            if let Some(slot) = self.effect_enabled_mut(effect_id) {
                *slot = en;
            }
        }

        let is_now_enabled = self
            .effect_enabled_mut(effect_id)
            .map(|b| *b)
            .unwrap_or(false);

        let mut disabled_conflicts = BTreeMap::new();
        if is_now_enabled && resolve_conflicts {
            let mut turned_off = Vec::new();
            for group in INCOMPATIBLE_GROUPS {
                if group.contains(&effect_id) {
                    for &other in *group {
                        if other == effect_id {
                            continue;
                        }
                        if let Some(other_slot) = self.effect_enabled_mut(other) {
                            if *other_slot {
                                *other_slot = false;
                                turned_off.push(other.to_string());
                            }
                        }
                    }
                }
            }
            if !turned_off.is_empty() {
                disabled_conflicts.insert(effect_id.to_string(), turned_off);
            }
        }

        self.sanitize();
        EffectApplyResult {
            applied: vec![effect_id.to_string()],
            disabled_conflicts,
            errors: Vec::new(),
        }
    }

    pub fn get_broadcast_state(&self) -> BroadcastState {
        BroadcastState {
            microphone: self.microphone.effects.clone(),
            speaker: self.speaker.effects.clone(),
            camera: self.camera.effects.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_ORACLE_JSON: &str = include_str!("../../../fixtures/config_oracle.json");

    fn oracle() -> Value {
        serde_json::from_str(CONFIG_ORACLE_JSON).expect("valid config_oracle.json")
    }

    #[test]
    fn default_config_matches_python_oracle() {
        let o = oracle();
        let expected = &o["default_config"];
        let actual = serde_json::to_value(BroadcastConfig::default()).unwrap();
        assert_eq!(actual, *expected);
        assert_eq!(actual, o["loaded_default_config"]);
    }

    #[test]
    fn enums_and_incompatible_groups_match_python_oracle() {
        let o = oracle();
        assert_eq!(
            serde_json::to_value(AUDIO_BACKENDS).unwrap(),
            o["enums"]["audio_backends"]
        );
        assert_eq!(
            serde_json::to_value(SEGMENTATION_BACKENDS).unwrap(),
            o["enums"]["segmentation_backends"]
        );
        assert_eq!(
            serde_json::to_value(STUDIO_VOICE_PROFILES).unwrap(),
            o["enums"]["studio_voice_profiles"]
        );
        assert_eq!(
            serde_json::to_value(BLUR_MODES).unwrap(),
            o["enums"]["blur_modes"]
        );
        assert_eq!(
            serde_json::to_value(BG_REPLACE_PRESETS).unwrap(),
            o["enums"]["bg_replace_presets"]
        );
        assert_eq!(
            serde_json::to_value(INCOMPATIBLE_GROUPS).unwrap(),
            o["incompatible_groups"]
        );

        let catalog_keys: Vec<&str> = o["effect_catalog"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        for eid in ALL_EFFECT_IDS {
            assert!(
                catalog_keys.contains(eid),
                "missing effect in python catalog: {eid}"
            );
        }
        assert_eq!(ALL_EFFECT_IDS.len(), catalog_keys.len());
    }

    #[test]
    fn sanitization_matches_python_oracle() {
        let o = oracle();
        let dirty_input = &o["sanitization_case"]["input"];
        let expected = &o["sanitization_case"]["expected"];

        let sanitized = BroadcastConfig::from_json_override(dirty_input);
        let actual = serde_json::to_value(&sanitized).unwrap();
        assert_eq!(actual, *expected);
    }

    #[test]
    fn conflict_resolution_matches_python_oracle() {
        let o = oracle();
        let cases = o["conflict_cases"].as_array().unwrap();
        assert!(!cases.is_empty());

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let mut cfg = BroadcastConfig::default();

            for step in case["setup"].as_array().unwrap() {
                let eid = step["effect_id"].as_str().unwrap();
                let enabled = step.get("enabled").and_then(Value::as_bool);
                let params = step.get("params").and_then(Value::as_object);
                cfg.set_effect(eid, enabled, params, true);
            }

            let action = &case["action"];
            let eid = action["effect_id"].as_str().unwrap();
            let enabled = action.get("enabled").and_then(Value::as_bool);
            let params = action.get("params").and_then(Value::as_object);
            let res = cfg.set_effect(eid, enabled, params, true);

            assert_eq!(
                serde_json::to_value(&res).unwrap(),
                case["result"],
                "conflict result mismatch for case {name}"
            );
            assert_eq!(
                serde_json::to_value(cfg.get_broadcast_state()).unwrap(),
                case["final_broadcast_state"],
                "final_broadcast_state mismatch for case {name}"
            );
        }
    }

    #[test]
    fn atomic_save_and_reload_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "nvb-core-test-{}",
            std::process::id()
        ));
        let path = dir.join("config.json");
        let mut cfg = BroadcastConfig::default();
        cfg.set_effect("studio_voice", Some(true), None, true);
        cfg.save_to_path(&path).unwrap();

        let reloaded = BroadcastConfig::load_from_path(&path);
        assert_eq!(cfg, reloaded);
        assert!(reloaded.microphone.effects.studio_voice.enabled);
        assert!(!reloaded.microphone.effects.noise_removal.enabled);

        let _ = fs::remove_dir_all(&dir);
    }
}
