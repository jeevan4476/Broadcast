//! PipeWire virtual device manager and real-time Rust audio bridge engine.
//!
//! Uses typed `PhysicalDevice` / `VirtualDevice` invariants from `nvb-core` plus
//! isolated `pw-cat` (`node.autoconnect = false`, `node.dont-reconnect = true`)
//! and explicit `pw-link` routing to guarantee zero acoustic/digital feedback loops.

use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use nvb_core::config::BroadcastConfig;
use nvb_core::devices::{list_audio_devices, run_cmd};
use serde_json::{json, Value};

use crate::{
    build_isolated_pw_cat_args, compute_waveform_peaks, generate_sample_clip, is_pw_link_active,
    is_virtual_broadcast_node, parse_pw_links, select_port_pairs, write_wav_mono16,
    AudioMeterReading, AudioProcessConfig, AudioProcessor, PhysicalDevice, PwCatMode,
    DEFAULT_LATENCY_MS, RNNOISE_FRAME_SAMPLES, SAMPLE_RATE, VIRTUAL_MIC_SINK_NAME,
    VIRTUAL_MIC_SOURCE_NAME, VIRTUAL_SPEAKER_SINK_NAME,
};

const MIC_CAPTURE_NODE: &str = "nv_broadcast_mic_capture";
const MIC_PLAYBACK_NODE: &str = "nv_broadcast_mic_playback";
const SPK_CAPTURE_NODE: &str = "nv_broadcast_spk_capture";
const SPK_PLAYBACK_NODE: &str = "nv_broadcast_spk_playback";

#[derive(Debug, Default)]
pub struct PipeWireVirtualManager {
    mic_sink_mod: Option<String>,
    mic_src_mod: Option<String>,
    spk_sink_mod: Option<String>,
}

impl PipeWireVirtualManager {
    pub fn find_existing_virtual_nodes() -> Value {
        let (_, src_out, _) = run_cmd("pactl", &["list", "short", "sources"]);
        let (_, sink_out, _) = run_cmd("pactl", &["list", "short", "sinks"]);
        json!({
            "virtual_mic_source": src_out.contains(VIRTUAL_MIC_SOURCE_NAME),
            "virtual_mic_sink": sink_out.contains(VIRTUAL_MIC_SINK_NAME),
            "virtual_speaker_sink": sink_out.contains(VIRTUAL_SPEAKER_SINK_NAME),
        })
    }

    pub fn ensure_virtual_mic(&mut self) -> Result<(), String> {
        let nodes = Self::find_existing_virtual_nodes();
        if !nodes["virtual_mic_sink"].as_bool().unwrap_or(false) {
            let (ok, out, err) = run_cmd(
                "pactl",
                &[
                    "load-module",
                    "module-null-sink",
                    &format!("sink_name={VIRTUAL_MIC_SINK_NAME}"),
                    "sink_properties=device.description=NVBroadcast_Mic_Bus",
                    "rate=48000",
                    "channels=1",
                ],
            );
            if !ok {
                return Err(format!("Failed to create virtual mic bus: {err}"));
            }
            self.mic_sink_mod = Some(out);
        }
        if !nodes["virtual_mic_source"].as_bool().unwrap_or(false) {
            let (ok, out, err) = run_cmd(
                "pactl",
                &[
                    "load-module",
                    "module-remap-source",
                    &format!("master={VIRTUAL_MIC_SINK_NAME}.monitor"),
                    &format!("source_name={VIRTUAL_MIC_SOURCE_NAME}"),
                    "source_properties=device.description=\"NVIDIA Broadcast (Microphone)\"",
                    "rate=48000",
                    "channels=1",
                ],
            );
            if !ok {
                return Err(format!("Failed to create virtual mic source: {err}"));
            }
            self.mic_src_mod = Some(out);
        }
        Ok(())
    }

    pub fn ensure_virtual_speakers(&mut self) -> Result<(), String> {
        let nodes = Self::find_existing_virtual_nodes();
        if !nodes["virtual_speaker_sink"].as_bool().unwrap_or(false) {
            let (ok, out, err) = run_cmd(
                "pactl",
                &[
                    "load-module",
                    "module-null-sink",
                    &format!("sink_name={VIRTUAL_SPEAKER_SINK_NAME}"),
                    "sink_properties=device.description=\"NVIDIA Broadcast (Speakers)\"",
                    "rate=48000",
                    "channels=2",
                ],
            );
            if !ok {
                return Err(format!("Failed to create virtual speaker sink: {err}"));
            }
            self.spk_sink_mod = Some(out);
        }
        Ok(())
    }

    pub fn link_nodes(src_node: &str, dst_node: &str) -> Result<(), String> {
        if src_node.is_empty() || dst_node.is_empty() {
            return Err("Empty PipeWire node name".to_string());
        }
        if src_node == dst_node {
            return Err(format!("Refusing self-loop link on {src_node}"));
        }
        if is_virtual_broadcast_node(src_node) && is_virtual_broadcast_node(dst_node) {
            return Err(format!(
                "Refusing virtual-to-virtual feedback loop: {src_node} -> {dst_node}"
            ));
        }

        let (_, out_raw, _) = run_cmd("pw-link", &["-o"]);
        let (_, in_raw, _) = run_cmd("pw-link", &["-i"]);
        let out_ports: Vec<String> = out_raw.lines().map(|l| l.trim().to_string()).collect();
        let in_ports: Vec<String> = in_raw.lines().map(|l| l.trim().to_string()).collect();
        let pairs = select_port_pairs(src_node, dst_node, &out_ports, &in_ports, "", "");
        if pairs.is_empty() {
            return Err(format!("No matching PipeWire ports for {src_node} -> {dst_node}"));
        }
        for (sp, dp) in pairs {
            let _ = run_cmd("pw-link", &[&sp, &dp]);
        }
        Ok(())
    }

    pub fn set_virtual_mic_as_default(enable: bool) -> (bool, String) {
        if enable {
            let (ok, _, err) = run_cmd("pactl", &["set-default-source", VIRTUAL_MIC_SOURCE_NAME]);
            if ok {
                (true, "Set NVIDIA Broadcast (Microphone) as default".to_string())
            } else {
                (false, format!("Failed setting default mic: {err}"))
            }
        } else {
            let (mics, _, _, _) = list_audio_devices(false, false);
            if let Some(phys) = mics.first() {
                let (ok, _, err) = run_cmd("pactl", &["set-default-source", &phys.name]);
                if ok {
                    (true, format!("Restored default microphone to {}", phys.description))
                } else {
                    (false, format!("Failed restoring default mic: {err}"))
                }
            } else {
                (false, "No physical microphone found".to_string())
            }
        }
    }

    pub fn set_virtual_speakers_as_default(enable: bool) -> (bool, String) {
        if enable {
            let (ok, _, err) = run_cmd("pactl", &["set-default-sink", VIRTUAL_SPEAKER_SINK_NAME]);
            if ok {
                (true, "Set NVIDIA Broadcast (Speakers) as default".to_string())
            } else {
                (false, format!("Failed setting default speaker: {err}"))
            }
        } else {
            let (_, spks, _, _) = list_audio_devices(false, false);
            if let Some(phys) = spks.first() {
                let (ok, _, err) = run_cmd("pactl", &["set-default-sink", &phys.name]);
                if ok {
                    (true, format!("Restored default speaker to {}", phys.description))
                } else {
                    (false, format!("Failed restoring default speaker: {err}"))
                }
            } else {
                (false, "No physical speaker found".to_string())
            }
        }
    }
}

fn spawn_pw_cat(mode: PwCatMode, node_name: &str, channels: u32) -> Result<Child, String> {
    let args = build_isolated_pw_cat_args(mode, node_name, SAMPLE_RATE, channels, DEFAULT_LATENCY_MS);
    let mut cmd = Command::new(&args[0]);
    cmd.args(&args[1..]).stderr(Stdio::null());
    match mode {
        PwCatMode::Record => {
            cmd.stdin(Stdio::null()).stdout(Stdio::piped());
        }
        PwCatMode::Playback => {
            cmd.stdin(Stdio::piped()).stdout(Stdio::null());
        }
    }
    cmd.spawn()
        .map_err(|e| format!("Failed to spawn pw-cat ({node_name}): {e}"))
}

fn mic_config_from_broadcast(cfg: &BroadcastConfig) -> AudioProcessConfig {
    let mic = &cfg.microphone;
    AudioProcessConfig {
        highpass_enabled: mic.highpass_enabled,
        highpass_hz: mic.highpass_hz as f64,
        noise_removal_enabled: mic.effects.noise_removal.enabled,
        noise_removal_strength: mic.effects.noise_removal.strength,
        room_echo_removal_enabled: mic.effects.room_echo_removal.enabled,
        room_echo_removal_strength: mic.effects.room_echo_removal.strength,
        studio_voice_enabled: mic.effects.studio_voice.enabled,
        studio_voice_profile: mic.effects.studio_voice.mic_profile.clone(),
        noise_gate_enabled: mic.noise_gate_enabled,
        noise_gate_db: mic.noise_gate_db,
        auto_gain_enabled: mic.auto_gain_enabled,
        auto_gain_target_db: mic.auto_gain_target_db,
    }
}

fn spk_config_from_broadcast(cfg: &BroadcastConfig) -> AudioProcessConfig {
    let spk = &cfg.speaker;
    AudioProcessConfig {
        highpass_enabled: false,
        highpass_hz: 85.0,
        noise_removal_enabled: spk.effects.speaker_noise_removal.enabled,
        noise_removal_strength: spk.effects.speaker_noise_removal.strength,
        room_echo_removal_enabled: spk.effects.speaker_echo_removal.enabled,
        room_echo_removal_strength: spk.effects.speaker_echo_removal.strength,
        studio_voice_enabled: false,
        studio_voice_profile: "default".to_string(),
        noise_gate_enabled: false,
        noise_gate_db: -46.0,
        auto_gain_enabled: false,
        auto_gain_target_db: -16.0,
    }
}

/// Real-time audio engine owning virtual microphone and speaker processing bridges.
pub struct AudioEngine {
    config: Arc<RwLock<BroadcastConfig>>,
    pw: Mutex<PipeWireVirtualManager>,
    mic_running: Arc<AtomicBool>,
    spk_running: Arc<AtomicBool>,
    mic_meter: Arc<RwLock<AudioMeterReading>>,
    spk_meter: Arc<RwLock<AudioMeterReading>>,
    mic_thread: Mutex<Option<JoinHandle<()>>>,
    spk_thread: Mutex<Option<JoinHandle<()>>>,
    playback_child: Mutex<Option<Child>>,
}

impl AudioEngine {
    pub fn new(config: Arc<RwLock<BroadcastConfig>>) -> Self {
        Self {
            config,
            pw: Mutex::new(PipeWireVirtualManager::default()),
            mic_running: Arc::new(AtomicBool::new(false)),
            spk_running: Arc::new(AtomicBool::new(false)),
            mic_meter: Arc::new(RwLock::new(AudioMeterReading {
                backend: "nnnoiseless".to_string(),
                ..AudioMeterReading::default()
            })),
            spk_meter: Arc::new(RwLock::new(AudioMeterReading {
                backend: "nnnoiseless".to_string(),
                ..AudioMeterReading::default()
            })),
            mic_thread: Mutex::new(None),
            spk_thread: Mutex::new(None),
            playback_child: Mutex::new(None),
        }
    }

    pub fn mic_running(&self) -> bool {
        self.mic_running.load(Ordering::Relaxed)
    }

    pub fn speaker_running(&self) -> bool {
        self.spk_running.load(Ordering::Relaxed)
    }

    pub fn mic_meter(&self) -> AudioMeterReading {
        if !self.mic_running() {
            return AudioMeterReading {
                backend: "nnnoiseless".to_string(),
                ..AudioMeterReading::default()
            };
        }
        self.mic_meter.read().map(|m| m.clone()).unwrap_or_default()
    }

    pub fn speaker_meter(&self) -> AudioMeterReading {
        if !self.speaker_running() {
            return AudioMeterReading {
                backend: "nnnoiseless".to_string(),
                ..AudioMeterReading::default()
            };
        }
        self.spk_meter.read().map(|m| m.clone()).unwrap_or_default()
    }

    fn resolve_physical_mic(&self) -> Result<PhysicalDevice, String> {
        let configured = self
            .config
            .read()
            .map(|c| c.microphone.device.clone())
            .unwrap_or_else(|_| "auto".to_string());
        if configured != "auto" && !configured.is_empty() {
            return PhysicalDevice::try_new(configured).map_err(|e| e.to_string());
        }
        let (mics, _, _, _) = list_audio_devices(false, false);
        let first = mics
            .first()
            .ok_or_else(|| "No physical microphone detected".to_string())?;
        PhysicalDevice::try_new(first.name.clone()).map_err(|e| e.to_string())
    }

    fn resolve_physical_speaker(&self) -> Result<PhysicalDevice, String> {
        let configured = self
            .config
            .read()
            .map(|c| c.speaker.device.clone())
            .unwrap_or_else(|_| "auto".to_string());
        if configured != "auto" && !configured.is_empty() {
            return PhysicalDevice::try_new(configured).map_err(|e| e.to_string());
        }
        let (_, spks, _, _) = list_audio_devices(false, false);
        let first = spks
            .first()
            .ok_or_else(|| "No physical speaker detected".to_string())?;
        PhysicalDevice::try_new(first.name.clone()).map_err(|e| e.to_string())
    }

    pub fn start_microphone(&self) -> (bool, String) {
        if self.mic_running() {
            return (true, "Virtual microphone already running".to_string());
        }
        let phys = match self.resolve_physical_mic() {
            Ok(p) => p,
            Err(e) => return (false, e),
        };
        if let Ok(mut pw) = self.pw.lock() {
            if let Err(e) = pw.ensure_virtual_mic() {
                return (false, e);
            }
        }

        let running = Arc::clone(&self.mic_running);
        let cfg_arc = Arc::clone(&self.config);
        let meter_arc = Arc::clone(&self.mic_meter);
        let phys_name = phys.as_str().to_string();
        running.store(true, Ordering::SeqCst);

        let handle = thread::spawn(move || {
            let mut proc = AudioProcessor::new(SAMPLE_RATE);
            let mut rec = match spawn_pw_cat(PwCatMode::Record, MIC_CAPTURE_NODE, 1) {
                Ok(c) => c,
                Err(_) => {
                    running.store(false, Ordering::SeqCst);
                    return;
                }
            };
            let mut play = match spawn_pw_cat(PwCatMode::Playback, MIC_PLAYBACK_NODE, 1) {
                Ok(c) => c,
                Err(_) => {
                    let _ = rec.kill();
                    let _ = rec.wait();
                    running.store(false, Ordering::SeqCst);
                    return;
                }
            };

            thread::sleep(Duration::from_millis(120));
            let _ = PipeWireVirtualManager::link_nodes(&phys_name, MIC_CAPTURE_NODE);
            let _ = PipeWireVirtualManager::link_nodes(MIC_PLAYBACK_NODE, VIRTUAL_MIC_SINK_NAME);

            let mut buf = [0u8; RNNOISE_FRAME_SAMPLES * 2];
            let mut last_link_check = Instant::now();

            if let (Some(mut stdout), Some(mut stdin)) = (rec.stdout.take(), play.stdin.take()) {
                while running.load(Ordering::Relaxed) {
                    if stdout.read_exact(&mut buf).is_err() {
                        break;
                    }
                    let dsp_cfg = cfg_arc
                        .read()
                        .map(|c| mic_config_from_broadcast(&c))
                        .unwrap_or_default();
                    let out = proc.process_pcm16(&buf, &dsp_cfg);
                    if stdin.write_all(&out).is_err() {
                        break;
                    }
                    if let Ok(mut m) = meter_arc.write() {
                        *m = proc.last_meter.clone();
                    }
                    if last_link_check.elapsed() >= Duration::from_secs(3) {
                        last_link_check = Instant::now();
                        let (_, raw_links, _) = run_cmd("pw-link", &["-l"]);
                        let links = parse_pw_links(&raw_links);
                        if !is_pw_link_active(&phys_name, MIC_CAPTURE_NODE, &links) {
                            let _ = PipeWireVirtualManager::link_nodes(&phys_name, MIC_CAPTURE_NODE);
                        }
                        if !is_pw_link_active(MIC_PLAYBACK_NODE, VIRTUAL_MIC_SINK_NAME, &links) {
                            let _ = PipeWireVirtualManager::link_nodes(MIC_PLAYBACK_NODE, VIRTUAL_MIC_SINK_NAME);
                        }
                    }
                }
            }

            let _ = rec.kill();
            let _ = rec.wait();
            let _ = play.kill();
            let _ = play.wait();
            running.store(false, Ordering::SeqCst);
        });

        if let Ok(mut slot) = self.mic_thread.lock() {
            *slot = Some(handle);
        }
        (
            true,
            format!("Started virtual microphone from {} (nnnoiseless)", phys.as_str()),
        )
    }

    pub fn stop_microphone(&self) {
        self.mic_running.store(false, Ordering::SeqCst);
        if let Ok(mut slot) = self.mic_thread.lock() {
            if let Some(h) = slot.take() {
                let _ = h.join();
            }
        }
    }

    pub fn start_speakers(&self) -> (bool, String) {
        if self.speaker_running() {
            return (true, "Virtual speakers already running".to_string());
        }
        let phys = match self.resolve_physical_speaker() {
            Ok(p) => p,
            Err(e) => return (false, e),
        };
        if let Ok(mut pw) = self.pw.lock() {
            if let Err(e) = pw.ensure_virtual_speakers() {
                return (false, e);
            }
        }

        let running = Arc::clone(&self.spk_running);
        let cfg_arc = Arc::clone(&self.config);
        let meter_arc = Arc::clone(&self.spk_meter);
        let phys_name = phys.as_str().to_string();
        running.store(true, Ordering::SeqCst);

        let handle = thread::spawn(move || {
            let mut proc = AudioProcessor::new(SAMPLE_RATE);
            let mut rec = match spawn_pw_cat(PwCatMode::Record, SPK_CAPTURE_NODE, 1) {
                Ok(c) => c,
                Err(_) => {
                    running.store(false, Ordering::SeqCst);
                    return;
                }
            };
            let mut play = match spawn_pw_cat(PwCatMode::Playback, SPK_PLAYBACK_NODE, 1) {
                Ok(c) => c,
                Err(_) => {
                    let _ = rec.kill();
                    let _ = rec.wait();
                    running.store(false, Ordering::SeqCst);
                    return;
                }
            };

            thread::sleep(Duration::from_millis(120));
            let _ = PipeWireVirtualManager::link_nodes(VIRTUAL_SPEAKER_SINK_NAME, SPK_CAPTURE_NODE);
            let _ = PipeWireVirtualManager::link_nodes(SPK_PLAYBACK_NODE, &phys_name);

            let mut buf = [0u8; RNNOISE_FRAME_SAMPLES * 2];
            if let (Some(mut stdout), Some(mut stdin)) = (rec.stdout.take(), play.stdin.take()) {
                while running.load(Ordering::Relaxed) {
                    if stdout.read_exact(&mut buf).is_err() {
                        break;
                    }
                    let dsp_cfg = cfg_arc
                        .read()
                        .map(|c| spk_config_from_broadcast(&c))
                        .unwrap_or_default();
                    let out = proc.process_pcm16(&buf, &dsp_cfg);
                    if stdin.write_all(&out).is_err() {
                        break;
                    }
                    if let Ok(mut m) = meter_arc.write() {
                        *m = proc.last_meter.clone();
                    }
                }
            }

            let _ = rec.kill();
            let _ = rec.wait();
            let _ = play.kill();
            let _ = play.wait();
            running.store(false, Ordering::SeqCst);
        });

        if let Ok(mut slot) = self.spk_thread.lock() {
            *slot = Some(handle);
        }
        (
            true,
            format!("Started virtual speakers -> {} (nnnoiseless)", phys.as_str()),
        )
    }

    pub fn stop_speakers(&self) {
        self.spk_running.store(false, Ordering::SeqCst);
        if let Ok(mut slot) = self.spk_thread.lock() {
            if let Some(h) = slot.take() {
                let _ = h.join();
            }
        }
    }

    pub fn build_speaker_sample(&self, kind: &str) -> Result<Value, String> {
        let raw = generate_sample_clip(kind, SAMPLE_RATE, 3.0);
        let dsp_cfg = self
            .config
            .read()
            .map(|c| spk_config_from_broadcast(&c))
            .unwrap_or_default();
        let mut proc = AudioProcessor::new(SAMPLE_RATE);
        let mut processed = Vec::with_capacity(raw.len());
        for chunk in raw.chunks(RNNOISE_FRAME_SAMPLES * 2) {
            processed.extend_from_slice(&proc.process_pcm16(chunk, &dsp_cfg));
        }
        let dir = std::env::temp_dir().join("nv-broadcast-samples");
        let raw_path = dir.join(format!("sample_{kind}_raw.wav"));
        let proc_path = dir.join(format!("sample_{kind}_processed.wav"));
        write_wav_mono16(&raw_path, &raw, SAMPLE_RATE).map_err(|e| e.to_string())?;
        write_wav_mono16(&proc_path, &processed, SAMPLE_RATE).map_err(|e| e.to_string())?;

        Ok(json!({
            "raw_path": raw_path.to_string_lossy(),
            "processed_path": proc_path.to_string_lossy(),
            "duration_sec": 3.0,
            "raw_peaks": compute_waveform_peaks(&raw, 120),
            "processed_peaks": compute_waveform_peaks(&processed, 120),
            "backend": proc.backend_name,
        }))
    }

    pub fn play_wav(&self, path: &str, target_sink: Option<&str>) -> (bool, String) {
        self.stop_playback();
        let sink = match target_sink {
            Some(s) if !s.is_empty() && !is_virtual_broadcast_node(s) => s.to_string(),
            _ => match self.resolve_physical_speaker() {
                Ok(p) => p.as_str().to_string(),
                Err(e) => return (false, e),
            },
        };
        let child = Command::new("pw-play")
            .args(["--target", &sink, path])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        match child {
            Ok(c) => {
                if let Ok(mut slot) = self.playback_child.lock() {
                    *slot = Some(c);
                }
                (true, format!("Playing {path} on {sink}"))
            }
            Err(e) => (false, format!("Failed to launch pw-play: {e}")),
        }
    }

    pub fn stop_playback(&self) {
        if let Ok(mut slot) = self.playback_child.lock() {
            if let Some(mut c) = slot.take() {
                let _ = c.kill();
                let _ = c.wait();
            }
        }
    }

    pub fn stop_all(&self) {
        self.stop_playback();
        self.stop_microphone();
        self.stop_speakers();
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        self.stop_all();
    }
}
