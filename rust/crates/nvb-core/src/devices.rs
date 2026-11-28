//! Linux audio, V4L2 camera, Bluetooth, and GPU capability probing.
//!
//! Port of `nv_broadcast/devices.py` using lightweight sysfs reads and bounded
//! CLI probes so the idle daemon stays fast and avoids heavy Python/GPU imports.

use crate::api_contract::{DeviceDefaults, DevicesPayload};
use crate::models::get_models_status;
use crate::routing::{VIRTUAL_MIC_SOURCE_NAME, VIRTUAL_SPEAKER_SINK_NAME};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub description: String,
    pub kind: String,
    pub sample_spec: String,
    pub state: String,
    pub is_monitor: bool,
    pub is_virtual: bool,
    pub is_default: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CameraDeviceSummary {
    pub path: String,
    pub driver: String,
    pub card: String,
    pub bus_info: String,
    pub is_capture: bool,
    pub is_output: bool,
    pub is_metadata: bool,
    pub is_loopback: bool,
    pub formats: Vec<String>,
    pub modes: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BluetoothCard {
    pub name: String,
    pub description: String,
    pub active_profile: String,
    pub profiles: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub available: bool,
    pub name: String,
    pub driver_version: String,
    pub vram_mib: u64,
    pub compute_capability: String,
    pub cuda_toolkit: String,
    pub rtx_capable: bool,
}

pub fn command_exists(bin: &str) -> bool {
    let Some(path_var) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path_var).any(|dir| {
        let p = dir.join(bin);
        p.is_file()
    })
}

pub fn run_cmd(program: &str, args: &[&str]) -> (bool, String, String) {
    match Command::new(program).args(args).output() {
        Ok(out) => (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).trim().to_string(),
            String::from_utf8_lossy(&out.stderr).trim().to_string(),
        ),
        Err(err) => (false, String::new(), err.to_string()),
    }
}

pub fn pretty_audio_name(node_name: &str) -> String {
    let low = node_name.to_ascii_lowercase();
    if node_name.contains("nv_broadcast_mic") {
        return "NVIDIA Broadcast (Microphone)".to_string();
    }
    if node_name.contains("nv_broadcast_speaker") {
        return "NVIDIA Broadcast (Speakers)".to_string();
    }
    if node_name.contains("HiFi__Mic") {
        return "Built-in Laptop Microphone".to_string();
    }
    if node_name.contains("HiFi__Headphones") {
        return "Wired Headphones (3.5mm)".to_string();
    }
    if node_name.contains("HiFi__Speaker") {
        return "Built-in Laptop Speakers".to_string();
    }
    if let Some((_, rest)) = node_name.split_once("HiFi__HDMI") {
        let tail = rest.split("__").next().unwrap_or("");
        return format!("HDMI Audio (HDMI{tail})");
    }
    if low.contains("bluez") {
        let tail = node_name.rsplit('.').next().unwrap_or(node_name);
        return format!("Bluetooth Audio ({tail})");
    }
    if low.contains("usb") {
        let mid = node_name.split('.').nth(1).unwrap_or(node_name);
        return format!("USB Audio ({mid})");
    }
    node_name.to_string()
}

pub fn parse_pactl_short(output: &str, kind: &str, default_name: &str) -> Vec<AudioDevice> {
    let mut devices = Vec::new();
    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            parts = line.split_whitespace().collect();
        }
        if parts.len() < 2 {
            continue;
        }
        let dev_id = parts[0].to_string();
        let name = parts[1].to_string();
        let sample_spec = parts.get(3).copied().unwrap_or("").to_string();
        let state = parts.get(4).copied().unwrap_or("").to_string();
        let low = name.to_ascii_lowercase();
        let is_monitor = name.ends_with(".monitor");
        let is_virtual = low.contains("nv_broadcast")
            || low.contains("nvbroadcast")
            || low.contains("nvidia_broadcast")
            || low.contains("effect_output");

        devices.push(AudioDevice {
            id: dev_id,
            description: pretty_audio_name(&name),
            is_default: name == default_name,
            name,
            kind: kind.to_string(),
            sample_spec,
            state,
            is_monitor,
            is_virtual,
        });
    }
    devices
}

pub fn list_audio_devices(include_monitors: bool, include_virtual: bool) -> (Vec<AudioDevice>, Vec<AudioDevice>, String, String) {
    if !command_exists("pactl") {
        return (Vec::new(), Vec::new(), String::new(), String::new());
    }
    let (_, def_src, _) = run_cmd("pactl", &["get-default-source"]);
    let (_, def_sink, _) = run_cmd("pactl", &["get-default-sink"]);
    let (_, src_out, _) = run_cmd("pactl", &["list", "short", "sources"]);
    let (_, sink_out, _) = run_cmd("pactl", &["list", "short", "sinks"]);

    let mut mics = parse_pactl_short(&src_out, "microphone", &def_src);
    let mut speakers = parse_pactl_short(&sink_out, "speaker", &def_sink);

    if !include_monitors {
        mics.retain(|d| !d.is_monitor);
    }
    if !include_virtual {
        mics.retain(|d| !d.is_virtual);
        speakers.retain(|d| !d.is_virtual);
    }
    (mics, speakers, def_src, def_sink)
}

pub fn list_bluetooth_audio_cards() -> Vec<BluetoothCard> {
    if !command_exists("pactl") {
        return Vec::new();
    }
    let (ok, out, _) = run_cmd("pactl", &["list", "cards"]);
    if !ok {
        return Vec::new();
    }
    let mut cards = Vec::new();
    let mut cur: Option<BluetoothCard> = None;
    let mut in_profiles = false;

    for raw in out.lines() {
        let line = raw.trim();
        if line.starts_with("Card #") {
            if let Some(c) = cur.take() {
                if c.name.starts_with("bluez_card.") {
                    cards.push(c);
                }
            }
            cur = Some(BluetoothCard {
                name: String::new(),
                description: String::new(),
                active_profile: String::new(),
                profiles: Vec::new(),
            });
            in_profiles = false;
        } else if let Some(ref mut c) = cur {
            if let Some(rest) = line.strip_prefix("Name:") {
                c.name = rest.trim().to_string();
            } else if let Some(rest) = line
                .strip_prefix("device.description =")
                .or_else(|| line.strip_prefix("device.alias ="))
            {
                if c.description.is_empty() {
                    c.description = rest.trim().trim_matches('"').to_string();
                }
            } else if let Some(rest) = line.strip_prefix("Active Profile:") {
                c.active_profile = rest.trim().to_string();
                in_profiles = false;
            } else if line == "Profiles:" {
                in_profiles = true;
            } else if line.starts_with("Ports:") || line.starts_with("Properties:") {
                in_profiles = false;
            } else if in_profiles {
                if let Some((prof, _)) = line.split_once(':') {
                    let pname = prof.trim();
                    if !pname.is_empty() && !line.to_ascii_lowercase().contains("available: no") {
                        c.profiles.push(pname.to_string());
                    }
                }
            }
        }
    }
    if let Some(c) = cur {
        if c.name.starts_with("bluez_card.") {
            cards.push(c);
        }
    }
    cards
}

pub fn set_bluetooth_headset_mode(card_name: &str, enable_mic: bool) -> (bool, String) {
    let cards = list_bluetooth_audio_cards();
    let Some(card) = cards.into_iter().find(|c| c.name == card_name) else {
        return (false, format!("Bluetooth audio card not found: {card_name}"));
    };
    let wanted = if enable_mic {
        card.profiles
            .iter()
            .find(|p| p.contains("headset-head-unit-msbc"))
            .or_else(|| {
                card.profiles
                    .iter()
                    .find(|p| p.contains("headset-head-unit") || p.contains("handsfree"))
            })
    } else {
        card.profiles.iter().find(|p| p.starts_with("a2dp-sink"))
    };
    let Some(profile) = wanted else {
        let mode = if enable_mic { "headset mic" } else { "A2DP stereo" };
        return (false, format!("No matching {mode} profile on {}", card.description));
    };
    let (ok, _, err) = run_cmd("pactl", &["set-card-profile", card_name, profile]);
    if ok {
        (true, format!("Switched {} to {profile}", card.description))
    } else {
        (false, format!("Failed to switch profile: {err}"))
    }
}

/// Fast sysfs V4L2 camera enumeration (zero subprocess overhead for basic inventory).
pub fn list_v4l2_sysfs_devices() -> Vec<CameraDeviceSummary> {
    let sys_dir = Path::new("/sys/class/video4linux");
    let Ok(entries) = fs::read_dir(sys_dir) else {
        return Vec::new();
    };
    let mut devs = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("video") {
            continue;
        }
        let path = format!("/dev/{name}");
        let card = fs::read_to_string(entry.path().join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| name.clone());
        let index: u32 = fs::read_to_string(entry.path().join("index"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let low_card = card.to_ascii_lowercase();
        let is_loopback = low_card.contains("nvidia broadcast")
            || low_card.contains("v4l2loopback")
            || low_card.contains("dummy video");
        let is_metadata = !is_loopback && index > 0;
        let driver = if is_loopback {
            "v4l2 loopback".to_string()
        } else {
            "uvcvideo".to_string()
        };
        devs.push(CameraDeviceSummary {
            path,
            driver,
            card,
            bus_info: String::new(),
            is_capture: !is_metadata,
            is_output: is_loopback,
            is_metadata,
            is_loopback,
            formats: if is_loopback {
                vec!["YUYV".to_string()]
            } else {
                vec!["MJPG".to_string(), "YUYV".to_string()]
            },
            modes: Vec::new(),
        });
    }
    devs.sort_by(|a, b| a.path.cmp(&b.path));
    devs
}

pub fn build_devices_payload(include_all: bool) -> DevicesPayload {
    let (mics, speakers, def_src, def_sink) = list_audio_devices(include_all, include_all);
    let all_cams = list_v4l2_sysfs_devices();
    let cameras: Vec<Value> = all_cams
        .iter()
        .filter(|c| include_all || (!c.is_metadata && !c.is_loopback && c.is_capture))
        .filter_map(|c| serde_json::to_value(c).ok())
        .collect();
    let loopback_cameras: Vec<Value> = all_cams
        .iter()
        .filter(|c| c.is_loopback)
        .filter_map(|c| serde_json::to_value(c).ok())
        .collect();
    let bluetooth_cards: Vec<Value> = list_bluetooth_audio_cards()
        .into_iter()
        .filter_map(|c| serde_json::to_value(c).ok())
        .collect();

    DevicesPayload {
        microphones: mics
            .into_iter()
            .filter_map(|d| serde_json::to_value(d).ok())
            .collect(),
        speakers: speakers
            .into_iter()
            .filter_map(|d| serde_json::to_value(d).ok())
            .collect(),
        cameras,
        loopback_cameras,
        bluetooth_cards,
        defaults: DeviceDefaults {
            mic_is_default: def_src.contains(VIRTUAL_MIC_SOURCE_NAME),
            speaker_is_default: def_sink.contains(VIRTUAL_SPEAKER_SINK_NAME),
            default_source: def_src,
            default_sink: def_sink,
        },
    }
}

pub fn probe_gpu() -> GpuInfo {
    if !command_exists("nvidia-smi") {
        return GpuInfo::default();
    }
    let (ok, out, _) = run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    );
    if !ok || out.is_empty() {
        return GpuInfo::default();
    }
    let first = out.lines().next().unwrap_or("");
    let parts: Vec<&str> = first.split(',').map(str::trim).collect();
    let name = parts.first().copied().unwrap_or("NVIDIA GPU").to_string();
    let driver_version = parts.get(1).copied().unwrap_or("").to_string();
    let vram_mib = parts
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .map_or(0, |v| v as u64);
    let compute_capability = parts.get(3).copied().unwrap_or("").to_string();
    let major_cc: u32 = compute_capability
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let rtx_capable = name.to_ascii_lowercase().contains("rtx") || major_cc >= 7;

    GpuInfo {
        available: true,
        name,
        driver_version,
        vram_mib,
        compute_capability,
        cuda_toolkit: String::new(),
        rtx_capable,
    }
}

pub fn query_gpu_live() -> Value {
    if !command_exists("nvidia-smi") {
        return json!({
            "available": false,
            "utilization_pct": 0,
            "memory_used_mib": 0,
            "memory_total_mib": 0,
            "temp_c": 0,
        });
    }
    let (ok, out, _) = run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
    );
    if !ok || out.is_empty() {
        return json!({
            "available": false,
            "utilization_pct": 0,
            "memory_used_mib": 0,
            "memory_total_mib": 0,
            "temp_c": 0,
        });
    }
    let parts: Vec<&str> = out.lines().next().unwrap_or("").split(',').map(str::trim).collect();
    let parse_i = |idx: usize| -> i64 {
        parts
            .get(idx)
            .and_then(|s| s.parse::<f64>().ok())
            .map_or(0, |v| v as i64)
    };
    json!({
        "available": true,
        "name": parts.first().copied().unwrap_or("").replace("NVIDIA GeForce ", ""),
        "utilization_pct": parse_i(1),
        "memory_used_mib": parse_i(2),
        "memory_total_mib": parse_i(3),
        "temp_c": parse_i(4),
    })
}

fn find_ladspa_plugin(candidates: &[&str]) -> String {
    let mut dirs: Vec<PathBuf> = vec![
        PathBuf::from("/usr/lib64/ladspa"),
        PathBuf::from("/usr/lib/ladspa"),
        PathBuf::from("/usr/local/lib/ladspa"),
        PathBuf::from("/usr/local/lib64/ladspa"),
    ];
    if let Some(home) = env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".ladspa"));
    }
    if let Ok(lp) = env::var("LADSPA_PATH") {
        for entry in lp.split(':').filter(|s| !s.is_empty()) {
            dirs.insert(0, PathBuf::from(entry));
        }
    }
    for d in dirs {
        for name in candidates {
            let p = d.join(name);
            if p.is_file() {
                return p.to_string_lossy().into_owned();
            }
        }
    }
    String::new()
}

pub fn probe_capabilities() -> Value {
    let gpu = probe_gpu();
    let mut pw_ver = String::new();
    if command_exists("pipewire") {
        let (ok, out, _) = run_cmd("pipewire", &["--version"]);
        if ok {
            for line in out.lines() {
                if line.contains("libpipewire") {
                    pw_ver = line.split_whitespace().last().unwrap_or("").to_string();
                    break;
                }
            }
        }
    }

    let webrtc_aec = [
        "/usr/lib64/spa-0.2/aec/libspa-aec-webrtc.so",
        "/usr/lib/x86_64-linux-gnu/spa-0.2/aec/libspa-aec-webrtc.so",
    ]
    .iter()
    .any(|p| Path::new(p).is_file());

    let rnnoise_path = find_ladspa_plugin(&["librnnoise_ladspa.so"]);
    let df_path = find_ladspa_plugin(&["libdeep_filter_ladspa.so"]);
    let has_ffmpeg = command_exists("ffmpeg");

    let has_v4l2loopback_mod = run_cmd("modinfo", &["v4l2loopback"]).0;
    let v4l2loopback_loaded = fs::read_to_string("/proc/modules")
        .map(|s| s.contains("v4l2loopback"))
        .unwrap_or(false);

    let models = get_models_status();
    let rvm_installed = models.get("rvm_mobilenetv3").is_some_and(|m| m.installed);
    let rec_seg = if gpu.available && rvm_installed {
        "rvm_cuda"
    } else if rvm_installed {
        "rvm_cpu"
    } else {
        "fast_portrait"
    };

    let mut rust_pkgs: BTreeMap<&str, bool> = BTreeMap::new();
    rust_pkgs.insert("nnnoiseless", true);
    rust_pkgs.insert("realfft", true);

    json!({
        "gpu": gpu,
        "pipewire_version": pw_ver,
        "has_pactl": command_exists("pactl"),
        "has_pw_cat": command_exists("pw-cat"),
        "has_pw_loopback": command_exists("pw-loopback"),
        "has_pipewire_webrtc_aec": webrtc_aec,
        "has_rnnoise_ladspa": !rnnoise_path.is_empty(),
        "rnnoise_ladspa_path": rnnoise_path,
        "has_deepfilter_ladspa": !df_path.is_empty(),
        "deepfilter_ladspa_path": df_path,
        "has_ffmpeg": has_ffmpeg,
        "ffmpeg_filters": BTreeSet::<String>::new(),
        "has_v4l2loopback_module": has_v4l2loopback_mod,
        "v4l2loopback_loaded": v4l2loopback_loaded,
        "python_packages": rust_pkgs,
        "onnx_providers": if gpu.available { vec!["CUDAExecutionProvider", "CPUExecutionProvider"] } else { vec!["CPUExecutionProvider"] },
        "models": models,
        "recommended_audio_backend": "nnnoiseless",
        "recommended_segmentation_backend": rec_seg,
        "engine": "rust",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pactl_short_output_and_virtual_flags() {
        let sample = "\
42\talsa_input.pci-0000_00_1f.3.HiFi__Mic1__source\tPipeWire\ts32le 2ch 48000Hz\tRUNNING
99\tnv_broadcast_mic\tPipeWire\ts16le 1ch 48000Hz\tIDLE
101\talsa_output.pci.monitor\tPipeWire\ts32le 2ch 48000Hz\tIDLE
";
        let devs = parse_pactl_short(sample, "microphone", "alsa_input.pci-0000_00_1f.3.HiFi__Mic1__source");
        assert_eq!(devs.len(), 3);
        assert!(devs[0].is_default);
        assert!(!devs[0].is_virtual);
        assert_eq!(devs[0].description, "Built-in Laptop Microphone");
        assert!(devs[1].is_virtual);
        assert!(devs[2].is_monitor);
    }
}
