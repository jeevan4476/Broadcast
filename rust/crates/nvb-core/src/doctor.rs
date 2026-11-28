//! Hardware, PipeWire, V4L2, and Neural AI environment doctor.
//!
//! Port of `nv_broadcast/doctor.py`.

use crate::devices::{build_devices_payload, probe_capabilities};
use crate::APP_VERSION;
use serde_json::{json, Value};

pub fn run_doctor() -> Value {
    let caps = probe_capabilities();
    let devs = build_devices_payload(false);
    let mut checks: Vec<Value> = Vec::new();
    let mut next_steps: Vec<String> = Vec::new();

    let gpu = &caps["gpu"];
    if gpu["available"].as_bool().unwrap_or(false) {
        checks.push(json!({
            "name": "nvidia_gpu",
            "status": "ok",
            "detail": format!(
                "{} (driver {}, {} MiB VRAM, CC {})",
                gpu["name"].as_str().unwrap_or("NVIDIA GPU"),
                gpu["driver_version"].as_str().unwrap_or(""),
                gpu["vram_mib"].as_u64().unwrap_or(0),
                gpu["compute_capability"].as_str().unwrap_or(""),
            ),
        }));
    } else {
        checks.push(json!({
            "name": "nvidia_gpu",
            "status": "warn",
            "detail": "No NVIDIA GPU detected via nvidia-smi",
        }));
    }

    let pw_ver = caps["pipewire_version"].as_str().unwrap_or("");
    let has_pactl = caps["has_pactl"].as_bool().unwrap_or(false);
    let has_pw_cat = caps["has_pw_cat"].as_bool().unwrap_or(false);
    if !pw_ver.is_empty() && has_pactl && has_pw_cat {
        checks.push(json!({
            "name": "pipewire_audio",
            "status": "ok",
            "detail": format!("PipeWire {pw_ver} ready (pure-Rust nnnoiseless RNNoise active)"),
        }));
    } else {
        checks.push(json!({
            "name": "pipewire_audio",
            "status": "fail",
            "detail": "PipeWire / pactl / pw-cat missing",
        }));
    }

    let n_mics = devs.microphones.len();
    let n_spks = devs.speakers.len();
    checks.push(json!({
        "name": "audio_devices",
        "status": if n_mics > 0 { "ok" } else { "warn" },
        "detail": format!("{n_mics} microphone(s), {n_spks} speaker output(s)"),
    }));

    if let Some(first_cam) = devs.cameras.first() {
        checks.push(json!({
            "name": "physical_camera",
            "status": "ok",
            "detail": format!(
                "{} ({})",
                first_cam["path"].as_str().unwrap_or("/dev/video0"),
                first_cam["card"].as_str().unwrap_or("webcam"),
            ),
        }));
    } else {
        checks.push(json!({
            "name": "physical_camera",
            "status": "warn",
            "detail": "No physical V4L2 camera detected",
        }));
    }

    let lb_loaded = caps["v4l2loopback_loaded"].as_bool().unwrap_or(false);
    let has_lb_mod = caps["has_v4l2loopback_module"].as_bool().unwrap_or(false);
    if lb_loaded && !devs.loopback_cameras.is_empty() {
        let paths: Vec<&str> = devs
            .loopback_cameras
            .iter()
            .filter_map(|c| c["path"].as_str())
            .collect();
        checks.push(json!({
            "name": "virtual_camera",
            "status": "ok",
            "detail": format!("v4l2loopback active at {}", paths.join(", ")),
        }));
    } else if has_lb_mod {
        checks.push(json!({
            "name": "virtual_camera",
            "status": "warn",
            "detail": "v4l2loopback installed but not loaded",
        }));
        next_steps.push("Load virtual camera: sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"NVIDIA Broadcast Camera\" exclusive_caps=1".to_string());
    } else {
        checks.push(json!({
            "name": "virtual_camera",
            "status": "warn",
            "detail": "v4l2loopback kernel module not installed",
        }));
    }

    checks.push(json!({
        "name": "neural_ai_runtime",
        "status": "ok",
        "detail": format!(
            "audio: nnnoiseless (pure Rust), segmentation: {}",
            caps["recommended_segmentation_backend"].as_str().unwrap_or("fast_portrait")
        ),
    }));

    json!({
        "engine": "rust",
        "version": APP_VERSION,
        "capabilities": caps,
        "audio_devices": {
            "microphones": devs.microphones,
            "speakers": devs.speakers,
        },
        "cameras": devs.cameras,
        "loopback_cameras": devs.loopback_cameras,
        "checks": checks,
        "next_steps": next_steps,
    })
}

pub fn format_doctor_text(report: &Value) -> String {
    let mut lines = vec!["=== NVIDIA Broadcast (Linux) Doctor [Rust] ===".to_string()];
    if let Some(checks) = report["checks"].as_array() {
        for chk in checks {
            let status = chk["status"].as_str().unwrap_or("info");
            let badge = match status {
                "ok" => "[OK]  ",
                "warn" => "[WARN]",
                "fail" => "[FAIL]",
                _ => "[INFO]",
            };
            lines.push(format!(
                "  {badge} {}: {}",
                chk["name"].as_str().unwrap_or(""),
                chk["detail"].as_str().unwrap_or(""),
            ));
        }
    }
    if let Some(steps) = report["next_steps"].as_array() {
        if !steps.is_empty() {
            lines.push("\nRecommended next steps:".to_string());
            for step in steps {
                if let Some(s) = step.as_str() {
                    lines.push(format!("  - {s}"));
                }
            }
        }
    }
    lines.join("\n")
}
