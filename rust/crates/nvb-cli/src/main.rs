//! NVIDIA Broadcast (Linux) Rust CLI (`nvb-cli`) and unified benchmark driver.
//!
//! Phase R0 exposes `bench` subcommands (`capabilities`, `config-bench`,
//! `pcm-bench`, `reliability`) so the shared `bench/` suite exercises live
//! Rust crates immediately while reporting pending phases (`R1..R3`) cleanly.

use std::env;
use std::f64::consts::PI;
use std::fs;
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nvb_audio::{
    build_isolated_pw_cat_args, compute_rms_db, f32_to_pcm16, is_pw_link_active, parse_pw_links,
    pcm16_to_f32, PhysicalDevice, PwCatMode, RoutingError, DEFAULT_LATENCY_MS,
    RNNOISE_FRAME_SAMPLES, SAMPLE_RATE,
};
use nvb_core::camera_state::CameraState;
use nvb_core::config::BroadcastConfig;
use nvb_core::{APP_NAME, APP_VERSION};
use serde_json::{json, Value};

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn summarize_us(mut samples_us: Vec<f64>) -> Value {
    samples_us.sort_by(|a, b| a.total_cmp(b));
    let total_us: f64 = samples_us.iter().sum();
    let count = samples_us.len();
    let mean_us = if count > 0 {
        total_us / count as f64
    } else {
        0.0
    };
    json!({
        "count": count,
        "total_ms": (total_us / 1000.0 * 1000.0).round() / 1000.0,
        "min_us": (percentile(&samples_us, 0.0) * 100.0).round() / 100.0,
        "mean_us": (mean_us * 100.0).round() / 100.0,
        "p50_us": (percentile(&samples_us, 0.50) * 100.0).round() / 100.0,
        "p95_us": (percentile(&samples_us, 0.95) * 100.0).round() / 100.0,
        "p99_us": (percentile(&samples_us, 0.99) * 100.0).round() / 100.0,
        "p999_us": (percentile(&samples_us, 0.999) * 100.0).round() / 100.0,
        "max_us": (percentile(&samples_us, 1.0) * 100.0).round() / 100.0,
    })
}

fn capabilities_payload() -> Value {
    json!({
        "target": "rust",
        "app": APP_NAME,
        "version": APP_VERSION,
        "phase": "R0",
        "capabilities": {
            "config_bench": true,
            "pcm_bench": true,
            "reliability_core": true,
            "audio_dsp": false,
            "daemon_http": false,
            "cli_full": false,
            "video_pipeline": false
        },
        "pending_phases": {
            "audio_quality": "pending_phase_r1",
            "audio_latency": "pending_phase_r1",
            "memory": "pending_phase_r2",
            "video": "pending_phase_r3",
            "api_cli": "pending_phase_r2",
            "system_impact": "pending_phase_r2"
        }
    })
}

fn run_config_bench(iterations: usize) -> Value {
    let dirty = json!({
        "audio_backend": "invalid_backend",
        "segmentation_backend": "invalid_seg",
        "microphone": {
            "highpass_hz": 9999,
            "noise_gate_db": -120.0,
            "auto_gain_target_db": 10.0,
            "effects": {
                "noise_removal": {"enabled": true, "strength": 1.9},
                "room_echo_removal": {"enabled": true, "strength": -0.3},
                "studio_voice": {"enabled": false, "micProfile": "alien"}
            }
        },
        "camera": {
            "split_position": 0.99,
            "idle_timeout_sec": 0.01,
            "effects": {
                "background_blur": {"enabled": true, "strength": 2.0, "mode": "ultra"}
            }
        }
    });

    let mut latencies_us = Vec::with_capacity(iterations);
    let wall_start = Instant::now();
    let mut last_conflicts = 0usize;

    for _ in 0..iterations {
        let t0 = Instant::now();
        let mut cfg = BroadcastConfig::from_json_override(&dirty);
        let res = cfg.set_effect("studio_voice", Some(true), None, true);
        last_conflicts = res.disabled_conflicts.values().map(Vec::len).sum();
        let _ = cfg.set_effect("background_remove", Some(true), None, true);
        latencies_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let ops_per_sec = if wall_ms > 0.0 {
        (iterations as f64) / (wall_ms / 1000.0)
    } else {
        0.0
    };

    json!({
        "status": "ok",
        "iterations": iterations,
        "wall_ms": (wall_ms * 1000.0).round() / 1000.0,
        "ops_per_sec": (ops_per_sec * 10.0).round() / 10.0,
        "last_conflicts_disabled": last_conflicts,
        "latency": summarize_us(latencies_us),
    })
}

fn run_pcm_bench(frames: usize) -> Value {
    let sine_f32: Vec<f32> = (0..RNNOISE_FRAME_SAMPLES)
        .map(|i| (0.5 * (2.0 * PI * 440.0 * i as f64 / f64::from(SAMPLE_RATE)).sin()) as f32)
        .collect();
    let pcm_in = f32_to_pcm16(&sine_f32);

    let mut latencies_us = Vec::with_capacity(frames);
    let wall_start = Instant::now();
    let mut last_rms = -90.0_f64;

    for _ in 0..frames {
        let t0 = Instant::now();
        let floats = pcm16_to_f32(&pcm_in);
        let (rms_db, _peak_db) = compute_rms_db(&floats);
        let pcm_out = f32_to_pcm16(&floats);
        std::hint::black_box(&pcm_out);
        last_rms = rms_db;
        latencies_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let wall_sec = wall_start.elapsed().as_secs_f64();
    let audio_sec = (frames * RNNOISE_FRAME_SAMPLES) as f64 / f64::from(SAMPLE_RATE);
    let rtf = if audio_sec > 0.0 {
        wall_sec / audio_sec
    } else {
        0.0
    };

    json!({
        "status": "ok",
        "frames": frames,
        "frame_samples": RNNOISE_FRAME_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "audio_sec": (audio_sec * 1000.0).round() / 1000.0,
        "wall_ms": (wall_sec * 1000.0 * 1000.0).round() / 1000.0,
        "rtf": (rtf * 1_000_000.0).round() / 1_000_000.0,
        "last_rms_db": (last_rms * 100.0).round() / 100.0,
        "latency": summarize_us(latencies_us),
    })
}

fn run_reliability_checks() -> Value {
    let mut checks = Vec::new();

    // 1. Feedback loop prevention via typed PhysicalDevice
    let virtual_names = [
        "nv_broadcast_mic",
        "nv_broadcast_mic_bus",
        "nv_broadcast_mic_bus.monitor",
        "nv_broadcast_speakers",
        "NV_BROADCAST_SPEAKERS.monitor",
        "",
    ];
    let all_virtual_rejected = virtual_names.iter().all(|name| {
        matches!(
            PhysicalDevice::try_new(*name),
            Err(RoutingError::VirtualNodeRejected(_) | RoutingError::EmptyName)
        )
    });
    let physical_accepted = PhysicalDevice::try_new("alsa_input.pci-0000_00_1f.3").is_ok();
    checks.push(json!({
        "name": "feedback_loop_rejection",
        "passed": all_virtual_rejected && physical_accepted,
        "detail": "PhysicalDevice newtype rejects all Broadcast virtual nodes and empty names"
    }));

    // 2. Isolated pw-cat stream properties
    let rec_args = build_isolated_pw_cat_args(
        PwCatMode::Record,
        "nv_broadcast_mic_capture",
        SAMPLE_RATE,
        1,
        DEFAULT_LATENCY_MS,
    );
    let play_args = build_isolated_pw_cat_args(
        PwCatMode::Playback,
        "nv_broadcast_spk_playback",
        SAMPLE_RATE,
        2,
        DEFAULT_LATENCY_MS,
    );
    let has_isolation = [rec_args, play_args].iter().all(|args| {
        args.iter().any(|a| {
            a.contains("node.autoconnect = false") && a.contains("node.dont-reconnect = true")
        })
    });
    checks.push(json!({
        "name": "isolated_stream_properties",
        "passed": has_isolation,
        "detail": "pw-cat args set node.autoconnect=false and node.dont-reconnect=true"
    }));

    // 3. Dropped link detection via parse_pw_links + is_pw_link_active
    let links_ok = parse_pw_links(
        "alsa_input.usb:capture_FL\n  |-> nv_broadcast_mic_capture:input_FL\n",
    );
    let links_dropped = parse_pw_links("alsa_input.usb:capture_FL\n");
    let drop_detected = is_pw_link_active("alsa_input.usb", "nv_broadcast_mic_capture", &links_ok)
        && !is_pw_link_active("alsa_input.usb", "nv_broadcast_mic_capture", &links_dropped);
    checks.push(json!({
        "name": "dropped_link_detection",
        "passed": drop_detected,
        "detail": "parse_pw_links detects missing active edge"
    }));

    // 4. Corrupt / truncated config.json resilience
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp_path = env::temp_dir().join(format!("nvb_corrupt_cfg_{nonce}.json"));
    let corrupt_ok = if fs::write(&tmp_path, "{broken json").is_ok() {
        let cfg = BroadcastConfig::load_from_path(&tmp_path);
        let _ = fs::remove_file(&tmp_path);
        cfg.version == 2 && cfg.audio_backend == "auto"
    } else {
        false
    };
    checks.push(json!({
        "name": "corrupt_config_recovery",
        "passed": corrupt_ok,
        "detail": "BroadcastConfig::load_from_path recovers cleanly to sanitized defaults"
    }));

    // 5. Camera on-demand standby state machine invariants
    let s1 = CameraState::Off.enable_on_demand();
    let s2 = s1.on_consumers_observed(1, false, true);
    let s3 = s2.on_consumers_observed(0, true, true);
    let cam_sm_ok = s1.is_standby()
        && s2.is_on_demand_active()
        && s2.is_capture_running()
        && s3.is_standby()
        && !s3.is_capture_running();
    checks.push(json!({
        "name": "camera_standby_state_machine",
        "passed": cam_sm_ok,
        "detail": "Standby -> OnDemandActive -> Standby releases physical webcam"
    }));

    let passed = checks
        .iter()
        .filter(|c| c["passed"].as_bool().unwrap_or(false))
        .count();
    json!({
        "status": "ok",
        "phase": "R0",
        "passed": passed,
        "total": checks.len(),
        "checks": checks,
    })
}

fn parse_usize_flag(args: &[String], flag: &str, default: usize) -> usize {
    for pair in args.windows(2) {
        if pair[0] == flag {
            if let Ok(val) = pair[1].parse::<usize>() {
                return val.max(1);
            }
        }
    }
    default
}

fn dispatch(args: &[String]) -> Result<String, String> {
    if args.is_empty() || args[0] == "--version" || args[0] == "-V" {
        return Ok(format!("{APP_NAME} nvb-cli {APP_VERSION}"));
    }

    if args[0] == "bench" {
        let sub = args.get(1).map(String::as_str).unwrap_or("capabilities");
        let rest = if args.len() > 2 { &args[2..] } else { &[] };
        let payload = match sub {
            "capabilities" => capabilities_payload(),
            "config-bench" => {
                let iters = parse_usize_flag(rest, "--iterations", 1000);
                run_config_bench(iters)
            }
            "pcm-bench" => {
                let frames = parse_usize_flag(rest, "--frames", 2000);
                run_pcm_bench(frames)
            }
            "reliability" => run_reliability_checks(),
            "audio-file" | "audio-loop" => json!({
                "status": "pending_phase_r1",
                "reason": "Pure-Rust AudioProcessor DSP lands in Phase R1"
            }),
            "video-frames" => json!({
                "status": "pending_phase_r3",
                "reason": "Native Rust camera worker lands in Phase R3"
            }),
            other => {
                return Err(format!("unknown bench subcommand: {other}"));
            }
        };
        return serde_json::to_string_pretty(&payload).map_err(|e| e.to_string());
    }

    Ok(format!("{APP_NAME} nvb-cli {APP_VERSION}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_capabilities_reports_r0_contract() {
        let out = dispatch(&["bench".to_string(), "capabilities".to_string()]).unwrap();
        let val: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(val["target"], "rust");
        assert_eq!(val["capabilities"]["config_bench"], true);
        assert_eq!(val["capabilities"]["pcm_bench"], true);
        assert_eq!(val["capabilities"]["reliability_core"], true);
        assert_eq!(val["pending_phases"]["audio_quality"], "pending_phase_r1");
    }

    #[test]
    fn bench_config_and_pcm_and_reliability_succeed() {
        let cfg = run_config_bench(25);
        assert_eq!(cfg["status"], "ok");
        assert_eq!(cfg["iterations"], 25);
        assert_eq!(cfg["last_conflicts_disabled"], 2);

        let pcm = run_pcm_bench(20);
        assert_eq!(pcm["status"], "ok");
        assert_eq!(pcm["frames"], 20);

        let rel = run_reliability_checks();
        assert_eq!(rel["status"], "ok");
        assert_eq!(rel["passed"], rel["total"]);
    }
}
