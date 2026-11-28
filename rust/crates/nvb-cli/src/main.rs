//! NVIDIA Broadcast (Linux) native Rust CLI (`nvb-cli`) and unified benchmark driver.
//!
//! Drop-in `clap` replacement for `nv-broadcast` (`doctor`, `status`, `devices`,
//! `effects`, `mic`, `speaker`, `camera`, `serve`, and `bench`).

use std::env;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use nvb_audio::{
    build_isolated_pw_cat_args, compute_rms_db, f32_to_pcm16, generate_sample_clip,
    is_pw_link_active, parse_pw_links, pcm16_to_f32, read_wav_mono16, write_wav_mono16,
    AudioProcessConfig, AudioProcessor, PhysicalDevice, PipeWireVirtualManager, PwCatMode,
    RoutingError, DEFAULT_LATENCY_MS, RNNOISE_FRAME_SAMPLES, SAMPLE_RATE,
};
use nvb_core::camera_state::CameraState;
use nvb_core::config::{clamp_float, BroadcastConfig, INCOMPATIBLE_GROUPS};
use nvb_core::devices::build_devices_payload;
use nvb_core::doctor::{format_doctor_text, run_doctor};
use nvb_core::{APP_NAME, APP_VERSION};
use serde_json::{json, Map, Value};

const CONFIG_ORACLE_JSON: &str = include_str!("../../../fixtures/config_oracle.json");

#[derive(Parser, Debug)]
#[command(
    name = "nv-broadcast",
    version = APP_VERSION,
    about = "NVIDIA Broadcast for Linux — PipeWire Neural Audio & GPU V4L2 Camera Suite (Rust)"
)]
struct Cli {
    /// Path to config.json
    #[arg(long)]
    config: Option<PathBuf>,

    /// Emit machine-readable JSON output
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inspect GPU, PipeWire, V4L2 camera, v4l2loopback, and neural AI backends
    Doctor,
    /// Show current configuration, effect state, and virtual devices
    Status,
    /// List microphones, speakers, and V4L2 cameras
    Devices {
        #[arg(long)]
        all: bool,
    },
    /// Describe all supported NVIDIA Broadcast effect IDs and parameters
    Effects,
    /// Configure microphone device and effects
    Mic {
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        noise_removal: Option<bool>,
        #[arg(long)]
        noise_strength: Option<f64>,
        #[arg(long)]
        echo_removal: Option<bool>,
        #[arg(long)]
        echo_strength: Option<f64>,
        #[arg(long)]
        studio_voice: Option<bool>,
        #[arg(long)]
        mic_profile: Option<String>,
    },
    /// Configure speaker output and incoming audio effects
    Speaker {
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        noise_removal: Option<bool>,
        #[arg(long)]
        noise_strength: Option<f64>,
        #[arg(long)]
        echo_removal: Option<bool>,
        #[arg(long)]
        echo_strength: Option<f64>,
    },
    /// Configure webcam, resolution, and video effects
    Camera {
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        output_device: Option<String>,
        #[arg(long)]
        on_demand: Option<bool>,
        #[arg(long)]
        idle_timeout: Option<f64>,
        #[arg(long)]
        blur: Option<bool>,
        #[arg(long)]
        blur_strength: Option<f64>,
        #[arg(long)]
        blur_mode: Option<String>,
        #[arg(long)]
        replace: Option<bool>,
        #[arg(long)]
        remove: Option<bool>,
    },
    /// Start the local HTTP/SSE/MJPEG daemon (nvb-daemon)
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 0)]
        port: u16,
        #[arg(long)]
        token: Option<String>,
        #[arg(long)]
        autostart: bool,
        #[arg(long)]
        daemon_file: bool,
    },
    /// Internal benchmark driver used by `bench/run_suite.py`
    Bench {
        #[command(subcommand)]
        sub: BenchCmd,
    },
}

#[derive(Subcommand, Debug)]
enum BenchCmd {
    Capabilities,
    ConfigBench {
        #[arg(long, default_value_t = 1000)]
        iterations: usize,
    },
    PcmBench {
        #[arg(long, default_value_t = 2000)]
        frames: usize,
    },
    AudioFile {
        #[arg(long = "in")]
        in_path: String,
        #[arg(long = "out")]
        out_path: String,
        #[arg(long, default_value = "{}")]
        config: String,
    },
    AudioLoop {
        #[arg(long, default_value_t = 2000)]
        frames: usize,
    },
    Reliability,
}

fn default_config_path() -> PathBuf {
    let base = env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            home.join(".config")
        });
    base.join("nv-broadcast").join("config.json")
}

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

fn summarize_latency_us_map(mut samples_us: Vec<f64>) -> Value {
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
        "min": (percentile(&samples_us, 0.0) * 1000.0).round() / 1000.0,
        "mean": (mean_us * 1000.0).round() / 1000.0,
        "p50": (percentile(&samples_us, 0.50) * 1000.0).round() / 1000.0,
        "p95": (percentile(&samples_us, 0.95) * 1000.0).round() / 1000.0,
        "p99": (percentile(&samples_us, 0.99) * 1000.0).round() / 1000.0,
        "p999": (percentile(&samples_us, 0.999) * 1000.0).round() / 1000.0,
        "max": (percentile(&samples_us, 1.0) * 1000.0).round() / 1000.0,
    })
}

fn read_self_rss_kb() -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(kb) = tok.parse::<u64>() {
                    return kb;
                }
            }
        }
    }
    0
}

fn capabilities_payload() -> Value {
    json!({
        "target": "rust",
        "app": APP_NAME,
        "version": APP_VERSION,
        "phase": "R2",
        "audio_backend": "nnnoiseless",
        "capabilities": {
            "config_bench": true,
            "pcm_bench": true,
            "reliability_core": true,
            "audio_dsp": true,
            "daemon_http": true,
            "cli_full": true,
            "video_pipeline": false
        },
        "pending_phases": {
            "video": "pending_phase_r3"
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

fn run_audio_file_bench(in_path: &str, out_path: &str, cfg_arg: &str) -> Result<Value, String> {
    let cfg_val: Value = if Path::new(cfg_arg).is_file() {
        let raw = fs::read_to_string(cfg_arg).map_err(|e| format!("failed reading config: {e}"))?;
        serde_json::from_str(&raw).map_err(|e| format!("invalid config json file: {e}"))?
    } else {
        serde_json::from_str(cfg_arg).map_err(|e| format!("invalid config json: {e}"))?
    };
    let cfg = AudioProcessConfig::from_json_value(&cfg_val);

    let (raw_pcm, sr) = read_wav_mono16(Path::new(in_path))
        .map_err(|e| format!("failed reading input WAV '{in_path}': {e}"))?;
    let mut proc = AudioProcessor::new(sr);
    let chunk_bytes = RNNOISE_FRAME_SAMPLES * 2;
    let mut out_pcm = Vec::with_capacity(raw_pcm.len());

    let t0 = Instant::now();
    for chunk in raw_pcm.chunks(chunk_bytes) {
        out_pcm.extend_from_slice(&proc.process_pcm16(chunk, &cfg));
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    write_wav_mono16(Path::new(out_path), &out_pcm, sr)
        .map_err(|e| format!("failed writing output WAV '{out_path}': {e}"))?;

    Ok(json!({
        "status": "ok",
        "backend": proc.backend_name,
        "sample_rate": sr,
        "elapsed_ms": (elapsed_ms * 1000.0).round() / 1000.0,
        "rss_kb": read_self_rss_kb(),
        "meter": proc.last_meter,
    }))
}

fn run_audio_loop_bench(frames: usize) -> Value {
    let clip = generate_sample_clip("fan_echo", SAMPLE_RATE, 2.0);
    let frame_bytes = RNNOISE_FRAME_SAMPLES * 2;
    let ring: Vec<&[u8]> = clip.chunks_exact(frame_bytes).collect();
    let audio_sec = (frames * RNNOISE_FRAME_SAMPLES) as f64 / f64::from(SAMPLE_RATE);

    let specs = [
        (
            "noise_removal",
            AudioProcessConfig {
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
            },
        ),
        (
            "room_echo_removal",
            AudioProcessConfig {
                highpass_enabled: true,
                highpass_hz: 85.0,
                noise_removal_enabled: true,
                noise_removal_strength: 0.85,
                room_echo_removal_enabled: true,
                room_echo_removal_strength: 0.6,
                studio_voice_enabled: false,
                studio_voice_profile: "default".to_string(),
                noise_gate_enabled: true,
                noise_gate_db: -46.0,
                auto_gain_enabled: true,
                auto_gain_target_db: -16.0,
            },
        ),
        (
            "studio_voice",
            AudioProcessConfig {
                highpass_enabled: true,
                highpass_hz: 85.0,
                noise_removal_enabled: false,
                noise_removal_strength: 0.85,
                room_echo_removal_enabled: false,
                room_echo_removal_strength: 0.5,
                studio_voice_enabled: true,
                studio_voice_profile: "warm".to_string(),
                noise_gate_enabled: true,
                noise_gate_db: -46.0,
                auto_gain_enabled: true,
                auto_gain_target_db: -16.0,
            },
        ),
    ];

    let mut workloads = Map::new();
    let mut backend_name = "nnnoiseless";

    for (wname, cfg) in &specs {
        let mut proc = AudioProcessor::new(SAMPLE_RATE);
        backend_name = proc.backend_name;
        for i in 0..20 {
            let chunk = ring[i % ring.len()];
            std::hint::black_box(proc.process_pcm16(chunk, cfg));
        }
        let mut lats_us = Vec::with_capacity(frames);
        let wall_start = Instant::now();
        for i in 0..frames {
            let chunk = ring[i % ring.len()];
            let t0 = Instant::now();
            let out = proc.process_pcm16(chunk, cfg);
            std::hint::black_box(&out);
            lats_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
        }
        let wall_sec = wall_start.elapsed().as_secs_f64();
        let rtf = wall_sec / audio_sec.max(1e-9);
        let misses_2ms = lats_us.iter().filter(|&&v| v > 2000.0).count();
        let misses_10ms = lats_us.iter().filter(|&&v| v > 10_000.0).count();

        workloads.insert(
            (*wname).to_string(),
            json!({
                "frames": frames,
                "audio_sec": (audio_sec * 1000.0).round() / 1000.0,
                "wall_ms": (wall_sec * 1000.0 * 1000.0).round() / 1000.0,
                "rtf": (rtf * 1_000_000.0).round() / 1_000_000.0,
                "single_core_cpu_pct": (rtf * 100.0 * 1000.0).round() / 1000.0,
                "budget_misses_2ms": misses_2ms,
                "budget_misses_10ms": misses_10ms,
                "latency_us": summarize_latency_us_map(lats_us),
            }),
        );
    }

    let pcm_raw = run_pcm_bench(frames);
    let pcm_lat = &pcm_raw["latency"];
    let pcm_primitive = json!({
        "status": "ok",
        "frames": pcm_raw["frames"],
        "audio_sec": pcm_raw["audio_sec"],
        "wall_ms": pcm_raw["wall_ms"],
        "rtf": pcm_raw["rtf"],
        "latency_us": {
            "count": pcm_lat["count"],
            "min": pcm_lat["min_us"],
            "mean": pcm_lat["mean_us"],
            "p50": pcm_lat["p50_us"],
            "p95": pcm_lat["p95_us"],
            "p99": pcm_lat["p99_us"],
            "p999": pcm_lat["p999_us"],
            "max": pcm_lat["max_us"],
        }
    });

    json!({
        "status": "ok",
        "backend": backend_name,
        "rss_kb": read_self_rss_kb(),
        "workloads": workloads,
        "pcm_primitive": pcm_primitive,
    })
}

fn run_reliability_checks() -> Value {
    let mut checks = Vec::new();

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
        "phase": "R2",
        "passed": passed,
        "total": checks.len(),
        "checks": checks,
    })
}

fn emit_json(val: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(val).unwrap_or_else(|_| "{}".to_string())
    );
}

fn find_sibling_daemon() -> PathBuf {
    if let Ok(exe) = env::current_exe() {
        let sibling = exe.with_file_name("nvb-daemon");
        if sibling.is_file() {
            return sibling;
        }
    }
    PathBuf::from("nvb-daemon")
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let cfg_path = cli.config.unwrap_or_else(default_config_path);

    let Some(cmd) = cli.command else {
        println!("{APP_NAME} nvb-cli {APP_VERSION}");
        return ExitCode::SUCCESS;
    };

    match cmd {
        Commands::Doctor => {
            let report = run_doctor();
            if cli.json {
                emit_json(&report);
            } else {
                println!("{}", format_doctor_text(&report));
            }
            ExitCode::SUCCESS
        }
        Commands::Status => {
            let cfg = BroadcastConfig::load_from_path(&cfg_path);
            let payload = json!({
                "version": APP_VERSION,
                "engine": "rust",
                "config_path": cfg_path.to_string_lossy(),
                "config": cfg,
                "broadcast_state": cfg.get_broadcast_state(),
                "virtual_nodes": PipeWireVirtualManager::find_existing_virtual_nodes(),
            });
            emit_json(&payload);
            ExitCode::SUCCESS
        }
        Commands::Devices { all } => {
            let payload = build_devices_payload(all);
            emit_json(&serde_json::to_value(payload).unwrap_or_else(|_| json!({})));
            ExitCode::SUCCESS
        }
        Commands::Effects => {
            let catalog = serde_json::from_str::<Value>(CONFIG_ORACLE_JSON)
                .ok()
                .and_then(|v| v.get("effect_catalog").cloned())
                .unwrap_or_else(|| json!({}));
            let payload = json!({
                "effects": catalog,
                "incompatible_groups": INCOMPATIBLE_GROUPS,
            });
            emit_json(&payload);
            ExitCode::SUCCESS
        }
        Commands::Mic {
            device,
            backend,
            noise_removal,
            noise_strength,
            echo_removal,
            echo_strength,
            studio_voice,
            mic_profile,
        } => {
            let mut cfg = BroadcastConfig::load_from_path(&cfg_path);
            if let Some(d) = device {
                cfg.microphone.device = d;
            }
            if let Some(b) = backend {
                cfg.audio_backend = b;
            }
            if noise_removal.is_some() || noise_strength.is_some() {
                let mut p = Map::new();
                if let Some(s) = noise_strength {
                    p.insert("strength".to_string(), json!(s));
                }
                let _ = cfg.set_effect("noise_removal", noise_removal, Some(&p), true);
            }
            if echo_removal.is_some() || echo_strength.is_some() {
                let mut p = Map::new();
                if let Some(s) = echo_strength {
                    p.insert("strength".to_string(), json!(s));
                }
                let _ = cfg.set_effect("room_echo_removal", echo_removal, Some(&p), true);
            }
            if studio_voice.is_some() || mic_profile.is_some() {
                let mut p = Map::new();
                if let Some(mp) = mic_profile {
                    p.insert("micProfile".to_string(), json!(mp));
                }
                let _ = cfg.set_effect("studio_voice", studio_voice, Some(&p), true);
            }
            let _ = cfg.save_to_path(&cfg_path);
            emit_json(&json!({"ok": true, "microphone": cfg.microphone}));
            ExitCode::SUCCESS
        }
        Commands::Speaker {
            device,
            noise_removal,
            noise_strength,
            echo_removal,
            echo_strength,
        } => {
            let mut cfg = BroadcastConfig::load_from_path(&cfg_path);
            if let Some(d) = device {
                cfg.speaker.device = d;
            }
            if noise_removal.is_some() || noise_strength.is_some() {
                let mut p = Map::new();
                if let Some(s) = noise_strength {
                    p.insert("strength".to_string(), json!(s));
                }
                let _ = cfg.set_effect("speaker_noise_removal", noise_removal, Some(&p), true);
            }
            if echo_removal.is_some() || echo_strength.is_some() {
                let mut p = Map::new();
                if let Some(s) = echo_strength {
                    p.insert("strength".to_string(), json!(s));
                }
                let _ = cfg.set_effect("speaker_echo_removal", echo_removal, Some(&p), true);
            }
            let _ = cfg.save_to_path(&cfg_path);
            emit_json(&json!({"ok": true, "speaker": cfg.speaker}));
            ExitCode::SUCCESS
        }
        Commands::Camera {
            device,
            output_device,
            on_demand,
            idle_timeout,
            blur,
            blur_strength,
            blur_mode,
            replace,
            remove,
        } => {
            let mut cfg = BroadcastConfig::load_from_path(&cfg_path);
            if let Some(d) = device {
                cfg.camera.device = d;
            }
            if let Some(o) = output_device {
                cfg.camera.output_device = o;
            }
            if let Some(od) = on_demand {
                cfg.camera.on_demand = od;
            }
            if let Some(t) = idle_timeout {
                cfg.camera.idle_timeout_sec = clamp_float(t, 0.5, 60.0, 3.0);
            }
            if blur.is_some() || blur_strength.is_some() || blur_mode.is_some() {
                let mut p = Map::new();
                if let Some(s) = blur_strength {
                    p.insert("strength".to_string(), json!(s));
                }
                if let Some(m) = blur_mode {
                    p.insert("mode".to_string(), json!(m));
                }
                let _ = cfg.set_effect("background_blur", blur, Some(&p), true);
            }
            if let Some(r) = replace {
                let _ = cfg.set_effect("background_replace", Some(r), None, true);
            }
            if let Some(r) = remove {
                let _ = cfg.set_effect("background_remove", Some(r), None, true);
            }
            let _ = cfg.save_to_path(&cfg_path);
            emit_json(&json!({"ok": true, "camera": cfg.camera}));
            ExitCode::SUCCESS
        }
        Commands::Serve {
            host,
            port,
            token,
            autostart,
            daemon_file,
        } => {
            let daemon = find_sibling_daemon();
            let mut c = Command::new(daemon);
            c.args([
                "--host",
                &host,
                "--port",
                &port.to_string(),
                "--config",
                &cfg_path.to_string_lossy(),
            ]);
            if let Some(tok) = token {
                c.args(["--token", &tok]);
            }
            if autostart {
                c.arg("--autostart");
            }
            if daemon_file {
                c.arg("--daemon-file");
            }
            match c.status() {
                Ok(st) if st.success() => ExitCode::SUCCESS,
                Ok(st) => ExitCode::from(st.code().unwrap_or(1) as u8),
                Err(e) => {
                    eprintln!("failed to launch nvb-daemon: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::Bench { sub } => {
            let res = match sub {
                BenchCmd::Capabilities => Ok(capabilities_payload()),
                BenchCmd::ConfigBench { iterations } => Ok(run_config_bench(iterations.max(1))),
                BenchCmd::PcmBench { frames } => Ok(run_pcm_bench(frames.max(1))),
                BenchCmd::AudioFile {
                    in_path,
                    out_path,
                    config,
                } => run_audio_file_bench(&in_path, &out_path, &config),
                BenchCmd::AudioLoop { frames } => Ok(run_audio_loop_bench(frames.max(1))),
                BenchCmd::Reliability => Ok(run_reliability_checks()),
            };
            match res {
                Ok(val) => {
                    emit_json(&val);
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_capabilities_reports_r2_contract() {
        let val = capabilities_payload();
        assert_eq!(val["target"], "rust");
        assert_eq!(val["capabilities"]["config_bench"], true);
        assert_eq!(val["capabilities"]["pcm_bench"], true);
        assert_eq!(val["capabilities"]["reliability_core"], true);
        assert_eq!(val["capabilities"]["audio_dsp"], true);
        assert_eq!(val["capabilities"]["daemon_http"], true);
        assert_eq!(val["capabilities"]["cli_full"], true);
        assert_eq!(val["pending_phases"]["video"], "pending_phase_r3");
    }

    #[test]
    fn bench_config_pcm_audio_loop_and_reliability_succeed() {
        let cfg = run_config_bench(25);
        assert_eq!(cfg["status"], "ok");
        assert_eq!(cfg["iterations"], 25);
        assert_eq!(cfg["last_conflicts_disabled"], 2);

        let pcm = run_pcm_bench(20);
        assert_eq!(pcm["status"], "ok");
        assert_eq!(pcm["frames"], 20);

        let aloop = run_audio_loop_bench(30);
        assert_eq!(aloop["status"], "ok");
        assert_eq!(aloop["backend"], "nnnoiseless");
        assert_eq!(aloop["workloads"]["noise_removal"]["frames"], 30);

        let rel = run_reliability_checks();
        assert_eq!(rel["status"], "ok");
        assert_eq!(rel["passed"], rel["total"]);
    }
}
