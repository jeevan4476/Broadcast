//! Native Rust isolated on-demand GPU/CPU camera worker (`nvb-camera-worker`).
//!
//! Spawned by `nvb-daemon` only while a video call or preview is active so the
//! OS reclaims 100% of ONNX Runtime / CUDA / cuDNN RAM and GPU VRAM the instant
//! the worker exits.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use nvb_core::config::BroadcastConfig;
use nvb_core::{APP_NAME, APP_VERSION};
use nvb_video::{
    build_preset_background, CameraEffectsConfig, RgbFrame, VideoEffectsPipeline,
};
use serde_json::{json, Map, Value};

#[derive(Parser, Debug)]
#[command(
    name = "nvb-camera-worker",
    version = APP_VERSION,
    about = "Isolated on-demand GPU/CPU camera effects worker for NVIDIA Broadcast (Linux)"
)]
struct WorkerCli {
    #[command(subcommand)]
    command: Option<WorkerCmd>,
}

#[derive(Subcommand, Debug)]
enum WorkerCmd {
    /// Stream frames from physical V4L2 camera through Rust effects to v4l2loopback
    Stream {
        #[arg(long, default_value = "/dev/video0")]
        input: String,
        #[arg(long, default_value = "/dev/video10")]
        output: String,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 1280)]
        width: usize,
        #[arg(long, default_value_t = 720)]
        height: usize,
        #[arg(long, default_value_t = 30)]
        fps: u32,
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// Run a real video inference burst and hold until SIGTERM (for memory lifecycle tests)
    Hold {
        #[arg(long, default_value = "auto")]
        backend: String,
        #[arg(long, default_value_t = 1280)]
        width: usize,
        #[arg(long, default_value_t = 720)]
        height: usize,
        #[arg(long, default_value_t = 5)]
        frames: usize,
    },
    /// Benchmark all 6 standard video effects on PNG frame sequences in `--in-dir`
    Bench {
        #[arg(long)]
        in_dir: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, default_value = "auto")]
        backend: String,
    },
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

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn summarize_ms(mut vals: Vec<f64>) -> Value {
    vals.sort_by(|a, b| a.total_cmp(b));
    let count = vals.len();
    let mean = if count > 0 {
        vals.iter().sum::<f64>() / count as f64
    } else {
        0.0
    };
    json!({
        "count": count,
        "min": (percentile(&vals, 0.0) * 1000.0).round() / 1000.0,
        "mean": (mean * 1000.0).round() / 1000.0,
        "p50": (percentile(&vals, 0.50) * 1000.0).round() / 1000.0,
        "p95": (percentile(&vals, 0.95) * 1000.0).round() / 1000.0,
        "p99": (percentile(&vals, 0.99) * 1000.0).round() / 1000.0,
        "p999": (percentile(&vals, 0.999) * 1000.0).round() / 1000.0,
        "max": (percentile(&vals, 1.0) * 1000.0).round() / 1000.0,
    })
}

fn load_frame_sequence(dir: &Path, prefix: &str) -> Result<Vec<RgbFrame>, String> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("failed reading '{}': {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".png"))
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no '{prefix}*.png' frames found in {}", dir.display()));
    }
    let mut frames = Vec::with_capacity(paths.len());
    for p in paths {
        frames.push(RgbFrame::load_from_path(&p)?);
    }
    Ok(frames)
}

fn save_alpha_png(alpha: &[f32], w: usize, h: usize, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let raw: Vec<u8> = alpha
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let img = image::GrayImage::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| "invalid alpha dimensions".to_string())?;
    img.save(path).map_err(|e| e.to_string())
}

fn run_bench(in_dir: &Path, out_dir: &Path, backend: &str) -> Result<Value, String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let clean_frames = load_frame_sequence(in_dir, "clean_")?;
    let noisy_frames = load_frame_sequence(in_dir, "noisy_").unwrap_or_else(|_| clean_frames.clone());
    let w = clean_frames[0].width;
    let h = clean_frames[0].height;

    let mut pipe = VideoEffectsPipeline::new(backend);
    let seg_backend = pipe.segmenter.backend_name;

    // Save quality alpha matte on first clean frame for exact Python-vs-Rust IoU/MAE parity
    let alpha0 = pipe.segmenter.segment_alpha(&clean_frames[0], None, "quality");
    let alpha_path = out_dir.join("alpha_matte.png");
    save_alpha_png(&alpha0, w, h, &alpha_path)?;

    let specs: [(&str, bool, Value); 6] = [
        (
            "background_blur_perf",
            false,
            json!({"effects": {"background_blur": {"enabled": true, "strength": 0.75, "mode": "performance"}}}),
        ),
        (
            "background_blur_quality",
            false,
            json!({"effects": {"background_blur": {"enabled": true, "strength": 0.75, "mode": "quality"}}}),
        ),
        (
            "background_replace",
            false,
            json!({"effects": {"background_replace": {"enabled": true, "preset": "studio_dark", "imagePath": ""}}}),
        ),
        (
            "background_remove",
            false,
            json!({"effects": {"background_remove": {"enabled": true, "chromaColor": "#00ff00"}}}),
        ),
        (
            "video_noise_removal",
            true,
            json!({"effects": {"video_noise_removal": {"enabled": true, "strength": 0.65}}}),
        ),
        (
            "combo_blur_keylight_autoframe",
            false,
            json!({"effects": {
                "background_blur": {"enabled": true, "strength": 0.7, "mode": "performance"},
                "virtual_key_light": {"enabled": true, "warmth": 0.5, "intensity": 0.55},
                "auto_frame": {"enabled": true, "zoom": 0.35}
            }}),
        ),
    ];

    let mut effects_out = Map::new();
    for (ename, use_noisy, cfg_val) in &specs {
        pipe.reset();
        let cfg = CameraEffectsConfig::from_json_value(cfg_val);
        let src_frames = if *use_noisy { &noisy_frames } else { &clean_frames };

        // 2 warmup frames
        for wf in src_frames.iter().take(2) {
            std::hint::black_box(pipe.process_frame(wf, &cfg));
        }

        let mut lats_ms = Vec::with_capacity(src_frames.len());
        let mut last_out = src_frames[0].clone();
        for fr in src_frames {
            let t0 = Instant::now();
            last_out = pipe.process_frame(fr, &cfg);
            lats_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let out_png = out_dir.join(format!("{ename}.png"));
        last_out.save_png(&out_png)?;

        let stats = summarize_ms(lats_ms.clone());
        let mean_ms = stats["mean"].as_f64().unwrap_or(1.0).max(0.001);
        let fps = ((1000.0 / mean_ms) * 100.0).round() / 100.0;
        let misses_33 = lats_ms.iter().filter(|&&v| v > 33.333).count();

        effects_out.insert(
            (*ename).to_string(),
            json!({
                "status": "ok",
                "frames": src_frames.len(),
                "fps": fps,
                "latency_ms": stats,
                "budget_misses_33ms": misses_33,
                "out_png": out_png.to_string_lossy(),
            }),
        );
    }

    Ok(json!({
        "status": "ok",
        "backend": seg_backend,
        "face_backend": "ycbcr_skin_tracker",
        "resolution": format!("{w}x{h}"),
        "rss_kb": read_self_rss_kb(),
        "alpha_png": alpha_path.to_string_lossy(),
        "effects": effects_out,
    }))
}

fn run_hold(backend: &str, width: usize, height: usize, frames: usize) -> Result<(), String> {
    let mut pipe = VideoEffectsPipeline::new(backend);
    let cfg = CameraEffectsConfig::from_json_value(&json!({
        "effects": {"background_blur": {"enabled": true, "strength": 0.75, "mode": "quality"}}
    }));
    let frame = build_preset_background("office_blur", width, height);
    for _ in 0..frames.max(1) {
        std::hint::black_box(pipe.process_frame(&frame, &cfg));
    }
    println!("READY {} rss_kb={}", pipe.segmenter.backend_name, read_self_rss_kb());
    let _ = std::io::stdout().flush();
    loop {
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn run_stream(
    input: &str,
    output: &str,
    config_path: Option<&Path>,
    width: usize,
    height: usize,
    fps: u32,
    backend: &str,
) -> Result<(), String> {
    let mut pipe = VideoEffectsPipeline::new(backend);
    let mut cap = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
            "-i",
            input,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed launching ffmpeg capture: {e}"))?;

    let mut out_proc = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuyv422",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
            "-i",
            "-",
            "-f",
            "v4l2",
            "-pix_fmt",
            "yuyv422",
            output,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed launching ffmpeg v4l2loopback writer: {e}"))?;

    let frame_bytes = width * height * 3;
    let mut raw_buf = vec![0u8; frame_bytes];
    let mut last_cfg_poll = Instant::now() - Duration::from_secs(5);
    let mut cam_cfg = CameraEffectsConfig::default();

    if let (Some(mut stdout), Some(mut stdin)) = (cap.stdout.take(), out_proc.stdin.take()) {
        while stdout.read_exact(&mut raw_buf).is_ok() {
            if let Some(cp) = config_path {
                if last_cfg_poll.elapsed() >= Duration::from_millis(500) {
                    last_cfg_poll = Instant::now();
                    let bc = BroadcastConfig::load_from_path(cp);
                    if let Ok(val) = serde_json::to_value(&bc.camera) {
                        cam_cfg = CameraEffectsConfig::from_json_value(&val);
                    }
                }
            }
            if let Some(frame) = RgbFrame::from_rgb_vec(width, height, raw_buf.clone()) {
                let processed = pipe.process_frame(&frame, &cam_cfg);
                let yuyv = processed.to_yuyv();
                if stdin.write_all(&yuyv).is_err() {
                    break;
                }
            }
        }
    }

    let _ = cap.kill();
    let _ = cap.wait();
    let _ = out_proc.kill();
    let _ = out_proc.wait();
    Ok(())
}

fn main() -> ExitCode {
    let cli = WorkerCli::parse();
    let Some(cmd) = cli.command else {
        println!("{APP_NAME} nvb-camera-worker {APP_VERSION}");
        return ExitCode::SUCCESS;
    };
    let res = match cmd {
        WorkerCmd::Stream {
            input,
            output,
            config,
            width,
            height,
            fps,
            backend,
        } => run_stream(&input, &output, config.as_deref(), width, height, fps, &backend).map(|()| None),
        WorkerCmd::Hold {
            backend,
            width,
            height,
            frames,
        } => run_hold(&backend, width, height, frames).map(|()| None),
        WorkerCmd::Bench {
            in_dir,
            out_dir,
            backend,
        } => run_bench(&in_dir, &out_dir, &backend).map(Some),
    };

    match res {
        Ok(Some(val)) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&val).unwrap_or_else(|_| "{}".to_string())
            );
            ExitCode::SUCCESS
        }
        Ok(None) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
