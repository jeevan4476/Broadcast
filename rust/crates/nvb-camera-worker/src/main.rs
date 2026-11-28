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
    /// Offline process a local video file through Rust neural effects (`PROGRESS <0..1>` on stdout)
    ProcessVideo {
        #[arg(long = "in")]
        in_path: PathBuf,
        #[arg(long = "out")]
        out_path: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
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

#[derive(Debug, Clone, PartialEq)]
struct VideoMeta {
    width: usize,
    height: usize,
    fps: f64,
    total_frames: usize,
}

fn parse_rate_fraction(s: &str) -> Option<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.trim().parse().ok()?;
        let d: f64 = den.trim().parse().ok()?;
        if d > 0.0 && n > 0.0 {
            return Some(n / d);
        }
    }
    s.trim().parse::<f64>().ok().filter(|&v| v > 0.0)
}

fn parse_ffprobe_meta(json_text: &str) -> Result<VideoMeta, String> {
    let val: Value =
        serde_json::from_str(json_text).map_err(|e| format!("invalid ffprobe JSON: {e}"))?;
    let stream = val
        .get("streams")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .ok_or_else(|| "no video stream found in input".to_string())?;
    let width = stream
        .get("width")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing video width".to_string())? as usize;
    let height = stream
        .get("height")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing video height".to_string())? as usize;
    let fps = stream
        .get("avg_frame_rate")
        .and_then(Value::as_str)
        .and_then(parse_rate_fraction)
        .or_else(|| {
            stream
                .get("r_frame_rate")
                .and_then(Value::as_str)
                .and_then(parse_rate_fraction)
        })
        .unwrap_or(30.0)
        .clamp(1.0, 240.0);

    let nb_frames = stream
        .get("nb_frames")
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| {
            stream
                .get("duration")
                .and_then(Value::as_str)
                .and_then(|s| s.parse::<f64>().ok())
                .map(|d| (d * fps).round() as usize)
        })
        .unwrap_or(1)
        .max(1);

    Ok(VideoMeta {
        width,
        height,
        fps,
        total_frames: nb_frames,
    })
}

fn probe_video_meta(path: &Path) -> Result<VideoMeta, String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            &path.to_string_lossy(),
        ])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("ffprobe failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "ffprobe failed for '{}': {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    parse_ffprobe_meta(&String::from_utf8_lossy(&out.stdout))
}

fn ffmpeg_encoder_list() -> String {
    let Ok(out) = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .stdin(Stdio::null())
        .output()
    else {
        return String::new();
    };
    if !out.status.success() {
        return String::new();
    }
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn select_video_encoder(allow_nvenc: bool) -> &'static str {
    let encs = ffmpeg_encoder_list();
    if allow_nvenc && encs.contains("h264_nvenc") {
        "h264_nvenc"
    } else if encs.contains("libx264") {
        "libx264"
    } else if encs.contains("libopenh264") {
        "libopenh264"
    } else {
        "mpeg4"
    }
}

fn spawn_video_encoder(
    out_path: &Path,
    width: usize,
    height: usize,
    fps: f64,
    encoder: &str,
) -> Result<std::process::Child, String> {
    let size = format!("{width}x{height}");
    let fps_str = format!("{fps:.3}");
    let out_str = out_path.to_string_lossy().into_owned();
    let mut args = vec![
        "-nostdin".to_string(),
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-video_size".to_string(),
        size,
        "-framerate".to_string(),
        fps_str,
        "-i".to_string(),
        "-".to_string(),
    ];
    match encoder {
        "h264_nvenc" => args.extend([
            "-c:v".into(),
            "h264_nvenc".into(),
            "-preset".into(),
            "p4".into(),
            "-cq".into(),
            "20".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            out_str,
        ]),
        "libx264" => args.extend([
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            "fast".into(),
            "-crf".into(),
            "20".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            out_str,
        ]),
        "libopenh264" => args.extend([
            "-c:v".into(),
            "libopenh264".into(),
            "-b:v".into(),
            "4M".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            out_str,
        ]),
        _ => args.extend([
            "-c:v".into(),
            "mpeg4".into(),
            "-q:v".into(),
            "4".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            out_str,
        ]),
    }
    Command::new("ffmpeg")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed spawning ffmpeg video encoder ({encoder}): {e}"))
}

fn run_process_video(
    in_path: &Path,
    out_path: &Path,
    config_path: Option<&Path>,
    backend: &str,
) -> Result<(), String> {
    let meta = probe_video_meta(in_path)?;
    let mut cam_cfg = if let Some(cp) = config_path {
        let bc = BroadcastConfig::load_from_path(cp);
        serde_json::to_value(&bc.camera)
            .map(|v| CameraEffectsConfig::from_json_value(&v))
            .unwrap_or_default()
    } else {
        CameraEffectsConfig::default()
    };
    cam_cfg.split_preview = false;

    let mut pipe = VideoEffectsPipeline::new(backend);
    let mut dec = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &in_path.to_string_lossy(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed spawning ffmpeg video decoder: {e}"))?;

    let encoder = select_video_encoder(backend != "cpu");
    let mut enc = spawn_video_encoder(out_path, meta.width, meta.height, meta.fps, encoder)?;

    let frame_bytes = meta.width * meta.height * 3;
    let mut raw_buf = vec![0u8; frame_bytes];
    let mut idx = 0usize;

    if let (Some(mut dec_out), Some(mut enc_in)) = (dec.stdout.take(), enc.stdin.take()) {
        while dec_out.read_exact(&mut raw_buf).is_ok() {
            if let Some(frame) = RgbFrame::from_rgb_vec(meta.width, meta.height, raw_buf.clone()) {
                let processed = pipe.process_frame(&frame, &cam_cfg);
                if enc_in.write_all(&processed.data).is_err() {
                    break;
                }
            }
            idx += 1;
            if idx.is_multiple_of(5) {
                let frac = (idx as f64 / meta.total_frames as f64).min(1.0);
                println!("PROGRESS {frac:.4}");
                let _ = std::io::stdout().flush();
            }
        }
    }

    let _ = dec.wait();
    let enc_out = enc
        .wait_with_output()
        .map_err(|e| format!("encoder wait failed: {e}"))?;
    if !enc_out.status.success() {
        // Retry once with software encoder if NVENC failed (e.g. headless/no NVENC session)
        if encoder == "h264_nvenc" {
            return run_process_video(in_path, out_path, config_path, "cpu");
        }
        let err = String::from_utf8_lossy(&enc_out.stderr).trim().to_string();
        return Err(if err.is_empty() {
            "ffmpeg video encode failed".to_string()
        } else {
            err
        });
    }
    println!("PROGRESS 1.0000");
    let _ = std::io::stdout().flush();
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
        WorkerCmd::ProcessVideo {
            in_path,
            out_path,
            config,
            backend,
        } => run_process_video(&in_path, &out_path, config.as_deref(), &backend).map(|()| None),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ffprobe_metadata() {
        let sample = r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"60000/1001","nb_frames":"180"}]}"#;
        let meta = parse_ffprobe_meta(sample).unwrap();
        assert_eq!(meta.width, 1920);
        assert_eq!(meta.height, 1080);
        assert!((meta.fps - 59.94).abs() < 0.05);
        assert_eq!(meta.total_frames, 180);
    }
}
