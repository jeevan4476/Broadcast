//! Native Rust offline audio & video file processor (`FileProcessor`).
//!
//! Mirrors `nv_broadcast/file_processor.py` (NVIDIA Broadcast v2.3.0 Files tab):
//!   - Audio files (`.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`):
//!     `ffmpeg` decode -> pure-Rust `AudioProcessor` (`nnnoiseless` + STFT echo + Studio Voice)
//!     -> `.wav` copy or `ffmpeg` encode.
//!   - Video / screen-recording files (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`):
//!     1. Clean voice track with `AudioProcessor` when `apply_audio` is enabled.
//!     2. If `apply_video` is enabled and at least one camera effect is active,
//!        delegate video frames to an isolated `nvb-camera-worker process-video` child
//!        so ONNX Runtime / CUDA RAM & VRAM are 100% reclaimed by the OS on exit.
//!     3. Otherwise stream-copy video (`-c:v copy`) for instant lossless screen-recording cleanup.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use nvb_core::config::BroadcastConfig;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{write_wav_mono16, AudioProcessConfig, AudioProcessor, SAMPLE_RATE};

pub const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus"];
pub const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "webm", "mov", "avi"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileJobState {
    pub active: bool,
    pub status: String, // idle | running | completed | cancelled | failed
    pub progress: f64,
    pub input_path: String,
    pub output_path: String,
    pub media_kind: String,
    pub message: String,
}

impl Default for FileJobState {
    fn default() -> Self {
        Self {
            active: false,
            status: "idle".to_string(),
            progress: 0.0,
            input_path: String::new(),
            output_path: String::new(),
            media_kind: String::new(),
            message: String::new(),
        }
    }
}

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(prefix: &str) -> Result<Self, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = env::temp_dir().join(format!("{prefix}{}_{}", std::process::id(), nanos));
        fs::create_dir_all(&dir).map_err(|e| format!("failed creating temp dir: {e}"))?;
        Ok(Self(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn expand_user(raw: &str) -> PathBuf {
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if raw == "~" {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(raw)
}

fn command_exists(bin: &str) -> bool {
    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&paths).any(|dir| {
        let cand = dir.join(bin);
        cand.is_file()
    })
}

fn media_kind_for_ext(ext: &str) -> Option<&'static str> {
    let low = ext.to_ascii_lowercase();
    if AUDIO_EXTENSIONS.contains(&low.as_str()) {
        Some("audio")
    } else if VIDEO_EXTENSIONS.contains(&low.as_str()) {
        Some("video")
    } else {
        None
    }
}

/// Returns true when at least one visual effect in `cfg.camera` is turned on.
pub fn has_enabled_video_effects(cfg: &BroadcastConfig) -> bool {
    if cfg.camera.mirror {
        return true;
    }
    let Ok(effects_val) = serde_json::to_value(&cfg.camera.effects) else {
        return false;
    };
    let Some(effects) = effects_val.as_object() else {
        return false;
    };
    for key in [
        "background_blur",
        "background_replace",
        "background_remove",
        "auto_frame",
        "video_noise_removal",
        "virtual_key_light",
        "vignette",
    ] {
        if effects
            .get(key)
            .and_then(Value::as_object)
            .and_then(|o| o.get("enabled"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

fn find_camera_worker_bin() -> Option<PathBuf> {
    if let Ok(p) = env::var("NVB_CAMERA_WORKER") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            let pb = PathBuf::from(trimmed);
            if pb.is_file() {
                return Some(pb);
            }
        }
    }
    env::current_exe()
        .ok()
        .map(|p| p.with_file_name("nvb-camera-worker"))
        .filter(|p| p.is_file())
}

fn terminate_pid(pid: u32) {
    let _ = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[derive(Clone)]
pub struct FileProcessor {
    state: Arc<Mutex<FileJobState>>,
    cancel_flag: Arc<AtomicBool>,
    active_child_pid: Arc<Mutex<Option<u32>>>,
}

impl Default for FileProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl FileProcessor {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(FileJobState::default())),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            active_child_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub fn get_state(&self) -> FileJobState {
        self.state
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        if let Ok(guard) = self.active_child_pid.lock() {
            if let Some(pid) = *guard {
                terminate_pid(pid);
            }
        }
    }

    fn update_state(&self, f: impl FnOnce(&mut FileJobState)) {
        if let Ok(mut guard) = self.state.lock() {
            f(&mut guard);
        }
    }

    fn set_child_pid(&self, pid: Option<u32>) {
        if let Ok(mut guard) = self.active_child_pid.lock() {
            *guard = pid;
        }
    }

    fn prepare_job(
        &self,
        input_path: &str,
        output_path: Option<&str>,
    ) -> Result<(PathBuf, PathBuf, String), String> {
        {
            let guard = self
                .state
                .lock()
                .map_err(|_| "file processor state lock poisoned".to_string())?;
            if guard.active {
                return Err("A file processing job is already running".to_string());
            }
        }

        let src = expand_user(input_path);
        if !src.is_file() {
            return Err(format!("Input file not found: {}", src.display()));
        }

        let ext = src
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let Some(kind) = media_kind_for_ext(&ext) else {
            return Err(format!("Unsupported media format: .{ext}"));
        };

        if !command_exists("ffmpeg") {
            return Err("ffmpeg is required for offline file processing".to_string());
        }

        let dst = if let Some(out) = output_path.filter(|s| !s.trim().is_empty()) {
            expand_user(out)
        } else {
            let home = env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            let out_dir = if kind == "video" {
                home.join("Videos").join("NVIDIA Broadcast")
            } else {
                home.join("Music").join("NVIDIA Broadcast")
            };
            fs::create_dir_all(&out_dir)
                .map_err(|e| format!("failed creating output directory: {e}"))?;
            let stem = src
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let out_ext = if kind == "audio" && ext == "wav" {
                "wav"
            } else if kind == "audio" {
                "mp3"
            } else {
                "mp4"
            };
            out_dir.join(format!("{stem}_broadcast.{out_ext}"))
        };

        Ok((src, dst, kind.to_string()))
    }

    /// Start an asynchronous background file processing job (used by `nvb-daemon`).
    pub fn start_job(
        &self,
        cfg: BroadcastConfig,
        input_path: &str,
        output_path: Option<&str>,
        apply_audio: bool,
        apply_video: bool,
    ) -> (bool, String) {
        let (src, dst, kind) = match self.prepare_job(input_path, output_path) {
            Ok(v) => v,
            Err(msg) => return (false, msg),
        };

        self.cancel_flag.store(false, Ordering::SeqCst);
        let src_name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();
        let dst_disp = dst.display().to_string();

        self.update_state(|s| {
            s.active = true;
            s.status = "running".to_string();
            s.progress = 0.0;
            s.input_path = src.display().to_string();
            s.output_path = dst_disp.clone();
            s.media_kind = kind.clone();
            s.message = format!("Processing {src_name}...");
        });

        let this = self.clone();
        thread::Builder::new()
            .name("nvb-fileproc".to_string())
            .spawn(move || {
                this.run_job(cfg, &src, &dst, &kind, apply_audio, apply_video);
            })
            .ok();

        (true, format!("Started processing {src_name} -> {dst_disp}"))
    }

    /// Synchronously process a file and return the terminal `FileJobState` (used by `nvb-cli`).
    pub fn process_sync(
        &self,
        cfg: BroadcastConfig,
        input_path: &str,
        output_path: Option<&str>,
        apply_audio: bool,
        apply_video: bool,
    ) -> FileJobState {
        let (src, dst, kind) = match self.prepare_job(input_path, output_path) {
            Ok(v) => v,
            Err(msg) => {
                let st = FileJobState {
                    active: false,
                    status: "failed".to_string(),
                    progress: 0.0,
                    input_path: input_path.to_string(),
                    output_path: output_path.unwrap_or("").to_string(),
                    media_kind: String::new(),
                    message: msg,
                };
                self.update_state(|s| *s = st.clone());
                return st;
            }
        };

        self.cancel_flag.store(false, Ordering::SeqCst);
        self.update_state(|s| {
            s.active = true;
            s.status = "running".to_string();
            s.progress = 0.0;
            s.input_path = src.display().to_string();
            s.output_path = dst.display().to_string();
            s.media_kind = kind.clone();
            s.message = "Starting...".to_string();
        });

        self.run_job(cfg, &src, &dst, &kind, apply_audio, apply_video);
        self.get_state()
    }

    fn run_job(
        &self,
        mut cfg: BroadcastConfig,
        src: &Path,
        dst: &Path,
        kind: &str,
        apply_audio: bool,
        apply_video: bool,
    ) {
        let res = (|| -> Result<(), String> {
            if let Some(parent) = dst.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("failed creating output parent dir: {e}"))?;
                }
            }
            if kind == "audio" {
                self.process_audio_file(&cfg, src, dst)
            } else {
                self.process_video_file(&mut cfg, src, dst, apply_audio, apply_video)
            }
        })();

        self.set_child_pid(None);

        if self.cancel_flag.load(Ordering::SeqCst) {
            self.update_state(|s| {
                s.active = false;
                s.status = "cancelled".to_string();
                s.message = "Cancelled by user".to_string();
            });
            return;
        }

        match res {
            Ok(()) => {
                let dst_name = dst
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("output")
                    .to_string();
                self.update_state(|s| {
                    s.active = false;
                    s.status = "completed".to_string();
                    s.progress = 100.0;
                    s.message = format!("Saved {dst_name}");
                });
            }
            Err(err) => {
                self.update_state(|s| {
                    s.active = false;
                    s.status = "failed".to_string();
                    s.message = err;
                });
            }
        }
    }

    fn denoise_pcm_file(
        &self,
        cfg: &BroadcastConfig,
        src: &Path,
        out_wav: &Path,
        p_start: f64,
        p_end: f64,
    ) -> Result<bool, String> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                &src.to_string_lossy(),
                "-vn",
                "-ac",
                "1",
                "-ar",
                &SAMPLE_RATE.to_string(),
                "-f",
                "s16le",
                "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed spawning ffmpeg audio decode: {e}"))?;

        self.set_child_pid(Some(child.id()));
        let mut raw = Vec::new();
        if let Some(mut stdout) = child.stdout.take() {
            let _ = stdout.read_to_end(&mut raw);
        }
        let status = child
            .wait()
            .map_err(|e| format!("ffmpeg decode wait failed: {e}"))?;
        self.set_child_pid(None);

        if self.cancel_flag.load(Ordering::SeqCst) {
            return Ok(false);
        }
        if !status.success() || raw.is_empty() {
            return Ok(false);
        }

        let mic_val = serde_json::to_value(&cfg.microphone).unwrap_or(Value::Null);
        let dsp_cfg = AudioProcessConfig::from_json_value(&mic_val);
        let mut dsp = AudioProcessor::new(SAMPLE_RATE);

        let total = raw.len();
        let step = 480 * 2 * 10; // 100 ms blocks (4800 mono s16le samples)
        let mut cleaned = Vec::with_capacity(total);

        let mut off = 0usize;
        while off < total {
            if self.cancel_flag.load(Ordering::SeqCst) {
                return Ok(false);
            }
            let end = (off + step).min(total);
            let mut block = &raw[off..end];
            if block.len() % 2 == 1 {
                block = &block[..block.len() - 1];
            }
            let out_block = dsp.process_pcm16(block, &dsp_cfg);
            cleaned.extend_from_slice(&out_block);
            off = end;

            let frac = (off as f64 / total.max(1) as f64).min(1.0);
            let pct = ((p_start + (p_end - p_start) * frac) * 10.0).round() / 10.0;
            self.update_state(|s| s.progress = pct);
        }

        write_wav_mono16(out_wav, &cleaned, SAMPLE_RATE)
            .map_err(|e| format!("failed writing cleaned WAV: {e}"))?;
        Ok(true)
    }

    fn process_audio_file(&self, cfg: &BroadcastConfig, src: &Path, dst: &Path) -> Result<(), String> {
        let td = TempDirGuard::new("nvb-audio-")?;
        let clean_wav = td.path().join("clean.wav");

        self.update_state(|s| {
            s.message = "Cleaning audio with pure-Rust neural RNNoise...".to_string();
        });
        let ok = self.denoise_pcm_file(cfg, src, &clean_wav, 0.0, 90.0)?;
        if self.cancel_flag.load(Ordering::SeqCst) {
            return Ok(());
        }
        if !ok {
            return Err("Failed to decode audio stream with ffmpeg".to_string());
        }

        let dst_ext = dst
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if dst_ext == "wav" {
            fs::copy(&clean_wav, dst).map_err(|e| format!("failed saving wav: {e}"))?;
        } else {
            self.update_state(|s| {
                s.message = "Encoding cleaned audio...".to_string();
                s.progress = 94.0;
            });
            let child = Command::new("ffmpeg")
                .args([
                    "-nostdin",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    &clean_wav.to_string_lossy(),
                    &dst.to_string_lossy(),
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("failed spawning ffmpeg audio encode: {e}"))?;
            self.set_child_pid(Some(child.id()));
            let out = child
                .wait_with_output()
                .map_err(|e| format!("ffmpeg encode wait failed: {e}"))?;
            self.set_child_pid(None);
            if self.cancel_flag.load(Ordering::SeqCst) {
                return Ok(());
            }
            if !out.status.success() {
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                return Err(if err.is_empty() {
                    "ffmpeg audio encode failed".to_string()
                } else {
                    err
                });
            }
        }
        self.update_state(|s| s.progress = 100.0);
        Ok(())
    }

    fn process_video_file(
        &self,
        cfg: &mut BroadcastConfig,
        src: &Path,
        dst: &Path,
        apply_audio: bool,
        apply_video: bool,
    ) -> Result<(), String> {
        let td = TempDirGuard::new("nvb-video-")?;
        let clean_wav = td.path().join("clean_audio.wav");
        let run_video_fx = apply_video && has_enabled_video_effects(cfg);

        let mut has_clean_audio = false;
        if apply_audio {
            self.update_state(|s| {
                s.message = "Cleaning audio track with pure-Rust neural RNNoise...".to_string();
            });
            let audio_end = if run_video_fx { 25.0 } else { 85.0 };
            has_clean_audio = self.denoise_pcm_file(cfg, src, &clean_wav, 0.0, audio_end)?;
            if self.cancel_flag.load(Ordering::SeqCst) {
                return Ok(());
            }
        }

        let mut video_src = src.to_path_buf();
        let mut video_was_processed = false;
        if run_video_fx {
            let worker_bin = find_camera_worker_bin().ok_or_else(|| {
                "nvb-camera-worker binary not found for offline video effects".to_string()
            })?;
            let cfg_file = td.path().join("camera_cfg.json");
            cfg.save_to_path(&cfg_file)
                .map_err(|e| format!("failed writing worker config: {e}"))?;
            let proc_vid = td.path().join("processed_video.mp4");

            self.update_state(|s| {
                s.message = "Applying neural video effects in isolated worker...".to_string();
            });

            let mut child = Command::new(&worker_bin)
                .args([
                    "process-video",
                    "--in",
                    &src.to_string_lossy(),
                    "--out",
                    &proc_vid.to_string_lossy(),
                    "--config",
                    &cfg_file.to_string_lossy(),
                    "--backend",
                    &cfg.segmentation_backend,
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("failed spawning nvb-camera-worker: {e}"))?;

            self.set_child_pid(Some(child.id()));
            if let Some(stdout) = child.stdout.take() {
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    if let Some(rest) = line.strip_prefix("PROGRESS ") {
                        if let Ok(frac) = rest.trim().parse::<f64>() {
                            let pct = ((25.0 + 65.0 * frac.clamp(0.0, 1.0)) * 10.0).round() / 10.0;
                            self.update_state(|s| s.progress = pct);
                        }
                    }
                }
            }
            let out = child
                .wait_with_output()
                .map_err(|e| format!("nvb-camera-worker wait failed: {e}"))?;
            self.set_child_pid(None);

            if self.cancel_flag.load(Ordering::SeqCst) {
                return Ok(());
            }
            if !out.status.success() || !proc_vid.is_file() {
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                return Err(if err.is_empty() {
                    "offline video effects worker failed".to_string()
                } else {
                    err
                });
            }
            video_src = proc_vid;
            video_was_processed = true;
        }

        self.update_state(|s| {
            s.message = "Muxing final output...".to_string();
            s.progress = 92.0;
        });

        let mut args: Vec<String> = vec![
            "-nostdin".into(),
            "-y".into(),
            "-hide_banner".into(),
            "-loglevel".into(),
            "error".into(),
            "-i".into(),
            video_src.to_string_lossy().into_owned(),
        ];

        if has_clean_audio {
            args.extend([
                "-i".into(),
                clean_wav.to_string_lossy().into_owned(),
                "-map".into(),
                "0:v:0".into(),
                "-map".into(),
                "1:a:0".into(),
                "-c:v".into(),
                "copy".into(),
                "-c:a".into(),
                "aac".into(),
                "-b:a".into(),
                "192k".into(),
                "-shortest".into(),
                dst.to_string_lossy().into_owned(),
            ]);
        } else if video_was_processed {
            // Preserve original audio track from `src` when `apply_audio == false`
            args.extend([
                "-i".into(),
                src.to_string_lossy().into_owned(),
                "-map".into(),
                "0:v:0".into(),
                "-map".into(),
                "1:a?".into(),
                "-c:v".into(),
                "copy".into(),
                "-c:a".into(),
                "copy".into(),
                "-shortest".into(),
                dst.to_string_lossy().into_owned(),
            ]);
        } else {
            args.extend([
                "-c".into(),
                "copy".into(),
                dst.to_string_lossy().into_owned(),
            ]);
        }

        let child = Command::new("ffmpeg")
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed spawning ffmpeg mux: {e}"))?;
        self.set_child_pid(Some(child.id()));
        let out = child
            .wait_with_output()
            .map_err(|e| format!("ffmpeg mux wait failed: {e}"))?;
        self.set_child_pid(None);

        if self.cancel_flag.load(Ordering::SeqCst) {
            return Ok(());
        }
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            return Err(if err.is_empty() {
                "ffmpeg mux failed".to_string()
            } else {
                err
            });
        }
        self.update_state(|s| s.progress = 100.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_sample_clip, read_wav_mono16};

    #[test]
    fn rejects_missing_and_unsupported_files() {
        let fp = FileProcessor::new();
        let cfg = BroadcastConfig::default();

        let (ok, msg) = fp.start_job(cfg.clone(), "/nonexistent/audio.wav", None, true, true);
        assert!(!ok);
        assert!(msg.contains("not found"));

        let td = TempDirGuard::new("nvb-fp-badext-").unwrap();
        let bad = td.path().join("notes.txt");
        fs::write(&bad, "hello").unwrap();
        let (ok2, msg2) = fp.start_job(cfg, &bad.to_string_lossy(), None, true, true);
        assert!(!ok2);
        assert!(msg2.contains("Unsupported"));
    }

    #[test]
    fn processes_wav_and_screen_recording_mp4_offline() {
        if !command_exists("ffmpeg") {
            return;
        }
        let td = TempDirGuard::new("nvb-fp-test-").unwrap();
        let in_wav = td.path().join("noisy.wav");
        let out_wav = td.path().join("clean.wav");

        let pcm = generate_sample_clip("fan", SAMPLE_RATE, 1.0);
        write_wav_mono16(&in_wav, &pcm, SAMPLE_RATE).unwrap();

        let fp = FileProcessor::new();
        let cfg = BroadcastConfig::default();
        let st = fp.process_sync(
            cfg.clone(),
            &in_wav.to_string_lossy(),
            Some(&out_wav.to_string_lossy()),
            true,
            false,
        );
        assert_eq!(st.status, "completed", "wav job failed: {}", st.message);
        assert_eq!(st.progress, 100.0);
        assert_eq!(st.media_kind, "audio");
        assert!(out_wav.is_file());
        let (out_pcm, sr) = read_wav_mono16(&out_wav).unwrap();
        assert_eq!(sr, SAMPLE_RATE);
        assert!(!out_pcm.is_empty());

        // Synthetic 1s 320x240 screen-recording MP4 + noisy voice track -> fast lossless video stream-copy + cleaned voice
        let in_mp4 = td.path().join("screenrec.mp4");
        let out_mp4 = td.path().join("screenrec_clean.mp4");
        let gen = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=15",
                "-i",
                &in_wav.to_string_lossy(),
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                &in_mp4.to_string_lossy(),
            ])
            .status()
            .unwrap();
        assert!(gen.success(), "failed generating synthetic test mp4");
        let st2 = fp.process_sync(
            cfg,
            &in_mp4.to_string_lossy(),
            Some(&out_mp4.to_string_lossy()),
            true,
            false,
        );
        assert_eq!(st2.status, "completed", "mp4 job failed: {}", st2.message);
        assert_eq!(st2.media_kind, "video");
        assert!(out_mp4.is_file());
        assert!(fs::metadata(&out_mp4).unwrap().len() > 1000);
    }
}
