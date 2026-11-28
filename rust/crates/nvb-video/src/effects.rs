//! Pure-Rust GPU-accelerated (ONNX Runtime RVM CUDA/CPU) and fixed-point
//! camera effects pipeline (`nvb-video`).
//!
//! Zero OpenCV, MediaPipe, or Python runtime dependencies:
//!   - Portrait matting: `ort` 2.x (`load-dynamic`) running `rvm_mobilenetv3_fp32.onnx`
//!     on RTX CUDA (`CUDAExecutionProvider`) -> CPU (`CPUExecutionProvider`) ->
//!     `fast_portrait` pure-Rust fallback when ONNX Runtime or model is absent.
//!   - Video effects: `background_blur` (performance/quality), `background_replace`,
//!     `background_remove`, `auto_frame`, `video_noise_removal` (edge-preserving
//!     spatial + motion-aware temporal filter), `virtual_key_light`, `vignette`,
//!     `mirror`, and `split_preview`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use nvb_core::models::model_path;
use ort::environment::Environment;
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Packed 8-bit RGB24 video frame.
#[derive(Debug, Clone)]
pub struct RgbFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbFrame {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; width * height * 3],
        }
    }

    pub fn from_rgb_vec(width: usize, height: usize, data: Vec<u8>) -> Option<Self> {
        if data.len() == width * height * 3 {
            Some(Self {
                width,
                height,
                data,
            })
        } else {
            None
        }
    }

    pub fn load_from_path(path: &Path) -> Result<Self, String> {
        let img = image::open(path)
            .map_err(|e| format!("failed opening image '{}': {e}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        Ok(Self {
            width: w as usize,
            height: h as usize,
            data: img.into_raw(),
        })
    }

    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let img = image::RgbImage::from_raw(self.width as u32, self.height as u32, self.data.clone())
            .ok_or_else(|| "invalid RGB buffer dimensions".to_string())?;
        img.save(path).map_err(|e| e.to_string())
    }

    /// Pack RGB24 frame into V4L2 YUYV (4:2:2) bytes for `v4l2loopback` output.
    pub fn to_yuyv(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.width * self.height * 2);
        for row in 0..self.height {
            let row_off = row * self.width * 3;
            let mut x = 0usize;
            while x + 1 < self.width {
                let i0 = row_off + x * 3;
                let i1 = i0 + 3;
                let (r0, g0, b0) = (
                    i32::from(self.data[i0]),
                    i32::from(self.data[i0 + 1]),
                    i32::from(self.data[i0 + 2]),
                );
                let (r1, g1, b1) = (
                    i32::from(self.data[i1]),
                    i32::from(self.data[i1 + 1]),
                    i32::from(self.data[i1 + 2]),
                );
                let y0 = (((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16).clamp(0, 255) as u8;
                let y1 = (((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16).clamp(0, 255) as u8;
                let ra = (r0 + r1) >> 1;
                let ga = (g0 + g1) >> 1;
                let ba = (b0 + b1) >> 1;
                let u = (((-38 * ra - 74 * ga + 112 * ba + 128) >> 8) + 128).clamp(0, 255) as u8;
                let v = (((112 * ra - 94 * ga - 18 * ba + 128) >> 8) + 128).clamp(0, 255) as u8;
                out.extend_from_slice(&[y0, u, y1, v]);
                x += 2;
            }
        }
        out
    }
}

/// Configuration snapshot matching Python `camera_cfg` JSON passed to `VideoEffectsPipeline`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CameraEffectsConfig {
    pub mirror: bool,
    pub split_preview: bool,
    pub split_position: f32,
    pub blur_enabled: bool,
    pub blur_strength: f32,
    pub blur_mode: String,
    pub replace_enabled: bool,
    pub replace_preset: String,
    pub replace_image_path: String,
    pub remove_enabled: bool,
    pub remove_chroma_color: String,
    pub auto_frame_enabled: bool,
    pub auto_frame_zoom: f32,
    pub video_noise_removal_enabled: bool,
    pub video_noise_removal_strength: f32,
    pub key_light_enabled: bool,
    pub key_light_warmth: f32,
    pub key_light_intensity: f32,
    pub vignette_enabled: bool,
    pub vignette_strength: f32,
    pub vignette_face_tracking: bool,
}

impl CameraEffectsConfig {
    pub fn from_json_value(val: &Value) -> Self {
        let mut cfg = Self {
            split_position: 0.5,
            blur_strength: 0.7,
            blur_mode: "quality".to_string(),
            replace_preset: "studio_dark".to_string(),
            remove_chroma_color: "#00ff00".to_string(),
            auto_frame_zoom: 0.35,
            video_noise_removal_strength: 0.6,
            key_light_warmth: 0.5,
            key_light_intensity: 0.55,
            vignette_strength: 0.4,
            vignette_face_tracking: true,
            ..Self::default()
        };
        let Some(obj) = val.as_object() else {
            return cfg;
        };
        cfg.mirror = obj.get("mirror").and_then(Value::as_bool).unwrap_or(false);
        cfg.split_preview = obj.get("split_preview").and_then(Value::as_bool).unwrap_or(false);
        if let Some(p) = obj.get("split_position").and_then(Value::as_f64) {
            cfg.split_position = (p as f32).clamp(0.05, 0.95);
        }
        if let Some(eff) = obj.get("effects").and_then(Value::as_object) {
            if let Some(e) = eff.get("background_blur").and_then(Value::as_object) {
                cfg.blur_enabled = e.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(s) = e.get("strength").and_then(Value::as_f64) {
                    cfg.blur_strength = (s as f32).clamp(0.0, 1.0);
                }
                if let Some(m) = e.get("mode").and_then(Value::as_str) {
                    cfg.blur_mode = m.to_string();
                }
            }
            if let Some(e) = eff.get("background_replace").and_then(Value::as_object) {
                cfg.replace_enabled = e.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(p) = e.get("preset").and_then(Value::as_str) {
                    cfg.replace_preset = p.to_string();
                }
                if let Some(p) = e.get("imagePath").and_then(Value::as_str) {
                    cfg.replace_image_path = p.to_string();
                }
            }
            if let Some(e) = eff.get("background_remove").and_then(Value::as_object) {
                cfg.remove_enabled = e.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(c) = e.get("chromaColor").and_then(Value::as_str) {
                    cfg.remove_chroma_color = c.to_string();
                }
            }
            if let Some(e) = eff.get("auto_frame").and_then(Value::as_object) {
                cfg.auto_frame_enabled = e.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(z) = e.get("zoom").and_then(Value::as_f64) {
                    cfg.auto_frame_zoom = (z as f32).clamp(0.0, 1.0);
                }
            }
            if let Some(e) = eff.get("video_noise_removal").and_then(Value::as_object) {
                cfg.video_noise_removal_enabled =
                    e.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(s) = e.get("strength").and_then(Value::as_f64) {
                    cfg.video_noise_removal_strength = (s as f32).clamp(0.0, 1.0);
                }
            }
            if let Some(e) = eff.get("virtual_key_light").and_then(Value::as_object) {
                cfg.key_light_enabled = e.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(w) = e.get("warmth").and_then(Value::as_f64) {
                    cfg.key_light_warmth = (w as f32).clamp(0.0, 1.0);
                }
                if let Some(i) = e.get("intensity").and_then(Value::as_f64) {
                    cfg.key_light_intensity = (i as f32).clamp(0.0, 1.0);
                }
            }
            if let Some(e) = eff.get("vignette").and_then(Value::as_object) {
                cfg.vignette_enabled = e.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                if let Some(s) = e.get("strength").and_then(Value::as_f64) {
                    cfg.vignette_strength = (s as f32).clamp(0.0, 1.0);
                }
                if let Some(ft) = e.get("faceTracking").and_then(Value::as_bool) {
                    cfg.vignette_face_tracking = ft;
                }
            }
        }
        cfg
    }
}

pub fn parse_hex_color(hex_str: &str) -> [u8; 3] {
    let s = hex_str.trim().trim_start_matches('#');
    if s.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return [r, g, b];
        }
    }
    [0, 255, 0]
}

#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Fast damped face tracker using YCbCr skin-tone cluster detection (<0.2 ms per 720p frame).
pub struct AutoFrameTracker {
    pub cx: f32,
    pub cy: f32,
    deadzone: f32,
    damping: f32,
}

impl Default for AutoFrameTracker {
    fn default() -> Self {
        Self {
            cx: 0.5,
            cy: 0.45,
            deadzone: 0.02,
            damping: 0.14,
        }
    }
}

impl AutoFrameTracker {
    pub fn detect_face(&self, frame: &RgbFrame) -> Option<FaceBox> {
        let w = frame.width;
        let h = frame.height;
        if w == 0 || h == 0 {
            return None;
        }
        let step_x = (w / 80).max(1);
        let step_y = (h / 45).max(1);
        let mut sum_x = 0.0_f32;
        let mut sum_y = 0.0_f32;
        let mut count = 0u32;
        let (mut min_x, mut max_x) = (w, 0usize);
        let (mut min_y, mut max_y) = (h, 0usize);

        let mut y = 0usize;
        while y < h {
            let row = y * w * 3;
            let mut x = 0usize;
            while x < w {
                let idx = row + x * 3;
                let r = f32::from(frame.data[idx]);
                let g = f32::from(frame.data[idx + 1]);
                let b = f32::from(frame.data[idx + 2]);
                let yy = 0.299 * r + 0.587 * g + 0.114 * b;
                let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
                let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
                if yy > 60.0 && (77.0..=127.0).contains(&cb) && (133.0..=173.0).contains(&cr) {
                    sum_x += x as f32;
                    sum_y += y as f32;
                    count += 1;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
                x += step_x;
            }
            y += step_y;
        }

        if count < 12 {
            return None;
        }
        let fw = ((max_x.saturating_sub(min_x) + step_x) as f32 / w as f32).clamp(0.10, 0.65);
        let fh = ((max_y.saturating_sub(min_y) + step_y) as f32 / h as f32).clamp(0.12, 0.70);
        Some(FaceBox {
            cx: (sum_x / count as f32) / w as f32,
            cy: (sum_y / count as f32) / h as f32,
            w: fw,
            h: fh,
        })
    }

    pub fn update(&mut self, face: Option<FaceBox>) -> (f32, f32) {
        if let Some(f) = face {
            let dx = f.cx - self.cx;
            let dy = f.cy - self.cy;
            if dx.abs() > self.deadzone {
                self.cx += dx * self.damping;
            }
            if dy.abs() > self.deadzone {
                self.cy += dy * self.damping;
            }
        } else {
            self.cx += (0.5 - self.cx) * 0.03;
            self.cy += (0.45 - self.cy) * 0.03;
        }
        self.cx = self.cx.clamp(0.15, 0.85);
        self.cy = self.cy.clamp(0.15, 0.85);
        (self.cx, self.cy)
    }
}

static ORT_ENV: OnceLock<Option<Environment>> = OnceLock::new();

/// Locate `libonnxruntime.so*` dynamically at runtime (zero build-time ONNX dependency).
pub fn find_onnxruntime_dylib() -> Option<PathBuf> {
    if let Ok(explicit) = env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(explicit.trim());
        if p.is_file() {
            return Some(p);
        }
    }

    let mut search_dirs: Vec<PathBuf> = vec![
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/local/lib"),
    ];

    if let Ok(xdg) = env::var("XDG_DATA_HOME") {
        if !xdg.is_empty() {
            search_dirs.push(PathBuf::from(xdg).join("nv-broadcast/venv/lib"));
        }
    }
    if let Ok(home) = env::var("HOME") {
        search_dirs.push(PathBuf::from(&home).join(".local/share/nv-broadcast/venv/lib"));
    }
    if let Ok(cwd) = env::current_dir() {
        search_dirs.push(cwd.join(".venv/lib"));
        if let Some(parent) = cwd.parent() {
            search_dirs.push(parent.join("Broadcast/.venv/lib"));
            search_dirs.push(parent.join(".venv/lib"));
        }
    }

    for dir in search_dirs {
        if !dir.is_dir() {
            continue;
        }
        // Check direct libonnxruntime.so*
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with("libonnxruntime.so") && p.is_file() {
                    return Some(p);
                }
                // Check python3.X/site-packages/onnxruntime/capi/libonnxruntime.so*
                if name.starts_with("python3.") && p.is_dir() {
                    let capi = p.join("site-packages/onnxruntime/capi");
                    if let Ok(c_entries) = fs::read_dir(&capi) {
                        for ce in c_entries.flatten() {
                            let cname = ce.file_name().to_string_lossy().into_owned();
                            if cname.starts_with("libonnxruntime.so") && ce.path().is_file() {
                                return Some(ce.path());
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn get_ort_env() -> Option<&'static Environment> {
    ORT_ENV
        .get_or_init(|| {
            let dylib = find_onnxruntime_dylib()?;
            ort::init_from(&dylib).ok()?.build().ok()
        })
        .as_ref()
}

struct RvmRecurrentState {
    r: [(Vec<usize>, Vec<f32>); 4],
    state_key: Option<(usize, usize, u32)>,
}

impl Default for RvmRecurrentState {
    fn default() -> Self {
        Self {
            r: [
                (vec![1, 1, 1, 1], vec![0.0]),
                (vec![1, 1, 1, 1], vec![0.0]),
                (vec![1, 1, 1, 1], vec![0.0]),
                (vec![1, 1, 1, 1], vec![0.0]),
            ],
            state_key: None,
        }
    }
}

impl RvmRecurrentState {
    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Real neural portrait matting via `ort` (`rvm_mobilenetv3_fp32.onnx`) with pure-Rust fallback.
pub struct PortraitSegmenter {
    pub backend_name: &'static str,
    session: Option<Session>,
    rvm_state: RvmRecurrentState,
    input_nchw_buf: Vec<f32>,
}

impl PortraitSegmenter {
    pub fn new(preferred_backend: &str) -> Self {
        let mut seg = Self {
            backend_name: "fast_portrait",
            session: None,
            rvm_state: RvmRecurrentState::default(),
            input_nchw_buf: Vec::new(),
        };
        seg.init_backend(preferred_backend);
        seg
    }

    fn init_backend(&mut self, preferred: &str) {
        if preferred == "fast_portrait" {
            return;
        }
        let Some(rvm_path) = model_path("rvm_mobilenetv3") else {
            return;
        };
        if !rvm_path.is_file() {
            return;
        }
        let Some(ort_env) = get_ort_env() else {
            return;
        };

        // 1. Try RTX CUDA first unless CPU was explicitly requested
        if matches!(preferred, "auto" | "rvm_cuda" | "onnx_cuda") {
            if let Ok(builder) = Session::builder(ort_env) {
                if let Ok(mut builder) = builder.with_execution_providers([ep::CUDA::default().build().error_on_failure()]) {
                    if let Ok(sess) = builder.commit_from_file(&rvm_path) {
                        self.session = Some(sess);
                        self.backend_name = "rvm_cuda";
                        return;
                    }
                }
            }
        }

        // 2. Fall back to ONNX Runtime CPU Execution Provider
        if matches!(preferred, "auto" | "rvm_cpu" | "onnx_cpu" | "rvm_cuda" | "onnx_cuda") {
            if let Ok(builder) = Session::builder(ort_env) {
                if let Ok(mut builder) = builder.with_intra_threads(4) {
                    if let Ok(sess) = builder.commit_from_file(&rvm_path) {
                        self.session = Some(sess);
                        self.backend_name = "rvm_cpu";
                    }
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.rvm_state.reset();
    }

    /// Compute an `H x W` float32 alpha matte in `[0.0, 1.0]`.
    pub fn segment_alpha(&mut self, frame: &RgbFrame, face: Option<FaceBox>, quality: &str) -> Vec<f32> {
        let w = frame.width;
        let h = frame.height;

        if let Some(ref mut sess) = self.session {
            let tw = if quality == "quality" { 640usize } else { 480usize };
            let th = (((tw * h) / w.max(1)) / 16 * 16).max(160);
            let dr_val: f32 = if self.backend_name == "rvm_cuda" && quality == "quality" {
                0.6
            } else {
                0.4
            };
            let state_key = (tw, th, (dr_val * 100.0) as u32);
            if self.rvm_state.state_key != Some(state_key) {
                self.rvm_state.reset();
                self.rvm_state.state_key = Some(state_key);
            }

            // Bilinear downsample + RGB HWC u8 -> NCHW f32 [0..1] in one fused pass!
            let plane = tw * th;
            self.input_nchw_buf.resize(3 * plane, 0.0);
            let inv255 = 1.0_f32 / 255.0_f32;
            let scale_x = w as f32 / tw as f32;
            let scale_y = h as f32 / th as f32;
            for ty in 0..th {
                let sy = ((ty as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (h - 1) as f32);
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(h - 1);
                let fy = sy - y0 as f32;
                let row0 = y0 * w * 3;
                let row1 = y1 * w * 3;
                for tx in 0..tw {
                    let sx = ((tx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (w - 1) as f32);
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let fx = sx - x0 as f32;
                    let i00 = row0 + x0 * 3;
                    let i10 = row0 + x1 * 3;
                    let i01 = row1 + x0 * 3;
                    let i11 = row1 + x1 * 3;
                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx * (1.0 - fy);
                    let w01 = (1.0 - fx) * fy;
                    let w11 = fx * fy;
                    let dst_idx = ty * tw + tx;
                    for c in 0..3 {
                        let val = f32::from(frame.data[i00 + c]) * w00
                            + f32::from(frame.data[i10 + c]) * w10
                            + f32::from(frame.data[i01 + c]) * w01
                            + f32::from(frame.data[i11 + c]) * w11;
                        self.input_nchw_buf[c * plane + dst_idx] = val * inv255;
                    }
                }
            }

            let src_res = Tensor::from_array(([1usize, 3, th, tw], self.input_nchw_buf.clone()));
            let r1_res = Tensor::from_array((self.rvm_state.r[0].0.clone(), self.rvm_state.r[0].1.clone()));
            let r2_res = Tensor::from_array((self.rvm_state.r[1].0.clone(), self.rvm_state.r[1].1.clone()));
            let r3_res = Tensor::from_array((self.rvm_state.r[2].0.clone(), self.rvm_state.r[2].1.clone()));
            let r4_res = Tensor::from_array((self.rvm_state.r[3].0.clone(), self.rvm_state.r[3].1.clone()));
            let dr_res = Tensor::from_array(([1usize], vec![dr_val]));

            if let (Ok(src), Ok(r1i), Ok(r2i), Ok(r3i), Ok(r4i), Ok(dr)) =
                (src_res, r1_res, r2_res, r3_res, r4_res, dr_res)
            {
                let run_res = sess.run(ort::inputs![
                    "src" => src,
                    "r1i" => r1i,
                    "r2i" => r2i,
                    "r3i" => r3i,
                    "r4i" => r4i,
                    "downsample_ratio" => dr,
                ]);
                if let Ok(outputs) = run_res {
                    // Outputs: fgr(0), pha(1), r1o(2), r2o(3), r3o(4), r4o(5)
                    let mut next_r: Option<[(Vec<usize>, Vec<f32>); 4]> = None;
                    if let (Ok((s1, d1)), Ok((s2, d2)), Ok((s3, d3)), Ok((s4, d4))) = (
                        outputs[2].try_extract_tensor::<f32>(),
                        outputs[3].try_extract_tensor::<f32>(),
                        outputs[4].try_extract_tensor::<f32>(),
                        outputs[5].try_extract_tensor::<f32>(),
                    ) {
                        next_r = Some([
                            (s1.iter().map(|&v| v as usize).collect(), d1.to_vec()),
                            (s2.iter().map(|&v| v as usize).collect(), d2.to_vec()),
                            (s3.iter().map(|&v| v as usize).collect(), d3.to_vec()),
                            (s4.iter().map(|&v| v as usize).collect(), d4.to_vec()),
                        ]);
                    }
                    if let Some(nr) = next_r {
                        self.rvm_state.r = nr;
                    }
                    if let Ok((_pha_shape, pha_data)) = outputs[1].try_extract_tensor::<f32>() {
                        if pha_data.len() == tw * th {
                            return resize_f32_bilinear(pha_data, tw, th, w, h);
                        }
                    }
                }
            }
        }

        // Pure-Rust fallback portrait matte (ellipse head + shoulders)
        let cx = face.map_or(0.5, |f| f.cx);
        let cy = face.map_or(0.42, |f| f.cy);
        let fw = face.map_or(0.24, |f| f.w);
        let fh = face.map_or(0.30, |f| f.h);
        let sw = 96usize;
        let sh = ((sw * h) / w.max(1)).max(54);
        let mut small = vec![0.0_f32; sw * sh];
        let head_rx = (sw as f32 * 0.14).max(sw as f32 * fw * 0.72);
        let head_ry = (sh as f32 * 0.19).max(sh as f32 * fh * 0.80);
        let hx = cx * sw as f32;
        let hy = cy * sh as f32;
        let sh_rx = (sw as f32 * 0.34).max(head_rx * 2.35);
        let sy = (sh as f32 * 0.98).min(hy + head_ry * 1.65);
        let sh_ry = sh as f32 * 0.55;

        for py in 0..sh {
            for px in 0..sw {
                let d_head = ((px as f32 - hx) / head_rx).powi(2) + ((py as f32 - hy) / head_ry).powi(2);
                let d_sh = ((px as f32 - hx) / sh_rx).powi(2) + ((py as f32 - sy) / sh_ry).powi(2);
                let d = d_head.min(d_sh);
                let a = (1.0 - (d - 0.85) / 0.30).clamp(0.0, 1.0);
                small[py * sw + px] = a;
            }
        }
        resize_f32_bilinear(&small, sw, sh, w, h)
    }
}

fn resize_f32_bilinear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    }
    let mut dst = vec![0.0_f32; dw * dh];
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;
    for y in 0..dh {
        let sy = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = sy - y0 as f32;
        let row0 = y0 * sw;
        let row1 = y1 * sw;
        let dst_row = y * dw;
        for x in 0..dw {
            let sx = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = sx - x0 as f32;
            let v = src[row0 + x0] * (1.0 - fx) * (1.0 - fy)
                + src[row0 + x1] * fx * (1.0 - fy)
                + src[row1 + x0] * (1.0 - fx) * fy
                + src[row1 + x1] * fx * fy;
            dst[dst_row + x] = v.clamp(0.0, 1.0);
        }
    }
    dst
}

fn resize_rgb_bilinear(src: &RgbFrame, dw: usize, dh: usize) -> RgbFrame {
    if src.width == dw && src.height == dh {
        return src.clone();
    }
    let mut dst = RgbFrame::new(dw, dh);
    let sw = src.width;
    let sh = src.height;
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    for y in 0..dh {
        let sy = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = ((sy - y0 as f32) * 256.0) as u32;
        let ify = 256 - fy;
        let r0 = y0 * sw * 3;
        let r1 = y1 * sw * 3;
        let dr = y * dw * 3;
        for x in 0..dw {
            let sx = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = ((sx - x0 as f32) * 256.0) as u32;
            let ifx = 256 - fx;
            let w00 = ifx * ify;
            let w10 = fx * ify;
            let w01 = ifx * fy;
            let w11 = fx * fy;
            let i00 = r0 + x0 * 3;
            let i10 = r0 + x1 * 3;
            let i01 = r1 + x0 * 3;
            let i11 = r1 + x1 * 3;
            let di = dr + x * 3;
            for c in 0..3 {
                let v = u32::from(src.data[i00 + c]) * w00
                    + u32::from(src.data[i10 + c]) * w10
                    + u32::from(src.data[i01 + c]) * w01
                    + u32::from(src.data[i11 + c]) * w11;
                dst.data[di + c] = ((v + 32_768) >> 16).min(255) as u8;
            }
        }
    }
    dst
}

/// Fast separable box blur (3 passes approximate Gaussian blur in O(N) time!).
fn fast_blur_rgb(src: &RgbFrame, radius: usize) -> RgbFrame {
    let w = src.width;
    let h = src.height;
    let r = radius.max(1);
    let mut cur = src.data.clone();
    let mut tmp = vec![0u8; cur.len()];

    for _pass in 0..3 {
        // Horizontal pass with running window sum
        let win = (2 * r + 1) as u32;
        for y in 0..h {
            let row = y * w * 3;
            let mut sum = [0u32; 3];
            for dx in -(r as isize)..=(r as isize) {
                let sx = dx.clamp(0, (w - 1) as isize) as usize;
                let idx = row + sx * 3;
                sum[0] += u32::from(cur[idx]);
                sum[1] += u32::from(cur[idx + 1]);
                sum[2] += u32::from(cur[idx + 2]);
            }
            for x in 0..w {
                let di = row + x * 3;
                tmp[di] = (sum[0] / win) as u8;
                tmp[di + 1] = (sum[1] / win) as u8;
                tmp[di + 2] = (sum[2] / win) as u8;

                let old_x = (x as isize - r as isize).clamp(0, (w - 1) as isize) as usize;
                let new_x = (x as isize + r as isize + 1).clamp(0, (w - 1) as isize) as usize;
                let oi = row + old_x * 3;
                let ni = row + new_x * 3;
                for c in 0..3 {
                    sum[c] = sum[c] + u32::from(cur[ni + c]) - u32::from(cur[oi + c]);
                }
            }
        }

        // Vertical pass with running window sum
        for x in 0..w {
            let mut sum = [0u32; 3];
            for dy in -(r as isize)..=(r as isize) {
                let sy = dy.clamp(0, (h - 1) as isize) as usize;
                let idx = (sy * w + x) * 3;
                sum[0] += u32::from(tmp[idx]);
                sum[1] += u32::from(tmp[idx + 1]);
                sum[2] += u32::from(tmp[idx + 2]);
            }
            for y in 0..h {
                let di = (y * w + x) * 3;
                cur[di] = (sum[0] / win) as u8;
                cur[di + 1] = (sum[1] / win) as u8;
                cur[di + 2] = (sum[2] / win) as u8;

                let old_y = (y as isize - r as isize).clamp(0, (h - 1) as isize) as usize;
                let new_y = (y as isize + r as isize + 1).clamp(0, (h - 1) as isize) as usize;
                let oi = (old_y * w + x) * 3;
                let ni = (new_y * w + x) * 3;
                for c in 0..3 {
                    sum[c] = sum[c] + u32::from(tmp[ni + c]) - u32::from(tmp[oi + c]);
                }
            }
        }
    }

    RgbFrame {
        width: w,
        height: h,
        data: cur,
    }
}

pub fn build_preset_background(preset: &str, width: usize, height: usize) -> RgbFrame {
    let mut frame = RgbFrame::new(width, height);
    let wf = width.max(1) as f32;
    let hf = height.max(1) as f32;
    for y in 0..height {
        let yn = y as f32 / hf;
        let row = y * width * 3;
        for x in 0..width {
            let xn = x as f32 / wf;
            let (r, g, b) = match preset {
                "neon_green" => {
                    let glow = (-((xn - 0.5).powi(2) / 0.22 + (yn - 0.45).powi(2) / 0.25)).exp();
                    (16.0 + 18.0 * glow, 20.0 + 68.0 * glow, 18.0 + 14.0 * glow)
                }
                "soft_daylight" => {
                    let glow = (-((xn - 0.72).powi(2) / 0.25 + (yn - 0.30).powi(2) / 0.35)).exp();
                    (
                        195.0 - 28.0 * yn + 32.0 * glow,
                        202.0 - 26.0 * yn + 30.0 * glow,
                        212.0 - 20.0 * yn + 26.0 * glow,
                    )
                }
                "office_blur" => {
                    let g1 = (-((xn - 0.22).powi(2) / 0.015 + (yn - 0.34).powi(2) / 0.02)).exp();
                    let g2 = (-((xn - 0.80).powi(2) / 0.02 + (yn - 0.26).powi(2) / 0.025)).exp();
                    (
                        52.0 + 24.0 * (1.0 - yn) + 120.0 * g1 + 90.0 * g2,
                        56.0 + 22.0 * (1.0 - yn) + 95.0 * g1 + 100.0 * g2,
                        66.0 + 18.0 * (1.0 - yn) + 50.0 * g1 + 115.0 * g2,
                    )
                }
                _ => {
                    let glow = (-((xn - 0.5).powi(2) / 0.28 + (yn - 0.42).powi(2) / 0.30)).exp();
                    (22.0 + 28.0 * glow, 24.0 + 30.0 * glow, 28.0 + 34.0 * glow)
                }
            };
            let idx = row + x * 3;
            frame.data[idx] = r.clamp(0.0, 255.0) as u8;
            frame.data[idx + 1] = g.clamp(0.0, 255.0) as u8;
            frame.data[idx + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
    frame
}

fn alpha_blend_in_place(fg: &mut RgbFrame, bg: &RgbFrame, alpha: &[f32]) {
    let n_pix = fg.width * fg.height;
    for (i, &av) in alpha.iter().enumerate().take(n_pix) {
        let a = (av.clamp(0.0, 1.0) * 256.0) as u16;
        let inv = 256 - a;
        let p = i * 3;
        fg.data[p] = ((u16::from(fg.data[p]) * a + u16::from(bg.data[p]) * inv) >> 8) as u8;
        fg.data[p + 1] = ((u16::from(fg.data[p + 1]) * a + u16::from(bg.data[p + 1]) * inv) >> 8) as u8;
        fg.data[p + 2] = ((u16::from(fg.data[p + 2]) * a + u16::from(bg.data[p + 2]) * inv) >> 8) as u8;
    }
}

/// Pure-Rust real-time video effects pipeline matching `nv_broadcast/video_effects.py`.
pub struct VideoEffectsPipeline {
    pub tracker: AutoFrameTracker,
    pub segmenter: PortraitSegmenter,
    prev_denoise: Option<Vec<u8>>,
    cached_bg_key: Option<(String, String, usize, usize)>,
    cached_bg: Option<RgbFrame>,
    pub last_alpha: Vec<f32>,
}

impl VideoEffectsPipeline {
    pub fn new(preferred_backend: &str) -> Self {
        Self {
            tracker: AutoFrameTracker::default(),
            segmenter: PortraitSegmenter::new(preferred_backend),
            prev_denoise: None,
            cached_bg_key: None,
            cached_bg: None,
            last_alpha: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.segmenter.reset();
        self.prev_denoise = None;
        self.cached_bg_key = None;
        self.cached_bg = None;
        self.last_alpha.clear();
    }

    pub fn process_frame(&mut self, input: &RgbFrame, cfg: &CameraEffectsConfig) -> RgbFrame {
        let mut work = if cfg.mirror {
            mirror_horizontal(input)
        } else {
            input.clone()
        };
        let raw_out = if cfg.split_preview {
            Some(work.clone())
        } else {
            None
        };

        // 1. Video Noise Removal (fast edge-preserving spatial 3x3 + motion-aware temporal filter)
        if cfg.video_noise_removal_enabled {
            work = self.apply_video_noise_removal(&work, cfg.video_noise_removal_strength);
        }

        // 2. Face tracking (only when needed)
        let need_face = cfg.auto_frame_enabled
            || cfg.key_light_enabled
            || cfg.vignette_enabled
            || (self.segmenter.backend_name == "fast_portrait"
                && (cfg.blur_enabled || cfg.replace_enabled || cfg.remove_enabled));
        let mut face = if need_face {
            self.tracker.detect_face(&work)
        } else {
            None
        };
        let (mut fcx, mut fcy) = self.tracker.update(face);

        // 3. Auto Frame crop & zoom
        if cfg.auto_frame_enabled {
            work = apply_auto_frame(&work, fcx, fcy, cfg.auto_frame_zoom);
            fcx = 0.5;
            fcy = 0.45;
            if let Some(ref mut f) = face {
                f.cx = 0.5;
                f.cy = 0.45;
                f.w = (f.w * 1.3).min(0.5);
                f.h = (f.h * 1.3).min(0.55);
            }
        }

        // 4. Portrait Matting (ONNX RVM CUDA/CPU or fallback)
        let need_matte =
            cfg.blur_enabled || cfg.replace_enabled || cfg.remove_enabled || cfg.key_light_enabled;
        let alpha = if need_matte {
            let quality = if cfg.blur_enabled {
                cfg.blur_mode.as_str()
            } else {
                "quality"
            };
            let a = self.segmenter.segment_alpha(&work, face, quality);
            self.last_alpha = a.clone();
            Some(a)
        } else {
            None
        };

        // 5. Background Blur / Replace / Remove
        if cfg.blur_enabled {
            if let Some(ref a) = alpha {
                let bw = (work.width / 4).max(160);
                let bh = (work.height / 4).max(90);
                let small = resize_rgb_bilinear(&work, bw, bh);
                let radius = (2.0 + 7.0 * cfg.blur_strength.clamp(0.0, 1.0)).round() as usize;
                let blurred_small = fast_blur_rgb(&small, radius);
                let blurred = resize_rgb_bilinear(&blurred_small, work.width, work.height);
                alpha_blend_in_place(&mut work, &blurred, a);
            }
        } else if cfg.replace_enabled {
            if let Some(ref a) = alpha {
                let bg = self.get_replacement_bg(
                    &cfg.replace_image_path,
                    &cfg.replace_preset,
                    work.width,
                    work.height,
                );
                alpha_blend_in_place(&mut work, &bg, a);
            }
        } else if cfg.remove_enabled {
            if let Some(ref a) = alpha {
                let rgb = parse_hex_color(&cfg.remove_chroma_color);
                let n_pix = work.width * work.height;
                for (i, &af) in a.iter().enumerate().take(n_pix) {
                    let av = (af.clamp(0.0, 1.0) * 256.0) as u16;
                    let inv = 256 - av;
                    let p = i * 3;
                    work.data[p] = ((u16::from(work.data[p]) * av + u16::from(rgb[0]) * inv) >> 8) as u8;
                    work.data[p + 1] =
                        ((u16::from(work.data[p + 1]) * av + u16::from(rgb[1]) * inv) >> 8) as u8;
                    work.data[p + 2] =
                        ((u16::from(work.data[p + 2]) * av + u16::from(rgb[2]) * inv) >> 8) as u8;
                }
            }
        }

        // 6. Subject-masked Virtual Key Light
        if cfg.key_light_enabled {
            apply_virtual_key_light(
                &mut work,
                alpha.as_deref(),
                fcx,
                fcy,
                cfg.key_light_warmth,
                cfg.key_light_intensity,
            );
        }

        // 7. Face-tracked Vignette
        if cfg.vignette_enabled {
            let (vcx, vcy) = if cfg.vignette_face_tracking {
                (fcx, fcy)
            } else {
                (0.5, 0.5)
            };
            apply_vignette(&mut work, cfg.vignette_strength, vcx, vcy);
        }

        if let Some(raw) = raw_out {
            render_split_frame(&raw, &mut work, cfg.split_position);
        }
        work
    }

    fn get_replacement_bg(&mut self, image_path: &str, preset: &str, w: usize, h: usize) -> RgbFrame {
        let key = (image_path.to_string(), preset.to_string(), w, h);
        if self.cached_bg_key.as_ref() == Some(&key) {
            if let Some(ref bg) = self.cached_bg {
                return bg.clone();
            }
        }
        let bg = if !image_path.is_empty() && Path::new(image_path).is_file() {
            RgbFrame::load_from_path(Path::new(image_path))
                .map(|im| resize_rgb_bilinear(&im, w, h))
                .unwrap_or_else(|_| build_preset_background(preset, w, h))
        } else {
            build_preset_background(preset, w, h)
        };
        self.cached_bg_key = Some(key);
        self.cached_bg = Some(bg.clone());
        bg
    }

    /// Fast edge-preserving 3x3 bilateral-weighted spatial + motion-aware temporal denoiser.
    fn apply_video_noise_removal(&mut self, src: &RgbFrame, strength: f32) -> RgbFrame {
        let w = src.width;
        let h = src.height;
        let s = strength.clamp(0.0, 1.0);
        let range_thresh = (26.0 + 52.0 * s) as i32;
        let mut out = src.data.clone();

        for y in 1..h.saturating_sub(1) {
            let row = y * w;
            for x in 1..w.saturating_sub(1) {
                let ci = (row + x) * 3;
                let cr = i32::from(src.data[ci]);
                let cg = i32::from(src.data[ci + 1]);
                let cb = i32::from(src.data[ci + 2]);
                let mut acc_r = cr * 4;
                let mut acc_g = cg * 4;
                let mut acc_b = cb * 4;
                let mut w_sum = 4i32;

                for dy in [-1isize, 0, 1] {
                    let nrow = ((y as isize + dy) as usize) * w;
                    for dx in [-1isize, 0, 1] {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ni = (nrow + (x as isize + dx) as usize) * 3;
                        let nr = i32::from(src.data[ni]);
                        let ng = i32::from(src.data[ni + 1]);
                        let nb = i32::from(src.data[ni + 2]);
                        let diff = (nr - cr).abs() + (ng - cg).abs() + (nb - cb).abs();
                        if diff < range_thresh * 3 {
                            let wt = if dx == 0 || dy == 0 { 2 } else { 1 };
                            acc_r += nr * wt;
                            acc_g += ng * wt;
                            acc_b += nb * wt;
                            w_sum += wt;
                        }
                    }
                }
                out[ci] = (acc_r / w_sum) as u8;
                out[ci + 1] = (acc_g / w_sum) as u8;
                out[ci + 2] = (acc_b / w_sum) as u8;
            }
        }

        // Motion-aware temporal IIR blend (zero ghosting on moving edges, deep noise cut on static pixels)
        if let Some(ref prev) = self.prev_denoise {
            if prev.len() == out.len() {
                let base_a = (0.78 - 0.42 * s).max(0.32);
                let n_pix = w * h;
                for i in 0..n_pix {
                    let p = i * 3;
                    let diff = ((i32::from(out[p]) - i32::from(prev[p])).abs()
                        + (i32::from(out[p + 1]) - i32::from(prev[p + 1])).abs()
                        + (i32::from(out[p + 2]) - i32::from(prev[p + 2])).abs())
                        as f32
                        / 3.0;
                    let motion = ((diff - 6.0) / 18.0).clamp(0.0, 1.0);
                    let eff_a = base_a + (1.0 - base_a) * motion;
                    let a_u16 = (eff_a * 256.0) as u16;
                    let inv_u16 = 256 - a_u16;
                    for c in 0..3 {
                        out[p + c] = ((u16::from(out[p + c]) * a_u16 + u16::from(prev[p + c]) * inv_u16)
                            >> 8) as u8;
                    }
                }
            }
        }
        self.prev_denoise = Some(out.clone());
        RgbFrame {
            width: w,
            height: h,
            data: out,
        }
    }
}

fn mirror_horizontal(src: &RgbFrame) -> RgbFrame {
    let mut dst = RgbFrame::new(src.width, src.height);
    for y in 0..src.height {
        let row = y * src.width * 3;
        for x in 0..src.width {
            let si = row + (src.width - 1 - x) * 3;
            let di = row + x * 3;
            dst.data[di..di + 3].copy_from_slice(&src.data[si..si + 3]);
        }
    }
    dst
}

fn apply_auto_frame(src: &RgbFrame, cx: f32, cy: f32, zoom: f32) -> RgbFrame {
    let w = src.width;
    let h = src.height;
    let scale = 1.0 + zoom.clamp(0.0, 1.0) * 0.95;
    let crop_w = ((w as f32 / scale).round() as usize).clamp(64, w);
    let crop_h = ((h as f32 / scale).round() as usize).clamp(48, h);
    let left = ((cx * w as f32 - crop_w as f32 * 0.5).round() as isize)
        .clamp(0, (w - crop_w) as isize) as usize;
    let top = ((cy * h as f32 - crop_h as f32 * 0.5).round() as isize)
        .clamp(0, (h - crop_h) as isize) as usize;

    let mut cropped = RgbFrame::new(crop_w, crop_h);
    for y in 0..crop_h {
        let s_off = ((top + y) * w + left) * 3;
        let d_off = y * crop_w * 3;
        cropped.data[d_off..d_off + crop_w * 3]
            .copy_from_slice(&src.data[s_off..s_off + crop_w * 3]);
    }
    resize_rgb_bilinear(&cropped, w, h)
}

fn apply_virtual_key_light(
    img: &mut RgbFrame,
    alpha: Option<&[f32]>,
    cx: f32,
    cy: f32,
    warmth: f32,
    intensity: f32,
) {
    let intensity = intensity.clamp(0.0, 1.0);
    if intensity <= 0.01 {
        return;
    }
    let warmth = warmth.clamp(0.0, 1.0);
    let w = img.width;
    let h = img.height;
    let sw = 64usize;
    let sh = ((sw * h) / w.max(1)).max(36);
    let lx = (cx - 0.06) * sw as f32;
    let ly = (cy - 0.08).max(0.15) * sh as f32;
    let mut falloff_s = vec![0.0_f32; sw * sh];
    for y in 0..sh {
        for x in 0..sw {
            let d2 = ((x as f32 - lx) / (sw as f32 * 0.48)).powi(2)
                + ((y as f32 - ly) / (sh as f32 * 0.55)).powi(2);
            falloff_s[y * sw + x] = (-0.95 * d2).exp();
        }
    }
    let falloff = resize_f32_bilinear(&falloff_s, sw, sh, w, h);

    let g_extra = [
        (intensity * (0.28 + 0.22 * warmth) * 256.0).round() as u32,
        (intensity * (0.28 + 0.06 * (1.0 - (warmth - 0.5).abs() * 2.0)) * 256.0).round() as u32,
        (intensity * (0.34 - 0.22 * warmth) * 256.0).round() as u32,
    ];
    let add_u16 = [
        (intensity * (14.0 + 14.0 * warmth)).round() as u32,
        (intensity * 14.0).round() as u32,
        (intensity * (20.0 - 12.0 * warmth)).round() as u32,
    ];

    let n_pix = w * h;
    for i in 0..n_pix {
        let subj = alpha.map_or(1.0, |a| a[i]);
        let m = (falloff[i] * (0.15 + 0.85 * subj) * 256.0) as u32;
        let p = i * 3;
        for c in 0..3 {
            let scale = 256 + ((m * g_extra[c]) >> 8);
            let lift = (m * add_u16[c]) >> 8;
            let val = ((u32::from(img.data[p + c]) * scale) >> 8) + lift;
            img.data[p + c] = val.min(255) as u8;
        }
    }
}

fn apply_vignette(img: &mut RgbFrame, strength: f32, cx: f32, cy: f32) {
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.01 {
        return;
    }
    let w = img.width;
    let h = img.height;
    let sw = 64usize;
    let sh = ((sw * h) / w.max(1)).max(36);
    let mut att_s = vec![0.0_f32; sw * sh];
    for y in 0..sh {
        let ny = (y as f32 / (sh - 1).max(1) as f32 - cy) / 0.62;
        for x in 0..sw {
            let nx = (x as f32 / (sw - 1).max(1) as f32 - cx) / 0.62;
            att_s[y * sw + x] = (1.0 - s * 0.72 * (nx * nx + ny * ny).min(1.4)).clamp(0.0, 1.0);
        }
    }
    let att = resize_f32_bilinear(&att_s, sw, sh, w, h);
    for (i, &a) in att.iter().enumerate() {
        let au = (a * 256.0) as u16;
        let p = i * 3;
        img.data[p] = ((u16::from(img.data[p]) * au) >> 8) as u8;
        img.data[p + 1] = ((u16::from(img.data[p + 1]) * au) >> 8) as u8;
        img.data[p + 2] = ((u16::from(img.data[p + 2]) * au) >> 8) as u8;
    }
}

fn render_split_frame(raw: &RgbFrame, proc: &mut RgbFrame, split_pos: f32) {
    let pos = split_pos.clamp(0.05, 0.95);
    let split_x = ((proc.width as f32 * pos).round() as usize).min(proc.width);
    let x0 = split_x.saturating_sub(1);
    let x1 = (split_x + 2).min(proc.width);
    for y in 0..proc.height {
        let row = y * proc.width * 3;
        proc.data[row..row + split_x * 3].copy_from_slice(&raw.data[row..row + split_x * 3]);
        for x in x0..x1 {
            let p = row + x * 3;
            proc.data[p] = 118;
            proc.data[p + 1] = 185;
            proc.data[p + 2] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_rust_video_pipeline_processes_all_effects() {
        let mut pipe = VideoEffectsPipeline::new("fast_portrait");
        let mut frame = build_preset_background("office_blur", 320, 180);
        // Add a skin-colored face patch in the center
        for y in 45..105 {
            for x in 130..190 {
                let p = (y * 320 + x) * 3;
                frame.data[p] = 212;
                frame.data[p + 1] = 168;
                frame.data[p + 2] = 140;
            }
        }

        let cfg = CameraEffectsConfig::from_json_value(&serde_json::json!({
            "mirror": true,
            "split_preview": true,
            "effects": {
                "background_blur": {"enabled": true, "strength": 0.75, "mode": "performance"},
                "virtual_key_light": {"enabled": true, "warmth": 0.5, "intensity": 0.55},
                "auto_frame": {"enabled": true, "zoom": 0.35},
                "video_noise_removal": {"enabled": true, "strength": 0.6},
                "vignette": {"enabled": true, "strength": 0.4, "faceTracking": true}
            }
        }));

        let out = pipe.process_frame(&frame, &cfg);
        assert_eq!(out.width, 320);
        assert_eq!(out.height, 180);
        assert_eq!(out.to_yuyv().len(), 320 * 180 * 2);
        assert_ne!(out.data, frame.data);
    }
}
