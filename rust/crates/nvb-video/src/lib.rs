//! On-demand webcam debounce, V4L2 mode selection, and pure-Rust GPU/CPU video effects.

pub mod effects;

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub use effects::{
    build_preset_background, find_onnxruntime_dylib, parse_hex_color, CameraEffectsConfig, FaceBox,
    PortraitSegmenter, RgbFrame, VideoEffectsPipeline,
};
pub use nvb_core::camera_state::CameraState;

/// Tracks continuous idle time after v4l2loopback consumers drop to 0 before releasing the camera.
/// Matches `nv_broadcast/loopback_monitor.py::OnDemandDebounce`.
#[derive(Debug, Clone, PartialEq)]
pub struct OnDemandDebounce {
    idle_timeout_sec: f64,
    idle_since: Option<f64>,
}

impl OnDemandDebounce {
    pub fn new(idle_timeout_sec: f64) -> Self {
        Self {
            idle_timeout_sec: idle_timeout_sec.max(0.0),
            idle_since: None,
        }
    }

    pub fn idle_timeout_sec(&self) -> f64 {
        self.idle_timeout_sec
    }

    pub fn set_idle_timeout_sec(&mut self, value: f64) {
        self.idle_timeout_sec = value.max(0.0);
    }

    pub fn idle_since(&self) -> Option<f64> {
        self.idle_since
    }

    pub fn reset(&mut self) {
        self.idle_since = None;
    }

    pub fn idle_elapsed(&self, now: f64) -> f64 {
        match self.idle_since {
            Some(since) => (now - since).max(0.0),
            None => 0.0,
        }
    }

    /// Advance debounce state with the current external consumer count and monotonic timestamp.
    /// Returns `true` once `consumer_count` has stayed at `0` for at least `idle_timeout_sec`.
    pub fn update(&mut self, consumer_count: usize, now: f64) -> bool {
        if consumer_count > 0 {
            self.idle_since = None;
            return false;
        }
        match self.idle_since {
            None => {
                self.idle_since = Some(now);
                self.idle_timeout_sec <= 0.0
            }
            Some(since) => (now - since) >= self.idle_timeout_sec,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V4l2Mode {
    pub pixel_format: String,
    pub width: u32,
    pub height: u32,
    pub fps_options: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V4l2DeviceInfo {
    pub path: String,
    pub driver: String,
    pub card: String,
    pub bus_info: String,
    pub is_capture: bool,
    pub is_output: bool,
    pub is_metadata: bool,
    pub is_loopback: bool,
    pub formats: Vec<String>,
    pub modes: Vec<V4l2Mode>,
}

fn max_fps(options: &[f64]) -> f64 {
    options
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| if v > acc { v } else { acc })
}

/// Pick MJPG over YUYV for HD modes so USB webcams run at 30fps instead of 10fps.
/// Matches `nv_broadcast/v4l2_ctl.py::choose_best_pixel_format`.
pub fn choose_best_pixel_format(
    device: &V4l2DeviceInfo,
    width: u32,
    height: u32,
    prefer_mjpg: bool,
) -> (String, f64) {
    let matching: Vec<&V4l2Mode> = device
        .modes
        .iter()
        .filter(|m| m.width == width && m.height == height)
        .collect();

    if matching.is_empty() {
        if prefer_mjpg && device.formats.iter().any(|f| f == "MJPG") {
            return ("MJPG".to_string(), 30.0);
        }
        let fmt = device
            .formats
            .first()
            .cloned()
            .unwrap_or_else(|| "MJPG".to_string());
        return (fmt, 30.0);
    }

    let best = matching
        .into_iter()
        .max_by(|a, b| {
            let a_fps = max_fps(&a.fps_options);
            let b_fps = max_fps(&b.fps_options);
            let a_bonus = u8::from(prefer_mjpg && a.pixel_format == "MJPG");
            let b_bonus = u8::from(prefer_mjpg && b.pixel_format == "MJPG");
            a_fps
                .partial_cmp(&b_fps)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a_bonus.cmp(&b_bonus))
        })
        .expect("matching is non-empty");

    let fps = if best.fps_options.is_empty() {
        30.0
    } else {
        max_fps(&best.fps_options)
    };
    (best.pixel_format.clone(), fps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const DSP_ORACLE_JSON: &str = include_str!("../../../fixtures/dsp_oracle.json");

    #[test]
    fn v4l2_mode_selection_matches_python_oracle() {
        let oracle: Value = serde_json::from_str(DSP_ORACLE_JSON).unwrap();
        let v4l2 = &oracle["v4l2"];
        let dev: V4l2DeviceInfo = serde_json::from_value(v4l2["device"].clone()).unwrap();

        let (fmt_hd, fps_hd) = choose_best_pixel_format(&dev, 1280, 720, true);
        assert_eq!(fmt_hd, v4l2["best_hd"][0].as_str().unwrap());
        assert_eq!(fps_hd, v4l2["best_hd"][1].as_f64().unwrap());

        let (fmt_vga, fps_vga) = choose_best_pixel_format(&dev, 640, 480, true);
        assert_eq!(fmt_vga, v4l2["best_vga_prefer_mjpg"][0].as_str().unwrap());
        assert_eq!(fps_vga, v4l2["best_vga_prefer_mjpg"][1].as_f64().unwrap());

        let (fmt_fb, fps_fb) = choose_best_pixel_format(&dev, 1920, 1080, true);
        assert_eq!(fmt_fb, v4l2["best_fallback_1080p"][0].as_str().unwrap());
        assert_eq!(fps_fb, v4l2["best_fallback_1080p"][1].as_f64().unwrap());
    }

    #[test]
    fn debounce_waits_for_continuous_idle_and_resets_on_reconnect() {
        let mut deb = OnDemandDebounce::new(3.0);
        assert!(!deb.update(1, 10.0));
        assert_eq!(deb.idle_since(), None);

        // Consumers drop to 0 at t=12.0
        assert!(!deb.update(0, 12.0));
        assert_eq!(deb.idle_since(), Some(12.0));
        assert!(!deb.update(0, 14.5));
        assert!((deb.idle_elapsed(14.5) - 2.5).abs() < 1e-9);

        // Consumer reconnects before timeout -> resets timer
        assert!(!deb.update(1, 14.8));
        assert_eq!(deb.idle_since(), None);
        assert_eq!(deb.idle_elapsed(14.8), 0.0);

        // Consumers drop again at t=20.0 -> expires at t=23.0
        assert!(!deb.update(0, 20.0));
        assert!(!deb.update(0, 22.99));
        assert!(deb.update(0, 23.0));
    }

    #[test]
    fn zero_timeout_triggers_immediately() {
        let mut deb = OnDemandDebounce::new(-1.0);
        assert_eq!(deb.idle_timeout_sec(), 0.0);
        assert!(deb.update(0, 5.0));
    }
}
