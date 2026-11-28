//! Neural model registry, XDG path resolution, and SHA-256 verification.
//!
//! Port of `nv_broadcast/models.py` with added SHA-256 integrity checking.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub key: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub purpose: &'static str,
    pub size_mb: f64,
    pub expected_sha256: Option<&'static str>,
}

pub const MODEL_REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        key: "rvm_mobilenetv3",
        filename: "rvm_mobilenetv3_fp32.onnx",
        url: "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        purpose: "Recurrent neural video matting (CUDA/CPU) for Background Blur/Replace/Remove & Key Light",
        size_mb: 14.4,
        expected_sha256: None,
    },
    ModelSpec {
        key: "selfie_segmenter",
        filename: "selfie_segmenter.tflite",
        url: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        purpose: "MediaPipe fast portrait segmentation fallback",
        size_mb: 0.25,
        expected_sha256: None,
    },
    ModelSpec {
        key: "face_landmarker",
        filename: "face_landmarker.task",
        url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        purpose: "MediaPipe 478-point face & iris landmarker for Auto Frame, Eye Contact, Key Light & Vignette",
        size_mb: 3.6,
        expected_sha256: None,
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub path: String,
    pub installed: bool,
    pub size_bytes: u64,
    pub purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256_verified: Option<bool>,
}

pub fn find_model_spec(key: &str) -> Option<&'static ModelSpec> {
    MODEL_REGISTRY.iter().find(|spec| spec.key == key)
}

pub fn models_dir() -> PathBuf {
    let base = env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
            home.join(".local").join("share")
        });
    base.join("nv-broadcast").join("models")
}

pub fn model_path(key: &str) -> Option<PathBuf> {
    find_model_spec(key).map(|spec| models_dir().join(spec.filename))
}

/// Compute lowercase hexadecimal SHA-256 digest of a file on disk.
pub fn compute_file_sha256(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65_536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    Ok(hex)
}

/// Verify a file against an expected lowercase hex SHA-256 digest.
pub fn verify_file_sha256(path: &Path, expected_hex: &str) -> io::Result<bool> {
    let actual = compute_file_sha256(path)?;
    Ok(actual.eq_ignore_ascii_case(expected_hex.trim()))
}

pub fn get_models_status_in(dir: &Path, include_hash: bool) -> BTreeMap<String, ModelStatus> {
    let mut out = BTreeMap::new();
    for spec in MODEL_REGISTRY {
        let path = dir.join(spec.filename);
        let meta = fs::metadata(&path).ok();
        let size_bytes = meta.as_ref().filter(|m| m.is_file()).map_or(0, |m| m.len());
        let installed = size_bytes > 1024;
        let sha256 = if installed && include_hash {
            compute_file_sha256(&path).ok()
        } else {
            None
        };
        let sha256_verified = match (&sha256, spec.expected_sha256) {
            (Some(actual), Some(expected)) => Some(actual.eq_ignore_ascii_case(expected)),
            _ => None,
        };
        out.insert(
            spec.key.to_string(),
            ModelStatus {
                path: path.to_string_lossy().into_owned(),
                installed,
                size_bytes: if installed { size_bytes } else { 0 },
                purpose: spec.purpose.to_string(),
                sha256,
                sha256_verified,
            },
        );
    }
    out
}

pub fn get_models_status() -> BTreeMap<String, ModelStatus> {
    get_models_status_in(&models_dir(), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn sha256_verification_and_registry_status() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("nvb_models_test_{nonce}"));
        fs::create_dir_all(&dir).unwrap();

        let sample_path = dir.join("selfie_segmenter.tflite");
        let payload = vec![b'A'; 2048];
        fs::write(&sample_path, &payload).unwrap();

        let digest = compute_file_sha256(&sample_path).unwrap();
        assert_eq!(digest.len(), 64);
        assert!(verify_file_sha256(&sample_path, &digest).unwrap());
        assert!(!verify_file_sha256(&sample_path, "0000").unwrap());

        let status = get_models_status_in(&dir, true);
        assert_eq!(status.len(), 3);
        assert!(status["selfie_segmenter"].installed);
        assert_eq!(status["selfie_segmenter"].size_bytes, 2048);
        assert_eq!(status["selfie_segmenter"].sha256.as_deref(), Some(digest.as_str()));
        assert!(!status["rvm_mobilenetv3"].installed);

        let _ = fs::remove_dir_all(&dir);
    }
}
