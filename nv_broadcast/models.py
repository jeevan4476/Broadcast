"""Neural model registry and downloader for NVIDIA Broadcast (Linux)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "rvm_mobilenetv3": {
        "filename": "rvm_mobilenetv3_fp32.onnx",
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        "purpose": "Recurrent neural video matting (CUDA/CPU) for Background Blur/Replace/Remove & Key Light",
        "size_mb": 14.4,
    },
    "selfie_segmenter": {
        "filename": "selfie_segmenter.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "purpose": "MediaPipe fast portrait segmentation fallback",
        "size_mb": 0.25,
    },
    "face_landmarker": {
        "filename": "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "purpose": "MediaPipe 478-point face & iris landmarker for Auto Frame, Eye Contact, Key Light & Vignette",
        "size_mb": 3.6,
    },
}


def models_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "nv-broadcast" / "models"


def model_path(key: str) -> Path:
    spec = MODEL_REGISTRY[key]
    return models_dir() / spec["filename"]


def get_models_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, spec in MODEL_REGISTRY.items():
        p = model_path(key)
        exists = p.is_file() and p.stat().st_size > 1024
        out[key] = {
            "path": str(p),
            "installed": exists,
            "size_bytes": p.stat().st_size if exists else 0,
            "purpose": spec["purpose"],
        }
    return out


def download_model(key: str, timeout: float = 45.0) -> tuple[bool, str]:
    if key not in MODEL_REGISTRY:
        return False, f"Unknown model key: {key}"
    spec = MODEL_REGISTRY[key]
    dest = model_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1024:
        return True, f"{key} already installed at {dest}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "nv-broadcast-linux/2.3"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as wf:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                wf.write(chunk)
        tmp.replace(dest)
        return True, f"Downloaded {key} to {dest}"
    except Exception as exc:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False, f"Failed to download {key}: {exc}"


def ensure_models(keys: tuple[str, ...] = ("rvm_mobilenetv3", "selfie_segmenter", "face_landmarker")) -> dict[str, tuple[bool, str]]:
    results: dict[str, tuple[bool, str]] = {}
    for key in keys:
        results[key] = download_model(key)
    return results
