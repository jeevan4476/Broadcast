"""Hardware, PipeWire, V4L2, and Neural AI environment doctor."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from .devices import find_loopback_cameras, list_audio_devices, list_cameras, probe_capabilities


def find_electron_binary() -> str:
    here = Path(__file__).resolve().parent.parent
    candidates = [
        here / "ui" / "node_modules" / ".bin" / "electron",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    found = shutil.which("electron")
    return found or ""


def run_doctor() -> dict[str, Any]:
    caps = probe_capabilities()
    audio = list_audio_devices(include_monitors=False, include_virtual=False)
    cams = list_cameras(include_metadata=False, include_loopback=False)
    loopbacks = find_loopback_cameras()
    electron_bin = find_electron_binary()

    checks: list[dict[str, Any]] = []

    if caps.gpu.available:
        checks.append(
            {
                "name": "nvidia_gpu",
                "status": "ok",
                "detail": (
                    f"{caps.gpu.name} (driver {caps.gpu.driver_version}, "
                    f"{caps.gpu.vram_mib} MiB VRAM, CC {caps.gpu.compute_capability}"
                    + (f", CUDA {caps.gpu.cuda_toolkit}" if caps.gpu.cuda_toolkit else "")
                    + ")"
                ),
            }
        )
    else:
        checks.append({"name": "nvidia_gpu", "status": "warn", "detail": "No NVIDIA GPU detected via nvidia-smi"})

    if caps.pipewire_version and caps.has_pactl and caps.has_pw_cat:
        checks.append(
            {
                "name": "pipewire_audio",
                "status": "ok",
                "detail": (
                    f"PipeWire {caps.pipewire_version} ready "
                    f"(RNNoise neural: {'yes' if caps.python_packages.get('pyrnnoise') else 'no'}, "
                    f"WebRTC AEC: {'yes' if caps.has_pipewire_webrtc_aec else 'no'}, "
                    f"LADSPA RNNoise: {'yes' if caps.has_rnnoise_ladspa else 'no'})"
                ),
            }
        )
    else:
        checks.append({"name": "pipewire_audio", "status": "fail", "detail": "PipeWire / pactl / pw-cat missing"})

    checks.append(
        {
            "name": "audio_devices",
            "status": "ok" if audio["microphones"] else "warn",
            "detail": f"{len(audio['microphones'])} microphone(s), {len(audio['speakers'])} speaker output(s)",
        }
    )

    if cams:
        primary = cams[0]
        fmt_items = primary.formats.values() if isinstance(primary.formats, dict) else primary.formats
        fmts = "/".join(getattr(f, "pixel_format", str(f)) for f in fmt_items) or "unknown"
        checks.append(
            {
                "name": "physical_camera",
                "status": "ok",
                "detail": f"{primary.path} ({primary.card}, {fmts})",
            }
        )
    else:
        checks.append({"name": "physical_camera", "status": "warn", "detail": "No physical V4L2 camera detected"})

    if caps.v4l2loopback_loaded and loopbacks:
        checks.append(
            {
                "name": "virtual_camera",
                "status": "ok",
                "detail": f"v4l2loopback active at {', '.join(d.path for d in loopbacks)}",
            }
        )
    elif caps.has_v4l2loopback_module:
        checks.append(
            {
                "name": "virtual_camera",
                "status": "warn",
                "detail": (
                    "v4l2loopback installed but not loaded. Run: "
                    'sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast Camera" exclusive_caps=1'
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "virtual_camera",
                "status": "warn",
                "detail": "v4l2loopback kernel module not installed (preview works; virtual webcam output needs akmod-v4l2loopback)",
            }
        )

    installed_models = [k for k, v in caps.models.items() if v.get("installed")]
    has_real_ai = bool(caps.python_packages.get("pyrnnoise")) and bool(
        caps.python_packages.get("onnxruntime") or caps.python_packages.get("mediapipe")
    )
    checks.append(
        {
            "name": "neural_ai_runtime",
            "status": "ok" if has_real_ai else "warn",
            "detail": (
                f"audio: {caps.recommended_audio_backend}, segmentation: {caps.recommended_segmentation_backend}, "
                f"models: {', '.join(installed_models) or 'none'} "
                f"(Python {sys.version.split()[0]})"
            ),
        }
    )

    checks.append(
        {
            "name": "desktop_gui",
            "status": "ok" if (electron_bin or caps.python_packages.get("gi")) else "warn",
            "detail": (
                f"Electron Broadcast 2.3.0 UI ready ({electron_bin})"
                if electron_bin
                else ("GTK4 + libadwaita fallback available" if caps.python_packages.get("gi") else "CLI only")
            ),
        }
    )

    next_steps: list[str] = []
    if not caps.has_v4l2loopback_module:
        next_steps.append("Install v4l2loopback on Fedora (RPM Fusion): sudo dnf install akmod-v4l2loopback v4l2loopback v4l-utils")
    elif not caps.v4l2loopback_loaded:
        next_steps.append(
            'Load virtual camera: sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast Camera" exclusive_caps=1'
        )
    if not installed_models:
        next_steps.append("Download neural models: nv-broadcast download-models")

    return {
        "python": sys.version.split()[0],
        "capabilities": caps.to_dict(),
        "audio_devices": {
            "microphones": [d.to_dict() for d in audio["microphones"]],
            "speakers": [d.to_dict() for d in audio["speakers"]],
        },
        "cameras": [c.to_dict() for c in cams],
        "loopback_cameras": [c.to_dict() for c in loopbacks],
        "electron_binary": electron_bin,
        "checks": checks,
        "next_steps": next_steps,
    }


def format_doctor_text(report: dict[str, Any]) -> str:
    lines = ["=== NVIDIA Broadcast (Linux) Doctor ==="]
    badges = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]", "info": "[INFO]"}
    for chk in report["checks"]:
        lines.append(f"  {badges.get(chk['status'], '[INFO]')} {chk['name']}: {chk['detail']}")
    if report["next_steps"]:
        lines.append("\nRecommended next steps:")
        for step in report["next_steps"]:
            lines.append(f"  - {step}")
    return "\n".join(lines)
