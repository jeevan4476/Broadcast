"""Configuration model, persistence, and NVIDIA Broadcast v2.3.0 effect schema."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Canonical mutual-exclusion groups matching NVIDIA Broadcast v2.3.0
INCOMPATIBLE_GROUPS: list[tuple[str, ...]] = [
    ("background_blur", "background_replace", "background_remove"),
    ("studio_voice", "noise_removal"),
    ("studio_voice", "room_echo_removal"),
]

STUDIO_VOICE_PROFILES = ("default", "bright", "full", "warm")
BLUR_MODES = ("performance", "quality")
BG_REPLACE_PRESETS = ("studio_dark", "office_blur", "neon_green", "soft_daylight")
AUDIO_BACKENDS = ("auto", "pyrnnoise", "deepfilternet", "rnnoise_ladspa", "webrtc_aec", "ffmpeg_afftdn", "dsp")
SEGMENTATION_BACKENDS = ("auto", "rvm_cuda", "onnx_cuda", "mediapipe", "rvm_cpu", "onnx_cpu", "fast_portrait")

EFFECT_CATALOG: dict[str, dict[str, Any]] = {
    # Microphone
    "noise_removal": {
        "section": "microphone",
        "label": "Noise Removal",
        "gpu_load": "Low",
        "description": "Uses neural AI (RNNoise) to eliminate fan, keyboard, and ambient background noise.",
        "params": {"strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.85}},
    },
    "room_echo_removal": {
        "section": "microphone",
        "label": "Room Echo Removal",
        "gpu_load": "Low",
        "description": "Attenuates room reverberation and hollow wall reflections.",
        "params": {"strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}},
    },
    "studio_voice": {
        "section": "microphone",
        "label": "Studio Voice",
        "gpu_load": "Medium",
        "description": "Neural cleanup paired with broadcast vocal EQ, warmth, and dynamic leveling.",
        "params": {
            "micProfile": {"type": "enum", "enum": list(STUDIO_VOICE_PROFILES), "default": "default"},
        },
    },
    # Speaker
    "speaker_noise_removal": {
        "section": "speaker",
        "label": "Noise Removal",
        "gpu_load": "Low",
        "description": "Cleans background noise from incoming voice calls and meeting audio.",
        "params": {"strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.75}},
    },
    "speaker_echo_removal": {
        "section": "speaker",
        "label": "Room Echo Removal",
        "gpu_load": "Low",
        "description": "Reduces room echo on incoming call audio.",
        "params": {"strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}},
    },
    # Camera
    "background_blur": {
        "section": "camera",
        "label": "Background Blur",
        "gpu_load": "Medium",
        "description": "Uses neural portrait matting to smoothly blur your background.",
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
            "mode": {"type": "enum", "enum": list(BLUR_MODES), "default": "quality"},
        },
    },
    "background_replace": {
        "section": "camera",
        "label": "Background Replacement",
        "gpu_load": "Medium",
        "description": "Replaces your background with a custom image or studio backdrop.",
        "params": {
            "imagePath": {"type": "path", "default": ""},
            "preset": {"type": "enum", "enum": list(BG_REPLACE_PRESETS), "default": "studio_dark"},
        },
    },
    "background_remove": {
        "section": "camera",
        "label": "Background Removal",
        "gpu_load": "Medium",
        "description": "Replaces your background with chroma green (#00FF00) for OBS compositing.",
        "params": {
            "chromaColor": {"type": "string", "default": "#00ff00"},
        },
    },
    "auto_frame": {
        "section": "camera",
        "label": "Auto Frame",
        "gpu_load": "Low",
        "description": "Tracks your face and smoothly crops and zooms to keep you centered.",
        "params": {
            "zoom": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35},
        },
    },
    "eye_contact": {
        "section": "camera",
        "label": "Eye Contact",
        "gpu_load": "Medium",
        "description": "Redirects your gaze toward the camera using real-time iris tracking.",
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
            "lookAway": {"type": "bool", "default": True},
        },
    },
    "video_noise_removal": {
        "section": "camera",
        "label": "Video Noise Removal",
        "gpu_load": "Low",
        "description": "Strips low-light webcam grain using motion-aware temporal and bilateral denoising.",
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.6},
        },
    },
    "virtual_key_light": {
        "section": "camera",
        "label": "Virtual Key Light",
        "gpu_load": "Medium",
        "description": "Relights your face with adjustable cool-to-warm studio key lighting.",
        "params": {
            "warmth": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "intensity": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.55},
        },
    },
    "vignette": {
        "section": "camera",
        "label": "Vignette",
        "gpu_load": "Low",
        "description": "Applies a soft studio edge darkening that can follow your face.",
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "faceTracking": {"type": "bool", "default": True},
        },
    },
}


def default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nv-broadcast" / "config.json"


def default_config() -> dict[str, Any]:
    return {
        "version": 2,
        "audio_backend": "auto",
        "segmentation_backend": "auto",
        "ui": {
            "show_gpu_indicator": True,
            "start_minimized": False,
        },
        "microphone": {
            "device": "auto",
            "virtual_name": "NVIDIA Broadcast (Microphone)",
            "sample_rate": 48000,
            "highpass_enabled": True,
            "highpass_hz": 85,
            "noise_gate_enabled": True,
            "noise_gate_db": -46.0,
            "auto_gain_enabled": True,
            "auto_gain_target_db": -16.0,
            "effects": {
                "noise_removal": {"enabled": True, "strength": 0.85},
                "room_echo_removal": {"enabled": False, "strength": 0.5},
                "studio_voice": {"enabled": False, "micProfile": "default"},
            },
        },
        "speaker": {
            "device": "auto",
            "virtual_name": "NVIDIA Broadcast (Speakers)",
            "sample_rate": 48000,
            "effects": {
                "speaker_noise_removal": {"enabled": False, "strength": 0.75},
                "speaker_echo_removal": {"enabled": False, "strength": 0.5},
            },
        },
        "camera": {
            "device": "auto",
            "output_device": "auto",
            "virtual_name": "NVIDIA Broadcast Camera",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "prefer_mjpg": True,
            "mirror": False,
            "split_preview": False,
            "split_position": 0.5,
            "v4l2_controls": {},
            "effects": {
                "background_blur": {"enabled": False, "strength": 0.7, "mode": "quality"},
                "background_replace": {"enabled": False, "imagePath": "", "preset": "studio_dark"},
                "background_remove": {"enabled": False, "chromaColor": "#00ff00"},
                "auto_frame": {"enabled": False, "zoom": 0.35},
                "eye_contact": {"enabled": False, "strength": 0.7, "lookAway": True},
                "video_noise_removal": {"enabled": False, "strength": 0.6},
                "virtual_key_light": {"enabled": False, "warmth": 0.5, "intensity": 0.55},
                "vignette": {"enabled": False, "strength": 0.4, "faceTracking": True},
            },
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def clamp_float(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


@dataclass
class EffectApplyResult:
    applied: list[str]
    disabled_conflicts: dict[str, list[str]]
    errors: list[str]


class BroadcastConfig:
    """Thread-safe JSON configuration store with v2.3.0 schema validation."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_config_path()
        self.data: dict[str, Any] = default_config()
        self.load()

    def load(self) -> dict[str, Any]:
        base = default_config()
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    base = _deep_merge(base, loaded)
            except (OSError, json.JSONDecodeError):
                pass
        self.data = base
        self._sanitize()
        return self.data

    def save(self) -> Path:
        self._sanitize()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    def _sanitize(self) -> None:
        if self.data.get("audio_backend") not in AUDIO_BACKENDS:
            self.data["audio_backend"] = "auto"
        if self.data.get("segmentation_backend") not in SEGMENTATION_BACKENDS:
            self.data["segmentation_backend"] = "auto"

        mic = self.data["microphone"]
        mic["highpass_hz"] = int(clamp_float(mic.get("highpass_hz", 85), 20, 300, 85))
        mic["noise_gate_db"] = clamp_float(mic.get("noise_gate_db", -46.0), -80.0, -10.0, -46.0)
        mic["auto_gain_target_db"] = clamp_float(mic.get("auto_gain_target_db", -16.0), -30.0, -3.0, -16.0)

        cam = self.data["camera"]
        cam["split_preview"] = bool(cam.get("split_preview", False))
        cam["split_position"] = clamp_float(cam.get("split_position", 0.5), 0.05, 0.95, 0.5)

        for eid, meta in EFFECT_CATALOG.items():
            section = meta["section"]
            eff = self.data[section]["effects"].setdefault(eid, {"enabled": False})
            eff["enabled"] = bool(eff.get("enabled", False))
            for pname, pspec in meta["params"].items():
                ptype = pspec["type"]
                if ptype == "float":
                    eff[pname] = clamp_float(eff.get(pname, pspec["default"]), pspec["min"], pspec["max"], pspec["default"])
                elif ptype == "bool":
                    eff[pname] = bool(eff.get(pname, pspec["default"]))
                elif ptype == "enum":
                    if eff.get(pname) not in pspec["enum"]:
                        eff[pname] = pspec["default"]
                elif ptype in ("path", "string"):
                    val = eff.get(pname, pspec["default"])
                    eff[pname] = str(val) if val is not None else pspec["default"]

    def get_effect(self, effect_id: str) -> dict[str, Any]:
        if effect_id not in EFFECT_CATALOG:
            raise KeyError(f"Unknown effectId: {effect_id}")
        section = EFFECT_CATALOG[effect_id]["section"]
        return copy.deepcopy(self.data[section]["effects"][effect_id])

    def set_effect(
        self,
        effect_id: str,
        enabled: bool | None = None,
        params: dict[str, Any] | None = None,
        resolve_conflicts: bool = True,
    ) -> EffectApplyResult:
        """Update an effect and enforce NVIDIA Broadcast mutual-exclusion rules."""
        if effect_id not in EFFECT_CATALOG:
            return EffectApplyResult([], {}, [f"Unknown effectId: {effect_id}"])

        meta = EFFECT_CATALOG[effect_id]
        section = meta["section"]
        eff = self.data[section]["effects"][effect_id]
        disabled_conflicts: dict[str, list[str]] = {}

        if params:
            for k, v in params.items():
                if k not in meta["params"]:
                    continue
                pspec = meta["params"][k]
                if pspec["type"] == "float":
                    eff[k] = clamp_float(v, pspec["min"], pspec["max"], pspec["default"])
                elif pspec["type"] == "bool":
                    eff[k] = bool(v)
                elif pspec["type"] == "enum":
                    if v in pspec["enum"]:
                        eff[k] = v
                else:
                    eff[k] = str(v)

        if enabled is not None:
            eff["enabled"] = bool(enabled)

        if eff["enabled"] and resolve_conflicts:
            turned_off: list[str] = []
            for group in INCOMPATIBLE_GROUPS:
                if effect_id in group:
                    for other in group:
                        if other == effect_id:
                            continue
                        other_sec = EFFECT_CATALOG[other]["section"]
                        other_eff = self.data[other_sec]["effects"][other]
                        if other_eff.get("enabled"):
                            other_eff["enabled"] = False
                            turned_off.append(other)
            if turned_off:
                disabled_conflicts[effect_id] = turned_off

        self._sanitize()
        return EffectApplyResult([effect_id], disabled_conflicts, [])

    def get_broadcast_state(self) -> dict[str, Any]:
        """Return state in NVIDIA Broadcast v2.3.0 public schema shape."""
        return {
            "microphone": copy.deepcopy(self.data["microphone"]["effects"]),
            "speaker": copy.deepcopy(self.data["speaker"]["effects"]),
            "camera": copy.deepcopy(self.data["camera"]["effects"]),
        }
