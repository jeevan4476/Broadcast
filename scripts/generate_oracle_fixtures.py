#!/usr/bin/env python3
"""Generate deterministic golden Python oracle fixtures for Rust port parity tests.

Following Anthropic's Oracle Verification Pattern (Messages API Rust port), the
existing Python implementation acts as the ground-truth oracle during migration.
This script emits deterministic JSON fixtures under `rust/fixtures/` covering:

  1. `config_oracle.json`: default config, clamping edge cases, mutual-exclusion
     conflict resolution, EFFECT_CATALOG, and INCOMPATIBLE_GROUPS.
  2. `api_contract_oracle.json`: exact JSON key contracts for `/api/health`,
     `/api/state`, `/api/devices`, `/api/doctor`, and `POST /api/effect`.
  3. `dsp_oracle.json`: PCM16 roundtrip vectors, `compute_rms_db` reference
     values, `build_isolated_pw_cat_cmd`, `parse_pw_links`, `select_port_pairs`,
     and `generate_sample_clip` metrics.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nv_broadcast import __app_name__, __version__
from nv_broadcast.audio_dsp import (
    AudioMeterReading,
    _floats_to_pcm16,
    _pcm16_to_floats,
    compute_rms_db,
    generate_sample_clip,
)
from nv_broadcast.config import (
    AUDIO_BACKENDS,
    BG_REPLACE_PRESETS,
    BLUR_MODES,
    EFFECT_CATALOG,
    INCOMPATIBLE_GROUPS,
    SEGMENTATION_BACKENDS,
    STUDIO_VOICE_PROFILES,
    BroadcastConfig,
    default_config,
)
from nv_broadcast.pipewire import (
    VIRTUAL_MIC_SINK_NAME,
    VIRTUAL_MIC_SOURCE_NAME,
    VIRTUAL_NODE_NAMES,
    VIRTUAL_SPEAKER_SINK_NAME,
    build_isolated_pw_cat_cmd,
    is_pw_link_active,
    is_virtual_broadcast_node,
    parse_pw_links,
    select_port_pairs,
)
from nv_broadcast.server import BroadcastRuntime, make_handler
from nv_broadcast.v4l2_ctl import V4L2DeviceInfo, V4L2Mode, choose_best_pixel_format

FIXTURES_DIR = REPO_ROOT / "rust" / "fixtures"


def build_config_oracle() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Fresh default config
        fresh = BroadcastConfig(tmp / "missing.json")

        # 2. Clamping & invalid value sanitization
        dirty_path = tmp / "dirty.json"
        dirty_input = {
            "audio_backend": "not_a_real_backend",
            "segmentation_backend": "invalid_seg",
            "microphone": {
                "highpass_hz": 9999,
                "noise_gate_db": -120.5,
                "auto_gain_target_db": 12.0,
                "effects": {
                    "noise_removal": {"enabled": True, "strength": 1.8},
                    "room_echo_removal": {"enabled": False, "strength": -0.4},
                    "studio_voice": {"enabled": False, "micProfile": "alien"},
                },
            },
            "camera": {
                "split_preview": 1,
                "split_position": 0.99,
                "on_demand": None,
                "idle_timeout_sec": 0.05,
                "effects": {
                    "background_blur": {"enabled": True, "strength": -5.0, "mode": "ultra"},
                    "background_replace": {"enabled": False, "preset": "mars"},
                    "auto_frame": {"enabled": True, "zoom": 2.5},
                },
            },
        }
        dirty_path.write_text(json.dumps(dirty_input), encoding="utf-8")
        sanitized = BroadcastConfig(dirty_path)

        # 3. Conflict resolution scenarios
        conflict_cases: list[dict[str, Any]] = []

        # Case A: enabling studio_voice disables noise_removal and room_echo_removal
        cfg_a = BroadcastConfig(tmp / "case_a.json")
        cfg_a.set_effect("noise_removal", enabled=True)
        cfg_a.set_effect("room_echo_removal", enabled=True)
        res_a = cfg_a.set_effect("studio_voice", enabled=True, params={"micProfile": "warm"})
        conflict_cases.append(
            {
                "name": "studio_voice_disables_mic_noise_and_echo",
                "setup": [
                    {"effect_id": "noise_removal", "enabled": True},
                    {"effect_id": "room_echo_removal", "enabled": True},
                ],
                "action": {"effect_id": "studio_voice", "enabled": True, "params": {"micProfile": "warm"}},
                "result": {
                    "applied": res_a.applied,
                    "disabled_conflicts": res_a.disabled_conflicts,
                    "errors": res_a.errors,
                },
                "final_broadcast_state": cfg_a.get_broadcast_state(),
            }
        )

        # Case B: enabling background_remove disables background_blur
        cfg_b = BroadcastConfig(tmp / "case_b.json")
        cfg_b.set_effect("background_blur", enabled=True, params={"strength": 0.85, "mode": "performance"})
        res_b = cfg_b.set_effect("background_remove", enabled=True, params={"chromaColor": "#00ff00"})
        conflict_cases.append(
            {
                "name": "background_remove_disables_background_blur",
                "setup": [
                    {
                        "effect_id": "background_blur",
                        "enabled": True,
                        "params": {"strength": 0.85, "mode": "performance"},
                    }
                ],
                "action": {
                    "effect_id": "background_remove",
                    "enabled": True,
                    "params": {"chromaColor": "#00ff00"},
                },
                "result": {
                    "applied": res_b.applied,
                    "disabled_conflicts": res_b.disabled_conflicts,
                    "errors": res_b.errors,
                },
                "final_broadcast_state": cfg_b.get_broadcast_state(),
            }
        )

        # Case C: enabling noise_removal disables studio_voice
        cfg_c = BroadcastConfig(tmp / "case_c.json")
        cfg_c.set_effect("studio_voice", enabled=True, params={"micProfile": "bright"})
        res_c = cfg_c.set_effect("noise_removal", enabled=True, params={"strength": 0.9})
        conflict_cases.append(
            {
                "name": "noise_removal_disables_studio_voice",
                "setup": [
                    {"effect_id": "studio_voice", "enabled": True, "params": {"micProfile": "bright"}},
                ],
                "action": {"effect_id": "noise_removal", "enabled": True, "params": {"strength": 0.9}},
                "result": {
                    "applied": res_c.applied,
                    "disabled_conflicts": res_c.disabled_conflicts,
                    "errors": res_c.errors,
                },
                "final_broadcast_state": cfg_c.get_broadcast_state(),
            }
        )

        # Case D: unknown effect returns error
        cfg_d = BroadcastConfig(tmp / "case_d.json")
        res_d = cfg_d.set_effect("does_not_exist", enabled=True)
        conflict_cases.append(
            {
                "name": "unknown_effect_returns_error",
                "setup": [],
                "action": {"effect_id": "does_not_exist", "enabled": True},
                "result": {
                    "applied": res_d.applied,
                    "disabled_conflicts": res_d.disabled_conflicts,
                    "errors": res_d.errors,
                },
                "final_broadcast_state": cfg_d.get_broadcast_state(),
            }
        )

    return {
        "schema_version": 2,
        "app": {"name": __app_name__, "version": __version__},
        "enums": {
            "audio_backends": list(AUDIO_BACKENDS),
            "segmentation_backends": list(SEGMENTATION_BACKENDS),
            "studio_voice_profiles": list(STUDIO_VOICE_PROFILES),
            "blur_modes": list(BLUR_MODES),
            "bg_replace_presets": list(BG_REPLACE_PRESETS),
        },
        "incompatible_groups": [list(g) for g in INCOMPATIBLE_GROUPS],
        "effect_catalog": EFFECT_CATALOG,
        "default_config": default_config(),
        "loaded_default_config": fresh.data,
        "sanitization_case": {
            "input": dirty_input,
            "expected": sanitized.data,
        },
        "conflict_cases": conflict_cases,
    }


def _keys_sorted(obj: dict[str, Any]) -> list[str]:
    return sorted(obj.keys())


def build_api_contract_oracle() -> dict[str, Any]:
    token = "oracle-test-token"
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = BroadcastConfig(Path(tmpdir) / "config.json")
        runtime = BroadcastRuntime(cfg)
        handler_cls = make_handler(runtime, token)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            def get_json(path: str, with_token: bool = True) -> tuple[int, dict[str, Any]]:
                req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
                if with_token:
                    req.add_header("X-NVB-Token", token)
                try:
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        return resp.status, json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as err:
                    return err.code, json.loads(err.read().decode("utf-8"))

            def post_json(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}{path}",
                    data=data,
                    method="POST",
                    headers={"Content-Type": "application/json", "X-NVB-Token": token},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status, json.loads(resp.read().decode("utf-8"))

            unauth_status, unauth_body = get_json("/api/health", with_token=False)
            health_status, health_body = get_json("/api/health")
            state_status, state_body = get_json("/api/state")
            devices_status, devices_body = get_json("/api/devices")
            effect_status, effect_body = post_json(
                "/api/effect",
                {"effect_id": "studio_voice", "enabled": True, "params": {"micProfile": "warm"}},
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
            runtime.shutdown()

    return {
        "auth_header": "X-NVB-Token",
        "auth_query_param": "token",
        "unauthorized_response": {
            "status": unauth_status,
            "keys": _keys_sorted(unauth_body),
            "body": unauth_body,
        },
        "health": {
            "status": health_status,
            "keys": _keys_sorted(health_body),
            "body": health_body,
        },
        "state": {
            "status": state_status,
            "top_level_keys": _keys_sorted(state_body),
            "app_keys": _keys_sorted(state_body["app"]),
            "telemetry_keys": _keys_sorted(state_body["telemetry"]),
            "telemetry_mic_keys": _keys_sorted(state_body["telemetry"]["mic"]),
            "telemetry_speaker_keys": _keys_sorted(state_body["telemetry"]["speaker"]),
            "telemetry_camera_keys": _keys_sorted(state_body["telemetry"]["camera"]),
            "meter_keys": _keys_sorted(state_body["telemetry"]["mic"]["meter"]),
        },
        "devices": {
            "status": devices_status,
            "top_level_keys": _keys_sorted(devices_body),
            "defaults_keys": _keys_sorted(devices_body["defaults"]),
        },
        "post_effect": {
            "status": effect_status,
            "keys": _keys_sorted(effect_body),
            "ok": effect_body["ok"],
            "applied": effect_body["applied"],
            "disabled_conflicts": effect_body["disabled_conflicts"],
            "errors": effect_body["errors"],
        },
    }


def build_dsp_oracle() -> dict[str, Any]:
    # Deterministic float vectors for compute_rms_db and PCM16 roundtrip
    sine_480 = [0.5 * math.sin(2.0 * math.pi * 440.0 * i / 48000.0) for i in range(480)]
    vectors = {
        "empty": [],
        "silence": [0.0] * 64,
        "full_scale_dc": [1.0] * 16,
        "half_scale_alt": [0.5 if i % 2 == 0 else -0.5 for i in range(32)],
        "sine_440hz_480": sine_480,
    }
    rms_cases = []
    for name, samples in vectors.items():
        rms_db, peak_db = compute_rms_db(samples)
        pcm = _floats_to_pcm16(samples)
        roundtrip = _pcm16_to_floats(pcm)
        rms_cases.append(
            {
                "name": name,
                "samples": [] if name == "sine_440hz_480" else [round(s, 8) for s in samples],
                "sample_count": len(samples),
                "is_sine_440hz_480": name == "sine_440hz_480",
                "rms_db": round(rms_db, 6),
                "peak_db": round(peak_db, 6),
                "pcm16_len_bytes": len(pcm),
                "pcm16_head_hex": pcm[:16].hex(),
                "roundtrip_head": [round(s, 8) for s in roundtrip[:8]],
            }
        )

    # Sample clip baseline RMS metrics
    clip_metrics = {}
    for kind in ("fan", "typing", "echo"):
        pcm = generate_sample_clip(kind, sample_rate=48000, duration_sec=1.0)
        floats = _pcm16_to_floats(pcm)
        rms_db, peak_db = compute_rms_db(floats)
        clip_metrics[kind] = {
            "sample_rate": 48000,
            "duration_sec": 1.0,
            "byte_len": len(pcm),
            "rms_db": round(rms_db, 4),
            "peak_db": round(peak_db, 4),
        }

    # PipeWire helper contracts
    sample_pw_links = "\n".join(
        [
            "alsa_input.pci-0000_00_1f.3:capture_FL",
            "  |-> nv_broadcast_mic_capture:input_FL",
            "alsa_input.pci-0000_00_1f.3:capture_FR",
            "  |-> nv_broadcast_mic_capture:input_FR",
            "nv_broadcast_speakers:monitor_FL",
            "  |-> nv_broadcast_spk_capture:input_FL",
            "  |-> other_app:input_FL",
        ]
    )
    parsed_links = parse_pw_links(sample_pw_links)
    out_ports = [
        "phys_mic:capture_FL",
        "phys_mic:capture_FR",
        "mono_mic:capture_MONO",
        "aux_src:output_AUX0",
        "aux_src:output_AUX1",
    ]
    in_ports = [
        "cap_stereo:input_FL",
        "cap_stereo:input_FR",
        "cap_mono:input_MONO",
        "cap_aux:input_AUX0",
        "cap_aux:input_AUX1",
    ]

    # V4L2 mode selection contract
    v4l2_device = V4L2DeviceInfo(
        path="/dev/video0",
        driver="uvcvideo",
        card="USB HD Webcam",
        bus_info="usb-0000:00:14.0-1",
        is_capture=True,
        is_output=False,
        is_metadata=False,
        is_loopback=False,
        formats=["YUYV", "MJPG"],
        modes=[
            V4L2Mode("YUYV", 1280, 720, [10.0, 5.0]),
            V4L2Mode("MJPG", 1280, 720, [30.0, 15.0]),
            V4L2Mode("YUYV", 640, 480, [30.0]),
            V4L2Mode("MJPG", 640, 480, [30.0]),
        ],
    )
    best_hd = choose_best_pixel_format(v4l2_device, 1280, 720, prefer_mjpg=True)
    best_vga_prefer_mjpg = choose_best_pixel_format(v4l2_device, 640, 480, prefer_mjpg=True)
    best_fallback = choose_best_pixel_format(v4l2_device, 1920, 1080, prefer_mjpg=True)

    return {
        "audio_constants": {
            "sample_rate": 48000,
            "rnnoise_frame_samples": 480,
            "default_meter": AudioMeterReading().__dict__,
        },
        "rms_cases": rms_cases,
        "sample_clips": clip_metrics,
        "pipewire": {
            "virtual_mic_sink_name": VIRTUAL_MIC_SINK_NAME,
            "virtual_mic_source_name": VIRTUAL_MIC_SOURCE_NAME,
            "virtual_speaker_sink_name": VIRTUAL_SPEAKER_SINK_NAME,
            "virtual_node_names": list(VIRTUAL_NODE_NAMES),
            "virtual_checks": {
                "nv_broadcast_mic": is_virtual_broadcast_node("nv_broadcast_mic"),
                "nv_broadcast_mic_bus.monitor": is_virtual_broadcast_node("nv_broadcast_mic_bus.monitor"),
                "NV_BROADCAST_SPEAKERS": is_virtual_broadcast_node("NV_BROADCAST_SPEAKERS"),
                "alsa_input.pci-0000_00_1f.3": is_virtual_broadcast_node("alsa_input.pci-0000_00_1f.3"),
                "": is_virtual_broadcast_node(""),
            },
            "isolated_pw_cat_record": build_isolated_pw_cat_cmd(
                "record", "nv_broadcast_mic_capture", rate=48000, channels=1, latency_ms=20
            ),
            "isolated_pw_cat_playback": build_isolated_pw_cat_cmd(
                "playback", "nv_broadcast_spk_playback", rate=48000, channels=2, latency_ms=20
            ),
            "sample_pw_links_raw": sample_pw_links,
            "parsed_pw_links": [list(edge) for edge in parsed_links],
            "link_active_true": is_pw_link_active(
                "alsa_input.pci-0000_00_1f.3", "nv_broadcast_mic_capture", parsed_links
            ),
            "link_active_false": is_pw_link_active(
                "alsa_input.pci-0000_00_1f.3", "other_app", parsed_links
            ),
            "port_pairing": {
                "out_ports": out_ports,
                "in_ports": in_ports,
                "stereo_to_stereo": [
                    list(p) for p in select_port_pairs("phys_mic", "cap_stereo", out_ports, in_ports)
                ],
                "stereo_to_mono": [
                    list(p) for p in select_port_pairs("phys_mic", "cap_mono", out_ports, in_ports)
                ],
                "mono_to_stereo": [
                    list(p) for p in select_port_pairs("mono_mic", "cap_stereo", out_ports, in_ports)
                ],
                "aux_positional": [
                    list(p) for p in select_port_pairs("aux_src", "cap_aux", out_ports, in_ports)
                ],
            },
        },
        "v4l2": {
            "device": v4l2_device.to_dict(),
            "best_hd": list(best_hd),
            "best_vga_prefer_mjpg": list(best_vga_prefer_mjpg),
            "best_fallback_1080p": list(best_fallback),
        },
    }


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "config_oracle.json": build_config_oracle(),
        "api_contract_oracle.json": build_api_contract_oracle(),
        "dsp_oracle.json": build_dsp_oracle(),
    }
    for filename, payload in outputs.items():
        out_path = FIXTURES_DIR / filename
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        print(f"Wrote {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
