"""Localhost HTTP + SSE + MJPEG backend server for the NVIDIA Broadcast Electron UI."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __app_name__, __version__
from .audio_dsp import AudioMeterReading
from .audio_engine import AudioEngine
from .config import EFFECT_CATALOG, INCOMPATIBLE_GROUPS, BroadcastConfig, clamp_float
from .devices import (
    find_loopback_cameras,
    list_audio_devices,
    list_bluetooth_audio_cards,
    list_cameras,
    probe_capabilities,
    query_gpu_live,
    set_bluetooth_headset_mode,
)
from .doctor import run_doctor
from .file_processor import FileProcessor
from .models import ensure_models, get_models_status
from .pipewire import (
    VIRTUAL_MIC_SOURCE_NAME,
    VIRTUAL_SPEAKER_SINK_NAME,
    _run_pactl,
    generate_deepfilter_filter_chain_conf,
    generate_rnnoise_filter_chain_conf,
    generate_webrtc_echo_cancel_conf,
    write_persistent_filter_conf,
)
from .v4l2_ctl import list_v4l2_controls, set_v4l2_control
from .video_engine import VideoEngine


class BroadcastRuntime:
    """Owns all live engines (Audio, Video, FileProcessor) and telemetry snapshots."""

    def __init__(self, config: BroadcastConfig) -> None:
        self.config = config
        self.audio = AudioEngine(config)
        self.video = VideoEngine(config)
        self.files = FileProcessor(config)

        self._lock = threading.Lock()
        self._mic_meter = AudioMeterReading(backend=self.audio.mic_processor.backend_name)
        self._spk_meter = AudioMeterReading(backend=self.audio.speaker_processor.backend_name)
        self._gpu_live: dict[str, Any] = query_gpu_live()
        self._last_gpu_ts: float = time.monotonic()
        self._device_sig: str = ""
        self._defaults_cache: dict[str, bool] = {"mic_is_default": False, "speaker_is_default": False}

        self.audio.add_meter_callback(self._on_mic_meter)
        self.audio.add_speaker_meter_callback(self._on_spk_meter)

    def _on_mic_meter(self, reading: AudioMeterReading) -> None:
        with self._lock:
            self._mic_meter = reading

    def _on_spk_meter(self, reading: AudioMeterReading) -> None:
        with self._lock:
            self._spk_meter = reading

    def get_devices_payload(self) -> dict[str, Any]:
        audio_devs = list_audio_devices(include_monitors=False, include_virtual=False)
        cams = list_cameras(include_metadata=False, include_loopback=False)
        loopbacks = find_loopback_cameras()
        bt_cards = list_bluetooth_audio_cards()
        _, def_src, _ = _run_pactl(["get-default-source"])
        _, def_sink, _ = _run_pactl(["get-default-sink"])
        defaults = {
            "default_source": def_src,
            "default_sink": def_sink,
            "mic_is_default": VIRTUAL_MIC_SOURCE_NAME in (def_src or ""),
            "speaker_is_default": VIRTUAL_SPEAKER_SINK_NAME in (def_sink or ""),
        }
        return {
            "microphones": [d.to_dict() for d in audio_devs["microphones"]],
            "speakers": [d.to_dict() for d in audio_devs["speakers"]],
            "cameras": [c.to_dict() for c in cams],
            "loopback_cameras": [c.to_dict() for c in loopbacks],
            "bluetooth_cards": bt_cards,
            "defaults": defaults,
        }

    def _compute_device_sig(self) -> tuple[str, dict[str, bool]]:
        _, src_out, _ = _run_pactl(["list", "short", "sources"])
        _, sink_out, _ = _run_pactl(["list", "short", "sinks"])
        _, def_src, _ = _run_pactl(["get-default-source"])
        _, def_sink, _ = _run_pactl(["get-default-sink"])
        bt_cards = list_bluetooth_audio_cards()
        bt_part = "|".join(f"{c['name']}:{c['active_profile']}" for c in bt_cards)
        sig = f"{src_out}||{sink_out}||{def_src}||{def_sink}||{bt_part}"
        defaults = {
            "mic_is_default": VIRTUAL_MIC_SOURCE_NAME in (def_src or ""),
            "speaker_is_default": VIRTUAL_SPEAKER_SINK_NAME in (def_sink or ""),
        }
        return sig, defaults

    def get_telemetry(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if now - self._last_gpu_ts >= 2.0:
                self._last_gpu_ts = now
                refresh_slow = True
            else:
                refresh_slow = False
            mic_m = asdict(self._mic_meter) if self.audio.mic_running else asdict(AudioMeterReading(backend=self.audio.active_mic_backend))
            spk_m = asdict(self._spk_meter) if self.audio.speaker_running else asdict(AudioMeterReading(backend=self.audio.speaker_processor.backend_name))
            gpu = dict(self._gpu_live)
            sig = self._device_sig
            defaults = dict(self._defaults_cache)

        if refresh_slow:
            fresh_gpu = query_gpu_live()
            fresh_sig, fresh_def = self._compute_device_sig()
            with self._lock:
                self._gpu_live = fresh_gpu
                self._device_sig = fresh_sig
                self._defaults_cache = fresh_def
                gpu = dict(fresh_gpu)
                sig = fresh_sig
                defaults = dict(fresh_def)

        cam_cfg = self.config.data["camera"]
        return {
            "mic": {
                "running": self.audio.mic_running,
                "backend": self.audio.active_mic_backend,
                "is_default": defaults.get("mic_is_default", False),
                "meter": mic_m,
            },
            "speaker": {
                "running": self.audio.speaker_running,
                "backend": self.audio.speaker_processor.backend_name,
                "is_default": defaults.get("speaker_is_default", False),
                "meter": spk_m,
            },
            "camera": {
                "running": self.video.running,
                "fps": self.video.fps,
                "input": self.video.active_input,
                "output": self.video.active_output,
                "resolution": self.video.active_resolution,
                "segmentation_backend": self.video.segmentation_backend,
                "face_backend": self.video.face_backend,
                "last_error": self.video.last_error,
                "on_demand": bool(cam_cfg.get("on_demand", True)),
                "idle_timeout_sec": float(cam_cfg.get("idle_timeout_sec", 3.0)),
                "on_demand_active": bool(self.video.on_demand_active),
                "external_consumers": list(self.video.external_consumers),
                "standby_active": bool(self.video.standby_active),
            },
            "gpu": gpu,
            "device_sig": sig,
            "file_job": self.files.get_state(),
        }

    def get_full_state(self) -> dict[str, Any]:
        caps = probe_capabilities()
        return {
            "app": {"name": __app_name__, "version": __version__},
            "config": self.config.data,
            "catalog": EFFECT_CATALOG,
            "incompatible_groups": INCOMPATIBLE_GROUPS,
            "devices": self.get_devices_payload(),
            "virtual_nodes": self.audio.pw.find_existing_virtual_nodes(),
            "capabilities": caps.to_dict(),
            "models": get_models_status(),
            "telemetry": self.get_telemetry(),
        }

    def shutdown(self) -> None:
        self.files.cancel()
        self.audio.stop_all()
        self.video.stop(shutdown=True)


def make_handler(runtime: BroadcastRuntime, auth_token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _fmt: str, *_args: Any) -> None:
            return  # keep stdout clean for READY handshake

        def _check_auth(self, parsed: urllib.parse.ParseResult) -> bool:
            if not auth_token:
                return True
            hdr = self.headers.get("X-NVB-Token", "")
            if hdr == auth_token:
                return True
            qs = urllib.parse.parse_qs(parsed.query)
            return qs.get("token", [""])[0] == auth_token

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                val = json.loads(raw.decode("utf-8"))
                return val if isinstance(val, dict) else {}
            except Exception:
                return {}

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if not self._check_auth(parsed):
                self._send_json({"error": "unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return

            if parsed.path == "/api/health":
                self._send_json({"ok": True, "version": __version__})
                return

            if parsed.path == "/api/state":
                self._send_json(runtime.get_full_state())
                return

            if parsed.path == "/api/devices":
                self._send_json(runtime.get_devices_payload())
                return

            if parsed.path == "/api/doctor":
                self._send_json(run_doctor())
                return

            if parsed.path == "/api/files/state":
                self._send_json(runtime.files.get_state())
                return

            if parsed.path == "/api/camera/v4l2":
                qs = urllib.parse.parse_qs(parsed.query)
                dev = qs.get("device", [""])[0]
                if not dev or dev == "auto":
                    cams = list_cameras(include_metadata=False, include_loopback=False)
                    dev = cams[0].path if cams else "/dev/video0"
                ctrls = list_v4l2_controls(dev)
                self._send_json({"device": dev, "controls": [asdict(c) for c in ctrls]})
                return

            if parsed.path == "/api/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        data = json.dumps(runtime.get_telemetry())
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            if parsed.path == "/api/camera/stream.mjpeg":
                boundary = "nvbframe"
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_seq = -1
                try:
                    while runtime.video.running:
                        seq, jpeg = runtime.video.get_latest_jpeg(quality=84)
                        if jpeg and seq != last_seq:
                            last_seq = seq
                            hdr = (
                                f"--{boundary}\r\n"
                                f"Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(jpeg)}\r\n\r\n"
                            ).encode("ascii")
                            self.wfile.write(hdr)
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        else:
                            time.sleep(0.015)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                return

            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if not self._check_auth(parsed):
                self._send_json({"error": "unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return

            body = self._read_json()

            if parsed.path == "/api/effect":
                eid = str(body.get("effect_id", ""))
                enabled = body.get("enabled")
                params = body.get("params") if isinstance(body.get("params"), dict) else None
                res = runtime.config.set_effect(eid, enabled=enabled, params=params, resolve_conflicts=True)
                runtime.config.save()
                self._send_json(
                    {
                        "ok": not res.errors,
                        "applied": res.applied,
                        "disabled_conflicts": res.disabled_conflicts,
                        "errors": res.errors,
                        "config": runtime.config.data,
                    }
                )
                return

            if parsed.path == "/api/audio/mic/start":
                ok, msg = runtime.audio.start_microphone()
                self._send_json({"ok": ok, "message": msg, "telemetry": runtime.get_telemetry()})
                return

            if parsed.path == "/api/audio/mic/stop":
                runtime.audio.stop_microphone()
                self._send_json({"ok": True, "message": "Stopped virtual microphone", "telemetry": runtime.get_telemetry()})
                return

            if parsed.path == "/api/audio/mic/config":
                mic = runtime.config.data["microphone"]
                restart = False
                for key in ("device", "highpass_enabled", "highpass_hz", "noise_gate_enabled", "noise_gate_db", "auto_gain_enabled", "auto_gain_target_db"):
                    if key in body:
                        if key == "device" and mic.get("device") != body[key]:
                            restart = True
                        mic[key] = body[key]
                if "audio_backend" in body:
                    runtime.config.data["audio_backend"] = body["audio_backend"]
                    restart = True
                runtime.config.save()
                if restart and runtime.audio.mic_running:
                    runtime.audio.stop_microphone()
                    runtime.audio.start_microphone()
                self._send_json({"ok": True, "config": runtime.config.data})
                return

            if parsed.path == "/api/audio/mic/test":
                secs = max(1.0, min(10.0, float(body.get("seconds", 5.0))))
                try:
                    res = runtime.audio.record_mic_test(duration_sec=secs)
                    self._send_json({"ok": True, **asdict(res)})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

            if parsed.path == "/api/audio/speaker/start":
                ok, msg = runtime.audio.start_speakers()
                self._send_json({"ok": ok, "message": msg, "telemetry": runtime.get_telemetry()})
                return

            if parsed.path == "/api/audio/speaker/stop":
                runtime.audio.stop_speakers()
                self._send_json({"ok": True, "message": "Stopped virtual speakers", "telemetry": runtime.get_telemetry()})
                return

            if parsed.path == "/api/audio/speaker/config":
                spk = runtime.config.data["speaker"]
                restart = False
                if "device" in body:
                    if spk.get("device") != body["device"]:
                        restart = True
                    spk["device"] = body["device"]
                runtime.config.save()
                if restart and runtime.audio.speaker_running:
                    runtime.audio.stop_speakers()
                    runtime.audio.start_speakers()
                self._send_json({"ok": True, "config": runtime.config.data})
                return

            if parsed.path == "/api/audio/speaker/sample":
                kind = str(body.get("kind", "fan"))
                res = runtime.audio.build_speaker_sample(kind=kind)
                self._send_json({"ok": True, **asdict(res)})
                return

            if parsed.path == "/api/audio/play":
                path = str(body.get("path", ""))
                sink = body.get("target_sink")
                ok, msg = runtime.audio.play_wav(path, target_sink=sink)
                self._send_json({"ok": ok, "message": msg})
                return

            if parsed.path == "/api/audio/stop-play":
                runtime.audio.stop_playback()
                self._send_json({"ok": True})
                return

            if parsed.path == "/api/audio/set-default":
                target = str(body.get("target", ""))
                enable = bool(body.get("enable", True))
                if target == "mic":
                    ok, msg = runtime.audio.set_virtual_mic_as_default(enable)
                elif target == "speaker":
                    ok, msg = runtime.audio.set_virtual_speakers_as_default(enable)
                else:
                    ok, msg = False, f"Invalid target: {target}"
                self._send_json({"ok": ok, "message": msg, "devices": runtime.get_devices_payload()})
                return

            if parsed.path == "/api/audio/bluetooth-profile":
                card = str(body.get("card", ""))
                enable_mic = bool(body.get("enable_mic", True))
                ok, msg = set_bluetooth_headset_mode(card, enable_mic=enable_mic)
                self._send_json({"ok": ok, "message": msg, "devices": runtime.get_devices_payload()})
                return

            if parsed.path == "/api/audio/install-filter":
                caps = probe_capabilities()
                mic = runtime.config.data["microphone"]
                target = mic.get("device", "auto")
                if caps.has_deepfilter_ladspa:
                    conf = generate_deepfilter_filter_chain_conf(ladspa_plugin=caps.deepfilter_ladspa_path, target_source=target)
                    used = "deepfilternet"
                elif caps.has_rnnoise_ladspa:
                    conf = generate_rnnoise_filter_chain_conf(ladspa_plugin=caps.rnnoise_ladspa_path, target_source=target)
                    used = "rnnoise_ladspa"
                else:
                    conf = generate_webrtc_echo_cancel_conf(target_source=target)
                    used = "webrtc_aec"
                out_path = write_persistent_filter_conf(conf)
                self._send_json({"ok": True, "backend": used, "path": str(out_path)})
                return

            if parsed.path == "/api/camera/start":
                ok, msg = runtime.video.start(preview_only=bool(body.get("preview_only", False)))
                self._send_json({"ok": ok, "message": msg, "telemetry": runtime.get_telemetry()})
                return

            if parsed.path == "/api/camera/stop":
                runtime.video.stop()
                self._send_json({"ok": True, "message": "Stopped camera", "telemetry": runtime.get_telemetry()})
                return

            if parsed.path == "/api/camera/config":
                cam = runtime.config.data["camera"]
                restart = False
                for key in ("device", "output_device", "width", "height", "fps", "prefer_mjpg", "mirror", "split_preview", "split_position"):
                    if key in body:
                        if key in ("device", "output_device", "width", "height", "fps") and cam.get(key) != body[key]:
                            restart = True
                        cam[key] = body[key]
                on_demand_changed = False
                if "on_demand" in body:
                    cam["on_demand"] = bool(body["on_demand"])
                    on_demand_changed = True
                if "idle_timeout_sec" in body:
                    cam["idle_timeout_sec"] = clamp_float(body["idle_timeout_sec"], 0.5, 60.0, 3.0)
                    on_demand_changed = True
                runtime.config.save()
                if on_demand_changed:
                    runtime.video.set_on_demand(
                        bool(cam.get("on_demand", True)),
                        idle_timeout_sec=float(cam.get("idle_timeout_sec", 3.0)),
                    )
                if restart and runtime.video.running:
                    was_on_demand = runtime.video.on_demand_active and not runtime.video.manual_active
                    runtime.video.stop()
                    runtime.video.start(_from_on_demand=was_on_demand)
                elif restart and runtime.video.standby_active:
                    runtime.video.stop_standby(permanent=False)
                    runtime.video.start_standby()
                self._send_json({"ok": True, "config": runtime.config.data})
                return

            if parsed.path == "/api/camera/snapshot":
                ok, path_or_err = runtime.video.save_snapshot()
                self._send_json({"ok": ok, "path": path_or_err if ok else "", "error": "" if ok else path_or_err})
                return

            if parsed.path == "/api/camera/v4l2":
                dev = str(body.get("device", "/dev/video0"))
                if not dev or dev == "auto":
                    cams = list_cameras(include_metadata=False, include_loopback=False)
                    dev = cams[0].path if cams else "/dev/video0"
                ctrl = str(body.get("control", ""))
                val = int(body.get("value", 0))
                ok, msg = set_v4l2_control(dev, ctrl, val)
                if ok:
                    runtime.config.data["camera"].setdefault("v4l2_controls", {})[ctrl] = val
                    runtime.config.save()
                self._send_json({"ok": ok, "message": msg})
                return

            if parsed.path == "/api/files/start":
                ok, msg = runtime.files.start_job(
                    input_path=str(body.get("input_path", "")),
                    output_path=body.get("output_path") or None,
                    apply_audio=bool(body.get("apply_audio", True)),
                    apply_video=bool(body.get("apply_video", True)),
                )
                self._send_json({"ok": ok, "message": msg, "state": runtime.files.get_state()})
                return

            if parsed.path == "/api/files/cancel":
                runtime.files.cancel()
                self._send_json({"ok": True, "state": runtime.files.get_state()})
                return

            if parsed.path == "/api/models/download":
                res = ensure_models()
                self._send_json(
                    {
                        "ok": all(v[0] for v in res.values()),
                        "results": {k: {"ok": v[0], "message": v[1]} for k, v in res.items()},
                        "models": get_models_status(),
                    }
                )
                return

            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    return Handler


def daemon_state_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nv-broadcast" / "daemon.json"


def run_server(
    host: str = "127.0.0.1",
    port: int = 0,
    config_path: str | None = None,
    token: str | None = None,
    autostart: bool = False,
    write_daemon_file: bool = False,
) -> int:
    cfg = BroadcastConfig(config_path)
    runtime = BroadcastRuntime(cfg)
    auth_token = token if token is not None else secrets.token_hex(16)
    httpd = ThreadingHTTPServer((host, port), make_handler(runtime, auth_token))
    httpd.daemon_threads = True
    actual_port = httpd.server_address[1]
    base_url = f"http://{host}:{actual_port}"

    if autostart:
        runtime.audio.start_microphone()
        runtime.audio.start_speakers()
        runtime.video.start_standby()

    state_file = daemon_state_path() if write_daemon_file else None
    if state_file is not None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"url": base_url, "token": auth_token, "pid": os.getpid()}) + "\n",
            encoding="utf-8",
        )
        os.chmod(tmp, 0o600)
        tmp.replace(state_file)

    def _stop(_sig: int, _frame: Any) -> None:
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"READY {base_url} {auth_token}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        if state_file is not None and state_file.is_file():
            try:
                state_file.unlink()
            except OSError:
                pass
        httpd.server_close()
        runtime.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NVIDIA Broadcast (Linux) local API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--autostart", action="store_true", help="Auto-start virtual mic, speakers, and camera standby")
    p.add_argument("--daemon-file", action="store_true", help="Publish connection info to ~/.config/nv-broadcast/daemon.json")
    args = p.parse_args(argv)
    return run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        token=args.token,
        autostart=args.autostart,
        write_daemon_file=args.daemon_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
