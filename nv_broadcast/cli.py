"""Command-line interface for NVIDIA Broadcast (Linux) with --json output."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .audio_engine import AudioEngine
from .config import (
    AUDIO_BACKENDS,
    BG_REPLACE_PRESETS,
    BLUR_MODES,
    EFFECT_CATALOG,
    INCOMPATIBLE_GROUPS,
    STUDIO_VOICE_PROFILES,
    BroadcastConfig,
)
from .devices import find_loopback_cameras, list_audio_devices, list_cameras, probe_capabilities
from .doctor import find_electron_binary, format_doctor_text, run_doctor
from .file_processor import FileProcessor
from .models import ensure_models, get_models_status
from .pipewire import (
    PipeWireVirtualManager,
    generate_deepfilter_filter_chain_conf,
    generate_rnnoise_filter_chain_conf,
    generate_webrtc_echo_cancel_conf,
    write_persistent_filter_conf,
)
from .v4l2_ctl import list_v4l2_controls, set_v4l2_control
from .video_engine import VideoEngine


def _emit(payload: Any, as_json: bool, text: str | None = None) -> int:
    if as_json:
        print(json.dumps(payload, indent=2))
    elif text is not None:
        print(text)
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _parse_bool(val: str) -> bool:
    low = val.strip().lower()
    if low in ("1", "true", "on", "yes", "enable", "enabled"):
        return True
    if low in ("0", "false", "off", "no", "disable", "disabled"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nv-broadcast",
        description="NVIDIA Broadcast for Linux — PipeWire Neural Audio & GPU V4L2 Camera Suite",
    )
    p.add_argument("--version", action="version", version=f"nv-broadcast {__version__}")
    p.add_argument("--config", type=str, default=None, help="Path to config.json")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")

    sub = p.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Inspect GPU, PipeWire, V4L2 camera, v4l2loopback, and neural AI backends")
    sub.add_parser("status", help="Show current configuration, effect state, and virtual devices")

    dev_p = sub.add_parser("devices", help="List microphones, speakers, and V4L2 cameras")
    dev_p.add_argument("--all", action="store_true", help="Include monitor/metadata/virtual devices")

    sub.add_parser("effects", help="Describe all supported NVIDIA Broadcast effect IDs and parameters")
    sub.add_parser("download-models", help="Download/verify neural segmentation and face/iris landmark models")

    # Microphone config subcommand
    mic_p = sub.add_parser("mic", help="Configure microphone device and effects")
    mic_p.add_argument("--device", type=str, help="PipeWire source name or 'auto'")
    mic_p.add_argument("--backend", choices=AUDIO_BACKENDS, help="Preferred audio denoiser backend")
    mic_p.add_argument("--noise-removal", type=_parse_bool, help="Enable/disable noise_removal")
    mic_p.add_argument("--noise-strength", type=float, help="Noise removal strength 0.0-1.0")
    mic_p.add_argument("--echo-removal", type=_parse_bool, help="Enable/disable room_echo_removal")
    mic_p.add_argument("--echo-strength", type=float, help="Room echo removal strength 0.0-1.0")
    mic_p.add_argument("--studio-voice", type=_parse_bool, help="Enable/disable studio_voice")
    mic_p.add_argument("--mic-profile", choices=STUDIO_VOICE_PROFILES, help="Studio Voice profile")
    mic_p.add_argument("--highpass", type=_parse_bool, help="Enable/disable fan rumble high-pass filter")
    mic_p.add_argument("--highpass-hz", type=int, help="High-pass cutoff frequency (20-300 Hz)")
    mic_p.add_argument("--noise-gate", type=_parse_bool, help="Enable/disable noise gate")
    mic_p.add_argument("--auto-gain", type=_parse_bool, help="Enable/disable voice auto-gain/compressor")

    # Speaker config subcommand
    spk_p = sub.add_parser("speaker", help="Configure speaker output and incoming audio effects")
    spk_p.add_argument("--device", type=str, help="PipeWire sink name or 'auto'")
    spk_p.add_argument("--noise-removal", type=_parse_bool, help="Enable/disable speaker_noise_removal")
    spk_p.add_argument("--noise-strength", type=float, help="Speaker noise removal strength 0.0-1.0")
    spk_p.add_argument("--echo-removal", type=_parse_bool, help="Enable/disable speaker_echo_removal")
    spk_p.add_argument("--echo-strength", type=float, help="Speaker echo removal strength 0.0-1.0")

    # Camera config subcommand
    cam_p = sub.add_parser("camera", help="Configure webcam, resolution, and video effects")
    cam_p.add_argument("--device", type=str, help="Input V4L2 device path (e.g. /dev/video0) or 'auto'")
    cam_p.add_argument("--output-device", type=str, help="Output v4l2loopback device or 'auto'")
    cam_p.add_argument("--width", type=int, help="Capture width")
    cam_p.add_argument("--height", type=int, help="Capture height")
    cam_p.add_argument("--fps", type=int, help="Capture frame rate")
    cam_p.add_argument("--mirror", type=_parse_bool, help="Mirror camera horizontally")
    cam_p.add_argument("--split", type=_parse_bool, help="Enable Before/After split preview")
    cam_p.add_argument("--on-demand", type=_parse_bool, help="Auto-wake physical webcam when apps open virtual camera")
    cam_p.add_argument("--idle-timeout", type=float, help="Idle seconds before releasing webcam after apps close it (0.5-60.0)")
    cam_p.add_argument("--blur", type=_parse_bool, help="Enable/disable background_blur")
    cam_p.add_argument("--blur-strength", type=float, help="Background blur strength 0.0-1.0")
    cam_p.add_argument("--blur-mode", choices=BLUR_MODES, help="Background blur mode")
    cam_p.add_argument("--replace", type=_parse_bool, help="Enable/disable background_replace")
    cam_p.add_argument("--replace-image", type=str, help="Absolute path to replacement background image")
    cam_p.add_argument("--replace-preset", choices=BG_REPLACE_PRESETS, help="Built-in background preset")
    cam_p.add_argument("--remove", type=_parse_bool, help="Enable/disable background_remove (chroma green)")
    cam_p.add_argument("--auto-frame", type=_parse_bool, help="Enable/disable auto_frame")
    cam_p.add_argument("--zoom", type=float, help="Auto-frame zoom 0.0-1.0")
    cam_p.add_argument("--eye-contact", type=_parse_bool, help="Enable/disable eye_contact gaze redirection")
    cam_p.add_argument("--eye-contact-strength", type=float, help="Eye contact strength 0.0-1.0")
    cam_p.add_argument("--look-away", type=_parse_bool, help="Disable eye contact when looking far away")
    cam_p.add_argument("--denoise", type=_parse_bool, help="Enable/disable video_noise_removal")
    cam_p.add_argument("--key-light", type=_parse_bool, help="Enable/disable virtual_key_light")
    cam_p.add_argument("--warmth", type=float, help="Virtual Key Light warmth 0.0 (cool) - 1.0 (warm)")
    cam_p.add_argument("--vignette", type=_parse_bool, help="Enable/disable vignette")
    cam_p.add_argument("--vignette-strength", type=float, help="Vignette strength 0.0-1.0")
    cam_p.add_argument("--face-tracking", type=_parse_bool, help="Track face for vignette")

    # Hardware V4L2 camera controls
    v4l_p = sub.add_parser("v4l2", help="Inspect or set physical webcam hardware controls")
    v4l_p.add_argument("--device", type=str, default="/dev/video0", help="V4L2 webcam path")
    v4l_p.add_argument("--set", nargs=2, metavar=("CONTROL", "VALUE"), action="append", help="Set control value")

    # Start / Stop runtime pipelines
    start_p = sub.add_parser("start", help="Start virtual microphone, speakers, and/or camera pipeline")
    start_p.add_argument("--mic", action="store_true", help="Start virtual microphone")
    start_p.add_argument("--speaker", action="store_true", help="Start virtual speakers")
    start_p.add_argument("--camera", action="store_true", help="Start virtual camera")
    start_p.add_argument("--all", action="store_true", help="Start mic, speakers, and camera")
    start_p.add_argument("--duration", type=float, default=0.0, help="Auto-stop after N seconds (0 = until Ctrl+C)")

    sub.add_parser("stop", help="Tear down any active nv-broadcast PipeWire virtual audio modules")

    # 5s Mic test
    test_p = sub.add_parser("test-mic", help="Record raw vs processed microphone test clip")
    test_p.add_argument("--seconds", type=float, default=5.0, help="Recording duration in seconds")
    test_p.add_argument("--play", choices=("none", "raw", "processed", "both"), default="none")

    # Offline media file processing
    fp_p = sub.add_parser("process-file", help="Enhance an audio or video file with Broadcast effects")
    fp_p.add_argument("input", type=str, help="Input audio/video file")
    fp_p.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    fp_p.add_argument("--no-audio", action="store_true", help="Skip audio processing")
    fp_p.add_argument("--no-video", action="store_true", help="Skip video processing")

    # PipeWire filter-chain config
    pw_p = sub.add_parser("install-filter", help="Write native PipeWire filter-chain config to ~/.config/pipewire/")
    pw_p.add_argument("--backend", choices=("auto", "webrtc_aec", "rnnoise_ladspa", "deepfilternet"), default="auto")
    pw_p.add_argument("--dry-run", action="store_true", help="Print generated config without writing")

    dep_p = sub.add_parser("install-deps", help="Show Linux dependency setup instructions")
    dep_p.add_argument("--print-only", action="store_true", default=True, help="Print exact commands")

    # GUI
    gui_p = sub.add_parser("gui", help="Launch the NVIDIA Broadcast 2.3.0 desktop interface")
    gui_p.add_argument("--tab", choices=("mic", "speakers", "camera", "files", "settings"), default="mic", help="Initial tab to open")
    gui_p.add_argument("--gtk", action="store_true", help="Force legacy GTK4 fallback UI instead of Electron")

    # Local API server (used by Electron UI & persistent background daemon)
    srv_p = sub.add_parser("serve", help="Run localhost HTTP/SSE/MJPEG backend server")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=0)
    srv_p.add_argument("--token", default=None)
    srv_p.add_argument("--autostart", action="store_true", help="Auto-start virtual mic, speakers, and camera standby")
    srv_p.add_argument("--daemon-file", action="store_true", help="Publish connection info to ~/.config/nv-broadcast/daemon.json")

    return p


def _launch_electron_ui(initial_tab: str, config_path: str | None) -> int:
    electron_bin = find_electron_binary()
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if not electron_bin or not (ui_dir / "main.js").is_file():
        return -1

    env = os.environ.copy()
    env["NVB_PYTHON_BIN"] = sys.executable
    env["NVB_INITIAL_TAB"] = initial_tab
    if config_path:
        env["NVB_CONFIG_PATH"] = config_path

    # Allow clean launch under Wayland/X11
    args = [electron_bin, str(ui_dir), f"--tab={initial_tab}"]
    proc = subprocess.run(args, env=env, check=False)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = BroadcastConfig(args.config)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "doctor":
        report = run_doctor()
        return _emit(report, args.json, format_doctor_text(report))

    if args.command == "download-models":
        res = ensure_models()
        payload = {
            "ok": all(v[0] for v in res.values()),
            "results": {k: {"ok": v[0], "message": v[1]} for k, v in res.items()},
            "models": get_models_status(),
        }
        lines = [f"  [{'OK' if v[0] else 'FAIL'}] {k}: {v[1]}" for k, v in res.items()]
        return _emit(payload, args.json, "\n".join(lines))

    if args.command == "status":
        pw = PipeWireVirtualManager()
        payload = {
            "version": __version__,
            "config_path": str(cfg.path),
            "virtual_nodes": pw.find_existing_virtual_nodes(),
            "loopback_cameras": [c.to_dict() for c in find_loopback_cameras()],
            "broadcast_state": cfg.get_broadcast_state(),
            "config": cfg.data,
        }
        if args.json:
            return _emit(payload, True)
        vn = payload["virtual_nodes"]
        lines = [
            f"NVIDIA Broadcast (Linux) v{__version__}",
            f"Config: {cfg.path}",
            f"Virtual Mic active:      {'yes' if vn['virtual_mic'] else 'no'}",
            f"Virtual Speakers active: {'yes' if vn['virtual_speakers'] else 'no'}",
            f"Virtual Camera devices:  {len(payload['loopback_cameras'])}",
            "",
            "Effects:",
        ]
        for sec, effs in payload["broadcast_state"].items():
            active = [k for k, v in effs.items() if v.get("enabled")]
            lines.append(f"  {sec:10s}: {', '.join(active) if active else 'none'}")
        return _emit(payload, False, "\n".join(lines))

    if args.command == "devices":
        audio = list_audio_devices(include_monitors=args.all, include_virtual=args.all)
        cams = list_cameras(include_metadata=args.all, include_loopback=args.all)
        payload = {
            "microphones": [d.to_dict() for d in audio["microphones"]],
            "speakers": [d.to_dict() for d in audio["speakers"]],
            "cameras": [c.to_dict() for c in cams],
        }
        if args.json:
            return _emit(payload, True)
        lines = ["Microphones:"]
        for m in audio["microphones"]:
            mark = " *" if m.is_default else ""
            lines.append(f"  - {m.description}{mark}  [{m.name}]")
        lines.append("\nSpeakers:")
        for s in audio["speakers"]:
            mark = " *" if s.is_default else ""
            lines.append(f"  - {s.description}{mark}  [{s.name}]")
        lines.append("\nCameras:")
        for c in cams:
            fmt_items = c.formats.values() if isinstance(c.formats, dict) else c.formats
            fmts = ", ".join(
                f"{f.pixel_format}({len(f.resolutions)} modes)" if hasattr(f, "pixel_format") else str(f)
                for f in fmt_items
            )
            lines.append(f"  - {c.path}: {c.card}  [{fmts or c.driver}]")
        return _emit(payload, False, "\n".join(lines))

    if args.command == "effects":
        payload = {"effects": EFFECT_CATALOG, "incompatible_groups": INCOMPATIBLE_GROUPS}
        if args.json:
            return _emit(payload, True)
        lines = ["Supported Effects (NVIDIA Broadcast v2.3.0 parity):"]
        for eid, meta in EFFECT_CATALOG.items():
            params = ", ".join(meta["params"].keys())
            lines.append(f"  [{meta['section']:10s}] {eid:22s} ({params}) - {meta['description']}")
        return _emit(payload, False, "\n".join(lines))

    if args.command == "mic":
        mic = cfg.data["microphone"]
        conflicts: dict[str, list[str]] = {}
        if args.device is not None:
            mic["device"] = args.device
        if args.backend is not None:
            cfg.data["audio_backend"] = args.backend
        if args.highpass is not None:
            mic["highpass_enabled"] = args.highpass
        if args.highpass_hz is not None:
            mic["highpass_hz"] = args.highpass_hz
        if args.noise_gate is not None:
            mic["noise_gate_enabled"] = args.noise_gate
        if args.auto_gain is not None:
            mic["auto_gain_enabled"] = args.auto_gain
        if args.noise_removal is not None or args.noise_strength is not None:
            p_dict = {"strength": args.noise_strength} if args.noise_strength is not None else None
            r = cfg.set_effect("noise_removal", enabled=args.noise_removal, params=p_dict)
            conflicts.update(r.disabled_conflicts)
        if args.echo_removal is not None or args.echo_strength is not None:
            p_dict = {"strength": args.echo_strength} if args.echo_strength is not None else None
            r = cfg.set_effect("room_echo_removal", enabled=args.echo_removal, params=p_dict)
            conflicts.update(r.disabled_conflicts)
        if args.studio_voice is not None or args.mic_profile is not None:
            p_dict = {"micProfile": args.mic_profile} if args.mic_profile is not None else None
            r = cfg.set_effect("studio_voice", enabled=args.studio_voice, params=p_dict)
            conflicts.update(r.disabled_conflicts)
        cfg.save()
        return _emit({"microphone": cfg.data["microphone"], "disabled_conflicts": conflicts}, args.json)

    if args.command == "speaker":
        spk = cfg.data["speaker"]
        if args.device is not None:
            spk["device"] = args.device
        if args.noise_removal is not None or args.noise_strength is not None:
            p_dict = {"strength": args.noise_strength} if args.noise_strength is not None else None
            cfg.set_effect("speaker_noise_removal", enabled=args.noise_removal, params=p_dict)
        if args.echo_removal is not None or args.echo_strength is not None:
            p_dict = {"strength": args.echo_strength} if args.echo_strength is not None else None
            cfg.set_effect("speaker_echo_removal", enabled=args.echo_removal, params=p_dict)
        cfg.save()
        return _emit({"speaker": cfg.data["speaker"]}, args.json)

    if args.command == "camera":
        cam = cfg.data["camera"]
        conflicts = {}
        for attr, key in (
            ("device", "device"),
            ("output_device", "output_device"),
            ("width", "width"),
            ("height", "height"),
            ("fps", "fps"),
            ("mirror", "mirror"),
            ("split", "split_preview"),
            ("on_demand", "on_demand"),
            ("idle_timeout", "idle_timeout_sec"),
        ):
            val = getattr(args, attr)
            if val is not None:
                cam[key] = val

        if args.blur is not None or args.blur_strength is not None or args.blur_mode is not None:
            p_dict = {}
            if args.blur_strength is not None:
                p_dict["strength"] = args.blur_strength
            if args.blur_mode is not None:
                p_dict["mode"] = args.blur_mode
            r = cfg.set_effect("background_blur", enabled=args.blur, params=p_dict or None)
            conflicts.update(r.disabled_conflicts)
        if args.replace is not None or args.replace_image is not None or args.replace_preset is not None:
            p_dict = {}
            if args.replace_image is not None:
                p_dict["imagePath"] = args.replace_image
            if args.replace_preset is not None:
                p_dict["preset"] = args.replace_preset
            r = cfg.set_effect("background_replace", enabled=args.replace, params=p_dict or None)
            conflicts.update(r.disabled_conflicts)
        if args.remove is not None:
            r = cfg.set_effect("background_remove", enabled=args.remove)
            conflicts.update(r.disabled_conflicts)
        if args.auto_frame is not None or args.zoom is not None:
            p_dict = {"zoom": args.zoom} if args.zoom is not None else None
            cfg.set_effect("auto_frame", enabled=args.auto_frame, params=p_dict)
        if args.eye_contact is not None or args.eye_contact_strength is not None or args.look_away is not None:
            p_dict = {}
            if args.eye_contact_strength is not None:
                p_dict["strength"] = args.eye_contact_strength
            if args.look_away is not None:
                p_dict["lookAway"] = args.look_away
            cfg.set_effect("eye_contact", enabled=args.eye_contact, params=p_dict or None)
        if args.denoise is not None:
            cfg.set_effect("video_noise_removal", enabled=args.denoise)
        if args.key_light is not None or args.warmth is not None:
            p_dict = {"warmth": args.warmth} if args.warmth is not None else None
            cfg.set_effect("virtual_key_light", enabled=args.key_light, params=p_dict)
        if args.vignette is not None or args.vignette_strength is not None or args.face_tracking is not None:
            p_dict = {}
            if args.vignette_strength is not None:
                p_dict["strength"] = args.vignette_strength
            if args.face_tracking is not None:
                p_dict["faceTracking"] = args.face_tracking
            cfg.set_effect("vignette", enabled=args.vignette, params=p_dict or None)

        cfg.save()
        return _emit({"camera": cfg.data["camera"], "disabled_conflicts": conflicts}, args.json)

    if args.command == "v4l2":
        results = []
        if args.set:
            for ctrl_name, raw_val in args.set:
                ok, msg = set_v4l2_control(args.device, ctrl_name, int(raw_val))
                results.append({"control": ctrl_name, "value": int(raw_val), "ok": ok, "message": msg})
                if ok:
                    cfg.data["camera"].setdefault("v4l2_controls", {})[ctrl_name] = int(raw_val)
            cfg.save()
        ctrls = list_v4l2_controls(args.device)
        payload = {"device": args.device, "changes": results, "controls": [c.__dict__ for c in ctrls]}
        if args.json:
            return _emit(payload, True)
        lines = [f"V4L2 Hardware Controls for {args.device}:"]
        for c in ctrls:
            lines.append(f"  {c.name:28s} = {c.value:5d}  (min={c.minimum}, max={c.maximum}, default={c.default})")
        return _emit(payload, False, "\n".join(lines))

    if args.command == "process-file":
        fp = FileProcessor(cfg)
        state = fp.process_sync(
            args.input,
            args.output,
            apply_audio=not args.no_audio,
            apply_video=not args.no_video,
        )
        return _emit(state.to_dict(), args.json, f"[{state.status}] {state.message} -> {state.output_path}")

    if args.command == "install-filter":
        caps = probe_capabilities()
        mic = cfg.data["microphone"]
        target = mic.get("device", "auto")
        backend = args.backend
        if backend == "auto":
            if caps.has_deepfilter_ladspa:
                backend = "deepfilternet"
            elif caps.has_rnnoise_ladspa:
                backend = "rnnoise_ladspa"
            else:
                backend = "webrtc_aec"

        if backend == "deepfilternet":
            conf = generate_deepfilter_filter_chain_conf(
                ladspa_plugin=caps.deepfilter_ladspa_path or "libdeep_filter_ladspa",
                target_source=target,
            )
        elif backend == "rnnoise_ladspa":
            conf = generate_rnnoise_filter_chain_conf(
                ladspa_plugin=caps.rnnoise_ladspa_path or "librnnoise_ladspa",
                target_source=target,
            )
        else:
            conf = generate_webrtc_echo_cancel_conf(
                noise_suppression=bool(mic["effects"]["noise_removal"].get("enabled", True)),
                echo_cancellation=bool(mic["effects"]["room_echo_removal"].get("enabled", False)),
                high_pass=bool(mic.get("highpass_enabled", True)),
                auto_gain=bool(mic.get("auto_gain_enabled", True)),
                target_source=target,
            )

        if args.dry_run:
            return _emit({"backend": backend, "config": conf}, args.json, conf)
        out_path = write_persistent_filter_conf(conf)
        return _emit(
            {"backend": backend, "written_path": str(out_path)},
            args.json,
            f"Wrote PipeWire {backend} config to {out_path}\nRestart PipeWire to activate: systemctl --user restart pipewire",
        )

    if args.command == "stop":
        pw = PipeWireVirtualManager()
        pw.teardown_all()
        return _emit({"stopped": True, "virtual_nodes": pw.find_existing_virtual_nodes()}, args.json, "Stopped virtual audio devices.")

    if args.command == "start":
        want_mic = args.all or args.mic or (not args.speaker and not args.camera)
        want_spk = args.all or args.speaker
        want_cam = args.all or args.camera

        audio = AudioEngine(cfg)
        video = VideoEngine(cfg)
        started: list[str] = []

        if want_mic:
            ok, msg = audio.start_microphone()
            started.append(f"mic: {msg}" if ok else f"mic FAILED: {msg}")
        if want_spk:
            ok, msg = audio.start_speakers()
            started.append(f"speaker: {msg}" if ok else f"speaker FAILED: {msg}")
        if want_cam:
            ok, msg = video.start()
            started.append(f"camera: {msg}" if ok else f"camera FAILED: {msg}")

        print("\n".join(started), flush=True)
        stop_flag = False

        def _handle_sig(_sig: int, _frame: Any) -> None:
            nonlocal stop_flag
            stop_flag = True

        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)

        t_end = time.monotonic() + args.duration if args.duration > 0 else None
        try:
            while not stop_flag:
                if t_end is not None and time.monotonic() >= t_end:
                    break
                time.sleep(0.2)
        finally:
            audio.stop_all()
            video.stop()
        return 0

    if args.command == "test-mic":
        audio = AudioEngine(cfg)
        res = audio.record_mic_test(duration_sec=args.seconds)
        if args.play in ("raw", "both"):
            audio.play_wav(res.raw_wav_path)
            if args.play == "both":
                time.sleep(args.seconds + 0.4)
        if args.play in ("processed", "both"):
            audio.play_wav(res.processed_wav_path)
        payload = {
            "duration_sec": res.duration_sec,
            "raw_wav": res.raw_wav_path,
            "processed_wav": res.processed_wav_path,
            "raw_rms_db": res.raw_rms_db,
            "processed_rms_db": res.processed_rms_db,
            "noise_floor_reduction_db": res.noise_floor_reduction_db,
        }
        txt = (
            f"Recorded {res.duration_sec}s mic test:\n"
            f"  Raw:       {res.raw_wav_path} ({res.raw_rms_db} dBFS)\n"
            f"  Processed: {res.processed_wav_path} ({res.processed_rms_db} dBFS, -{res.noise_floor_reduction_db} dB noise)\n"
            f"Play back with: pw-play {res.raw_wav_path} && pw-play {res.processed_wav_path}"
        )
        return _emit(payload, args.json, txt)

    if args.command == "install-deps":
        script_path = Path(__file__).resolve().parent.parent / "install-deps.sh"
        info = {
            "script": str(script_path),
            "fedora_packages": "sudo dnf install akmod-v4l2loopback v4l2loopback v4l-utils ladspa-noise-suppression-for-voice",
            "load_virtual_camera": 'sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast Camera" exclusive_caps=1',
            "python_gpu_venv": f"bash {script_path} --venv",
        }
        txt = (
            "Optional Linux Dependencies for Virtual Webcam:\n"
            f"  1. Virtual camera + RNNoise LADSPA:\n     {info['fedora_packages']}\n"
            f"  2. Load virtual camera device:\n     {info['load_virtual_camera']}\n"
            f"  3. Python 3.12 GPU AI venv:\n     {info['python_gpu_venv']}"
        )
        return _emit(info, args.json, txt)

    if args.command == "serve":
        from .server import run_server

        return run_server(
            host=args.host,
            port=args.port,
            config_path=args.config,
            token=args.token,
            autostart=args.autostart,
            write_daemon_file=args.daemon_file,
        )

    if args.command == "gui":
        if not args.gtk:
            rc = _launch_electron_ui(args.tab, args.config)
            if rc != -1:
                return rc
        from .gui import launch_gui

        return launch_gui(cfg, initial_tab=args.tab)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
