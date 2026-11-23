"""Comprehensive tests for nv-broadcast (config, PipeWire, DSP, video, V4L2, CLI)."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

NV_BCAST_ROOT = Path(__file__).resolve().parent.parent
if str(NV_BCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(NV_BCAST_ROOT))

from nv_broadcast.audio_dsp import (
    AudioProcessor,
    _floats_to_pcm16,
    _pcm16_to_floats,
    build_ffmpeg_audio_filter,
    compute_rms_db,
)
from nv_broadcast.cli import main as cli_main
from nv_broadcast.config import BroadcastConfig, default_config
from nv_broadcast.devices import parse_pactl_short
from nv_broadcast.pipewire import (
    generate_deepfilter_filter_chain_conf,
    generate_rnnoise_filter_chain_conf,
    generate_webrtc_echo_cancel_conf,
)
from nv_broadcast.v4l2_ctl import V4L2DeviceInfo, V4L2Mode, choose_best_pixel_format
from nv_broadcast.video_effects import VideoEffectsPipeline


class TestConfigAndSchema(unittest.TestCase):
    def test_persistence_and_clamping(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg = BroadcastConfig(cfg_path)
            res = cfg.set_effect("noise_removal", enabled=True, params={"strength": 1.8})
            self.assertEqual(res.applied, ["noise_removal"])
            self.assertEqual(cfg.get_effect("noise_removal")["strength"], 1.0)
            cfg.save()

            reloaded = BroadcastConfig(cfg_path)
            self.assertTrue(reloaded.get_effect("noise_removal")["enabled"])
            self.assertEqual(reloaded.get_effect("noise_removal")["strength"], 1.0)

    def test_camera_background_mutual_exclusion(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = BroadcastConfig(Path(tmp) / "config.json")
            cfg.set_effect("background_blur", enabled=True, params={"strength": 0.6})
            self.assertTrue(cfg.get_effect("background_blur")["enabled"])

            res = cfg.set_effect("background_replace", enabled=True, params={"preset": "office_blur"})
            self.assertIn("background_blur", res.disabled_conflicts.get("background_replace", []))
            self.assertFalse(cfg.get_effect("background_blur")["enabled"])
            self.assertTrue(cfg.get_effect("background_replace")["enabled"])

            res2 = cfg.set_effect("background_remove", enabled=True)
            self.assertIn("background_replace", res2.disabled_conflicts.get("background_remove", []))
            self.assertFalse(cfg.get_effect("background_replace")["enabled"])
            self.assertTrue(cfg.get_effect("background_remove")["enabled"])

    def test_studio_voice_mutual_exclusion(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = BroadcastConfig(Path(tmp) / "config.json")
            cfg.set_effect("noise_removal", enabled=True)
            cfg.set_effect("room_echo_removal", enabled=True)
            res = cfg.set_effect("studio_voice", enabled=True, params={"micProfile": "warm"})
            conflicts = set(res.disabled_conflicts.get("studio_voice", []))
            self.assertEqual(conflicts, {"noise_removal", "room_echo_removal"})
            self.assertFalse(cfg.get_effect("noise_removal")["enabled"])
            self.assertFalse(cfg.get_effect("room_echo_removal")["enabled"])
            self.assertEqual(cfg.get_effect("studio_voice")["micProfile"], "warm")


class TestDeviceAndPipeWireParsing(unittest.TestCase):
    def test_parse_pactl_short_sources(self) -> None:
        sample = (
            "58\talsa_output.pci-0000_00_1f.3.HiFi__Speaker__sink.monitor\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n"
            "59\talsa_input.pci-0000_00_1f.3.HiFi__Mic1__source\tPipeWire\ts32le 2ch 48000Hz\tRUNNING\n"
            "99\tnv_broadcast_mic\tPipeWire\ts16le 1ch 48000Hz\tRUNNING\n"
        )
        devs = parse_pactl_short(sample, "microphone", default_name="alsa_input.pci-0000_00_1f.3.HiFi__Mic1__source")
        self.assertEqual(len(devs), 3)
        self.assertTrue(devs[0].is_monitor)
        self.assertTrue(devs[1].is_default)
        self.assertFalse(devs[1].is_monitor)
        self.assertTrue(devs[2].is_virtual)

    def test_pipewire_filter_chain_configs(self) -> None:
        rn = generate_rnnoise_filter_chain_conf(vad_threshold=65.0, target_source="alsa_input.mic1")
        self.assertIn("libpipewire-module-filter-chain", rn)
        self.assertIn("librnnoise_ladspa", rn)
        self.assertIn('"VAD Threshold (%)" 65.0', rn)
        self.assertIn('target.object = "alsa_input.mic1"', rn)

        df = generate_deepfilter_filter_chain_conf(attenuation_limit_db=85.0)
        self.assertIn("libdeep_filter_ladspa", df)
        self.assertIn('"Attenuation Limit (dB)" 85.0', df)

        aec = generate_webrtc_echo_cancel_conf(noise_suppression=True, echo_cancellation=True, high_pass=True)
        self.assertIn("libpipewire-module-echo-cancel", aec)
        self.assertIn("aec/libspa-aec-webrtc", aec)
        self.assertIn("webrtc.noise_suppression     = true", aec)

    def test_v4l2_prefers_mjpg_30fps_over_yuyv_10fps(self) -> None:
        cam = V4L2DeviceInfo(
            path="/dev/video0",
            driver="uvcvideo",
            card="HD User Facing Webcam",
            bus_info="usb-0000:00:14.0-6",
            is_capture=True,
            is_output=False,
            is_metadata=False,
            is_loopback=False,
            formats=["MJPG", "YUYV"],
            modes=[
                V4L2Mode("MJPG", 1280, 720, [30.0]),
                V4L2Mode("YUYV", 1280, 720, [10.0]),
            ],
        )
        fmt, fps = choose_best_pixel_format(cam, 1280, 720, prefer_mjpg=True)
        self.assertEqual(fmt, "MJPG")
        self.assertEqual(fps, 30.0)


class TestAudioDSP(unittest.TestCase):
    def test_noise_removal_suppresses_stationary_fan_noise(self) -> None:
        sr = 48000
        rng = random.Random(42)
        # Generate 0.4s of stationary laptop fan hum (60Hz + broadband hiss)
        noisy = [
            0.08 * math.sin(2.0 * math.pi * 60.0 * i / sr) + rng.uniform(-0.04, 0.04)
            for i in range(int(sr * 0.4))
        ]
        cfg = default_config()["microphone"]
        cfg["highpass_enabled"] = True
        cfg["highpass_hz"] = 100
        cfg["noise_gate_enabled"] = False
        cfg["auto_gain_enabled"] = False
        cfg["effects"]["noise_removal"] = {"enabled": True, "strength": 0.9}

        proc = AudioProcessor(sample_rate=sr, frame_size=512)
        out: list[float] = []
        for off in range(0, len(noisy), 512):
            out.extend(proc.process_samples(noisy[off : off + 512], cfg))

        # Measure after initial noise floor adaptation window
        in_db, _ = compute_rms_db(noisy[4096:])
        out_db, _ = compute_rms_db(out[4096:])
        self.assertLess(out_db, in_db - 8.0, f"Expected >=8dB suppression, got in={in_db:.1f} out={out_db:.1f}")

    def test_speech_preserved_over_noise_and_pcm16_roundtrip(self) -> None:
        sr = 48000
        rng = random.Random(7)
        # 0.2s noise calibration followed by 0.2s 440Hz speech-like tone + low noise
        noise_only = [rng.uniform(-0.02, 0.02) for _ in range(9600)]
        speech = [
            0.35 * math.sin(2.0 * math.pi * 440.0 * i / sr) + rng.uniform(-0.02, 0.02)
            for i in range(9600)
        ]
        cfg = default_config()["microphone"]
        cfg["noise_gate_enabled"] = False
        cfg["auto_gain_enabled"] = False
        cfg["effects"]["noise_removal"] = {"enabled": True, "strength": 0.85}

        proc = AudioProcessor(sample_rate=sr, frame_size=512)
        raw_pcm = _floats_to_pcm16(noise_only + speech)
        out_chunks = [proc.process_pcm16(raw_pcm[i : i + 1024], cfg) for i in range(0, len(raw_pcm), 1024)]
        out_floats = _pcm16_to_floats(b"".join(out_chunks))

        noise_out_db, _ = compute_rms_db(out_floats[3000:9000])
        speech_out_db, _ = compute_rms_db(out_floats[11000:18000])
        self.assertGreater(speech_out_db, noise_out_db + 12.0)

    def test_studio_voice_and_ffmpeg_filter_graph(self) -> None:
        cfg = default_config()["microphone"]
        cfg["effects"]["noise_removal"]["enabled"] = True
        cfg["effects"]["studio_voice"] = {"enabled": True, "micProfile": "warm"}
        af = build_ffmpeg_audio_filter(cfg)
        self.assertIn("highpass=f=85", af)
        self.assertIn("afftdn=", af)
        self.assertIn("acompressor=", af)


class TestVideoEffectsPipeline(unittest.TestCase):
    def _make_synthetic_portrait(self, w: int = 320, h: int = 180, face_cx: float = 0.5) -> Image.Image:
        img = Image.new("RGB", (w, h), (25, 40, 80))
        draw = ImageDraw.Draw(img)
        # Checkerboard high-frequency background so blur/replace/remove changes are measurable
        for y in range(0, h, 16):
            for x in range(0, w, 16):
                if ((x // 16) + (y // 16)) % 2 == 0:
                    draw.rectangle([x, y, x + 15, y + 15], fill=(20, 30, 50))
                else:
                    draw.rectangle([x, y, x + 15, y + 15], fill=(60, 80, 120))
        # Synthetic skin-tone face oval + torso
        fx = int(w * face_cx)
        fy = int(h * 0.42)
        draw.ellipse([fx - 26, fy - 32, fx + 26, fy + 32], fill=(215, 162, 130))
        draw.ellipse([fx - 60, fy + 26, fx + 60, h + 20], fill=(40, 45, 55))
        return img

    def test_background_blur_replace_remove(self) -> None:
        pipe = VideoEffectsPipeline(segmentation_backend="fast_portrait")
        src = self._make_synthetic_portrait()

        # 1. Blur
        cam_cfg = default_config()["camera"]
        cam_cfg["effects"]["background_blur"] = {"enabled": True, "strength": 0.9, "mode": "quality"}
        blurred = pipe.process_frame(src, cam_cfg)
        self.assertEqual(blurred.size, src.size)
        self.assertNotEqual(blurred.tobytes(), src.tobytes())

        # 2. Replace with preset
        cam_cfg = default_config()["camera"]
        cam_cfg["effects"]["background_replace"] = {"enabled": True, "imagePath": "", "preset": "soft_daylight"}
        replaced = pipe.process_frame(src, cam_cfg)
        # Top-left background corner should become bright daylight
        corner_r, corner_g, corner_b = replaced.getpixel((5, 5))
        self.assertGreater(corner_r + corner_g + corner_b, 450)

        # 3. Remove (chroma green)
        cam_cfg = default_config()["camera"]
        cam_cfg["effects"]["background_remove"] = {"enabled": True, "chromaColor": "#00ff00"}
        removed = pipe.process_frame(src, cam_cfg)
        cr, cg, cb = removed.getpixel((5, 5))
        self.assertGreater(cg, 200)
        self.assertLess(cr, 40)

    def test_auto_frame_denoise_keylight_vignette(self) -> None:
        pipe = VideoEffectsPipeline(segmentation_backend="fast_portrait")
        src = self._make_synthetic_portrait(face_cx=0.7)

        cam_cfg = default_config()["camera"]
        cam_cfg["effects"]["auto_frame"] = {"enabled": True, "zoom": 0.5}
        cam_cfg["effects"]["video_noise_removal"] = {"enabled": True, "strength": 0.7}
        cam_cfg["effects"]["virtual_key_light"] = {"enabled": True, "warmth": 0.85, "intensity": 0.6}
        cam_cfg["effects"]["vignette"] = {"enabled": True, "strength": 0.6, "faceTracking": True}

        # Run two frames so temporal denoise and tracker damping both execute
        out1 = pipe.process_frame(src, cam_cfg)
        out2 = pipe.process_frame(src, cam_cfg)
        self.assertEqual(out1.size, src.size)
        self.assertEqual(out2.size, src.size)
        # Verify smooth damped face tracking toward a right-shifted face box
        from nv_broadcast.video_effects import FaceBox

        cx1, _ = pipe.tracker.update(FaceBox(cx=0.75, cy=0.45, w=0.22, h=0.28))
        cx2, _ = pipe.tracker.update(FaceBox(cx=0.75, cy=0.45, w=0.22, h=0.28))
        self.assertGreater(cx1, 0.5)
        self.assertGreater(cx2, cx1)

    def test_split_preview_and_eye_contact_and_mode_switch(self) -> None:
        from nv_broadcast.video_effects import render_split_frame

        pipe = VideoEffectsPipeline()
        src = self._make_synthetic_portrait()

        # Verify switching quality -> performance never crashes RVM recurrent shapes
        cam_cfg = default_config()["camera"]
        cam_cfg["effects"]["background_blur"] = {"enabled": True, "strength": 0.7, "mode": "quality"}
        cam_cfg["effects"]["eye_contact"] = {"enabled": True, "strength": 0.7, "lookAway": True}
        out_q = pipe.process_frame(src, cam_cfg)
        cam_cfg["effects"]["background_blur"]["mode"] = "performance"
        out_p = pipe.process_frame(src, cam_cfg)
        self.assertEqual(out_q.size, src.size)
        self.assertEqual(out_p.size, src.size)

        # Verify split preview draws the NVIDIA green divider (#76b900)
        split = render_split_frame(src, out_p, split_pos=0.5)
        r, g, b = split.getpixel((src.size[0] // 2, src.size[1] // 2))
        self.assertEqual((r, g, b), (118, 185, 0))


class TestAudioEnhancementsAndServer(unittest.TestCase):
    def test_ambient_noise_suppressed_without_agc_boost(self) -> None:
        import random
        from nv_broadcast.audio_dsp import compute_waveform_peaks, generate_sample_clip

        rng = random.Random(123)
        # Low-level ambient noise (~ -38 dBFS) must be attenuated, never boosted by AGC
        noise = [rng.uniform(-0.02, 0.02) for _ in range(48000)]
        cfg = default_config()["microphone"]
        proc = AudioProcessor(sample_rate=48000, frame_size=512)
        out = []
        for off in range(0, len(noise), 480):
            out.extend(proc.process_samples(noise[off : off + 480], cfg))
        in_db, _ = compute_rms_db(noise[9600:])
        out_db, _ = compute_rms_db(out[9600:])
        # Must attenuate ambient noise cleanly (never amplify like an ungated AGC)
        self.assertLess(out_db, in_db - 4.0)

        sample = generate_sample_clip("fan", sample_rate=48000, duration_sec=0.5)
        peaks = compute_waveform_peaks(sample, buckets=60)
        self.assertEqual(len(peaks), 60)
        self.assertGreater(max(peaks), 0.1)

    def test_file_processor_audio_and_api_server(self) -> None:
        import os
        import urllib.request
        from nv_broadcast.audio_dsp import generate_sample_clip, save_pcm16_wav
        from nv_broadcast.file_processor import FileProcessor

        with TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg = BroadcastConfig(cfg_path)
            cfg.set_effect("background_blur", enabled=True)
            cfg.save()

            # Offline WAV processing
            in_wav = Path(tmp) / "noisy.wav"
            out_wav = Path(tmp) / "clean.wav"
            save_pcm16_wav(in_wav, generate_sample_clip("fan", 48000, 0.5), 48000)
            fp = FileProcessor(cfg)
            st = fp.process_sync(str(in_wav), str(out_wav), apply_audio=True, apply_video=False)
            self.assertEqual(st.status, "completed")
            self.assertTrue(out_wav.is_file())

            # Local HTTP API server handshake + conflict resolution
            srv = subprocess.Popen(
                [sys.executable, "-m", "nv_broadcast.server", "--host", "127.0.0.1", "--port", "0", "--config", str(cfg_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONPATH": str(NV_BCAST_ROOT)},
            )
            try:
                assert srv.stdout is not None
                ready = srv.stdout.readline().strip()
                self.assertTrue(ready.startswith("READY http://127.0.0.1:"))
                _, base_url, token = ready.split()
                req = urllib.request.Request(f"{base_url}/api/state", headers={"X-NVB-Token": token})
                state = json.loads(urllib.request.urlopen(req, timeout=5).read())
                self.assertIn("config", state)
                post_req = urllib.request.Request(
                    f"{base_url}/api/effect",
                    data=json.dumps({"effect_id": "background_replace", "enabled": True}).encode(),
                    headers={"X-NVB-Token": token, "Content-Type": "application/json"},
                    method="POST",
                )
                eff_res = json.loads(urllib.request.urlopen(post_req, timeout=5).read())
                self.assertTrue(eff_res["ok"])
                self.assertIn("background_blur", eff_res["disabled_conflicts"].get("background_replace", []))
            finally:
                srv.terminate()
                srv.wait(timeout=3)
                if srv.stdout:
                    srv.stdout.close()
                if srv.stderr:
                    srv.stderr.close()


class TestCLICommands(unittest.TestCase):
    def test_cli_json_commands(self) -> None:
        launcher = NV_BCAST_ROOT / "nv-broadcast"
        with TemporaryDirectory() as tmp:
            cfg_path = str(Path(tmp) / "cfg.json")
            for args in (
                ["--config", cfg_path, "--json", "doctor"],
                ["--config", cfg_path, "--json", "status"],
                ["--config", cfg_path, "--json", "devices"],
                ["--config", cfg_path, "--json", "effects"],
                ["--config", cfg_path, "--json", "mic", "--noise-removal", "on", "--noise-strength", "0.75"],
                ["--config", cfg_path, "--json", "camera", "--blur", "on", "--blur-strength", "0.6", "--key-light", "on", "--warmth", "0.7"],
                ["--config", cfg_path, "--json", "install-filter", "--dry-run"],
            ):
                proc = subprocess.run(
                    [sys.executable, "-m", "nv_broadcast.cli", *args],
                    cwd=str(NV_BCAST_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                parsed = json.loads(proc.stdout)
                self.assertIsInstance(parsed, dict)

            # Verify launcher script itself is executable and responds to --version
            ver_proc = subprocess.run([str(launcher), "--version"], capture_output=True, text=True, check=True)
            self.assertIn("nv-broadcast", ver_proc.stdout)


if __name__ == "__main__":
    unittest.main()
