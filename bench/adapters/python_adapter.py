"""Python Broadcast (`nv_broadcast`) target adapter.

Executes all 7 benchmark suites against the real Python implementation using
either the repository `.venv` (with NumPy, SciPy, pyrnnoise, OpenCV, MediaPipe,
ONNX Runtime CUDA) or an explicit `--python-bin` interpreter, always isolated
in temporary config directories and ephemeral localhost ports so any live
user `nv-broadcast.service` stays untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from bench.adapters.base import TargetAdapter
from bench.metrics.audio_metrics import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioScenario,
    evaluate_audio_clip,
    read_wav_mono16,
    write_wav_mono16,
)
from bench.metrics.system_metrics import (
    compute_percentiles,
    compute_process_delta,
    query_nvidia_gpu,
    query_pipewire_top,
    query_power_supply,
    sample_process_tree,
)


def detect_python_binary(repo_root: Path, explicit: str | None = None) -> str:
    """Resolve the best Python interpreter for `nv_broadcast`."""
    if explicit:
        return explicit
    candidates = [
        repo_root / ".venv" / "bin" / "python",
        repo_root.parent / "Broadcast" / ".venv" / "bin" / "python",
        Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        / "nv-broadcast"
        / "venv"
        / "bin"
        / "python",
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return sys.executable


class PythonAdapter(TargetAdapter):
    name = "python"
    display_name = "Python Broadcast (nv_broadcast)"

    def __init__(self, repo_root: Path, python_bin: str | None = None) -> None:
        self.repo_root = repo_root
        self.python_bin = detect_python_binary(repo_root, python_bin)
        self._env = dict(os.environ)
        existing_pp = self._env.get("PYTHONPATH", "")
        self._env["PYTHONPATH"] = f"{self.repo_root}:{existing_pp}" if existing_pp else str(self.repo_root)

    def _run_py_json(self, code: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
        proc = subprocess.run(
            [self.python_bin, "-c", code],
            input=json.dumps(payload or {}),
            capture_output=True,
            text=True,
            env=self._env,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Python helper exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("Python helper produced empty stdout")
        return json.loads(lines[-1])

    def capabilities(self) -> dict[str, Any]:
        probe_code = """
import importlib.util, json, sys
from nv_broadcast import __version__
from nv_broadcast.audio_dsp import AudioProcessor
ap = AudioProcessor()
mods = {m: bool(importlib.util.find_spec(m)) for m in ('numpy', 'scipy', 'pyrnnoise', 'cv2', 'mediapipe', 'onnxruntime')}
print(json.dumps({
    'target': 'python',
    'version': __version__,
    'python_bin': sys.executable,
    'audio_backend': ap.backend_name,
    'modules': mods,
}))
"""
        try:
            info = self._run_py_json(probe_code, timeout=15.0)
        except Exception as exc:
            info = {"target": "python", "error": str(exc), "python_bin": self.python_bin}
        info["display_name"] = self.display_name
        info["phase"] = "oracle"
        info["capabilities"] = {
            "audio_quality": True,
            "audio_latency": True,
            "memory": True,
            "video": True,
            "api_cli": True,
            "system_impact": True,
            "reliability": True,
        }
        return info

    def run_audio_quality(
        self,
        scenarios: list[AudioScenario],
        out_dir: Path,
        profile: str = "quick",
    ) -> dict[str, Any]:
        audio_dir = out_dir / "audio_ab"
        audio_dir.mkdir(parents=True, exist_ok=True)

        jobs = []
        with tempfile.TemporaryDirectory(prefix="nvb_bench_audio_") as tmpdir:
            tmp = Path(tmpdir)
            for sc in scenarios:
                in_wav = write_wav_mono16(tmp / f"{sc.name}_in.wav", sc.noisy_pcm, sc.sample_rate)
                out_wav = audio_dir / f"{sc.name}_python.wav"
                jobs.append(
                    {
                        "name": sc.name,
                        "in_wav": str(in_wav),
                        "out_wav": str(out_wav),
                        "sample_rate": sc.sample_rate,
                        "effect_cfg": sc.effect_cfg,
                    }
                )

            worker_code = """
import json, sys, time, wave
from pathlib import Path
from nv_broadcast.audio_dsp import AudioProcessor, save_pcm16_wav

req = json.loads(sys.stdin.read())
out = {'backend': 'dsp', 'jobs': {}}
for job in req['jobs']:
    with wave.open(job['in_wav'], 'rb') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    proc = AudioProcessor(sample_rate=sr)
    out['backend'] = proc.backend_name
    chunk_bytes = 480 * 2
    chunks_out = []
    t0 = time.perf_counter()
    for off in range(0, len(raw), chunk_bytes):
        chunks_out.append(proc.process_pcm16(raw[off:off + chunk_bytes], job['effect_cfg']))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    processed = b''.join(chunks_out)
    save_pcm16_wav(Path(job['out_wav']), processed, sample_rate=sr)
    out['jobs'][job['name']] = {
        'elapsed_ms': round(elapsed_ms, 3),
        'backend': proc.backend_name,
        'meter': proc.last_meter.__dict__,
    }
print(json.dumps(out))
"""
            worker_res = self._run_py_json(worker_code, {"jobs": jobs}, timeout=180.0)

        results: dict[str, Any] = {}
        delta_snrs: list[float] = []
        delta_sisdrs: list[float] = []
        silence_attens: list[float] = []

        for sc in scenarios:
            out_wav = audio_dir / f"{sc.name}_python.wav"
            proc_pcm, _ = read_wav_mono16(out_wav)
            metrics = evaluate_audio_clip(
                noisy_pcm=sc.noisy_pcm,
                processed_pcm=proc_pcm,
                clean_pcm=sc.clean_pcm,
                speech_mask=sc.speech_mask,
                sample_rate=sc.sample_rate,
            )
            jmeta = worker_res["jobs"][sc.name]
            metrics["elapsed_ms"] = jmeta["elapsed_ms"]
            metrics["meter"] = jmeta["meter"]
            metrics["wav_path"] = str(out_wav)
            metrics["description"] = sc.description
            results[sc.name] = metrics
            if "delta_snr_db" in metrics:
                delta_snrs.append(metrics["delta_snr_db"])
            if "delta_si_sdr_db" in metrics:
                delta_sisdrs.append(metrics["delta_si_sdr_db"])
            silence_attens.append(metrics["silence_attenuation_db"])

        return {
            "status": "ok",
            "backend": worker_res.get("backend", "dsp"),
            "scenarios": results,
            "summary": {
                "mean_delta_snr_db": round(sum(delta_snrs) / len(delta_snrs), 3) if delta_snrs else 0.0,
                "mean_delta_si_sdr_db": round(sum(delta_sisdrs) / len(delta_sisdrs), 3) if delta_sisdrs else 0.0,
                "mean_silence_attenuation_db": round(sum(silence_attens) / len(silence_attens), 3)
                if silence_attens
                else 0.0,
                "pure_noise_agc_safe": results.get("pure_noise", {}).get("agc_noise_safety_pass", True),
            },
        }

    def run_audio_latency(
        self,
        profile: str = "quick",
        frames: int | None = None,
    ) -> dict[str, Any]:
        n_frames = frames if frames is not None else (2000 if profile == "quick" else 10000)
        worker_code = """
import json, math, sys, time
from nv_broadcast.audio_dsp import (
    AudioProcessor,
    _floats_to_pcm16,
    _pcm16_to_floats,
    compute_rms_db,
    generate_sample_clip,
)

req = json.loads(sys.stdin.read())
n_frames = int(req['frames'])
clip = generate_sample_clip('fan_echo', sample_rate=48000, duration_sec=2.0)
frame_bytes = 480 * 2
ring = [clip[i:i + frame_bytes] for i in range(0, (len(clip) // frame_bytes) * frame_bytes, frame_bytes)]

def pct(vals, p):
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, int(round((len(vals) - 1) * p)))
    return vals[idx]

def summarize(vals):
    s = sorted(vals)
    return {
        'count': len(s),
        'min': round(s[0], 3),
        'mean': round(sum(s) / len(s), 3),
        'p50': round(pct(s, 0.50), 3),
        'p95': round(pct(s, 0.95), 3),
        'p99': round(pct(s, 0.99), 3),
        'p999': round(pct(s, 0.999), 3),
        'max': round(s[-1], 3),
    }

configs = {
    'noise_removal': {
        'highpass_enabled': True, 'highpass_hz': 85,
        'noise_gate_enabled': True, 'noise_gate_db': -46.0,
        'auto_gain_enabled': True, 'auto_gain_target_db': -16.0,
        'effects': {
            'noise_removal': {'enabled': True, 'strength': 0.85},
            'room_echo_removal': {'enabled': False, 'strength': 0.5},
            'studio_voice': {'enabled': False, 'micProfile': 'default'},
        },
    },
    'room_echo_removal': {
        'highpass_enabled': True, 'highpass_hz': 85,
        'noise_gate_enabled': True, 'noise_gate_db': -46.0,
        'auto_gain_enabled': True, 'auto_gain_target_db': -16.0,
        'effects': {
            'noise_removal': {'enabled': True, 'strength': 0.85},
            'room_echo_removal': {'enabled': True, 'strength': 0.6},
            'studio_voice': {'enabled': False, 'micProfile': 'default'},
        },
    },
    'studio_voice': {
        'highpass_enabled': True, 'highpass_hz': 85,
        'noise_gate_enabled': True, 'noise_gate_db': -46.0,
        'auto_gain_enabled': True, 'auto_gain_target_db': -16.0,
        'effects': {
            'noise_removal': {'enabled': False, 'strength': 0.85},
            'room_echo_removal': {'enabled': False, 'strength': 0.5},
            'studio_voice': {'enabled': True, 'micProfile': 'warm'},
        },
    },
}

workloads = {}
backend_name = 'dsp'
audio_sec = (n_frames * 480) / 48000.0
for name, cfg in configs.items():
    ap = AudioProcessor(sample_rate=48000)
    backend_name = ap.backend_name
    # Warm up 20 frames
    for i in range(20):
        ap.process_pcm16(ring[i % len(ring)], cfg)
    lats = []
    t_wall0 = time.perf_counter()
    for i in range(n_frames):
        chunk = ring[i % len(ring)]
        t0 = time.perf_counter()
        ap.process_pcm16(chunk, cfg)
        lats.append((time.perf_counter() - t0) * 1_000_000.0)
    wall_sec = time.perf_counter() - t_wall0
    workloads[name] = {
        'frames': n_frames,
        'audio_sec': round(audio_sec, 3),
        'wall_ms': round(wall_sec * 1000.0, 3),
        'rtf': round(wall_sec / max(audio_sec, 1e-9), 6),
        'single_core_cpu_pct': round((wall_sec / max(audio_sec, 1e-9)) * 100.0, 3),
        'budget_misses_2ms': sum(1 for v in lats if v > 2000.0),
        'budget_misses_10ms': sum(1 for v in lats if v > 10000.0),
        'latency_us': summarize(lats),
    }

# Also benchmark PCM16 <-> float + RMS primitive so Rust Phase R0 has an apples-to-apples comparison
sine_f = [0.5 * math.sin(2.0 * math.pi * 440.0 * i / 48000.0) for i in range(480)]
pcm_in = _floats_to_pcm16(sine_f)
pcm_lats = []
t_p0 = time.perf_counter()
for _ in range(n_frames):
    t0 = time.perf_counter()
    fl = _pcm16_to_floats(pcm_in)
    compute_rms_db(fl)
    _floats_to_pcm16(fl)
    pcm_lats.append((time.perf_counter() - t0) * 1_000_000.0)
pcm_wall = time.perf_counter() - t_p0
pcm_prim = {
    'status': 'ok',
    'frames': n_frames,
    'audio_sec': round(audio_sec, 3),
    'wall_ms': round(pcm_wall * 1000.0, 3),
    'rtf': round(pcm_wall / max(audio_sec, 1e-9), 6),
    'latency_us': summarize(pcm_lats),
}

print(json.dumps({'status': 'ok', 'backend': backend_name, 'workloads': workloads, 'pcm_primitive': pcm_prim}))
"""
        res = self._run_py_json(worker_code, {"frames": n_frames}, timeout=180.0)
        if profile == "full":
            res["pipewire_top"] = query_pipewire_top()
        return res

    def _spawn_temp_daemon(self, tmpdir: Path) -> tuple[subprocess.Popen[str], str, str, float]:
        cfg_path = tmpdir / "config.json"
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [
                self.python_bin,
                "-m",
                "nv_broadcast.server",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--config",
                str(cfg_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env,
        )
        base_url = ""
        token = ""
        assert proc.stdout is not None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    err = proc.stderr.read() if proc.stderr else ""
                    raise RuntimeError(f"Python daemon exited early ({proc.returncode}): {err}")
                time.sleep(0.02)
                continue
            if line.startswith("READY "):
                parts = line.strip().split()
                if len(parts) >= 3:
                    base_url, token = parts[1], parts[2]
                    break
        if not base_url:
            proc.terminate()
            raise RuntimeError("Timed out waiting for Python daemon READY line")
        boot_ms = (time.perf_counter() - t0) * 1000.0
        return proc, base_url, token, boot_ms

    def _measure_worker_memory(self, script: str, payload: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
        """Run a short-lived Python worker that signals READY, sample `/proc` + `nvidia-smi`, then release it."""
        proc = subprocess.Popen(
            [self.python_bin, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env,
        )
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            snapshots: dict[str, Any] = {}
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                line = line.strip()
                if line.startswith("CHECKPOINT "):
                    _, cp_name, rest = (line.split(" ", 2) + [""])[:3]
                    meta = json.loads(rest) if rest else {}
                    tree = sample_process_tree(proc.pid)
                    gpu = query_nvidia_gpu(filter_pids=tree["pids"])
                    snapshots[cp_name] = {
                        "status": "ok",
                        "process_count": tree["process_count"],
                        "rss_mb": tree["rss_mb"],
                        "pss_mb": tree["pss_mb"],
                        "private_dirty_mb": tree["private_dirty_mb"],
                        "threads": tree["threads"],
                        "open_fds": tree["open_fds"],
                        "process_vram_mib": gpu.get("process_vram_mib", 0.0),
                        "gpu_pstate": gpu.get("pstate", "unavailable"),
                        "meta": meta,
                    }
                    proc.stdin.write("CONTINUE\n")
                    proc.stdin.flush()
                elif line == "DONE":
                    break
            proc.wait(timeout=10)
            return snapshots
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def run_memory(
        self,
        profile: str = "quick",
        soak_sec: int = 0,
    ) -> dict[str, Any]:
        checkpoints: dict[str, Any] = {}

        # 1. Idle / Standby daemon
        with tempfile.TemporaryDirectory(prefix="nvb_mem_idle_") as tmpdir:
            proc, base_url, token, _ = self._spawn_temp_daemon(Path(tmpdir))
            try:
                req = urllib.request.Request(f"{base_url}/api/health", headers={"X-NVB-Token": token})
                with urllib.request.urlopen(req, timeout=5):
                    pass
                time.sleep(0.6)
                tree = sample_process_tree(proc.pid)
                gpu = query_nvidia_gpu(filter_pids=tree["pids"])
                checkpoints["idle_standby"] = {
                    "status": "ok",
                    "process_count": tree["process_count"],
                    "rss_mb": tree["rss_mb"],
                    "pss_mb": tree["pss_mb"],
                    "private_dirty_mb": tree["private_dirty_mb"],
                    "threads": tree["threads"],
                    "open_fds": tree["open_fds"],
                    "process_vram_mib": gpu.get("process_vram_mib", 0.0),
                    "gpu_pstate": gpu.get("pstate", "unavailable"),
                }

                # Optional soak leak check
                soak_report = None
                if soak_sec > 0:
                    samples = []
                    t_end = time.monotonic() + soak_sec
                    t_start = time.monotonic()
                    while time.monotonic() < t_end:
                        st = sample_process_tree(proc.pid)
                        samples.append(
                            {
                                "t_sec": round(time.monotonic() - t_start, 2),
                                "rss_mb": st["rss_mb"],
                                "pss_mb": st["pss_mb"],
                                "open_fds": st["open_fds"],
                            }
                        )
                        time.sleep(min(5.0, max(0.5, t_end - time.monotonic())))
                    if len(samples) >= 2:
                        dt_min = max(1e-6, (samples[-1]["t_sec"] - samples[0]["t_sec"]) / 60.0)
                        rss_slope = (samples[-1]["rss_mb"] - samples[0]["rss_mb"]) / dt_min
                        fd_drift = samples[-1]["open_fds"] - samples[0]["open_fds"]
                    else:
                        rss_slope = 0.0
                        fd_drift = 0
                    soak_report = {
                        "duration_sec": soak_sec,
                        "sample_count": len(samples),
                        "rss_slope_mb_per_min": round(rss_slope, 3),
                        "fd_drift": fd_drift,
                        "samples": samples,
                    }
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        # 2. Active Audio Call (Mic RNNoise + Speaker NR)
        audio_worker = """
import json, sys
from nv_broadcast.audio_dsp import AudioProcessor, generate_sample_clip
req = json.loads(sys.stdin.readline())
clip = generate_sample_clip('fan_echo', sample_rate=48000, duration_sec=2.0)
mic = AudioProcessor(sample_rate=48000)
spk = AudioProcessor(sample_rate=48000)
mic_cfg = {'highpass_enabled': True, 'auto_gain_enabled': True, 'effects': {'noise_removal': {'enabled': True, 'strength': 0.85}}}
spk_cfg = {'highpass_enabled': True, 'auto_gain_enabled': False, 'effects': {'speaker_noise_removal': {'enabled': True, 'strength': 0.85}}}
step = 480 * 2
for off in range(0, len(clip), step):
    mic.process_pcm16(clip[off:off+step], mic_cfg)
    spk.process_pcm16(clip[off:off+step], spk_cfg)
print('CHECKPOINT active_audio_call ' + json.dumps({'backend': mic.backend_name}), flush=True)
sys.stdin.readline()
print('DONE', flush=True)
"""
        checkpoints.update(self._measure_worker_memory(audio_worker, {}))

        # 3. Active Video Call (CPU / Low-RAM)
        video_cpu_worker = """
import json, sys
from bench.metrics.video_metrics import generate_synthetic_portrait
from nv_broadcast.video_effects import VideoEffectsPipeline
req = json.loads(sys.stdin.readline())
pipe = VideoEffectsPipeline(segmentation_backend='rvm_cpu')
if pipe.segmenter.backend_name == 'fast_portrait':
    pipe = VideoEffectsPipeline(segmentation_backend='mediapipe')
cam_cfg = {'effects': {'background_blur': {'enabled': True, 'strength': 0.7, 'mode': 'performance'}}}
frame, _ = generate_synthetic_portrait(1280, 720, frame_index=0)
for _ in range(req.get('frames', 5)):
    pipe.process_frame(frame, cam_cfg)
print('CHECKPOINT active_video_cpu ' + json.dumps({'backend': pipe.segmenter.backend_name}), flush=True)
sys.stdin.readline()
pipe.close()
print('DONE', flush=True)
"""
        checkpoints.update(self._measure_worker_memory(video_cpu_worker, {"frames": 4 if profile == "quick" else 12}))

        # 4 & 5. Active Video Call (RTX CUDA) -> Post-Call Idle reclamation check
        wait_after = 2.0 if profile == "quick" else 15.0
        video_cuda_worker = """
import gc, json, sys, time
from bench.metrics.video_metrics import generate_synthetic_portrait
from nv_broadcast.video_effects import VideoEffectsPipeline
req = json.loads(sys.stdin.readline())
pipe = VideoEffectsPipeline(segmentation_backend='rvm_cuda')
backend = pipe.segmenter.backend_name
cam_cfg = {'effects': {'background_blur': {'enabled': True, 'strength': 0.75, 'mode': 'quality'}}}
frame, _ = generate_synthetic_portrait(1280, 720, frame_index=0)
for _ in range(req.get('frames', 6)):
    pipe.process_frame(frame, cam_cfg)
print('CHECKPOINT active_video_cuda ' + json.dumps({'backend': backend}), flush=True)
sys.stdin.readline()
pipe.close()
del pipe
gc.collect()
time.sleep(float(req.get('wait_after', 2.0)))
print('CHECKPOINT post_call_idle ' + json.dumps({'prior_backend': backend, 'wait_sec': req.get('wait_after', 2.0)}), flush=True)
sys.stdin.readline()
print('DONE', flush=True)
"""
        checkpoints.update(
            self._measure_worker_memory(
                video_cuda_worker,
                {"frames": 5 if profile == "quick" else 15, "wait_after": wait_after},
                timeout=120.0,
            )
        )

        # Also inspect live systemd nv-broadcast.service if active (read-only!)
        live_service = None
        try:
            pid_res = subprocess.run(
                ["systemctl", "--user", "show", "-p", "MainPID", "--value", "nv-broadcast.service"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if pid_res.returncode == 0 and pid_res.stdout.strip().isdigit():
                main_pid = int(pid_res.stdout.strip())
                if main_pid > 0:
                    ltree = sample_process_tree(main_pid)
                    lgpu = query_nvidia_gpu(filter_pids=ltree["pids"])
                    live_service = {
                        "main_pid": main_pid,
                        "process_count": ltree["process_count"],
                        "rss_mb": ltree["rss_mb"],
                        "pss_mb": ltree["pss_mb"],
                        "private_dirty_mb": ltree["private_dirty_mb"],
                        "threads": ltree["threads"],
                        "open_fds": ltree["open_fds"],
                        "process_vram_mib": lgpu.get("process_vram_mib", 0.0),
                        "gpu_pstate": lgpu.get("pstate", "unavailable"),
                    }
        except (OSError, subprocess.SubprocessError):
            pass

        result: dict[str, Any] = {
            "status": "ok",
            "checkpoints": checkpoints,
        }
        if live_service is not None:
            result["live_systemd_service"] = live_service
        if soak_sec > 0 and soak_report is not None:
            result["soak"] = soak_report
        return result

    def run_video(
        self,
        profile: str = "quick",
        out_dir: Path | None = None,
        live_camera: bool = False,
    ) -> dict[str, Any]:
        w, h = (640, 360) if profile == "quick" else (1280, 720)
        n_frames = 10 if profile == "quick" else 30
        strip_dir = str((out_dir / "video_strips")) if out_dir is not None else ""

        worker_code = """
import json, sys, time
from pathlib import Path
from bench.metrics.system_metrics import compute_percentiles
from bench.metrics.video_metrics import (
    compute_bg_smoothness_delta,
    compute_mask_iou_mae,
    compute_psnr_db,
    compute_ssim,
    generate_synthetic_portrait,
    save_comparison_strip,
)
from nv_broadcast.video_effects import VideoEffectsPipeline

req = json.loads(sys.stdin.read())
w, h, n_frames = int(req['w']), int(req['h']), int(req['frames'])
strip_dir = Path(req['strip_dir']) if req.get('strip_dir') else None

pipe = VideoEffectsPipeline(segmentation_backend='auto')
seg_backend = pipe.segmenter.backend_name
face_backend = pipe.tracker.backend_name

clean_frames = [generate_synthetic_portrait(w, h, frame_index=i, add_sensor_noise=False) for i in range(n_frames)]
noisy_frames = [generate_synthetic_portrait(w, h, frame_index=i, add_sensor_noise=True) for i in range(n_frames)]

# Evaluate segmentation matte accuracy against synthetic ground truth
sample_rgb, sample_gt = clean_frames[0]
mask_metrics = None
try:
    if hasattr(pipe.segmenter, 'segment_alpha') and hasattr(sample_rgb, 'shape'):
        alpha = pipe.segmenter.segment_alpha(sample_rgb, None, quality='quality')
        mask_metrics = compute_mask_iou_mae(alpha, sample_gt)
except Exception:
    mask_metrics = None

effect_specs = {
    'background_blur_perf': (clean_frames, {'effects': {'background_blur': {'enabled': True, 'strength': 0.75, 'mode': 'performance'}}}),
    'background_blur_quality': (clean_frames, {'effects': {'background_blur': {'enabled': True, 'strength': 0.75, 'mode': 'quality'}}}),
    'background_replace': (clean_frames, {'effects': {'background_replace': {'enabled': True, 'preset': 'studio_dark', 'imagePath': ''}}}),
    'background_remove': (clean_frames, {'effects': {'background_remove': {'enabled': True, 'chromaColor': '#00ff00'}}}),
    'video_noise_removal': (noisy_frames, {'effects': {'video_noise_removal': {'enabled': True, 'strength': 0.65}}}),
    'combo_blur_keylight_autoframe': (clean_frames, {'effects': {
        'background_blur': {'enabled': True, 'strength': 0.7, 'mode': 'performance'},
        'virtual_key_light': {'enabled': True, 'warmth': 0.5, 'intensity': 0.55},
        'auto_frame': {'enabled': True, 'zoom': 0.35},
    }}),
}

effects_out = {}
strips = []
for ename, (src_frames, cam_cfg) in effect_specs.items():
    pipe.reset()
    # 2 warmup frames
    for wf, _ in src_frames[:2]:
        pipe.process_frame(wf, cam_cfg)
    lats_ms = []
    last_out = None
    for fr, _ in src_frames:
        t0 = time.perf_counter()
        last_out = pipe.process_frame(fr, cam_cfg)
        lats_ms.append((time.perf_counter() - t0) * 1000.0)
    stats = compute_percentiles(lats_ms)
    fps = round(1000.0 / max(stats['mean'], 0.001), 2)
    ref_raw, ref_mask = src_frames[-1]
    clean_ref, _ = clean_frames[-1]
    entry = {
        'status': 'ok',
        'frames': len(src_frames),
        'fps': fps,
        'latency_ms': stats,
        'budget_misses_33ms': sum(1 for v in lats_ms if v > 33.333),
        'psnr_vs_raw_db': round(compute_psnr_db(ref_raw, last_out), 2),
        'ssim_vs_raw': round(compute_ssim(ref_raw, last_out), 4),
        'bg_smoothness_delta_pct': compute_bg_smoothness_delta(ref_raw, last_out, ref_mask),
    }
    if ename == 'video_noise_removal':
        entry['psnr_noisy_vs_clean_db'] = round(compute_psnr_db(clean_ref, ref_raw), 2)
        entry['psnr_denoised_vs_clean_db'] = round(compute_psnr_db(clean_ref, last_out), 2)
        entry['denoise_psnr_gain_db'] = round(entry['psnr_denoised_vs_clean_db'] - entry['psnr_noisy_vs_clean_db'], 2)
    if strip_dir is not None and ename in ('background_blur_quality', 'background_replace', 'combo_blur_keylight_autoframe'):
        sp = save_comparison_strip([('Raw (Baseline)', ref_raw), (f'Python ({seg_backend})', last_out)], strip_dir / f'{ename}_python.png')
        if sp:
            strips.append(sp)
    effects_out[ename] = entry

pipe.close()
print(json.dumps({
    'status': 'ok',
    'backend': seg_backend,
    'face_backend': face_backend,
    'resolution': f'{w}x{h}',
    'mask_vs_synthetic_gt': mask_metrics,
    'effects': effects_out,
    'strips': strips,
}))
"""
        return self._run_py_json(
            worker_code,
            {"w": w, "h": h, "frames": n_frames, "strip_dir": strip_dir},
            timeout=240.0,
        )

    def run_api_cli(self, profile: str = "quick") -> dict[str, Any]:
        # 1. Config bench (same workload as Rust `nvb-cli bench config-bench`)
        cfg_iters = 500 if profile == "quick" else 2000
        cfg_bench_code = """
import json, sys, tempfile, time
from pathlib import Path
from nv_broadcast.config import BroadcastConfig
from bench.metrics.system_metrics import compute_percentiles

req = json.loads(sys.stdin.read())
iters = int(req['iterations'])
dirty = {
    'audio_backend': 'invalid_backend',
    'segmentation_backend': 'invalid_seg',
    'microphone': {
        'highpass_hz': 9999,
        'noise_gate_db': -120.0,
        'auto_gain_target_db': 10.0,
        'effects': {
            'noise_removal': {'enabled': True, 'strength': 1.9},
            'room_echo_removal': {'enabled': True, 'strength': -0.3},
            'studio_voice': {'enabled': False, 'micProfile': 'alien'},
        },
    },
    'camera': {
        'split_position': 0.99,
        'idle_timeout_sec': 0.01,
        'effects': {
            'background_blur': {'enabled': True, 'strength': 2.0, 'mode': 'ultra'},
        },
    },
}
lats_us = []
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'cfg.json'
    p.write_text(json.dumps(dirty), encoding='utf-8')
    t_wall0 = time.perf_counter()
    last_conflicts = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        cfg = BroadcastConfig(p)
        res = cfg.set_effect('studio_voice', enabled=True)
        last_conflicts = sum(len(v) for v in res.disabled_conflicts.values())
        cfg.set_effect('background_remove', enabled=True)
        lats_us.append((time.perf_counter() - t0) * 1_000_000.0)
    wall_ms = (time.perf_counter() - t_wall0) * 1000.0
stats = compute_percentiles(lats_us)
print(json.dumps({
    'status': 'ok',
    'iterations': iters,
    'wall_ms': round(wall_ms, 3),
    'ops_per_sec': round(iters / max(wall_ms / 1000.0, 1e-9), 1),
    'last_conflicts_disabled': last_conflicts,
    'latency': {
        'count': stats['count'],
        'min_us': stats['min'],
        'mean_us': stats['mean'],
        'p50_us': stats['p50'],
        'p95_us': stats['p95'],
        'p99_us': stats['p99'],
        'p999_us': stats['p999'],
        'max_us': stats['max'],
    },
}))
"""
        config_bench = self._run_py_json(cfg_bench_code, {"iterations": cfg_iters}, timeout=60.0)

        # 2. CLI cold startup latency
        cli_runs = 5 if profile == "quick" else 15
        cli_commands = {
            "version": ["--version"],
            "status": ["--json", "status"],
            "effects": ["--json", "effects"],
            "devices": ["--json", "devices"],
        }
        if profile == "full":
            cli_commands["doctor"] = ["--json", "doctor"]

        cli_cold: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="nvb_cli_bench_") as td:
            tmp_cfg = str(Path(td) / "config.json")
            for cname, cargs in cli_commands.items():
                times_ms: list[float] = []
                for _ in range(cli_runs):
                    cmd = [self.python_bin, "-m", "nv_broadcast.cli"]
                    if cname != "version":
                        cmd.extend(["--config", tmp_cfg])
                    cmd.extend(cargs)
                    t0 = time.perf_counter()
                    r = subprocess.run(cmd, capture_output=True, text=True, env=self._env, timeout=20, check=False)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    if r.returncode == 0:
                        times_ms.append(elapsed)
                cli_cold[cname] = compute_percentiles(times_ms)

        # 3. Daemon cold boot + HTTP endpoint latency + SSE fan-out + contract snapshot
        http_iters = 25 if profile == "quick" else 100
        endpoints_report: dict[str, Any] = {}
        sse_report: dict[str, Any] = {}
        contract_snapshot: dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="nvb_http_bench_") as td:
            proc, base_url, token, boot_ms = self._spawn_temp_daemon(Path(td))
            try:
                def http_call(method: str, path: str, body: dict[str, Any] | None = None, with_token: bool = True) -> tuple[int, dict[str, Any]]:
                    data = json.dumps(body).encode("utf-8") if body is not None else None
                    headers = {}
                    if with_token:
                        headers["X-NVB-Token"] = token
                    if data is not None:
                        headers["Content-Type"] = "application/json"
                    req = urllib.request.Request(f"{base_url}{path}", data=data, method=method, headers=headers)
                    try:
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            return resp.status, json.loads(resp.read().decode("utf-8"))
                    except urllib.error.HTTPError as err:
                        return err.code, json.loads(err.read().decode("utf-8"))

                # Contract snapshot for Python-vs-Rust API schema diffing
                unauth_code, unauth_body = http_call("GET", "/api/health", with_token=False)
                h_code, h_body = http_call("GET", "/api/health")
                s_code, s_body = http_call("GET", "/api/state")
                d_code, d_body = http_call("GET", "/api/devices")
                e_code, e_body = http_call(
                    "POST",
                    "/api/effect",
                    {"effect_id": "studio_voice", "enabled": True, "params": {"micProfile": "warm"}},
                )
                contract_snapshot = {
                    "unauthorized_status": unauth_code,
                    "unauthorized_keys": sorted(unauth_body.keys()),
                    "health_status": h_code,
                    "health_keys": sorted(h_body.keys()),
                    "state_status": s_code,
                    "state_keys": sorted(s_body.keys()),
                    "devices_status": d_code,
                    "devices_keys": sorted(d_body.keys()),
                    "post_effect_status": e_code,
                    "post_effect_keys": sorted(e_body.keys()),
                    "post_effect_applied": e_body.get("applied"),
                }

                # Benchmark endpoints
                specs = [
                    ("GET /api/health", "GET", "/api/health", None, http_iters),
                    ("GET /api/state", "GET", "/api/state", None, http_iters),
                    ("GET /api/devices", "GET", "/api/devices", None, max(5, http_iters // 5)),
                    (
                        "POST /api/effect",
                        "POST",
                        "/api/effect",
                        {"effect_id": "noise_removal", "enabled": True, "params": {"strength": 0.85}},
                        http_iters,
                    ),
                ]
                for label, method, path, body, count in specs:
                    lats_ms: list[float] = []
                    t_wall0 = time.perf_counter()
                    for i in range(count):
                        payload = body
                        if path == "/api/effect":
                            payload = (
                                {"effect_id": "studio_voice", "enabled": True, "params": {"micProfile": "warm"}}
                                if i % 2 == 0
                                else {"effect_id": "noise_removal", "enabled": True, "params": {"strength": 0.85}}
                            )
                        t0 = time.perf_counter()
                        http_call(method, path, payload)
                        lats_ms.append((time.perf_counter() - t0) * 1000.0)
                    wall_s = time.perf_counter() - t_wall0
                    endpoints_report[label] = {
                        "requests": count,
                        "req_per_sec": round(count / max(wall_s, 1e-9), 1),
                        "latency_ms": compute_percentiles(lats_ms),
                    }

                # SSE fan-out test (1 vs 5 concurrent subscribers for 1.5s)
                for n_subs in (1, 5):
                    stop_evt = threading.Event()
                    events_seen = [0] * n_subs

                    def _sub(idx: int) -> None:
                        req = urllib.request.Request(f"{base_url}/api/events?token={token}")
                        try:
                            with urllib.request.urlopen(req, timeout=3) as resp:
                                while not stop_evt.is_set():
                                    line = resp.readline()
                                    if not line:
                                        break
                                    if line.startswith(b"data:"):
                                        events_seen[idx] += 1
                        except Exception:
                            pass

                    threads = [threading.Thread(target=_sub, args=(i,), daemon=True) for i in range(n_subs)]
                    for th in threads:
                        th.start()
                    time.sleep(0.25)
                    s0 = sample_process_tree(proc.pid)
                    time.sleep(1.2)
                    s1 = sample_process_tree(proc.pid)
                    stop_evt.set()
                    for th in threads:
                        th.join(timeout=2.0)
                    delta = compute_process_delta(s0, s1)
                    sse_report[f"{n_subs}_clients"] = {
                        "events_received_total": sum(events_seen),
                        "rss_mb": s1["rss_mb"],
                        "cpu_ms_per_sec": delta["cpu_ms_per_sec"],
                        "total_ctxt_per_sec": delta["total_ctxt_per_sec"],
                    }
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        return {
            "status": "ok",
            "config_bench": config_bench,
            "cli_cold_start_ms": cli_cold,
            "daemon_boot_ms": round(boot_ms, 2),
            "http_endpoints": endpoints_report,
            "sse_fanout": sse_report,
            "contract_snapshot": contract_snapshot,
        }

    def run_system_impact(
        self,
        profile: str = "quick",
        window_sec: float | None = None,
    ) -> dict[str, Any]:
        dur = window_sec if window_sec is not None else (3.0 if profile == "quick" else 15.0)
        with tempfile.TemporaryDirectory(prefix="nvb_sys_impact_") as td:
            proc, base_url, token, _ = self._spawn_temp_daemon(Path(td))
            try:
                req = urllib.request.Request(f"{base_url}/api/health", headers={"X-NVB-Token": token})
                with urllib.request.urlopen(req, timeout=5):
                    pass
                time.sleep(0.5)
                gpu_start = query_nvidia_gpu(filter_pids=sample_process_tree(proc.pid)["pids"])
                pwr_start = query_power_supply()
                snap0 = sample_process_tree(proc.pid)
                time.sleep(dur)
                snap1 = sample_process_tree(proc.pid)
                gpu_end = query_nvidia_gpu(filter_pids=snap1["pids"])
                pwr_end = query_power_supply()
                delta = compute_process_delta(snap0, snap1)
                pw = query_pipewire_top() if profile == "full" else {"available": False, "reason": "skipped_in_quick_profile"}
                return {
                    "status": "ok",
                    "window_sec": delta["duration_sec"],
                    "process_overhead": {
                        "process_count": snap1["process_count"],
                        "rss_mb": snap1["rss_mb"],
                        "pss_mb": snap1["pss_mb"],
                        "threads": snap1["threads"],
                        "open_fds": snap1["open_fds"],
                        "cpu_ms_per_sec": delta["cpu_ms_per_sec"],
                        "cpu_core_pct": delta["cpu_core_pct"],
                        "voluntary_ctxt_per_sec": delta["voluntary_ctxt_per_sec"],
                        "nonvoluntary_ctxt_per_sec": delta["nonvoluntary_ctxt_per_sec"],
                        "total_ctxt_per_sec": delta["total_ctxt_per_sec"],
                    },
                    "gpu": {
                        "start": gpu_start,
                        "end": gpu_end,
                        "pstate": gpu_end.get("pstate", "unavailable"),
                        "power_draw_w": gpu_end.get("power_draw_w"),
                        "process_vram_mib": gpu_end.get("process_vram_mib", 0.0),
                    },
                    "power": {"start": pwr_start, "end": pwr_end},
                    "pipewire_top": pw,
                }
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def run_reliability(self, profile: str = "quick") -> dict[str, Any]:
        worker_code = """
import json, tempfile
from pathlib import Path
from nv_broadcast.config import BroadcastConfig
from nv_broadcast.pipewire import (
    build_isolated_pw_cat_cmd,
    is_pw_link_active,
    is_virtual_broadcast_node,
    parse_pw_links,
)

checks = []

# 1. Feedback loop rejection
virtuals = [
    'nv_broadcast_mic',
    'nv_broadcast_mic_bus',
    'nv_broadcast_mic_bus.monitor',
    'nv_broadcast_speakers',
    'NV_BROADCAST_SPEAKERS.monitor',
]
all_virt = all(is_virtual_broadcast_node(v) for v in virtuals)
phys_ok = not is_virtual_broadcast_node('alsa_input.pci-0000_00_1f.3') and not is_virtual_broadcast_node('')
checks.append({
    'name': 'feedback_loop_rejection',
    'passed': bool(all_virt and phys_ok),
    'detail': 'is_virtual_broadcast_node blocks all virtual Broadcast nodes and allows physical devices',
})

# 2. Isolated pw-cat stream properties
rec = build_isolated_pw_cat_cmd('record', 'nv_broadcast_mic_capture', rate=48000, channels=1, latency_ms=20)
play = build_isolated_pw_cat_cmd('playback', 'nv_broadcast_spk_playback', rate=48000, channels=2, latency_ms=20)
iso_ok = all(
    any(('node.autoconnect = false' in a and 'node.dont-reconnect = true' in a) for a in cmd)
    for cmd in (rec, play)
)
checks.append({
    'name': 'isolated_stream_properties',
    'passed': bool(iso_ok),
    'detail': 'build_isolated_pw_cat_cmd sets node.autoconnect=false and node.dont-reconnect=true',
})

# 3. Dropped link detection
links_ok = parse_pw_links('alsa_input.usb:capture_FL\\n  |-> nv_broadcast_mic_capture:input_FL\\n')
links_drop = parse_pw_links('alsa_input.usb:capture_FL\\n')
drop_ok = is_pw_link_active('alsa_input.usb', 'nv_broadcast_mic_capture', links_ok) and not is_pw_link_active('alsa_input.usb', 'nv_broadcast_mic_capture', links_drop)
checks.append({
    'name': 'dropped_link_detection',
    'passed': bool(drop_ok),
    'detail': 'parse_pw_links detects dropped link edge',
})

# 4. Corrupt / truncated config.json resilience
with tempfile.TemporaryDirectory() as td:
    bad = Path(td) / 'corrupt.json'
    bad.write_text('{broken json', encoding='utf-8')
    cfg = BroadcastConfig(bad)
    corrupt_ok = cfg.data.get('version') == 2 and cfg.data.get('audio_backend') == 'auto'
checks.append({
    'name': 'corrupt_config_recovery',
    'passed': bool(corrupt_ok),
    'detail': 'BroadcastConfig recovers cleanly from truncated JSON to sanitized defaults',
})

# 5. Camera on-demand standby defaults & config sanitization
with tempfile.TemporaryDirectory() as td:
    cfg = BroadcastConfig(Path(td) / 'c.json')
    cam = cfg.data['camera']
    sm_ok = bool(cam.get('on_demand', True)) and (0.5 <= float(cam.get('idle_timeout_sec', 3.0)) <= 60.0)
checks.append({
    'name': 'camera_standby_state_machine',
    'passed': bool(sm_ok),
    'detail': 'On-demand standby config defaults and idle_timeout clamping verified',
})

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({'status': 'ok', 'phase': 'oracle', 'passed': passed, 'total': len(checks), 'checks': checks}))
"""
        return self._run_py_json(worker_code, timeout=30.0)
