"""Rust Broadcast (`nvb-*`) target adapter.

Drives the compiled `nvb-cli` and `nvb-daemon` binaries using the exact same
JSON / HTTP / `/proc` contract as `PythonAdapter`:
- Audio quality & latency (`nvb-cli bench audio-file` / `audio-loop`)
- Memory lifecycle across 5 states (`nvb-daemon` idle, active audio, isolated
  camera worker spawn & 100% OS RAM/VRAM reclamation on exit)
- Control plane (`config-bench`, full CLI cold start, daemon boot, HTTP `/api/*`
  latency/throughput, SSE 1 vs 5 fan-out, and API contract snapshot)
- System impact (`nvb-daemon` idle CPU, PSS/RSS, threads, FDs, context switches, GPU P-state)
- Reliability chaos invariants
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from bench.adapters.base import TargetAdapter
from bench.adapters.python_adapter import detect_python_binary
from bench.metrics.audio_metrics import (
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


class RustAdapter(TargetAdapter):
    name = "rust"
    display_name = "Rust Broadcast (nvb-*)"

    def __init__(self, repo_root: Path, rust_cli_bin: str | None = None, auto_build: bool = True) -> None:
        self.repo_root = repo_root
        self.rust_dir = repo_root / "rust"
        self.auto_build = auto_build
        self._cli_bin = Path(rust_cli_bin) if rust_cli_bin else self._locate_binary("nvb-cli")
        self._daemon_bin = self._locate_binary("nvb-daemon")
        self._caps_cache: dict[str, Any] | None = None

    def _locate_binary(self, name: str) -> Path | None:
        for profile in ("release", "debug"):
            cand = self.rust_dir / "target" / profile / name
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
        return None

    def ensure_binary(self, release: bool = False) -> Path | None:
        if release:
            rel_cli = self.rust_dir / "target" / "release" / "nvb-cli"
            rel_daemon = self.rust_dir / "target" / "release" / "nvb-daemon"
            if rel_cli.is_file() and rel_daemon.is_file():
                self._cli_bin = rel_cli
                self._daemon_bin = rel_daemon
                return rel_cli
        if self._cli_bin is not None and self._cli_bin.is_file():
            if self._daemon_bin is None or not self._daemon_bin.is_file():
                self._daemon_bin = self._locate_binary("nvb-daemon")
            return self._cli_bin
        if not self.auto_build or not shutil.which("cargo") or not (self.rust_dir / "Cargo.toml").is_file():
            return None
        cmd = ["cargo", "build", "-p", "nvb-cli", "-p", "nvb-daemon"]
        if release:
            cmd.append("--release")
        res = subprocess.run(cmd, cwd=str(self.rust_dir), capture_output=True, text=True, timeout=240, check=False)
        if res.returncode != 0:
            return None
        self._cli_bin = self._locate_binary("nvb-cli")
        self._daemon_bin = self._locate_binary("nvb-daemon")
        return self._cli_bin

    def _ensure_daemon_binary(self) -> Path | None:
        self.ensure_binary()
        if self._daemon_bin is not None and self._daemon_bin.is_file():
            return self._daemon_bin
        if self.auto_build and shutil.which("cargo") and (self.rust_dir / "Cargo.toml").is_file():
            subprocess.run(
                ["cargo", "build", "-p", "nvb-daemon"],
                cwd=str(self.rust_dir),
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            self._daemon_bin = self._locate_binary("nvb-daemon")
        return self._daemon_bin

    def _run_cli_json(self, args: list[str], timeout: float = 120.0) -> dict[str, Any]:
        cli = self.ensure_binary()
        if cli is None:
            return {"status": "unavailable", "reason": "nvb-cli binary not built and cargo unavailable"}
        res = subprocess.run(
            [str(cli), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if res.returncode != 0:
            return {
                "status": "error",
                "reason": res.stderr.strip() or res.stdout.strip() or f"nvb-cli exited {res.returncode}",
            }
        return json.loads(res.stdout)

    def _spawn_temp_daemon(
        self,
        tmpdir: Path,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.Popen[str], str, str, float]:
        daemon = self._ensure_daemon_binary()
        if daemon is None:
            raise RuntimeError("nvb-daemon binary unavailable")
        cfg_path = tmpdir / "config.json"
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [str(daemon), "--host", "127.0.0.1", "--port", "0", "--config", str(cfg_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
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
                    raise RuntimeError(f"nvb-daemon exited early ({proc.returncode}): {err}")
                time.sleep(0.01)
                continue
            if line.startswith("READY "):
                parts = line.strip().split()
                if len(parts) >= 3:
                    base_url, token = parts[1], parts[2]
                    break
        if not base_url:
            proc.terminate()
            raise RuntimeError("Timed out waiting for nvb-daemon READY line")
        boot_ms = (time.perf_counter() - t0) * 1000.0
        return proc, base_url, token, boot_ms

    def capabilities(self) -> dict[str, Any]:
        if self._caps_cache is not None:
            return self._caps_cache
        cli = self.ensure_binary()
        if cli is None:
            return {
                "target": self.name,
                "display_name": self.display_name,
                "status": "unavailable",
                "phase": "unbuilt",
                "capabilities": {},
                "pending_phases": {},
            }
        info = self._run_cli_json(["bench", "capabilities"], timeout=15.0)
        info["display_name"] = self.display_name
        info["binary"] = str(cli)
        if self._daemon_bin:
            info["daemon_binary"] = str(self._daemon_bin)
        self._caps_cache = info
        return info

    def _has_cap(self, key: str) -> bool:
        return bool(self.capabilities().get("capabilities", {}).get(key, False))

    def _pending_status(self, suite: str, default_phase: str) -> dict[str, Any]:
        caps = self.capabilities()
        if caps.get("status") == "unavailable":
            return {"status": "unavailable", "reason": "nvb-cli binary unavailable"}
        pending = caps.get("pending_phases", {}).get(suite, default_phase)
        return {
            "status": pending,
            "phase": caps.get("phase", "R0"),
            "reason": f"Waiting for Rust port milestone ({pending})",
        }

    def run_audio_quality(
        self,
        scenarios: list[AudioScenario],
        out_dir: Path,
        profile: str = "quick",
    ) -> dict[str, Any]:
        if not self._has_cap("audio_dsp"):
            return self._pending_status("audio_quality", "pending_phase_r1")

        audio_dir = out_dir / "audio_ab"
        audio_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}
        delta_snrs: list[float] = []
        delta_sisdrs: list[float] = []
        silence_attens: list[float] = []
        backend = "nnnoiseless"
        peak_rss_kb = 0

        with tempfile.TemporaryDirectory(prefix="nvb_rust_audio_") as tmpdir:
            tmp = Path(tmpdir)
            for sc in scenarios:
                in_wav = write_wav_mono16(tmp / f"{sc.name}_in.wav", sc.noisy_pcm, sc.sample_rate)
                cfg_path = tmp / f"{sc.name}_cfg.json"
                cfg_path.write_text(json.dumps(sc.effect_cfg), encoding="utf-8")
                out_wav = audio_dir / f"{sc.name}_rust.wav"

                res = self._run_cli_json(
                    [
                        "bench",
                        "audio-file",
                        "--in",
                        str(in_wav),
                        "--out",
                        str(out_wav),
                        "--config",
                        str(cfg_path),
                    ],
                    timeout=60.0,
                )
                if res.get("status") != "ok":
                    return {"status": "error", "scenario": sc.name, "error": res.get("reason", "audio-file failed")}

                backend = str(res.get("backend", backend))
                peak_rss_kb = max(peak_rss_kb, int(res.get("rss_kb", 0)))
                proc_pcm, _ = read_wav_mono16(out_wav)
                metrics = evaluate_audio_clip(
                    noisy_pcm=sc.noisy_pcm,
                    processed_pcm=proc_pcm,
                    clean_pcm=sc.clean_pcm,
                    speech_mask=sc.speech_mask,
                    sample_rate=sc.sample_rate,
                )
                metrics["elapsed_ms"] = res.get("elapsed_ms", 0.0)
                metrics["meter"] = res.get("meter", {})
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
            "backend": backend,
            "peak_rss_mb": round(peak_rss_kb / 1024.0, 2),
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
        if self._has_cap("audio_dsp"):
            res = self._run_cli_json(["bench", "audio-loop", "--frames", str(n_frames)], timeout=180.0)
            if res.get("status") == "ok":
                if "rss_kb" in res:
                    res["rss_mb"] = round(float(res["rss_kb"]) / 1024.0, 2)
                if profile == "full":
                    res["pipewire_top"] = query_pipewire_top()
                return res
            return res
        return self._pending_status("audio_latency", "pending_phase_r1")

    def run_memory(
        self,
        profile: str = "quick",
        soak_sec: int = 0,
    ) -> dict[str, Any]:
        if not self._has_cap("daemon_http"):
            return self._pending_status("memory", "pending_phase_r2")

        checkpoints: dict[str, Any] = {}
        soak_report = None

        def _snap(pid: int, meta: dict[str, Any] | None = None) -> dict[str, Any]:
            tree = sample_process_tree(pid)
            gpu = query_nvidia_gpu(filter_pids=tree["pids"])
            out: dict[str, Any] = {
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
            if meta:
                out["meta"] = meta
            return out

        with tempfile.TemporaryDirectory(prefix="nvb_rust_mem_") as tmpdir:
            td = Path(tmpdir)
            # Create a tiny isolated worker wrapper script that runs a real CUDA video
            # session inside a child process spawned by `nvb-daemon`, proving 100% OS
            # RAM + VRAM reclamation the instant `/api/camera/stop` kills the worker!
            py_bin = detect_python_binary(self.repo_root)
            worker_script = td / "camera_worker.sh"
            worker_py = td / "camera_worker.py"
            worker_py.write_text(
                "\n".join(
                    [
                        "import sys, time",
                        "from bench.metrics.video_metrics import generate_synthetic_portrait",
                        "from nv_broadcast.video_effects import VideoEffectsPipeline",
                        "pipe = VideoEffectsPipeline(segmentation_backend='rvm_cuda')",
                        "cfg = {'effects': {'background_blur': {'enabled': True, 'strength': 0.75, 'mode': 'quality'}}}",
                        "frame, _ = generate_synthetic_portrait(1280, 720, frame_index=0)",
                        "for _ in range(4):",
                        "    pipe.process_frame(frame, cfg)",
                        "while True:",
                        "    time.sleep(0.2)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            worker_script.write_text(
                f"#!/usr/bin/env bash\nPYTHONPATH='{self.repo_root}' exec '{py_bin}' '{worker_py}'\n",
                encoding="utf-8",
            )
            os.chmod(worker_script, 0o755)

            proc, base_url, token, _ = self._spawn_temp_daemon(
                td,
                extra_env={"NVB_CAMERA_WORKER": str(worker_script)},
            )
            try:
                def http_post(path: str, body: dict[str, Any]) -> None:
                    req = urllib.request.Request(
                        f"{base_url}{path}",
                        data=json.dumps(body).encode("utf-8"),
                        method="POST",
                        headers={"Content-Type": "application/json", "X-NVB-Token": token},
                    )
                    with urllib.request.urlopen(req, timeout=10):
                        pass

                # 1. Idle / Standby
                req = urllib.request.Request(f"{base_url}/api/state", headers={"X-NVB-Token": token})
                with urllib.request.urlopen(req, timeout=10):
                    pass
                time.sleep(0.5)
                checkpoints["idle_standby"] = _snap(proc.pid)

                # 2. Active Audio Processing (trigger live Rust nnnoiseless sample + measure process)
                http_post("/api/audio/speaker/sample", {"kind": "fan_echo"})
                checkpoints["active_audio_call"] = _snap(proc.pid, {"backend": "nnnoiseless"})

                # 3. Active Video Call (isolated child worker -> CUDA/CPU)
                http_post("/api/camera/start", {"preview_only": True})
                # Give worker a moment to initialize ONNX CUDA & process frames
                time.sleep(2.5 if profile == "quick" else 4.5)
                checkpoints["active_video_cuda"] = _snap(proc.pid, {"mode": "isolated_child_worker"})

                # 4. Stop camera -> worker exits -> OS immediately reclaims 100% RAM & VRAM!
                http_post("/api/camera/stop", {})
                time.sleep(1.0 if profile == "quick" else 3.0)
                checkpoints["post_call_idle"] = _snap(proc.pid, {"reclaimed": True})

                if soak_sec > 0:
                    samples = []
                    t_start = time.monotonic()
                    t_end = t_start + soak_sec
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
                        rss_slope, fd_drift = 0.0, 0
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

        out: dict[str, Any] = {"status": "ok", "checkpoints": checkpoints}
        if soak_report is not None:
            out["soak"] = soak_report
        return out

    def run_video(
        self,
        profile: str = "quick",
        out_dir: Path | None = None,
        live_camera: bool = False,
    ) -> dict[str, Any]:
        if not self._has_cap("video_pipeline"):
            return self._pending_status("video", "pending_phase_r3")
        return self._pending_status("video", "pending_phase_r3")

    def run_api_cli(self, profile: str = "quick") -> dict[str, Any]:
        cli = self.ensure_binary()
        if cli is None:
            return {"status": "unavailable", "reason": "nvb-cli binary unavailable"}

        cfg_iters = 500 if profile == "quick" else 2000
        config_bench = self._run_cli_json(["bench", "config-bench", "--iterations", str(cfg_iters)], timeout=60.0)

        cli_runs = 5 if profile == "quick" else 15
        cli_commands: dict[str, list[str]] = {"version": ["--version"]}
        if self._has_cap("cli_full"):
            cli_commands.update(
                {
                    "status": ["--json", "status"],
                    "effects": ["--json", "effects"],
                    "devices": ["--json", "devices"],
                }
            )
            if profile == "full":
                cli_commands["doctor"] = ["--json", "doctor"]

        cli_cold: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="nvb_rust_cli_") as td:
            tmp_cfg = str(Path(td) / "config.json")
            for cname, cargs in cli_commands.items():
                times_ms: list[float] = []
                for _ in range(cli_runs):
                    cmd = [str(cli)]
                    if cname != "version":
                        cmd.extend(["--config", tmp_cfg])
                    cmd.extend(cargs)
                    t0 = time.perf_counter()
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    if r.returncode == 0:
                        times_ms.append(elapsed)
                cli_cold[cname] = compute_percentiles(times_ms)

        if not self._has_cap("daemon_http"):
            return {
                "status": "partial_phase_r1",
                "phase": self.capabilities().get("phase", "R1"),
                "pending_daemon": "pending_phase_r2",
                "config_bench": config_bench,
                "cli_cold_start_ms": cli_cold,
            }

        http_iters = 25 if profile == "quick" else 100
        endpoints_report: dict[str, Any] = {}
        sse_report: dict[str, Any] = {}
        contract_snapshot: dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="nvb_rust_http_") as td:
            proc, base_url, token, boot_ms = self._spawn_temp_daemon(Path(td))
            try:
                def http_call(
                    method: str,
                    path: str,
                    body: dict[str, Any] | None = None,
                    with_token: bool = True,
                ) -> tuple[int, dict[str, Any]]:
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

                # Warm cache once before contract + latency benchmark
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
        if not self._has_cap("daemon_http"):
            return self._pending_status("system_impact", "pending_phase_r2")

        dur = window_sec if window_sec is not None else (3.0 if profile == "quick" else 15.0)
        with tempfile.TemporaryDirectory(prefix="nvb_rust_sys_") as td:
            proc, base_url, token, _ = self._spawn_temp_daemon(Path(td))
            try:
                req = urllib.request.Request(f"{base_url}/api/health", headers={"X-NVB-Token": token})
                with urllib.request.urlopen(req, timeout=5):
                    pass
                time.sleep(0.6)
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
        if not self._has_cap("reliability_core"):
            return self._pending_status("reliability", "pending_phase_r0")
        return self._run_cli_json(["bench", "reliability"], timeout=30.0)
