"""Rust Broadcast (`nvb-*`) target adapter.

Drives the compiled `nvb-cli` (and future `nvb-daemon` / `nvb-camera-worker`)
binaries using the exact same JSON protocol as `PythonAdapter`.

Works immediately at Phase R0:
- Executes real Rust benchmarks for implemented subsystems (`config_bench`,
  `pcm_bench`, CLI cold start, and `reliability` invariants).
- Automatically queries `nvb-cli bench capabilities` and reports
  `pending_phase_r1..r3` for unported modules until each phase lands.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from bench.adapters.base import TargetAdapter
from bench.metrics.audio_metrics import AudioScenario
from bench.metrics.system_metrics import compute_percentiles


class RustAdapter(TargetAdapter):
    name = "rust"
    display_name = "Rust Broadcast (nvb-*)"

    def __init__(self, repo_root: Path, rust_cli_bin: str | None = None, auto_build: bool = True) -> None:
        self.repo_root = repo_root
        self.rust_dir = repo_root / "rust"
        self.auto_build = auto_build
        self._cli_bin = Path(rust_cli_bin) if rust_cli_bin else self._locate_cli_binary()
        self._caps_cache: dict[str, Any] | None = None

    def _locate_cli_binary(self) -> Path | None:
        for profile in ("release", "debug"):
            cand = self.rust_dir / "target" / profile / "nvb-cli"
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
        return None

    def ensure_binary(self, release: bool = False) -> Path | None:
        if self._cli_bin is not None and self._cli_bin.is_file():
            return self._cli_bin
        if not self.auto_build or not shutil.which("cargo") or not (self.rust_dir / "Cargo.toml").is_file():
            return None
        cmd = ["cargo", "build", "-p", "nvb-cli"]
        if release:
            cmd.append("--release")
        res = subprocess.run(cmd, cwd=str(self.rust_dir), capture_output=True, text=True, timeout=180, check=False)
        if res.returncode != 0:
            return None
        self._cli_bin = self._locate_cli_binary()
        return self._cli_bin

    def _run_cli_json(self, args: list[str], timeout: float = 60.0) -> dict[str, Any]:
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
        # Hook ready for Phase R1: `nvb-cli bench audio-file --in ... --out ... --config ...`
        return self._pending_status("audio_quality", "pending_phase_r1")

    def run_audio_latency(
        self,
        profile: str = "quick",
        frames: int | None = None,
    ) -> dict[str, Any]:
        n_frames = frames if frames is not None else (2000 if profile == "quick" else 10000)
        pcm_prim = None
        if self._has_cap("pcm_bench"):
            raw_pcm = self._run_cli_json(["bench", "pcm-bench", "--frames", str(n_frames)], timeout=60.0)
            if raw_pcm.get("status") == "ok":
                lat = raw_pcm.get("latency", {})
                pcm_prim = {
                    "status": "ok",
                    "frames": raw_pcm["frames"],
                    "audio_sec": raw_pcm["audio_sec"],
                    "wall_ms": raw_pcm["wall_ms"],
                    "rtf": raw_pcm["rtf"],
                    "latency_us": {
                        "count": lat.get("count", 0),
                        "min": lat.get("min_us", 0.0),
                        "mean": lat.get("mean_us", 0.0),
                        "p50": lat.get("p50_us", 0.0),
                        "p95": lat.get("p95_us", 0.0),
                        "p99": lat.get("p99_us", 0.0),
                        "p999": lat.get("p999_us", 0.0),
                        "max": lat.get("max_us", 0.0),
                    },
                }

        if not self._has_cap("audio_dsp"):
            out = self._pending_status("audio_latency", "pending_phase_r1")
            if pcm_prim is not None:
                out["pcm_primitive"] = pcm_prim
            return out
        return self._pending_status("audio_latency", "pending_phase_r1")

    def run_memory(
        self,
        profile: str = "quick",
        soak_sec: int = 0,
    ) -> dict[str, Any]:
        if not self._has_cap("daemon_http"):
            return self._pending_status("memory", "pending_phase_r2")
        return self._pending_status("memory", "pending_phase_r2")

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
        config_bench = (
            self._run_cli_json(["bench", "config-bench", "--iterations", str(cfg_iters)], timeout=60.0)
            if self._has_cap("config_bench")
            else {"status": "pending_phase_r0"}
        )

        # Measure real cold binary startup for `nvb-cli --version` even at Phase R0!
        cli_runs = 5 if profile == "quick" else 15
        ver_times: list[float] = []
        for _ in range(cli_runs):
            t0 = time.perf_counter()
            r = subprocess.run([str(cli), "--version"], capture_output=True, text=True, timeout=10, check=False)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if r.returncode == 0:
                ver_times.append(elapsed)

        if not self._has_cap("daemon_http"):
            return {
                "status": "partial_phase_r0",
                "phase": self.capabilities().get("phase", "R0"),
                "pending_daemon": "pending_phase_r2",
                "config_bench": config_bench,
                "cli_cold_start_ms": {
                    "version": compute_percentiles(ver_times),
                },
            }
        return self._pending_status("api_cli", "pending_phase_r2")

    def run_system_impact(
        self,
        profile: str = "quick",
        window_sec: float | None = None,
    ) -> dict[str, Any]:
        if not self._has_cap("daemon_http"):
            return self._pending_status("system_impact", "pending_phase_r2")
        return self._pending_status("system_impact", "pending_phase_r2")

    def run_reliability(self, profile: str = "quick") -> dict[str, Any]:
        if not self._has_cap("reliability_core"):
            return self._pending_status("reliability", "pending_phase_r0")
        return self._run_cli_json(["bench", "reliability"], timeout=30.0)
