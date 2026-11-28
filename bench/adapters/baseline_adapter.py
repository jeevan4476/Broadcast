"""Baseline (No Broadcast) target adapter.

Represents running the system with Broadcast absent or completely bypassed:
- Audio quality: identity passthrough of raw noisy signal (establishes Delta-SNR = 0 dB ground truth).
- Audio latency: raw 480-sample PCM buffer pass-through overhead (~0 us).
- Memory: 0 MB daemon RAM, 0 MiB Broadcast VRAM, 0 background threads/FDs.
- Video: raw unedited frames (0% background blur/denoise, identity frame pass-through).
- API / CLI: N/A (not installed/running).
- System impact: 0 background CPU/wakeups + live host GPU P-state / battery / PipeWire baseline.
- Reliability: N/A (no virtual routing active).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bench.adapters.base import TargetAdapter
from bench.metrics.audio_metrics import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioScenario,
    evaluate_audio_clip,
    write_wav_mono16,
)
from bench.metrics.system_metrics import (
    compute_percentiles,
    query_nvidia_gpu,
    query_pipewire_top,
    query_power_supply,
)
from bench.metrics.video_metrics import (
    compute_bg_smoothness_delta,
    compute_psnr_db,
    compute_ssim,
    generate_synthetic_portrait,
)


class BaselineAdapter(TargetAdapter):
    name = "baseline"
    display_name = "No Broadcast (Baseline)"

    def capabilities(self) -> dict[str, Any]:
        return {
            "target": self.name,
            "display_name": self.display_name,
            "phase": "baseline",
            "capabilities": {
                "audio_quality": True,
                "audio_latency": True,
                "memory": True,
                "video": True,
                "api_cli": True,
                "system_impact": True,
                "reliability": True,
            },
        }

    def run_audio_quality(
        self,
        scenarios: list[AudioScenario],
        out_dir: Path,
        profile: str = "quick",
    ) -> dict[str, Any]:
        audio_dir = out_dir / "audio_ab"
        results: dict[str, Any] = {}
        delta_snrs: list[float] = []
        delta_sisdrs: list[float] = []
        silence_attens: list[float] = []

        for sc in scenarios:
            wav_path = write_wav_mono16(audio_dir / f"{sc.name}_raw.wav", sc.noisy_pcm, sc.sample_rate)
            metrics = evaluate_audio_clip(
                noisy_pcm=sc.noisy_pcm,
                processed_pcm=sc.noisy_pcm,
                clean_pcm=sc.clean_pcm,
                speech_mask=sc.speech_mask,
                sample_rate=sc.sample_rate,
            )
            metrics["wav_path"] = str(wav_path)
            metrics["description"] = sc.description
            results[sc.name] = metrics
            if "delta_snr_db" in metrics:
                delta_snrs.append(metrics["delta_snr_db"])
            if "delta_si_sdr_db" in metrics:
                delta_sisdrs.append(metrics["delta_si_sdr_db"])
            silence_attens.append(metrics["silence_attenuation_db"])

        return {
            "status": "ok",
            "backend": "passthrough",
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
        dummy = bytes(FRAME_SAMPLES * 2)
        lat_us: list[float] = []
        t_wall0 = time.perf_counter()
        for _ in range(n_frames):
            t0 = time.perf_counter()
            out = bytes(dummy)
            _ = len(out)
            lat_us.append((time.perf_counter() - t0) * 1_000_000.0)
        wall_sec = time.perf_counter() - t_wall0
        audio_sec = (n_frames * FRAME_SAMPLES) / float(SAMPLE_RATE)
        stats = compute_percentiles(lat_us)
        workload = {
            "frames": n_frames,
            "audio_sec": round(audio_sec, 3),
            "wall_ms": round(wall_sec * 1000.0, 3),
            "rtf": round(wall_sec / max(audio_sec, 1e-9), 6),
            "single_core_cpu_pct": round((wall_sec / max(audio_sec, 1e-9)) * 100.0, 4),
            "budget_misses_2ms": sum(1 for v in lat_us if v > 2000.0),
            "budget_misses_10ms": sum(1 for v in lat_us if v > 10000.0),
            "latency_us": stats,
        }
        res: dict[str, Any] = {
            "status": "ok",
            "backend": "passthrough",
            "workloads": {
                "noise_removal": workload,
                "room_echo_removal": workload,
                "studio_voice": workload,
            },
        }
        if profile == "full":
            res["pipewire_top"] = query_pipewire_top()
        return res

    def run_memory(
        self,
        profile: str = "quick",
        soak_sec: int = 0,
    ) -> dict[str, Any]:
        gpu = query_nvidia_gpu()
        zero_cp = {
            "status": "ok",
            "process_count": 0,
            "rss_mb": 0.0,
            "pss_mb": 0.0,
            "private_dirty_mb": 0.0,
            "threads": 0,
            "open_fds": 0,
            "process_vram_mib": 0.0,
            "gpu_pstate": gpu.get("pstate", "unavailable"),
        }
        return {
            "status": "ok",
            "checkpoints": {
                "idle_standby": dict(zero_cp),
                "active_audio_call": dict(zero_cp),
                "active_video_cpu": dict(zero_cp),
                "active_video_cuda": dict(zero_cp),
                "post_call_idle": dict(zero_cp),
            },
            "host_gpu_baseline": gpu,
        }

    def run_video(
        self,
        profile: str = "quick",
        out_dir: Path | None = None,
        live_camera: bool = False,
    ) -> dict[str, Any]:
        w, h = (640, 360) if profile == "quick" else (1280, 720)
        n_frames = 10 if profile == "quick" else 30
        frames = [generate_synthetic_portrait(w, h, frame_index=i) for i in range(n_frames)]
        lat_ms: list[float] = []
        for frame, _ in frames:
            t0 = time.perf_counter()
            _ = frame.copy() if hasattr(frame, "copy") else frame
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
        stats = compute_percentiles(lat_ms)
        mean_ms = max(stats["mean"], 0.001)
        first_frame, first_mask = frames[0]
        raw_entry = {
            "status": "ok",
            "frames": n_frames,
            "fps": round(min(1000.0, 1000.0 / mean_ms), 2),
            "latency_ms": stats,
            "budget_misses_33ms": 0,
            "psnr_vs_raw_db": round(compute_psnr_db(first_frame, first_frame), 2),
            "ssim_vs_raw": round(compute_ssim(first_frame, first_frame), 4),
            "bg_smoothness_delta_pct": compute_bg_smoothness_delta(first_frame, first_frame, first_mask),
        }
        return {
            "status": "ok",
            "backend": "raw_passthrough",
            "resolution": f"{w}x{h}",
            "effects": {
                "background_blur_perf": dict(raw_entry),
                "background_blur_quality": dict(raw_entry),
                "background_replace": dict(raw_entry),
                "background_remove": dict(raw_entry),
                "video_noise_removal": dict(raw_entry),
                "combo_blur_keylight_autoframe": dict(raw_entry),
            },
        }

    def run_api_cli(self, profile: str = "quick") -> dict[str, Any]:
        return {
            "status": "not_applicable",
            "reason": "No Broadcast daemon or CLI is active in baseline mode",
        }

    def run_system_impact(
        self,
        profile: str = "quick",
        window_sec: float | None = None,
    ) -> dict[str, Any]:
        dur = window_sec if window_sec is not None else (2.0 if profile == "quick" else 10.0)
        gpu_start = query_nvidia_gpu()
        pwr_start = query_power_supply()
        time.sleep(min(dur, 3.0 if profile == "quick" else dur))
        gpu_end = query_nvidia_gpu()
        pwr_end = query_power_supply()
        pw = query_pipewire_top() if profile == "full" else {"available": False, "reason": "skipped_in_quick_profile"}
        return {
            "status": "ok",
            "window_sec": round(min(dur, 3.0 if profile == "quick" else dur), 2),
            "process_overhead": {
                "process_count": 0,
                "rss_mb": 0.0,
                "pss_mb": 0.0,
                "threads": 0,
                "open_fds": 0,
                "cpu_ms_per_sec": 0.0,
                "cpu_core_pct": 0.0,
                "voluntary_ctxt_per_sec": 0.0,
                "nonvoluntary_ctxt_per_sec": 0.0,
                "total_ctxt_per_sec": 0.0,
            },
            "gpu": {
                "start": gpu_start,
                "end": gpu_end,
                "pstate": gpu_end.get("pstate", "unavailable"),
                "power_draw_w": gpu_end.get("power_draw_w"),
                "process_vram_mib": 0.0,
            },
            "power": {
                "start": pwr_start,
                "end": pwr_end,
            },
            "pipewire_top": pw,
        }

    def run_reliability(self, profile: str = "quick") -> dict[str, Any]:
        return {
            "status": "not_applicable",
            "reason": "No Broadcast virtual audio/video routing active in baseline mode",
        }
