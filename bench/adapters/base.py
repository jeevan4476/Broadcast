"""Common TargetAdapter interface shared by baseline, python, and rust targets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from bench.metrics.audio_metrics import AudioScenario


class TargetAdapter(ABC):
    name: str = "base"
    display_name: str = "Base"

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return target metadata and suite readiness flags."""

    @abstractmethod
    def run_audio_quality(
        self,
        scenarios: list[AudioScenario],
        out_dir: Path,
        profile: str = "quick",
    ) -> dict[str, Any]:
        """Run Suite 1: Audio Signal Quality and Noise Reduction."""

    @abstractmethod
    def run_audio_latency(
        self,
        profile: str = "quick",
        frames: int | None = None,
    ) -> dict[str, Any]:
        """Run Suite 2: Real-Time Audio Latency, Jitter, and Throughput."""

    @abstractmethod
    def run_memory(
        self,
        profile: str = "quick",
        soak_sec: int = 0,
    ) -> dict[str, Any]:
        """Run Suite 3: Process-Tree RAM/PSS and GPU VRAM Lifecycle."""

    @abstractmethod
    def run_video(
        self,
        profile: str = "quick",
        out_dir: Path | None = None,
        live_camera: bool = False,
    ) -> dict[str, Any]:
        """Run Suite 4: Video Quality, FPS, Frame Jitter, and On-Demand Wake."""

    @abstractmethod
    def run_api_cli(self, profile: str = "quick") -> dict[str, Any]:
        """Run Suite 5: Control Plane, CLI Startup, and HTTP/SSE/MJPEG API."""

    @abstractmethod
    def run_system_impact(
        self,
        profile: str = "quick",
        window_sec: float | None = None,
    ) -> dict[str, Any]:
        """Run Suite 6: Background System, Power, and Idle Wakeup Overhead."""

    @abstractmethod
    def run_reliability(self, profile: str = "quick") -> dict[str, Any]:
        """Run Suite 7: Reliability, Anti-Loop, and Recovery Chaos Matrix."""
