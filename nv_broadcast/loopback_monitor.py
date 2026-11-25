"""Pure-Python /proc watcher for v4l2loopback external consumers and idle debounce."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable


def _read_proc_comm(pid_dir: Path) -> str:
    try:
        return (pid_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return ""


def list_loopback_consumers(
    device_path: str,
    exclude_pids: set[int] | None = None,
    proc_root: Path = Path("/proc"),
) -> list[dict[str, Any]]:
    """Return external processes holding open file descriptors to a v4l2loopback device.

    Scans ``proc_root/<pid>/fd/*`` directly with stdlib calls (no subprocess overhead).
    """
    if not device_path:
        return []

    target = os.path.realpath(device_path)
    excluded = {os.getpid()} if exclude_pids is None else set(exclude_pids)
    consumers: list[dict[str, Any]] = []

    try:
        entries = list(os.scandir(proc_root))
    except (FileNotFoundError, PermissionError, NotADirectoryError, OSError):
        return []

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue

        fd_dir = os.path.join(entry.path, "fd")
        matched = False
        try:
            with os.scandir(fd_dir) as fd_it:
                for fd_entry in fd_it:
                    try:
                        link = os.readlink(fd_entry.path)
                    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                        continue
                    if link == target or link == device_path:
                        matched = True
                        break
        except (FileNotFoundError, PermissionError, ProcessLookupError, NotADirectoryError, OSError):
            continue

        if matched:
            consumers.append({"pid": pid, "comm": _read_proc_comm(Path(entry.path))})

    consumers.sort(key=lambda item: int(item["pid"]))
    return consumers


def has_external_consumers(
    device_path: str,
    exclude_pids: set[int] | None = None,
    proc_root: Path = Path("/proc"),
) -> bool:
    """Return True if at least one external process has ``device_path`` open."""
    return bool(list_loopback_consumers(device_path, exclude_pids=exclude_pids, proc_root=proc_root))


class OnDemandDebounce:
    """Tracks continuous idle time after consumers drop to 0 before releasing the camera."""

    def __init__(
        self,
        idle_timeout_sec: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._idle_timeout_sec = max(0.0, float(idle_timeout_sec))
        self._clock = clock
        self._idle_since: float | None = None

    @property
    def idle_timeout_sec(self) -> float:
        return self._idle_timeout_sec

    @idle_timeout_sec.setter
    def idle_timeout_sec(self, value: float) -> None:
        self._idle_timeout_sec = max(0.0, float(value))

    @property
    def idle_since(self) -> float | None:
        return self._idle_since

    def reset(self) -> None:
        self._idle_since = None

    def idle_elapsed(self, now: float | None = None) -> float:
        if self._idle_since is None:
            return 0.0
        ts = self._clock() if now is None else now
        return max(0.0, ts - self._idle_since)

    def update(self, consumer_count: int, now: float | None = None) -> bool:
        """Advance debounce state with current consumer count.

        Returns True once consumer_count has stayed at 0 for at least ``idle_timeout_sec``.
        """
        if consumer_count > 0:
            self._idle_since = None
            return False
        ts = self._clock() if now is None else now
        if self._idle_since is None:
            self._idle_since = ts
            return self._idle_timeout_sec <= 0.0
        return (ts - self._idle_since) >= self._idle_timeout_sec
