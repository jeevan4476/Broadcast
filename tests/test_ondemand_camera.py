"""Tests for Issue #2: on-demand webcam activation via v4l2loopback consumer detection."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import unittest
import urllib.request
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

NV_BCAST_ROOT = Path(__file__).resolve().parent.parent
if str(NV_BCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(NV_BCAST_ROOT))

from nv_broadcast.cli import main as cli_main
from nv_broadcast.config import BroadcastConfig
from nv_broadcast.loopback_monitor import (
    OnDemandDebounce,
    has_external_consumers,
    list_loopback_consumers,
)
from nv_broadcast.server import BroadcastRuntime, make_handler
from nv_broadcast.video_engine import VideoEngine


def _wait_until(predicate, timeout: float = 2.5, step: float = 0.015) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self._now = float(start)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, delta: float) -> float:
        with self._lock:
            self._now += float(delta)
            return self._now


class _FakeVirtualWriterTracker:
    """Tracks virtual output writer lifecycles to verify clean handoff and zero FD leaks."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_concurrent = 0
        self.total_opens = 0
        self.total_closes = 0

    def open_writer(self, out_dev: str, width: int, height: int, fps: int) -> tuple[str, dict[str, Any]]:
        with self.lock:
            self.active += 1
            self.total_opens += 1
            if self.active > self.max_concurrent:
                self.max_concurrent = self.active
        return ("fake", {"dev": out_dev, "closed": False})

    def close_writer(self, writer: Any) -> None:
        if writer is None:
            return
        _, handle = writer
        with self.lock:
            if not handle["closed"]:
                handle["closed"] = True
                self.active -= 1
                self.total_closes += 1


class TestLoopbackMonitor(unittest.TestCase):
    def test_list_loopback_consumers_synthetic_proc(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_file = root / "video10"
            dev_file.write_text("", encoding="utf-8")
            dev_symlink = root / "by-id-virtual-cam"
            os.symlink(dev_file, dev_symlink)

            proc_root = root / "proc"
            proc_root.mkdir()
            # Non-PID entry should be ignored
            (proc_root / "self").mkdir()

            # Own PID opening the device must be excluded by default
            own_pid = os.getpid()
            own_fd = proc_root / str(own_pid) / "fd"
            own_fd.mkdir(parents=True)
            os.symlink(dev_file, own_fd / "3")
            (proc_root / str(own_pid) / "comm").write_text("python3\n", encoding="utf-8")

            # External consumer #1 with two FDs to the same device -> deduplicated to 1 entry
            ext1 = own_pid + 1001
            ext1_fd = proc_root / str(ext1) / "fd"
            ext1_fd.mkdir(parents=True)
            os.symlink(dev_file, ext1_fd / "4")
            os.symlink(dev_file, ext1_fd / "5")
            (proc_root / str(ext1) / "comm").write_text("Discord\n", encoding="utf-8")

            # External consumer #2 referencing the symlink path
            ext2 = own_pid + 1002
            ext2_fd = proc_root / str(ext2) / "fd"
            ext2_fd.mkdir(parents=True)
            os.symlink(dev_symlink, ext2_fd / "7")
            (proc_root / str(ext2) / "comm").write_text("chrome\n", encoding="utf-8")

            # External consumer #3 without comm file
            ext3 = own_pid + 1003
            ext3_fd = proc_root / str(ext3) / "fd"
            ext3_fd.mkdir(parents=True)
            os.symlink(dev_file, ext3_fd / "9")

            # Unrelated process opening /dev/null
            other = own_pid + 1004
            other_fd = proc_root / str(other) / "fd"
            other_fd.mkdir(parents=True)
            os.symlink("/dev/null", other_fd / "3")
            (proc_root / str(other) / "comm").write_text("bash\n", encoding="utf-8")

            # Query via canonical path: own_pid excluded, ext1 and ext3 matched
            found = list_loopback_consumers(str(dev_file), proc_root=proc_root)
            self.assertEqual(
                found,
                [
                    {"pid": ext1, "comm": "Discord"},
                    {"pid": ext3, "comm": ""},
                ],
            )
            self.assertTrue(has_external_consumers(str(dev_file), proc_root=proc_root))

            # Query via symlink path: matches both symlink target (ext1, ext3) and symlink itself (ext2)
            found_sym = list_loopback_consumers(str(dev_symlink), proc_root=proc_root)
            self.assertEqual(
                found_sym,
                [
                    {"pid": ext1, "comm": "Discord"},
                    {"pid": ext2, "comm": "chrome"},
                    {"pid": ext3, "comm": ""},
                ],
            )

            # Custom exclude_pids suppresses listed processes
            found_excl = list_loopback_consumers(
                str(dev_file),
                exclude_pids={own_pid, ext1, ext3},
                proc_root=proc_root,
            )
            self.assertEqual(found_excl, [])
            self.assertFalse(
                has_external_consumers(
                    str(dev_file),
                    exclude_pids={own_pid, ext1, ext3},
                    proc_root=proc_root,
                )
            )

    def test_on_demand_debounce(self) -> None:
        clock = _FakeClock(50.0)
        debounce = OnDemandDebounce(idle_timeout_sec=3.0, clock=clock)

        self.assertFalse(debounce.update(1))
        self.assertIsNone(debounce.idle_since)

        # Drops to 0 -> begins idle tracking
        self.assertFalse(debounce.update(0))
        self.assertEqual(debounce.idle_since, 50.0)

        clock.advance(2.5)
        self.assertFalse(debounce.update(0))
        self.assertAlmostEqual(debounce.idle_elapsed(), 2.5, places=3)

        # Consumer returns before timeout -> resets idle timer
        self.assertFalse(debounce.update(2))
        self.assertIsNone(debounce.idle_since)

        # Drops to 0 again and waits full 3.0s
        self.assertFalse(debounce.update(0))
        clock.advance(2.9)
        self.assertFalse(debounce.update(0))
        clock.advance(0.15)
        self.assertTrue(debounce.update(0))


class TestOnDemandConfig(unittest.TestCase):
    def test_defaults_migration_clamping_and_persistence(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"

            # Simulate an older config.json missing on_demand and idle_timeout_sec
            legacy = {
                "version": 2,
                "camera": {
                    "device": "/dev/video0",
                    "width": 1920,
                    "height": 1080,
                },
            }
            cfg_path.write_text(json.dumps(legacy), encoding="utf-8")

            cfg = BroadcastConfig(cfg_path)
            self.assertTrue(cfg.data["camera"]["on_demand"])
            self.assertEqual(cfg.data["camera"]["idle_timeout_sec"], 3.0)
            self.assertEqual(cfg.data["camera"]["width"], 1920)

            # Persist updated values and verify clamping bounds
            cfg.data["camera"]["on_demand"] = False
            cfg.data["camera"]["idle_timeout_sec"] = 0.05
            cfg.save()
            reloaded = BroadcastConfig(cfg_path)
            self.assertFalse(reloaded.data["camera"]["on_demand"])
            self.assertEqual(reloaded.data["camera"]["idle_timeout_sec"], 0.5)

            reloaded.data["camera"]["idle_timeout_sec"] = 200.0
            reloaded.save()
            reloaded2 = BroadcastConfig(cfg_path)
            self.assertEqual(reloaded2.data["camera"]["idle_timeout_sec"], 60.0)


class TestVideoEngineOnDemand(unittest.TestCase):
    def _build_mocked_engine(
        self,
        cfg: BroadcastConfig,
        consumers_ref: list[list[dict[str, Any]]],
        clock: _FakeClock,
    ) -> tuple[VideoEngine, _FakeVirtualWriterTracker, list[int]]:
        tracker = _FakeVirtualWriterTracker()
        capture_starts: list[int] = []

        def detect(device_path: str, exclude_pids: set[int]) -> list[dict[str, Any]]:
            self.assertIn(os.getpid(), exclude_pids)
            self.assertEqual(device_path, "/dev/video10")
            return list(consumers_ref[0])

        engine = VideoEngine(
            cfg,
            pipeline=mock.Mock(),
            consumer_detector=detect,
            watcher_poll_interval=0.025,
            clock=clock,
        )
        engine.resolve_output_device = mock.Mock(return_value="/dev/video10")
        engine.resolve_input_device = mock.Mock(return_value=("/dev/video0", "MJPG", 640, 360, 30))
        engine.apply_saved_v4l2_controls = mock.Mock()
        engine._open_virtual_output = tracker.open_writer  # type: ignore[method-assign]
        engine._close_virtual_output = tracker.close_writer  # type: ignore[method-assign]
        engine._write_virtual_output_np = mock.Mock()  # type: ignore[method-assign]
        engine._write_virtual_output_pil = mock.Mock()  # type: ignore[method-assign]

        def fake_capture(in_dev: str, pixfmt: str, width: int, height: int, fps: int, out_writer: Any) -> None:
            capture_starts.append(1)
            engine._stop.wait(5.0)

        engine._capture_opencv = fake_capture  # type: ignore[method-assign]
        engine._capture_ffmpeg = fake_capture  # type: ignore[method-assign]
        return engine, tracker, capture_starts

    def test_standby_wake_debounce_and_release_to_standby(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = BroadcastConfig(Path(tmp) / "config.json")
            cfg.data["camera"]["on_demand"] = True
            cfg.data["camera"]["idle_timeout_sec"] = 3.0
            consumers_ref: list[list[dict[str, Any]]] = [[]]
            clock = _FakeClock(1000.0)

            engine, tracker, capture_starts = self._build_mocked_engine(cfg, consumers_ref, clock)
            try:
                ok, _ = engine.start_standby()
                self.assertTrue(ok)
                self.assertTrue(engine.standby_active)
                self.assertFalse(engine.running)
                self.assertFalse(engine.on_demand_active)
                self.assertFalse(engine.manual_active)

                # External app opens /dev/video10 -> auto-wakes camera
                consumers_ref[0] = [{"pid": 5555, "comm": "Discord"}]
                self.assertTrue(_wait_until(lambda: engine.running and engine.on_demand_active))
                self.assertFalse(engine.standby_active)
                self.assertFalse(engine.manual_active)
                self.assertEqual(engine.external_consumers, [{"pid": 5555, "comm": "Discord"}])
                self.assertEqual(len(capture_starts), 1)
                self.assertEqual(tracker.max_concurrent, 1)

                # Consumer closes -> camera stays running during idle timeout
                consumers_ref[0] = []
                self.assertTrue(_wait_until(lambda: engine.external_consumers == []))
                clock.advance(2.0)
                time.sleep(0.08)
                self.assertTrue(engine.running)
                self.assertTrue(engine.on_demand_active)

                # Advance past 3.0s idle timeout -> camera releases back to standby
                clock.advance(1.5)
                self.assertTrue(_wait_until(lambda: (not engine.running) and engine.standby_active))
                self.assertTrue(_wait_until(lambda: tracker.active == 1))
                self.assertFalse(engine.on_demand_active)
                self.assertFalse(engine.manual_active)
                self.assertEqual(tracker.max_concurrent, 1)
                self.assertEqual(tracker.active, 1)  # standby writer active
            finally:
                engine.shutdown_all()

            self.assertFalse(engine.running)
            self.assertFalse(engine.standby_active)
            self.assertEqual(tracker.active, 0)
            self.assertEqual(tracker.total_opens, tracker.total_closes)

    def test_manual_start_not_torn_down_by_idle_timeout(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = BroadcastConfig(Path(tmp) / "config.json")
            cfg.data["camera"]["on_demand"] = True
            cfg.data["camera"]["idle_timeout_sec"] = 1.0
            consumers_ref: list[list[dict[str, Any]]] = [[]]
            clock = _FakeClock(2000.0)

            engine, tracker, _ = self._build_mocked_engine(cfg, consumers_ref, clock)
            try:
                engine.start_standby()
                ok, _ = engine.start(preview_only=False)
                self.assertTrue(ok)
                self.assertTrue(engine.running)
                self.assertTrue(engine.manual_active)
                self.assertFalse(engine.on_demand_active)

                # Even with 0 consumers and plenty of elapsed time, manual camera stays active
                clock.advance(10.0)
                time.sleep(0.1)
                self.assertTrue(engine.running)
                self.assertTrue(engine.manual_active)

                # Explicit manual stop clears manual_active and returns to standby + watcher
                engine.stop()
                self.assertFalse(engine.running)
                self.assertFalse(engine.manual_active)
                self.assertFalse(engine.on_demand_active)
                self.assertTrue(engine.standby_active)

                # Watcher is still alive: a new external consumer wakes the camera
                consumers_ref[0] = [{"pid": 7777, "comm": "obs"}]
                self.assertTrue(_wait_until(lambda: engine.running and engine.on_demand_active))

                # User manually starts while on-demand is active -> upgrades to manual
                ok2, _ = engine.start(preview_only=False)
                self.assertTrue(ok2)
                self.assertTrue(engine.manual_active)
                self.assertFalse(engine.on_demand_active)

                # Consumer leaves + timeout passes -> still running because manual_active is True
                consumers_ref[0] = []
                self.assertTrue(_wait_until(lambda: engine.external_consumers == []))
                clock.advance(10.0)
                time.sleep(0.1)
                self.assertTrue(engine.running)
                self.assertTrue(engine.manual_active)
            finally:
                engine.shutdown_all()

            self.assertEqual(tracker.active, 0)

    def test_set_on_demand_runtime_toggle(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = BroadcastConfig(Path(tmp) / "config.json")
            cfg.data["camera"]["on_demand"] = True
            cfg.data["camera"]["idle_timeout_sec"] = 2.0
            consumers_ref: list[list[dict[str, Any]]] = [[]]
            clock = _FakeClock(3000.0)

            engine, tracker, _ = self._build_mocked_engine(cfg, consumers_ref, clock)
            try:
                engine.start_standby()
                consumers_ref[0] = [{"pid": 8888, "comm": "zoom"}]
                self.assertTrue(_wait_until(lambda: engine.running and engine.on_demand_active))

                # Disabling on_demand stops watcher and releases on-demand-only capture back to standby
                engine.set_on_demand(False)
                self.assertFalse(engine.running)
                self.assertFalse(engine.on_demand_active)
                self.assertTrue(engine.standby_active)
                self.assertEqual(engine.external_consumers, [])

                # While disabled, external consumer does NOT wake the camera
                time.sleep(0.1)
                self.assertFalse(engine.running)

                # Re-enabling on_demand restarts watcher and wakes camera immediately
                engine.set_on_demand(True, idle_timeout_sec=1.5)
                self.assertTrue(_wait_until(lambda: engine.running and engine.on_demand_active))
                self.assertEqual(cfg.data["camera"]["idle_timeout_sec"], 1.5)
            finally:
                engine.shutdown_all()

            self.assertEqual(tracker.active, 0)


class TestServerAndCLIOnDemand(unittest.TestCase):
    def test_server_telemetry_and_camera_config_api(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg = BroadcastConfig(cfg_path)
            runtime = BroadcastRuntime(cfg)
            token = "testtoken"
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
            port = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            try:
                base = f"http://127.0.0.1:{port}"
                req = urllib.request.Request(f"{base}/api/state", headers={"X-NVB-Token": token})
                state = json.loads(urllib.request.urlopen(req, timeout=5).read())
                cam_tel = state["telemetry"]["camera"]
                self.assertIn("on_demand", cam_tel)
                self.assertIn("idle_timeout_sec", cam_tel)
                self.assertIn("on_demand_active", cam_tel)
                self.assertIn("external_consumers", cam_tel)
                self.assertIn("standby_active", cam_tel)
                self.assertTrue(cam_tel["on_demand"])
                self.assertEqual(cam_tel["idle_timeout_sec"], 3.0)

                # Update via POST /api/camera/config with out-of-range timeout to test clamping
                post_req = urllib.request.Request(
                    f"{base}/api/camera/config",
                    data=json.dumps({"on_demand": False, "idle_timeout_sec": 0.1}).encode("utf-8"),
                    headers={"X-NVB-Token": token, "Content-Type": "application/json"},
                    method="POST",
                )
                res = json.loads(urllib.request.urlopen(post_req, timeout=5).read())
                self.assertTrue(res["ok"])
                self.assertFalse(res["config"]["camera"]["on_demand"])
                self.assertEqual(res["config"]["camera"]["idle_timeout_sec"], 0.5)

                # Verify persisted to disk and reflected in telemetry
                reloaded = BroadcastConfig(cfg_path)
                self.assertFalse(reloaded.data["camera"]["on_demand"])
                self.assertEqual(reloaded.data["camera"]["idle_timeout_sec"], 0.5)
                tel_after = runtime.get_telemetry()["camera"]
                self.assertFalse(tel_after["on_demand"])
                self.assertEqual(tel_after["idle_timeout_sec"], 0.5)
            finally:
                httpd.shutdown()
                httpd.server_close()
                runtime.shutdown()
                t.join(timeout=2.0)

    def test_cli_camera_on_demand_options(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg_path = str(Path(tmp) / "config.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(
                    [
                        "--config",
                        cfg_path,
                        "--json",
                        "camera",
                        "--on-demand",
                        "off",
                        "--idle-timeout",
                        "7.5",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["camera"]["on_demand"])
            self.assertEqual(payload["camera"]["idle_timeout_sec"], 7.5)

            reloaded = BroadcastConfig(cfg_path)
            self.assertFalse(reloaded.data["camera"]["on_demand"])
            self.assertEqual(reloaded.data["camera"]["idle_timeout_sec"], 7.5)


if __name__ == "__main__":
    unittest.main()
