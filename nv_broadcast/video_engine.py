"""Real-time V4L2 Camera Capture + GPU Neural Effects + v4l2loopback + MJPEG Preview Engine."""

from __future__ import annotations

import datetime
import io
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from .config import BroadcastConfig
from .devices import find_loopback_cameras, list_cameras
from .v4l2_ctl import choose_best_pixel_format, set_v4l2_control
from .video_effects import HAS_CV2, HAS_NUMPY, VideoEffectsPipeline

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    import pyvirtualcam  # type: ignore

    HAS_PYVIRTUALCAM = True
except ImportError:
    pyvirtualcam = None  # type: ignore
    HAS_PYVIRTUALCAM = False


class VideoEngine:
    """Streams physical V4L2 camera -> GPU/Neural VideoEffectsPipeline -> preview + v4l2loopback."""

    def __init__(self, config: BroadcastConfig) -> None:
        self.config = config
        self.pipeline = VideoEffectsPipeline(
            segmentation_backend=config.data.get("segmentation_backend", "auto")
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = False
        self._standby_thread: threading.Thread | None = None
        self._standby_stop = threading.Event()
        self._standby_active = False
        self._keep_standby = False
        self._lock = threading.Lock()
        self._latest_frame_np: Any = None
        self._latest_frame_pil: Image.Image | None = None
        self._latest_jpeg: bytes = b""
        self._frame_seq: int = 0
        self._fps: float = 0.0
        self._active_input: str = ""
        self._active_output: str = ""
        self._active_resolution: str = ""
        self._last_error: str = ""
        self._frame_callbacks: list[Callable[[Image.Image], None]] = []

    @property
    def running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def active_input(self) -> str:
        return self._active_input

    @property
    def active_output(self) -> str:
        return self._active_output

    @property
    def active_resolution(self) -> str:
        return self._active_resolution

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def segmentation_backend(self) -> str:
        return self.pipeline.segmenter.backend_name

    @property
    def face_backend(self) -> str:
        return self.pipeline.tracker.backend_name

    def add_frame_callback(self, cb: Callable[[Image.Image], None]) -> None:
        self._frame_callbacks.append(cb)

    def get_latest_frame(self) -> Image.Image | None:
        with self._lock:
            if self._latest_frame_pil is not None:
                return self._latest_frame_pil.copy()
            if HAS_NUMPY and self._latest_frame_np is not None:
                return Image.fromarray(self._latest_frame_np, mode="RGB")
            return None

    def get_latest_jpeg(self, quality: int = 82) -> tuple[int, bytes]:
        with self._lock:
            if self._latest_jpeg:
                return self._frame_seq, self._latest_jpeg
            frame_np = self._latest_frame_np
            frame_pil = self._latest_frame_pil
            seq = self._frame_seq
        if HAS_CV2 and frame_np is not None:
            bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                data = enc.tobytes()
                with self._lock:
                    if self._frame_seq == seq:
                        self._latest_jpeg = data
                return seq, data
        if frame_pil is not None:
            buf = io.BytesIO()
            frame_pil.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            with self._lock:
                if self._frame_seq == seq:
                    self._latest_jpeg = data
            return seq, data
        return 0, b""

    def save_snapshot(self, output_dir: Path | None = None) -> tuple[bool, str]:
        img = self.get_latest_frame()
        if img is None:
            return False, "No camera frame available yet (start the camera first)"
        out_dir = output_dir or (Path.home() / "Pictures" / "NVIDIA Broadcast")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"NBX_Snapshot_{stamp}.png"
        img.save(path, format="PNG")
        return True, str(path)

    def resolve_input_device(self) -> tuple[str, str, int, int, int]:
        cam_cfg = self.config.data["camera"]
        wanted = cam_cfg.get("device", "auto")
        cams = list_cameras(include_metadata=False, include_loopback=False)
        selected = None
        if wanted and wanted != "auto":
            selected = next((c for c in cams if c.path == wanted), None)
        if selected is None and cams:
            selected = cams[0]
        if selected is None:
            raise RuntimeError("No physical V4L2 capture camera found")

        width = int(cam_cfg.get("width", 1280))
        height = int(cam_cfg.get("height", 720))
        fps = int(cam_cfg.get("fps", 30))
        pixfmt, best_fps = choose_best_pixel_format(
            selected, width, height, prefer_mjpg=bool(cam_cfg.get("prefer_mjpg", True))
        )
        effective_fps = int(min(fps, best_fps)) if best_fps > 0 else fps
        return selected.path, pixfmt, width, height, max(5, effective_fps)

    def resolve_output_device(self) -> str:
        cam_cfg = self.config.data["camera"]
        wanted = cam_cfg.get("output_device", "auto")
        loopbacks = find_loopback_cameras()
        if wanted and wanted != "auto":
            if any(d.path == wanted for d in loopbacks):
                return wanted
        return loopbacks[0].path if loopbacks else ""

    def apply_saved_v4l2_controls(self, device_path: str) -> None:
        controls = self.config.data["camera"].get("v4l2_controls", {})
        if not isinstance(controls, dict):
            return
        for key, val in controls.items():
            try:
                set_v4l2_control(device_path, key, int(val))
            except Exception:
                pass

    def start_standby(self) -> tuple[bool, str]:
        """Keep v4l2loopback visible to Discord/Meet/Brave (exclusive_caps=1) with webcam LED off."""
        self._keep_standby = True
        if self._running or self._standby_active:
            return True, "Virtual camera already active"
        out_dev = self.resolve_output_device()
        if not out_dev:
            return False, "No v4l2loopback device available for standby"
        cam_cfg = self.config.data["camera"]
        width = int(cam_cfg.get("width", 1280))
        height = int(cam_cfg.get("height", 720))
        fps = int(cam_cfg.get("fps", 30))
        self._active_output = out_dev
        self._standby_stop.clear()
        self._standby_active = True
        self._standby_thread = threading.Thread(
            target=self._run_standby_loop,
            args=(out_dev, width, height, fps),
            name="nv-broadcast-cam-standby",
            daemon=True,
        )
        self._standby_thread.start()
        return True, f"Virtual camera standby active on {out_dev}"

    def stop_standby(self, permanent: bool = False) -> None:
        if permanent:
            self._keep_standby = False
        self._standby_stop.set()
        if self._standby_thread is not None and self._standby_thread.is_alive():
            self._standby_thread.join(timeout=2.5)
        self._standby_thread = None
        self._standby_active = False

    def _run_standby_loop(self, out_dev: str, width: int, height: int, fps: int) -> None:
        # Build sleek dark NVIDIA Broadcast standby slate once (zero webcam access, zero GPU use)
        slate = Image.new("RGB", (width, height), (22, 24, 27))
        draw = ImageDraw.Draw(slate)
        cx, cy = width // 2, height // 2
        draw.rectangle([cx - 220, cy - 52, cx + 220, cy + 52], outline=(118, 185, 0), width=2)
        draw.text((cx - 125, cy - 18), "NVIDIA BROADCAST CAMERA", fill=(118, 185, 0))
        draw.text((cx - 155, cy + 8), "Camera is off - enable Camera in app", fill=(185, 190, 196))
        slate_np = np.asarray(slate, dtype=np.uint8) if HAS_NUMPY else None
        writer = self._open_virtual_output(out_dev, width, height, fps)
        try:
            if writer is None:
                return
            while not self._standby_stop.is_set():
                if slate_np is not None:
                    self._write_virtual_output_np(writer, slate_np)
                else:
                    self._write_virtual_output_pil(writer, slate)
                self._standby_stop.wait(1.0)
        finally:
            self._close_virtual_output(writer)
            self._standby_active = False

    def start(self, preview_only: bool = False) -> tuple[bool, str]:
        if self._running:
            return True, f"Camera already running ({self._active_input})"
        # Release standby holder before opening virtual camera for live stream
        if self._standby_active:
            self.stop_standby(permanent=False)
        try:
            in_dev, pixfmt, width, height, fps = self.resolve_input_device()
        except RuntimeError as exc:
            self._last_error = str(exc)
            return False, str(exc)

        out_dev = "" if preview_only else self.resolve_output_device()
        self.apply_saved_v4l2_controls(in_dev)

        self._active_input = in_dev
        self._active_output = out_dev
        self._active_resolution = f"{width}x{height}@{fps}fps ({pixfmt})"
        self._last_error = ""
        self._stop.clear()
        self.pipeline.reset()
        self._running = True

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(in_dev, pixfmt, width, height, fps, out_dev),
            name="nv-broadcast-video",
            daemon=True,
        )
        self._thread.start()

        out_note = f" -> {out_dev}" if out_dev else " (preview only; install v4l2loopback for virtual camera output)"
        return True, f"Started camera {in_dev} ({pixfmt} {width}x{height}@{fps}fps){out_note}"

    def stop(self, shutdown: bool = False) -> None:
        if shutdown:
            self._keep_standby = False
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._running = False
        self._fps = 0.0
        if shutdown:
            self.stop_standby(permanent=True)
        elif self._keep_standby:
            self.start_standby()

    def _run_loop(
        self,
        in_dev: str,
        pixfmt: str,
        width: int,
        height: int,
        fps: int,
        out_dev: str,
    ) -> None:
        out_writer = self._open_virtual_output(out_dev, width, height, fps) if out_dev else None
        try:
            if HAS_CV2 and HAS_NUMPY:
                self._capture_opencv(in_dev, pixfmt, width, height, fps, out_writer)
            elif shutil.which("ffmpeg"):
                self._capture_ffmpeg(in_dev, pixfmt, width, height, fps, out_writer)
            else:
                self._last_error = "Neither OpenCV nor ffmpeg is installed for V4L2 capture"
        finally:
            self._close_virtual_output(out_writer)
            self._running = False

    def _open_virtual_output(self, out_dev: str, width: int, height: int, fps: int) -> Any:
        if HAS_PYVIRTUALCAM and HAS_NUMPY:
            try:
                return ("pyvirtualcam", pyvirtualcam.Camera(width=width, height=height, fps=fps, device=out_dev))
            except Exception:
                pass
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuyv422",
                "-f",
                "v4l2",
                out_dev,
            ]
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                return ("ffmpeg", proc)
            except OSError:
                return None
        return None

    def _write_virtual_output_np(self, writer: Any, frame_rgb: Any) -> None:
        if writer is None:
            return
        kind, handle = writer
        try:
            if kind == "pyvirtualcam":
                handle.send(frame_rgb)
            elif kind == "ffmpeg" and handle.stdin is not None and handle.poll() is None:
                handle.stdin.write(frame_rgb.tobytes())
        except Exception:
            pass

    def _write_virtual_output_pil(self, writer: Any, frame: Image.Image) -> None:
        if writer is None:
            return
        kind, handle = writer
        try:
            if kind == "pyvirtualcam" and HAS_NUMPY:
                handle.send(np.asarray(frame, dtype=np.uint8))
            elif kind == "ffmpeg" and handle.stdin is not None and handle.poll() is None:
                handle.stdin.write(frame.tobytes())
        except Exception:
            pass

    def _close_virtual_output(self, writer: Any) -> None:
        if writer is None:
            return
        kind, handle = writer
        try:
            if kind == "pyvirtualcam":
                handle.close()
            elif kind == "ffmpeg":
                if handle.stdin is not None:
                    handle.stdin.close()
                if handle.poll() is None:
                    handle.terminate()
                    handle.wait(timeout=1.5)
        except Exception:
            pass

    def _publish_np(self, frame_rgb: Any) -> None:
        with self._lock:
            self._latest_frame_np = frame_rgb
            self._latest_frame_pil = None
            self._latest_jpeg = b""
            self._frame_seq += 1
        if self._frame_callbacks:
            pil = Image.fromarray(frame_rgb, mode="RGB")
            for cb in list(self._frame_callbacks):
                try:
                    cb(pil)
                except Exception:
                    pass

    def _publish_pil(self, frame: Image.Image) -> None:
        with self._lock:
            self._latest_frame_pil = frame
            self._latest_frame_np = None
            self._latest_jpeg = b""
            self._frame_seq += 1
        for cb in list(self._frame_callbacks):
            try:
                cb(frame)
            except Exception:
                pass

    def _capture_opencv(
        self,
        in_dev: str,
        pixfmt: str,
        width: int,
        height: int,
        fps: int,
        out_writer: Any,
    ) -> None:
        cap = cv2.VideoCapture(in_dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            self._last_error = f"OpenCV failed to open {in_dev}"
            return
        try:
            if pixfmt == "MJPG":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frames = 0
            warmed = False
            t0 = time.monotonic()
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    time.sleep(0.01)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cam_cfg = self.config.data["camera"]
                # Always send clean un-split processed frame to virtual webcam, split only on preview
                if cam_cfg.get("split_preview", False) and out_writer is not None:
                    no_split_cfg = dict(cam_cfg, split_preview=False)
                    raw_out, clean_proc = self.pipeline.process_frame_np(rgb, no_split_cfg)
                    self._write_virtual_output_np(out_writer, clean_proc)
                    from .video_effects import render_split_frame

                    preview = render_split_frame(raw_out, clean_proc, float(cam_cfg.get("split_position", 0.5)))
                    self._publish_np(preview)
                else:
                    _, processed = self.pipeline.process_frame_np(rgb, cam_cfg)
                    self._write_virtual_output_np(out_writer, processed)
                    self._publish_np(processed)

                if not warmed:
                    warmed = True
                    frames = 0
                    t0 = time.monotonic()
                    continue

                frames += 1
                dt = time.monotonic() - t0
                if dt >= 1.0:
                    self._fps = round(frames / dt, 1)
                    frames = 0
                    t0 = time.monotonic()
        finally:
            cap.release()

    def _capture_ffmpeg(
        self,
        in_dev: str,
        pixfmt: str,
        width: int,
        height: int,
        fps: int,
        out_writer: Any,
    ) -> None:
        in_fmt = "mjpeg" if pixfmt == "MJPG" else "yuyv422"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            in_fmt,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            in_dev,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = width * height * 3
        frames = 0
        t0 = time.monotonic()
        try:
            assert proc.stdout is not None
            while not self._stop.is_set():
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                raw_img = Image.frombytes("RGB", (width, height), buf)
                processed = self.pipeline.process_frame(raw_img, self.config.data["camera"])
                self._write_virtual_output_pil(out_writer, processed)
                self._publish_pil(processed)
                frames += 1
                dt = time.monotonic() - t0
                if dt >= 1.0:
                    self._fps = round(frames / dt, 1)
                    frames = 0
                    t0 = time.monotonic()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
