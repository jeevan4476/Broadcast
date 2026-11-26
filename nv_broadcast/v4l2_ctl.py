"""Pure-Python Linux V4L2 ioctl utilities (no v4l2-ctl dependency required)."""

from __future__ import annotations

import fcntl
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Linux V4L2 ioctl numbers (x86_64 / generic asm-generic/ioctl.h)
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xC0405602
VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C
VIDIOC_QUERYCTRL = 0xC0445624
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A
VIDIOC_ENUM_FRAMEINTERVALS = 0xC034564B

# Capability flags
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_OUTPUT = 0x00000002
V4L2_CAP_META_CAPTURE = 0x00800000
V4L2_CAP_DEVICE_CAPS = 0x80000000

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FRMSIZE_TYPE_DISCRETE = 1
V4L2_FRMIVAL_TYPE_DISCRETE = 1
V4L2_CTRL_FLAG_NEXT_CTRL = 0x80000000
V4L2_CTRL_FLAG_DISABLED = 0x0001
V4L2_CTRL_FLAG_READ_ONLY = 0x0004
V4L2_CTRL_TYPE_CTRL_CLASS = 6

# Friendly control aliases -> canonical V4L2 control IDs
COMMON_CONTROLS: dict[str, int] = {
    "brightness": 0x00980900,
    "contrast": 0x00980901,
    "saturation": 0x00980902,
    "hue": 0x00980903,
    "white_balance_auto": 0x0098090C,
    "gamma": 0x00980910,
    "gain": 0x00980913,
    "power_line_frequency": 0x00980918,
    "white_balance_temperature": 0x0098091A,
    "sharpness": 0x0098091B,
    "backlight_compensation": 0x0098091C,
    "auto_exposure": 0x009A0901,
    "exposure_time": 0x009A0902,
}


@dataclass
class V4L2Control:
    id: int
    name: str
    slug: str
    ctrl_type: int
    minimum: int
    maximum: int
    step: int
    default: int
    value: int
    flags: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class V4L2Mode:
    pixel_format: str
    width: int
    height: int
    fps_options: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class V4L2DeviceInfo:
    path: str
    driver: str
    card: str
    bus_info: str
    is_capture: bool
    is_output: bool
    is_metadata: bool
    is_loopback: bool
    formats: list[str] = field(default_factory=list)
    modes: list[V4L2Mode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modes"] = [m.to_dict() for m in self.modes]
        return d


def _slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def query_v4l2_device(device_path: str) -> V4L2DeviceInfo | None:
    """Inspect a /dev/videoX node via V4L2 ioctl without holding it open."""
    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        try:
            fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return None

    try:
        cap_buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap_buf)
        driver_b, card_b, bus_b, _ver, caps, dev_caps = struct.unpack("16s32s32sIII", cap_buf[:92])
        effective_caps = dev_caps if (caps & V4L2_CAP_DEVICE_CAPS) else caps

        driver = driver_b.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
        card = card_b.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
        bus = bus_b.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()

        is_capture = bool(effective_caps & V4L2_CAP_VIDEO_CAPTURE)
        is_output = bool(effective_caps & V4L2_CAP_VIDEO_OUTPUT)
        is_metadata = bool(effective_caps & V4L2_CAP_META_CAPTURE) and not is_capture
        is_loopback = driver.lower() == "v4l2 loopback" or "loopback" in bus.lower()

        formats: list[str] = []
        modes: list[V4L2Mode] = []
        if is_capture and not is_loopback:
            for idx in range(16):
                fbuf = bytearray(64)
                struct.pack_into("<II", fbuf, 0, idx, V4L2_BUF_TYPE_VIDEO_CAPTURE)
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FMT, fbuf)
                except OSError:
                    break
                _, _, _, _, pixfmt = struct.unpack_from("<III32sI", fbuf, 0)
                fourcc = struct.pack("<I", pixfmt).decode("ascii", errors="replace").strip()
                if fourcc and fourcc not in formats:
                    formats.append(fourcc)

                for sidx in range(32):
                    sbuf = bytearray(44)
                    struct.pack_into("<II", sbuf, 0, sidx, pixfmt)
                    try:
                        fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, sbuf)
                    except OSError:
                        break
                    _, _, ftype, w, h = struct.unpack_from("<IIIII", sbuf, 0)
                    if ftype != V4L2_FRMSIZE_TYPE_DISCRETE or w <= 0 or h <= 0:
                        continue
                    fps_list: list[float] = []
                    for ridx in range(16):
                        rbuf = bytearray(52)
                        struct.pack_into("<IIII", rbuf, 0, ridx, pixfmt, w, h)
                        try:
                            fcntl.ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, rbuf)
                        except OSError:
                            break
                        rtype, num, den = struct.unpack_from("<III", rbuf, 16)
                        if rtype == V4L2_FRMIVAL_TYPE_DISCRETE and num > 0:
                            fps = round(den / num, 1)
                            if fps not in fps_list:
                                fps_list.append(fps)
                    modes.append(V4L2Mode(fourcc, w, h, sorted(fps_list, reverse=True)))

        return V4L2DeviceInfo(
            path=device_path,
            driver=driver,
            card=card,
            bus_info=bus,
            is_capture=is_capture,
            is_output=is_output,
            is_metadata=is_metadata,
            is_loopback=is_loopback,
            formats=formats,
            modes=modes,
        )
    except OSError:
        return None
    finally:
        os.close(fd)


def list_v4l2_devices() -> list[V4L2DeviceInfo]:
    """Enumerate all /dev/video* nodes sorted numerically."""
    dev_dir = Path("/dev")
    nodes = sorted(
        (p for p in dev_dir.glob("video*") if p.name[5:].isdigit()),
        key=lambda p: int(p.name[5:]),
    )
    results: list[V4L2DeviceInfo] = []
    for node in nodes:
        info = query_v4l2_device(str(node))
        if info is not None:
            results.append(info)
    return results


def choose_best_pixel_format(device: V4L2DeviceInfo, width: int, height: int, prefer_mjpg: bool = True) -> tuple[str, float]:
    """Pick MJPG over YUYV for HD modes so USB webcams run at 30fps instead of 10fps."""
    matching = [m for m in device.modes if m.width == width and m.height == height]
    if not matching:
        if prefer_mjpg and "MJPG" in device.formats:
            return "MJPG", 30.0
        return (device.formats[0] if device.formats else "MJPG"), 30.0

    def score(mode: V4L2Mode) -> tuple[float, int]:
        max_fps = max(mode.fps_options) if mode.fps_options else 0.0
        mjpg_bonus = 1 if (prefer_mjpg and mode.pixel_format == "MJPG") else 0
        return (max_fps, mjpg_bonus)

    best = max(matching, key=score)
    return best.pixel_format, (max(best.fps_options) if best.fps_options else 30.0)


def list_v4l2_controls(device_path: str) -> list[V4L2Control]:
    """Enumerate adjustable user/camera controls for a V4L2 webcam."""
    controls: list[V4L2Control] = []
    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return controls

    try:
        cid = V4L2_CTRL_FLAG_NEXT_CTRL
        for _ in range(128):
            qbuf = bytearray(68)
            struct.pack_into("<I", qbuf, 0, cid)
            try:
                fcntl.ioctl(fd, VIDIOC_QUERYCTRL, qbuf)
            except OSError:
                break
            id_, ctype, name_b, mn, mx, step, default, flags = struct.unpack_from("<II32siiiiI", qbuf, 0)
            cid = id_ | V4L2_CTRL_FLAG_NEXT_CTRL
            if ctype == V4L2_CTRL_TYPE_CTRL_CLASS or (flags & V4L2_CTRL_FLAG_DISABLED):
                continue
            name = name_b.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
            val = default
            try:
                gbuf = bytearray(8)
                struct.pack_into("<Ii", gbuf, 0, id_, 0)
                fcntl.ioctl(fd, VIDIOC_G_CTRL, gbuf)
                _, val = struct.unpack("<Ii", gbuf)
            except OSError:
                pass
            controls.append(
                V4L2Control(
                    id=id_,
                    name=name,
                    slug=_slugify(name),
                    ctrl_type=ctype,
                    minimum=mn,
                    maximum=mx,
                    step=max(1, step),
                    default=default,
                    value=val,
                    flags=flags,
                )
            )
    finally:
        os.close(fd)
    return controls


def set_v4l2_control(device_path: str, control_id_or_name: int | str, value: int) -> tuple[bool, str]:
    """Set a V4L2 control by numeric ID, alias (e.g. 'brightness'), or slug."""
    controls = list_v4l2_controls(device_path)
    target: V4L2Control | None = None
    if isinstance(control_id_or_name, int):
        target = next((c for c in controls if c.id == control_id_or_name), None)
    else:
        key = control_id_or_name.strip().lower()
        cid = COMMON_CONTROLS.get(key)
        for c in controls:
            if (cid is not None and c.id == cid) or c.slug == key or c.name.lower() == key:
                target = c
                break
    if target is None:
        return False, f"Control '{control_id_or_name}' not found on {device_path}"
    if target.flags & V4L2_CTRL_FLAG_READ_ONLY:
        return False, f"Control '{target.name}' is read-only"

    clamped = max(target.minimum, min(target.maximum, int(value)))
    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        return False, f"Failed to open {device_path}: {exc}"
    try:
        buf = bytearray(8)
        struct.pack_into("<Ii", buf, 0, target.id, clamped)
        fcntl.ioctl(fd, VIDIOC_S_CTRL, buf)
        return True, f"Set {target.name} = {clamped}"
    except OSError as exc:
        return False, f"VIDIOC_S_CTRL failed for {target.name}: {exc}"
    finally:
        os.close(fd)
