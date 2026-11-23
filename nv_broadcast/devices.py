"""Device enumeration and hardware/backend capability probing for Linux."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import get_models_status
from .v4l2_ctl import V4L2DeviceInfo, list_v4l2_devices


@dataclass
class AudioDevice:
    id: str
    name: str
    description: str
    kind: str  # "microphone" | "speaker"
    sample_spec: str = ""
    state: str = ""
    is_monitor: bool = False
    is_virtual: bool = False
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GPUInfo:
    available: bool
    name: str = ""
    driver_version: str = ""
    vram_mib: int = 0
    compute_capability: str = ""
    cuda_toolkit: str = ""
    rtx_capable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeCapabilities:
    gpu: GPUInfo
    pipewire_version: str = ""
    has_pactl: bool = False
    has_pw_cat: bool = False
    has_pw_loopback: bool = False
    has_pipewire_webrtc_aec: bool = False
    has_rnnoise_ladspa: bool = False
    rnnoise_ladspa_path: str = ""
    has_deepfilter_ladspa: bool = False
    deepfilter_ladspa_path: str = ""
    has_ffmpeg: bool = False
    ffmpeg_filters: list[str] = field(default_factory=list)
    has_v4l2loopback_module: bool = False
    v4l2loopback_loaded: bool = False
    python_packages: dict[str, bool] = field(default_factory=dict)
    onnx_providers: list[str] = field(default_factory=list)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    recommended_audio_backend: str = "dsp"
    recommended_segmentation_backend: str = "fast_portrait"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gpu"] = self.gpu.to_dict()
        return d


def _run_cmd(args: list[str], timeout: float = 4.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def parse_pactl_short(output: str, kind: str, default_name: str = "") -> list[AudioDevice]:
    """Parse `pactl list short sources` or `pactl list short sinks` output."""
    devices: list[AudioDevice] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
        if len(parts) < 2:
            continue
        dev_id = parts[0]
        name = parts[1]
        sample_spec = parts[3] if len(parts) > 3 else ""
        state = parts[4] if len(parts) > 4 else ""
        is_monitor = name.endswith(".monitor")
        is_virtual = (
            "nv_broadcast" in name.lower()
            or "nvbroadcast" in name.lower()
            or "nvidia_broadcast" in name.lower()
            or "effect_output" in name.lower()
        )
        desc = _pretty_audio_name(name)
        devices.append(
            AudioDevice(
                id=dev_id,
                name=name,
                description=desc,
                kind=kind,
                sample_spec=sample_spec,
                state=state,
                is_monitor=is_monitor,
                is_virtual=is_virtual,
                is_default=(name == default_name),
            )
        )
    return devices


def _pretty_audio_name(node_name: str) -> str:
    if "nv_broadcast_mic" in node_name:
        return "NVIDIA Broadcast (Microphone)"
    if "nv_broadcast_speaker" in node_name:
        return "NVIDIA Broadcast (Speakers)"
    if "HiFi__Mic" in node_name:
        return "Built-in Laptop Microphone"
    if "HiFi__Speaker" in node_name:
        return "Built-in Laptop Speakers"
    if "HiFi__HDMI" in node_name:
        tail = node_name.split("HiFi__", 1)[-1].split("__", 1)[0]
        return f"HDMI Audio ({tail})"
    if "bluez" in node_name.lower():
        return f"Bluetooth Audio ({node_name.split('.')[-1]})"
    if "usb" in node_name.lower():
        return f"USB Audio ({node_name.split('.')[1] if '.' in node_name else node_name})"
    return node_name


def _fetch_pactl_metadata(kind: str) -> dict[str, dict[str, str]]:
    subcmd = "sources" if kind == "microphone" else "sinks"
    rc, out, _ = _run_cmd(["pactl", "list", subcmd])
    if rc != 0:
        return {}
    meta: dict[str, dict[str, str]] = {}
    current_name = ""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:"):
            current_name = stripped.split(":", 1)[1].strip()
            meta.setdefault(current_name, {})
        elif current_name:
            if stripped.startswith("Description:"):
                meta[current_name]["description"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Active Port:"):
                meta[current_name]["active_port"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("device.bus ="):
                meta[current_name]["bus"] = stripped.split("=", 1)[1].strip().strip('"')
            elif stripped.startswith("device.icon_name ="):
                meta[current_name]["icon"] = stripped.split("=", 1)[1].strip().strip('"')
    return meta


def list_bluetooth_audio_cards() -> list[dict[str, Any]]:
    """Detect connected Bluetooth audio cards and their available A2DP / HFP headset profiles."""
    if not shutil.which("pactl"):
        return []
    rc, out, _ = _run_cmd(["pactl", "list", "cards"])
    if rc != 0:
        return []
    cards: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    in_profiles = False
    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("Card #"):
            if cur and cur.get("name", "").startswith("bluez_card."):
                cards.append(cur)
            cur = {"name": "", "description": "", "active_profile": "", "profiles": []}
            in_profiles = False
        elif cur is not None:
            if line.startswith("Name:"):
                cur["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("device.description =") or line.startswith("device.alias ="):
                if not cur["description"]:
                    cur["description"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("Active Profile:"):
                cur["active_profile"] = line.split(":", 1)[1].strip()
                in_profiles = False
            elif line == "Profiles:":
                in_profiles = True
            elif line.startswith(("Ports:", "Properties:")):
                in_profiles = False
            elif in_profiles and ":" in line:
                prof_name = line.split(":", 1)[0].strip()
                if prof_name and "available: no" not in line.lower():
                    cur["profiles"].append(prof_name)
    if cur and cur.get("name", "").startswith("bluez_card."):
        cards.append(cur)
    return cards


def set_bluetooth_headset_mode(card_name: str, enable_mic: bool) -> tuple[bool, str]:
    """Switch a connected Bluetooth headphone card between HFP/mSBC (with mic) and A2DP (stereo)."""
    cards = list_bluetooth_audio_cards()
    card = next((c for c in cards if c["name"] == card_name), None)
    if not card:
        return False, f"Bluetooth audio card not found: {card_name}"
    profs = card["profiles"]
    if enable_mic:
        wanted = [p for p in profs if "headset-head-unit-msbc" in p] or [
            p for p in profs if "headset-head-unit" in p or "handsfree" in p
        ]
    else:
        wanted = [p for p in profs if p.startswith("a2dp-sink")]
    if not wanted:
        return False, f"No matching {'headset mic' if enable_mic else 'A2DP stereo'} profile on {card['description']}"
    rc, _, err = _run_cmd(["pactl", "set-card-profile", card_name, wanted[0]])
    if rc != 0:
        return False, f"Failed to switch profile: {err}"
    return True, f"Switched {card['description']} to {wanted[0]}"


def list_audio_devices(include_monitors: bool = False, include_virtual: bool = False) -> dict[str, list[AudioDevice]]:
    """Enumerate microphones and speakers from PipeWire via pactl, labeling headphones/USB/BT clearly."""
    if not shutil.which("pactl"):
        return {"microphones": [], "speakers": []}

    _, def_src, _ = _run_cmd(["pactl", "get-default-source"])
    _, def_sink, _ = _run_cmd(["pactl", "get-default-sink"])
    _, src_out, _ = _run_cmd(["pactl", "list", "short", "sources"])
    _, sink_out, _ = _run_cmd(["pactl", "list", "short", "sinks"])

    src_meta = _fetch_pactl_metadata("microphone")
    sink_meta = _fetch_pactl_metadata("speaker")

    mics = parse_pactl_short(src_out, "microphone", def_src)
    speakers = parse_pactl_short(sink_out, "speaker", def_sink)

    def _enrich(dev: AudioDevice, meta_map: dict[str, dict[str, str]]) -> None:
        m = meta_map.get(dev.name, {})
        base_desc = m.get("description") or dev.description
        port = m.get("active_port", "").lower()
        bus = m.get("bus", "").lower()
        icon = m.get("icon", "").lower()
        tags: list[str] = []
        if "bluez" in dev.name.lower() or bus == "bluetooth":
            tags.append("Bluetooth")
        elif "usb" in dev.name.lower() or bus == "usb":
            tags.append("USB")
        if "headphone" in port or "headset" in port or "headphone" in icon or "headset" in icon:
            if "headphone" not in base_desc.lower() and "headset" not in base_desc.lower():
                tags.append("Headphones")
        if tags:
            dev.description = f"{base_desc} [{' · '.join(tags)}]"
        else:
            dev.description = base_desc

    for d in mics:
        _enrich(d, src_meta)
    for d in speakers:
        _enrich(d, sink_meta)

    if not include_monitors:
        mics = [d for d in mics if not d.is_monitor]
    if not include_virtual:
        mics = [d for d in mics if not d.is_virtual]
        speakers = [d for d in speakers if not d.is_virtual]

    return {"microphones": mics, "speakers": speakers}


def list_cameras(include_metadata: bool = False, include_loopback: bool = False) -> list[V4L2DeviceInfo]:
    """Enumerate physical V4L2 webcams (excluding metadata nodes like /dev/video1 by default)."""
    all_devs = list_v4l2_devices()
    out: list[V4L2DeviceInfo] = []
    for dev in all_devs:
        if dev.is_metadata and not include_metadata:
            continue
        if dev.is_loopback and not include_loopback:
            continue
        if not dev.is_capture and not include_metadata and not (dev.is_loopback and include_loopback):
            continue
        out.append(dev)
    return out


def find_loopback_cameras() -> list[V4L2DeviceInfo]:
    return [d for d in list_v4l2_devices() if d.is_loopback]


def probe_gpu() -> GPUInfo:
    if not shutil.which("nvidia-smi"):
        return GPUInfo(available=False)
    rc, out, _ = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0 or not out:
        return GPUInfo(available=False)
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if len(parts) > 0 else "NVIDIA GPU"
    driver = parts[1] if len(parts) > 1 else ""
    try:
        vram = int(float(parts[2])) if len(parts) > 2 else 0
    except ValueError:
        vram = 0
    cc = parts[3] if len(parts) > 3 else ""
    cuda_ver = ""
    if shutil.which("nvcc"):
        crc, cout, _ = _run_cmd(["nvcc", "--version"])
        if crc == 0:
            for line in cout.splitlines():
                if "release" in line:
                    cuda_ver = line.split("release", 1)[1].split(",")[0].strip()
                    break
    rtx = "rtx" in name.lower() or (cc and float(cc.split(".")[0]) >= 7)
    return GPUInfo(
        available=True,
        name=name,
        driver_version=driver,
        vram_mib=vram,
        compute_capability=cc,
        cuda_toolkit=cuda_ver,
        rtx_capable=bool(rtx),
    )


def query_gpu_live() -> dict[str, Any]:
    """Read live GPU utilization, VRAM usage, and temperature for the header GPU pill."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "utilization_pct": 0, "memory_used_mib": 0, "memory_total_mib": 0, "temp_c": 0}
    rc, out, _ = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=2.0,
    )
    if rc != 0 or not out:
        return {"available": False, "utilization_pct": 0, "memory_used_mib": 0, "memory_total_mib": 0, "temp_c": 0}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    try:
        return {
            "available": True,
            "name": parts[0].replace("NVIDIA GeForce ", ""),
            "utilization_pct": int(float(parts[1])) if len(parts) > 1 else 0,
            "memory_used_mib": int(float(parts[2])) if len(parts) > 2 else 0,
            "memory_total_mib": int(float(parts[3])) if len(parts) > 3 else 0,
            "temp_c": int(float(parts[4])) if len(parts) > 4 else 0,
        }
    except ValueError:
        return {"available": True, "name": parts[0], "utilization_pct": 0, "memory_used_mib": 0, "memory_total_mib": 0, "temp_c": 0}


def _find_ladspa_plugin(candidates: list[str]) -> str:
    search_dirs = [
        Path("/usr/lib64/ladspa"),
        Path("/usr/lib/ladspa"),
        Path("/usr/local/lib/ladspa"),
        Path("/usr/local/lib64/ladspa"),
        Path.home() / ".ladspa",
    ]
    env_path = os.environ.get("LADSPA_PATH", "")
    for entry in env_path.split(":"):
        if entry:
            search_dirs.insert(0, Path(entry))
    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in candidates:
            p = d / name
            if p.is_file():
                return str(p)
    return ""


def probe_capabilities() -> RuntimeCapabilities:
    gpu = probe_gpu()
    pw_ver = ""
    if shutil.which("pipewire"):
        rc, out, _ = _run_cmd(["pipewire", "--version"])
        if rc == 0:
            for line in out.splitlines():
                if "libpipewire" in line:
                    pw_ver = line.split()[-1]
                    break

    webrtc_aec = any(
        Path(p).is_file()
        for p in (
            "/usr/lib64/spa-0.2/aec/libspa-aec-webrtc.so",
            "/usr/lib/x86_64-linux-gnu/spa-0.2/aec/libspa-aec-webrtc.so",
        )
    )
    rnnoise_path = _find_ladspa_plugin(["librnnoise_ladspa.so"])
    df_path = _find_ladspa_plugin(["libdeep_filter_ladspa.so"])

    has_ffmpeg = bool(shutil.which("ffmpeg"))
    ff_filters: list[str] = []
    if has_ffmpeg:
        rc, out, _ = _run_cmd(["ffmpeg", "-hide_banner", "-filters"])
        if rc == 0:
            wanted = ("arnndn", "afftdn", "anlmdn", "highpass", "acompressor", "agate", "dynaudnorm")
            for line in out.splitlines():
                tokens = line.split()
                if len(tokens) >= 2 and tokens[1] in wanted:
                    ff_filters.append(tokens[1])

    mod_rc, _, _ = _run_cmd(["modinfo", "v4l2loopback"])
    has_v4l2loopback_mod = mod_rc == 0
    v4l2loopback_loaded = False
    if Path("/proc/modules").is_file():
        try:
            v4l2loopback_loaded = "v4l2loopback" in Path("/proc/modules").read_text(encoding="utf-8")
        except OSError:
            pass

    pkg_names = [
        "numpy",
        "scipy",
        "cv2",
        "PIL",
        "onnxruntime",
        "mediapipe",
        "pyvirtualcam",
        "pyrnnoise",
        "df",
        "torch",
        "gi",
        "pyaudio",
    ]
    pkgs = {name: (importlib.util.find_spec(name) is not None) for name in pkg_names}
    models = get_models_status()

    onnx_providers: list[str] = []
    if pkgs["onnxruntime"]:
        try:
            import onnxruntime as ort  # type: ignore

            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
            onnx_providers = list(ort.get_available_providers())
        except Exception:
            pass

    if df_path or pkgs["df"]:
        rec_audio = "deepfilternet"
    elif pkgs["pyrnnoise"]:
        rec_audio = "pyrnnoise"
    elif rnnoise_path:
        rec_audio = "rnnoise_ladspa"
    elif webrtc_aec:
        rec_audio = "webrtc_aec"
    elif "afftdn" in ff_filters:
        rec_audio = "ffmpeg_afftdn"
    else:
        rec_audio = "dsp"

    if "CUDAExecutionProvider" in onnx_providers and models["rvm_mobilenetv3"]["installed"]:
        rec_seg = "rvm_cuda"
    elif pkgs["mediapipe"] and models["selfie_segmenter"]["installed"]:
        rec_seg = "mediapipe"
    elif pkgs["onnxruntime"] and models["rvm_mobilenetv3"]["installed"]:
        rec_seg = "rvm_cpu"
    else:
        rec_seg = "fast_portrait"

    return RuntimeCapabilities(
        gpu=gpu,
        pipewire_version=pw_ver,
        has_pactl=bool(shutil.which("pactl")),
        has_pw_cat=bool(shutil.which("pw-cat")),
        has_pw_loopback=bool(shutil.which("pw-loopback")),
        has_pipewire_webrtc_aec=webrtc_aec,
        has_rnnoise_ladspa=bool(rnnoise_path),
        rnnoise_ladspa_path=rnnoise_path,
        has_deepfilter_ladspa=bool(df_path),
        deepfilter_ladspa_path=df_path,
        has_ffmpeg=has_ffmpeg,
        ffmpeg_filters=sorted(set(ff_filters)),
        has_v4l2loopback_module=has_v4l2loopback_mod,
        v4l2loopback_loaded=v4l2loopback_loaded,
        python_packages=pkgs,
        onnx_providers=onnx_providers,
        models=models,
        recommended_audio_backend=rec_audio,
        recommended_segmentation_backend=rec_seg,
    )
