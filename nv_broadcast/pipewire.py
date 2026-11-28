"""PipeWire virtual microphone/speaker orchestration and filter-chain configs."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VIRTUAL_MIC_SINK_NAME = "nv_broadcast_mic_bus"
VIRTUAL_MIC_SOURCE_NAME = "nv_broadcast_mic"
VIRTUAL_SPEAKER_SINK_NAME = "nv_broadcast_speakers"
VIRTUAL_NODE_NAMES = (
    VIRTUAL_MIC_SINK_NAME,
    VIRTUAL_MIC_SOURCE_NAME,
    VIRTUAL_SPEAKER_SINK_NAME,
)


def is_virtual_broadcast_node(name: str) -> bool:
    """Return True if a PipeWire node/device name belongs to Broadcast's own virtual devices."""
    low = (name or "").strip().lower()
    if not low:
        return False
    return any(v in low for v in VIRTUAL_NODE_NAMES)


def build_isolated_pw_cat_cmd(
    mode: str,
    node_name: str,
    rate: int = 48000,
    channels: int = 2,
    latency_ms: int = 20,
) -> list[str]:
    """Build a pw-cat command isolated from WirePlumber auto-routing/reconnection.

    Setting both `node.autoconnect = false` and `node.dont-reconnect = true` ensures
    WirePlumber never auto-moves a Broadcast capture/playback stream onto another
    sink's monitor or back into a virtual sink (preventing acoustic/digital feedback loops).
    """
    flag = "--record" if mode == "record" else "--playback"
    props = f'{{ node.name = "{node_name}" node.autoconnect = false node.dont-reconnect = true }}'
    return [
        "pw-cat",
        flag,
        "--raw",
        "--format=s16",
        f"--rate={int(rate)}",
        f"--channels={int(channels)}",
        f"--latency={int(latency_ms)}ms",
        "-P",
        props,
        "-",
    ]


def list_pw_ports(direction: str = "output", timeout: float = 2.5) -> list[str]:
    """List PipeWire port names via `pw-link -o` (output) or `pw-link -i` (input)."""
    if not shutil.which("pw-link"):
        return []
    flag = "-o" if direction == "output" else "-i"
    try:
        proc = subprocess.run(
            ["pw-link", flag],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        return []


def parse_pw_links(output: str) -> list[tuple[str, str]]:
    """Parse `pw-link -l` output into directed `(src_port, dst_port)` tuples."""
    edges: list[tuple[str, str]] = []
    current_src = ""
    for line in output.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            current_src = line.strip()
        elif "|->" in line and current_src:
            dst = line.split("|->", 1)[1].strip()
            if dst:
                edges.append((current_src, dst))
    return edges


def list_pw_links(timeout: float = 2.5) -> list[tuple[str, str]]:
    """Return active PipeWire links as `(src_port, dst_port)` pairs."""
    if not shutil.which("pw-link"):
        return []
    try:
        proc = subprocess.run(
            ["pw-link", "-l"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return parse_pw_links(proc.stdout)
    except (OSError, subprocess.TimeoutExpired):
        return []


def is_pw_link_active(
    src_node: str,
    dst_node: str,
    links: list[tuple[str, str]] | None = None,
) -> bool:
    """Return True if at least one active PipeWire link connects `src_node` -> `dst_node`."""
    if not src_node or not dst_node:
        return False
    active = links if links is not None else list_pw_links()
    src_prefix = f"{src_node}:"
    dst_prefix = f"{dst_node}:"
    return any(s.startswith(src_prefix) and d.startswith(dst_prefix) for s, d in active)


def select_port_pairs(
    src_node: str,
    dst_node: str,
    out_ports: list[str],
    in_ports: list[str],
    *,
    src_prefix: str = "",
    dst_prefix: str = "",
) -> list[tuple[str, str]]:
    """Pair matching PipeWire ports between src_node and dst_node deterministically."""
    src_head = f"{src_node}:"
    dst_head = f"{dst_node}:"
    src_list = [p for p in out_ports if p.startswith(src_head) and p[len(src_head):].startswith(src_prefix)]
    dst_list = [p for p in in_ports if p.startswith(dst_head) and p[len(dst_head):].startswith(dst_prefix)]
    if not src_list or not dst_list:
        return []

    def _chan(port: str) -> str:
        suffix = port.split(":", 1)[1]
        return suffix.rsplit("_", 1)[-1].upper() if "_" in suffix else suffix.upper()

    src_by_ch = {_chan(p): p for p in src_list}
    dst_by_ch = {_chan(p): p for p in dst_list}

    # Exact stereo or mono channel matches first
    pairs: list[tuple[str, str]] = []
    for ch in ("FL", "FR", "MONO"):
        if ch in src_by_ch and ch in dst_by_ch:
            pairs.append((src_by_ch[ch], dst_by_ch[ch]))
    if pairs:
        return pairs

    # Stereo source -> single mono input (use FL / first channel to avoid +6 dB summing)
    if len(dst_list) == 1:
        return [(src_by_ch.get("FL", src_list[0]), dst_list[0])]

    # Mono source -> stereo sink (feed both L and R)
    if len(src_list) == 1:
        return [(src_list[0], dp) for dp in dst_list[:2]]

    # Positional fallback (e.g. AUX0/AUX1)
    return list(zip(src_list, dst_list))


def link_pw_nodes(
    src_node: str,
    dst_node: str,
    *,
    src_prefix: str = "",
    dst_prefix: str = "",
    allow_virtual_src: bool = False,
    allow_virtual_dst: bool = False,
) -> tuple[bool, list[tuple[str, str]]]:
    """Explicitly link two PipeWire nodes with hard anti-loop guards."""
    if not src_node or not dst_node or src_node == dst_node:
        return False, []
    if not allow_virtual_src and is_virtual_broadcast_node(src_node):
        return False, []
    if not allow_virtual_dst and is_virtual_broadcast_node(dst_node):
        return False, []
    if not shutil.which("pw-link"):
        return False, []

    pairs = select_port_pairs(
        src_node,
        dst_node,
        list_pw_ports("output"),
        list_pw_ports("input"),
        src_prefix=src_prefix,
        dst_prefix=dst_prefix,
    )
    if not pairs:
        return False, []

    linked: list[tuple[str, str]] = []
    for sp, dp in pairs:
        try:
            res = subprocess.run(
                ["pw-link", sp, dp],
                capture_output=True,
                text=True,
                timeout=2.5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, linked
        if res.returncode == 0 or "file exists" in (res.stderr or "").lower():
            linked.append((sp, dp))
        else:
            return False, linked
    return bool(linked), linked


@dataclass
class VirtualAudioState:
    mic_sink_module_id: int | None = None
    mic_source_module_id: int | None = None
    speaker_sink_module_id: int | None = None
    filter_chain_pid: int | None = None
    active_backend: str = ""
    warnings: list[str] = field(default_factory=list)


def _run_pactl(args: list[str], timeout: float = 4.0) -> tuple[int, str, str]:
    if not shutil.which("pactl"):
        return 1, "", "pactl not found"
    try:
        proc = subprocess.run(
            ["pactl", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def generate_rnnoise_filter_chain_conf(
    ladspa_plugin: str = "librnnoise_ladspa",
    vad_threshold: float = 50.0,
    node_description: str = "NVIDIA Broadcast (Microphone)",
    target_source: str | None = None,
) -> str:
    """Generate a native PipeWire filter-chain config for RNNoise LADSPA."""
    vad = max(0.0, min(95.0, float(vad_threshold)))
    target_line = f'\n                target.object = "{target_source}"' if target_source and target_source != "auto" else ""
    return f"""# Generated by nv-broadcast (RNNoise LADSPA filter-chain)
context.modules = [
    {{
        name = libpipewire-module-filter-chain
        flags = [ nofail ]
        args = {{
            node.description = "{node_description}"
            media.name       = "{node_description}"
            filter.graph = {{
                nodes = [
                    {{
                        type   = ladspa
                        name   = rnnoise
                        plugin = "{ladspa_plugin}"
                        label  = noise_suppressor_stereo
                        control = {{
                            "VAD Threshold (%)" {vad:.1f}
                            "VAD Grace Period (ms)" 200
                            "Retroactive VAD Grace (ms)" 0
                        }}
                    }}
                ]
            }}
            audio.rate     = 48000
            audio.position = [ FL FR ]
            capture.props = {{
                node.name    = "nv_broadcast_rnnoise_in"
                node.passive = true{target_line}
            }}
            playback.props = {{
                node.name   = "{VIRTUAL_MIC_SOURCE_NAME}"
                media.class = Audio/Source
            }}
        }}
    }}
]
"""


def generate_deepfilter_filter_chain_conf(
    ladspa_plugin: str = "libdeep_filter_ladspa",
    attenuation_limit_db: float = 80.0,
    node_description: str = "NVIDIA Broadcast (Microphone)",
    target_source: str | None = None,
) -> str:
    """Generate a native PipeWire filter-chain config for DeepFilterNet LADSPA."""
    atten = max(6.0, min(100.0, float(attenuation_limit_db)))
    target_line = f'\n                target.object = "{target_source}"' if target_source and target_source != "auto" else ""
    return f"""# Generated by nv-broadcast (DeepFilterNet LADSPA filter-chain)
context.modules = [
    {{
        name = libpipewire-module-filter-chain
        flags = [ nofail ]
        args = {{
            node.description = "{node_description}"
            media.name       = "{node_description}"
            filter.graph = {{
                nodes = [
                    {{
                        type   = ladspa
                        name   = deep_filter_mono
                        plugin = "{ladspa_plugin}"
                        label  = deep_filter_mono
                        control = {{
                            "Attenuation Limit (dB)" {atten:.1f}
                        }}
                    }}
                ]
            }}
            audio.rate     = 48000
            audio.position = [ MONO ]
            capture.props = {{
                node.name    = "nv_broadcast_df_in"
                node.passive = true{target_line}
            }}
            playback.props = {{
                node.name   = "{VIRTUAL_MIC_SOURCE_NAME}"
                media.class = Audio/Source
            }}
        }}
    }}
]
"""


def generate_webrtc_echo_cancel_conf(
    node_description: str = "NVIDIA Broadcast (Microphone)",
    noise_suppression: bool = True,
    echo_cancellation: bool = True,
    high_pass: bool = True,
    auto_gain: bool = True,
    target_source: str | None = None,
) -> str:
    """Generate a native PipeWire module-echo-cancel config (uses libspa-aec-webrtc.so)."""
    target_line = f'\n                target.object = "{target_source}"' if target_source and target_source != "auto" else ""
    return f"""# Generated by nv-broadcast (PipeWire native WebRTC NS/AEC/HPF/AGC)
context.modules = [
    {{
        name = libpipewire-module-echo-cancel
        args = {{
            library.name     = aec/libspa-aec-webrtc
            node.latency     = 1024/48000
            monitor.mode     = true
            aec.args = {{
                webrtc.gain_control          = {"true" if auto_gain else "false"}
                webrtc.extended_filter       = true
                webrtc.high_pass_filter      = {"true" if high_pass else "false"}
                webrtc.noise_suppression     = {"true" if noise_suppression else "false"}
                webrtc.voice_detection       = true
                webrtc.echo_cancellation     = {"true" if echo_cancellation else "false"}
            }}
            capture.props = {{
                node.name        = "nv_broadcast_webrtc_capture"
                node.description = "NVIDIA Broadcast Capture"{target_line}
            }}
            source.props = {{
                node.name        = "{VIRTUAL_MIC_SOURCE_NAME}"
                node.description = "{node_description}"
                media.class      = Audio/Source
            }}
            sink.props = {{
                node.name        = "nv_broadcast_webrtc_ref"
                node.description = "NVIDIA Broadcast AEC Reference"
            }}
            playback.props = {{
                node.name        = "nv_broadcast_webrtc_playback"
                node.description = "NVIDIA Broadcast AEC Playback"
            }}
        }}
    }}
]
"""


def pipewire_user_conf_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "pipewire" / "pipewire.conf.d"


def write_persistent_filter_conf(content: str, filename: str = "99-nv-broadcast.conf") -> Path:
    conf_dir = pipewire_user_conf_dir()
    conf_dir.mkdir(parents=True, exist_ok=True)
    path = conf_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def remove_persistent_filter_conf(filename: str = "99-nv-broadcast.conf") -> bool:
    path = pipewire_user_conf_dir() / filename
    if path.is_file():
        path.unlink()
        return True
    return False


class PipeWireVirtualManager:
    """Create and cleanly remove runtime PipeWire virtual mic and speaker nodes."""

    def __init__(self) -> None:
        self.state = VirtualAudioState()
        self._filter_proc: subprocess.Popen[bytes] | None = None

    def find_existing_virtual_nodes(self) -> dict[str, bool]:
        rc_s, out_s, _ = _run_pactl(["list", "short", "sources"])
        rc_k, out_k, _ = _run_pactl(["list", "short", "sinks"])
        return {
            "virtual_mic": (rc_s == 0 and VIRTUAL_MIC_SOURCE_NAME in out_s),
            "virtual_mic_bus": (rc_k == 0 and VIRTUAL_MIC_SINK_NAME in out_k),
            "virtual_speakers": (rc_k == 0 and VIRTUAL_SPEAKER_SINK_NAME in out_k),
        }

    def create_virtual_microphone(self, description: str = "NVIDIA Broadcast (Microphone)") -> tuple[bool, str]:
        """Create a clean PipeWire virtual mic (`Audio/Source/Virtual`) visible in Discord/Meet/Zoom without cluttering Sinks."""
        existing = self.find_existing_virtual_nodes()
        if existing["virtual_mic"]:
            return True, f"Virtual microphone '{description}' already active"

        # Escape spaces for pactl load-module so PipeWire displays clean names in Discord/Meet/Zoom
        esc_desc = description.replace('"', "").replace(" ", r"\ ")
        rc, out, err = _run_pactl(
            [
                "load-module",
                "module-null-sink",
                "media.class=Audio/Source/Virtual",
                f"sink_name={VIRTUAL_MIC_SOURCE_NAME}",
                f'sink_properties=device.description="{esc_desc}"',
                "channel_map=mono",
                "rate=48000",
            ]
        )
        if rc != 0:
            return False, f"Failed to create virtual mic: {err}"
        try:
            self.state.mic_source_module_id = int(out.strip())
        except ValueError:
            pass
        return True, f"Created virtual microphone '{description}' ({VIRTUAL_MIC_SOURCE_NAME})"

    def destroy_virtual_microphone(self, force: bool = False) -> None:
        if self._filter_proc is not None and self._filter_proc.poll() is None:
            self._filter_proc.terminate()
            try:
                self._filter_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._filter_proc.kill()
            self._filter_proc = None
            self.state.filter_chain_pid = None

        for mod_id in (self.state.mic_source_module_id, self.state.mic_sink_module_id):
            if mod_id is not None:
                _run_pactl(["unload-module", str(mod_id)])
        self.state.mic_source_module_id = None
        self.state.mic_sink_module_id = None

        if force:
            self._unload_matching_modules(VIRTUAL_MIC_SOURCE_NAME)
            self._unload_matching_modules(VIRTUAL_MIC_SINK_NAME)

    def create_virtual_speakers(self, description: str = "NVIDIA Broadcast (Speakers)") -> tuple[bool, str]:
        """Create a real PipeWire virtual speaker sink that apps can route audio to."""
        existing = self.find_existing_virtual_nodes()
        if existing["virtual_speakers"]:
            return True, f"Virtual speakers '{description}' already active"

        esc_desc = description.replace('"', "").replace(" ", r"\ ")
        rc, out, err = _run_pactl(
            [
                "load-module",
                "module-null-sink",
                f"sink_name={VIRTUAL_SPEAKER_SINK_NAME}",
                f'sink_properties=device.description="{esc_desc}"',
                "channels=2",
                "rate=48000",
            ]
        )
        if rc != 0:
            return False, f"Failed to create virtual speakers: {err}"
        try:
            self.state.speaker_sink_module_id = int(out.strip())
        except ValueError:
            pass
        return True, f"Created virtual speakers '{description}' ({VIRTUAL_SPEAKER_SINK_NAME})"

    def destroy_virtual_speakers(self, force: bool = False) -> None:
        if self.state.speaker_sink_module_id is not None:
            _run_pactl(["unload-module", str(self.state.speaker_sink_module_id)])
            self.state.speaker_sink_module_id = None
        if force:
            self._unload_matching_modules(VIRTUAL_SPEAKER_SINK_NAME)

    def launch_runtime_filter_chain(self, conf_text: str, runtime_dir: Path | None = None) -> tuple[bool, str]:
        """Run a standalone PipeWire filter-chain client (`pipewire -c <file>`) without restarting PipeWire."""
        if not shutil.which("pipewire"):
            return False, "pipewire binary not found"
        base = runtime_dir or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "nv-broadcast"
        base.mkdir(parents=True, exist_ok=True)
        conf_path = base / "filter-chain.conf"
        conf_path.write_text(conf_text, encoding="utf-8")
        try:
            self._filter_proc = subprocess.Popen(
                ["pipewire", "-c", str(conf_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.state.filter_chain_pid = self._filter_proc.pid
            return True, f"Started PipeWire filter-chain (pid {self._filter_proc.pid})"
        except OSError as exc:
            return False, f"Failed to start PipeWire filter-chain: {exc}"

    def _unload_matching_modules(self, needle: str) -> None:
        rc, out, _ = _run_pactl(["list", "short", "modules"])
        if rc != 0:
            return
        for line in out.splitlines():
            if needle in line:
                parts = line.split("\t")
                if parts and parts[0].strip().isdigit():
                    _run_pactl(["unload-module", parts[0].strip()])

    def teardown_all(self) -> None:
        self.destroy_virtual_microphone(force=True)
        self.destroy_virtual_speakers(force=True)
