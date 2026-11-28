"""Real-time PipeWire Microphone & Speaker neural audio engine + Waveform Mic/Speaker Tests."""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .audio_dsp import (
    HAS_NUMPY,
    AudioMeterReading,
    AudioProcessor,
    _pcm16_to_floats,
    compute_rms_db,
    compute_waveform_peaks,
    generate_sample_clip,
    save_pcm16_wav,
)
from .config import BroadcastConfig
from .devices import list_audio_devices, probe_capabilities
from .pipewire import (
    VIRTUAL_MIC_SINK_NAME,
    VIRTUAL_MIC_SOURCE_NAME,
    VIRTUAL_SPEAKER_SINK_NAME,
    PipeWireVirtualManager,
    _run_pactl,
    build_isolated_pw_cat_cmd,
    generate_deepfilter_filter_chain_conf,
    generate_rnnoise_filter_chain_conf,
    generate_webrtc_echo_cancel_conf,
    is_pw_link_active,
    is_virtual_broadcast_node,
    link_pw_nodes,
    list_pw_links,
)


def _read_chunk_nonblocking(
    fd: int,
    buf: bytearray,
    chunk_bytes: int,
    timeout: float = 0.5,
) -> tuple[bytes | None, bool]:
    """Read up to `chunk_bytes` from `fd` with a strict `select()` timeout.

    Returns `(data, False)` when a full chunk arrives, `(b"", False)` on EOF,
    or `(None, True)` when `timeout` elapses without enough bytes arriving.
    This guarantees audio worker loops stay responsive for hot-plug and
    suspend/resume link healing even while the PipeWire stream is idle.
    """
    deadline = time.monotonic() + max(0.01, timeout)
    while len(buf) < chunk_bytes:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            return None, True
        try:
            ready, _, _ = select.select([fd], [], [], remaining_time)
        except (OSError, ValueError):
            return b"", False
        if not ready:
            return None, True
        try:
            data = os.read(fd, chunk_bytes - len(buf))
        except BlockingIOError:
            return None, True
        except OSError:
            return b"", False
        if not data:
            return b"", False
        buf.extend(data)
    out = bytes(buf[:chunk_bytes])
    del buf[:chunk_bytes]
    return out, False

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore


@dataclass
class MicTestResult:
    duration_sec: float
    raw_wav_path: str
    processed_wav_path: str
    raw_rms_db: float
    processed_rms_db: float
    noise_floor_reduction_db: float
    raw_waveform: list[float] = field(default_factory=list)
    processed_waveform: list[float] = field(default_factory=list)


class AudioEngine:
    """Manages real-time PipeWire virtual microphone, virtual speakers, and A/B audio tests."""

    def __init__(self, config: BroadcastConfig) -> None:
        self.config = config
        self.pw = PipeWireVirtualManager()
        self.mic_processor = AudioProcessor(sample_rate=48000, frame_size=512)
        self.speaker_processor = AudioProcessor(sample_rate=48000, frame_size=512)
        self.speaker_processor_r = AudioProcessor(sample_rate=48000, frame_size=512)

        self._prev_default_source: str = ""
        self._prev_default_sink: str = ""

        self._mic_thread: threading.Thread | None = None
        self._mic_stop = threading.Event()
        self._mic_running = False
        self._mic_backend_used = ""

        self._spk_thread: threading.Thread | None = None
        self._spk_stop = threading.Event()
        self._spk_running = False

        self._play_proc: subprocess.Popen[bytes] | None = None
        self._meter_callbacks: list[Callable[[AudioMeterReading], None]] = []
        self._speaker_meter_callbacks: list[Callable[[AudioMeterReading], None]] = []

    @property
    def mic_running(self) -> bool:
        return self._mic_running

    @property
    def speaker_running(self) -> bool:
        return self._spk_running

    @property
    def active_mic_backend(self) -> str:
        return self._mic_backend_used or self.mic_processor.backend_name

    def add_meter_callback(self, cb: Callable[[AudioMeterReading], None]) -> None:
        self._meter_callbacks.append(cb)

    def add_speaker_meter_callback(self, cb: Callable[[AudioMeterReading], None]) -> None:
        self._speaker_meter_callbacks.append(cb)

    def _select_backend(self) -> str:
        wanted = self.config.data.get("audio_backend", "auto")
        if wanted != "auto":
            return wanted
        caps = probe_capabilities()
        if caps.has_deepfilter_ladspa:
            return "deepfilternet"
        if self.mic_processor.backend_name == "pyrnnoise" and caps.has_pw_cat and caps.has_pactl:
            return "pyrnnoise"
        if caps.has_rnnoise_ladspa:
            return "rnnoise_ladspa"
        if caps.has_pw_cat and caps.has_pactl:
            return "dsp"
        if caps.has_pipewire_webrtc_aec:
            return "webrtc_aec"
        return "dsp"

    def start_microphone(self) -> tuple[bool, str]:
        if self._mic_running:
            return True, f"Virtual microphone already running ({self._mic_backend_used})"

        mic_cfg = self.config.data["microphone"]
        backend = self._select_backend()
        caps = probe_capabilities()
        vname = mic_cfg.get("virtual_name", "NVIDIA Broadcast (Microphone)")
        target_dev = mic_cfg.get("device", "auto")

        # Optional native PipeWire LADSPA filter-chains (when installed & selected)
        if backend == "deepfilternet" and caps.has_deepfilter_ladspa:
            strength = float(mic_cfg["effects"]["noise_removal"].get("strength", 0.85))
            conf = generate_deepfilter_filter_chain_conf(
                ladspa_plugin=caps.deepfilter_ladspa_path,
                attenuation_limit_db=15.0 + 85.0 * strength,
                node_description=vname,
                target_source=target_dev,
            )
            ok, msg = self.pw.launch_runtime_filter_chain(conf)
            if ok:
                self._mic_running = True
                self._mic_backend_used = "deepfilternet"
                return True, msg

        if backend == "rnnoise_ladspa" and caps.has_rnnoise_ladspa:
            strength = float(mic_cfg["effects"]["noise_removal"].get("strength", 0.85))
            conf = generate_rnnoise_filter_chain_conf(
                ladspa_plugin=caps.rnnoise_ladspa_path,
                vad_threshold=15.0 + 75.0 * strength,
                node_description=vname,
                target_source=target_dev,
            )
            ok, msg = self.pw.launch_runtime_filter_chain(conf)
            if ok:
                self._mic_running = True
                self._mic_backend_used = "rnnoise_ladspa"
                return True, msg

        if backend == "webrtc_aec" and caps.has_pipewire_webrtc_aec:
            conf = generate_webrtc_echo_cancel_conf(
                node_description=vname,
                noise_suppression=bool(mic_cfg["effects"]["noise_removal"].get("enabled", True)),
                echo_cancellation=bool(mic_cfg["effects"]["room_echo_removal"].get("enabled", False)),
                high_pass=bool(mic_cfg.get("highpass_enabled", True)),
                auto_gain=bool(mic_cfg.get("auto_gain_enabled", True)),
                target_source=target_dev,
            )
            ok, msg = self.pw.launch_runtime_filter_chain(conf)
            if ok:
                self._mic_running = True
                self._mic_backend_used = "webrtc_aec"
                return True, msg

        # Live streaming PipeWire neural RNNoise + DSP pipeline (supports live sliders, Studio Voice, VU meter)
        if not shutil.which("pw-cat"):
            return False, "pw-cat not found; cannot start streaming virtual microphone"

        ok, msg = self.pw.create_virtual_microphone(description=vname)
        if not ok:
            return False, msg

        self._mic_stop.clear()
        self.mic_processor.reset()
        self._mic_running = True
        self._mic_backend_used = self.mic_processor.backend_name
        self._mic_thread = threading.Thread(
            target=self._run_mic_stream,
            name="nv-broadcast-mic",
            daemon=True,
        )
        self._mic_thread.start()
        return True, f"Started virtual microphone '{vname}' (backend: {self._mic_backend_used})"

    def resolve_physical_mic(self) -> str:
        """Always resolve to a real non-virtual PipeWire source (prevents self-feedback loops!)."""
        wanted = str(self.config.data["microphone"].get("device", "auto"))
        mics = list_audio_devices(include_monitors=False, include_virtual=False)["microphones"]
        if wanted and wanted != "auto":
            if any(m.name == wanted for m in mics):
                return wanted
        # Prefer the current default if it is a physical mic, else first real physical mic
        for m in mics:
            if m.is_default:
                return m.name
        return mics[0].name if mics else ""

    def resolve_physical_speaker(self) -> str:
        """Always resolve to a real non-virtual PipeWire sink (prevents self-feedback loops!)."""
        wanted = str(self.config.data["speaker"].get("device", "auto"))
        sinks = list_audio_devices(include_monitors=False, include_virtual=False)["speakers"]
        if wanted and wanted != "auto":
            if any(s.name == wanted for s in sinks):
                return wanted
        for s in sinks:
            if s.is_default:
                return s.name
        # Prefer analog/headphone/bluetooth/usb output over unused HDMI ports
        non_hdmi = [s for s in sinks if "hdmi" not in s.name.lower()]
        if non_hdmi:
            return non_hdmi[0].name
        return sinks[0].name if sinks else ""

    def set_virtual_mic_as_default(self, enable: bool) -> tuple[bool, str]:
        if enable:
            if not self._mic_running:
                ok, msg = self.start_microphone()
                if not ok:
                    return False, msg
            _, cur, _ = _run_pactl(["get-default-source"])
            if cur and VIRTUAL_MIC_SOURCE_NAME not in cur:
                self._prev_default_source = cur
            rc, _, err = _run_pactl(["set-default-source", VIRTUAL_MIC_SOURCE_NAME])
            if rc != 0:
                return False, f"Could not set default mic: {err}"
            return True, "Set NVIDIA Broadcast (Microphone) as Linux default input"
        restore = self._prev_default_source or self.resolve_physical_mic()
        if restore:
            _run_pactl(["set-default-source", restore])
        return True, f"Restored default input to {restore or 'system default'}"

    def set_virtual_speakers_as_default(self, enable: bool) -> tuple[bool, str]:
        if enable:
            if not self._spk_running:
                ok, msg = self.start_speakers()
                if not ok:
                    return False, msg
            _, cur, _ = _run_pactl(["get-default-sink"])
            if cur and VIRTUAL_SPEAKER_SINK_NAME not in cur:
                self._prev_default_sink = cur
            rc, _, err = _run_pactl(["set-default-sink", VIRTUAL_SPEAKER_SINK_NAME])
            if rc != 0:
                return False, f"Could not set default speakers: {err}"
            return True, "Set NVIDIA Broadcast (Speakers) as Linux default output"
        restore = self._prev_default_sink or self.resolve_physical_speaker()
        if restore:
            _run_pactl(["set-default-sink", restore])
        return True, f"Restored default output to {restore or 'system default'}"

    def stop_microphone(self) -> None:
        # Restore default physical mic first if NVIDIA Broadcast was set as default
        _, cur, _ = _run_pactl(["get-default-source"])
        if cur and VIRTUAL_MIC_SOURCE_NAME in cur:
            restore = self._prev_default_source or self.resolve_physical_mic()
            if restore:
                _run_pactl(["set-default-source", restore])
        self._mic_stop.set()
        if self._mic_thread is not None and self._mic_thread.is_alive():
            self._mic_thread.join(timeout=2.5)
        self._mic_thread = None
        self._mic_running = False
        self._mic_backend_used = ""
        self.pw.destroy_virtual_microphone()

    def _run_mic_stream(self) -> None:
        chunk_bytes = 480 * 2  # 480 mono s16 samples = exact 10.0 ms RNNoise frame @ 48 kHz
        try:
            while not self._mic_stop.is_set():
                mic_cfg = self.config.data["microphone"]
                rate = int(mic_cfg.get("sample_rate", 48000))
                target = self.resolve_physical_mic()
                if not target or is_virtual_broadcast_node(target):
                    self._mic_stop.wait(1.0)
                    continue

                # Isolate both capture and feed from WirePlumber auto-routing/reconnection
                # so processed mic audio can NEVER leak to speakers or loop into itself.
                cap_name = "nvb_mic_cap"
                feeder_name = "nvb_mic_feed"
                rec_cmd = build_isolated_pw_cat_cmd("record", cap_name, rate=rate, channels=1, latency_ms=20)
                play_cmd = build_isolated_pw_cat_cmd("playback", feeder_name, rate=rate, channels=1, latency_ms=20)

                rec_proc: subprocess.Popen[bytes] | None = None
                play_proc: subprocess.Popen[bytes] | None = None
                last_check = time.monotonic()
                cap_linked = False
                feed_linked = False
                try:
                    rec_proc = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    play_proc = subprocess.Popen(play_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    assert rec_proc.stdout is not None and play_proc.stdin is not None

                    def _ensure_links() -> tuple[bool, bool]:
                        ok_cap = cap_linked
                        ok_feed = feed_linked
                        if not ok_cap:
                            ok_cap, _ = link_pw_nodes(
                                target,
                                cap_name,
                                src_prefix="capture_",
                                dst_prefix="input_",
                                allow_virtual_src=False,
                                allow_virtual_dst=False,
                            )
                        if not ok_feed:
                            ok_feed, _ = link_pw_nodes(
                                feeder_name,
                                VIRTUAL_MIC_SOURCE_NAME,
                                src_prefix="output_",
                                dst_prefix="input_",
                                allow_virtual_src=False,
                                allow_virtual_dst=True,
                            )
                        return ok_cap, ok_feed

                    # Link BEFORE reading because node.autoconnect=false means pw-cat --record
                    # will not emit frames until its input port is linked.
                    deadline = time.monotonic() + 3.0
                    while not self._mic_stop.is_set() and not (cap_linked and feed_linked) and time.monotonic() < deadline:
                        cap_linked, feed_linked = _ensure_links()
                        if not (cap_linked and feed_linked):
                            self._mic_stop.wait(0.05)

                    rec_fd = rec_proc.stdout.fileno()
                    read_buf = bytearray()
                    last_audio_ts = time.monotonic()
                    meter_decayed = False

                    while not self._mic_stop.is_set():
                        raw, timed_out = _read_chunk_nonblocking(rec_fd, read_buf, chunk_bytes, timeout=0.5)
                        now = time.monotonic()
                        if not timed_out:
                            if not raw:
                                break
                            last_audio_ts = now
                            meter_decayed = False
                            if not (cap_linked and feed_linked):
                                cap_linked, feed_linked = _ensure_links()
                            processed = self.mic_processor.process_pcm16(raw, self.config.data["microphone"])
                            try:
                                play_proc.stdin.write(processed)
                                play_proc.stdin.flush()
                            except BrokenPipeError:
                                break
                            meter = self.mic_processor.last_meter
                            for cb in list(self._meter_callbacks):
                                try:
                                    cb(meter)
                                except Exception:
                                    pass
                        elif not meter_decayed and (now - last_audio_ts >= 1.0):
                            meter_decayed = True
                            idle_meter = AudioMeterReading(backend=self.mic_processor.backend_name)
                            self.mic_processor.last_meter = idle_meter
                            for cb in list(self._meter_callbacks):
                                try:
                                    cb(idle_meter)
                                except Exception:
                                    pass

                        # Hot-plug & suspend/resume self-healing (runs even when idle/silent!)
                        if now - last_check >= 1.5:
                            last_check = now
                            if rec_proc.poll() is not None or play_proc.poll() is not None:
                                break
                            if self.config.data["microphone"].get("device", "auto") == "auto":
                                new_target = self.resolve_physical_mic()
                                if new_target and new_target != target:
                                    break
                            links = list_pw_links()
                            if links or timed_out:
                                cap_ok = is_pw_link_active(target, cap_name, links=links)
                                feed_ok = is_pw_link_active(feeder_name, VIRTUAL_MIC_SOURCE_NAME, links=links)
                                if not cap_ok:
                                    cap_linked = False
                                if not feed_ok:
                                    feed_linked = False
                                if not (cap_linked and feed_linked):
                                    cap_linked, feed_linked = _ensure_links()
                                    if not (cap_linked and feed_linked) and rec_proc.poll() is not None:
                                        break
                finally:
                    for proc in (rec_proc, play_proc):
                        if proc is not None and proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=1.0)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait(timeout=1.0)
                if not self._mic_stop.is_set():
                    time.sleep(0.25)
        finally:
            self._mic_running = False

    def start_speakers(self) -> tuple[bool, str]:
        if self._spk_running:
            return True, "Virtual speakers already running"
        if not shutil.which("pw-cat"):
            return False, "pw-cat not found; cannot start virtual speakers"

        spk_cfg = self.config.data["speaker"]
        vname = spk_cfg.get("virtual_name", "NVIDIA Broadcast (Speakers)")
        ok, msg = self.pw.create_virtual_speakers(description=vname)
        if not ok:
            return False, msg

        self._spk_stop.clear()
        self.speaker_processor.reset()
        self.speaker_processor_r.reset()
        self._spk_running = True
        self._spk_thread = threading.Thread(
            target=self._run_speaker_stream,
            name="nv-broadcast-speakers",
            daemon=True,
        )
        self._spk_thread.start()
        return True, f"Started virtual speakers '{vname}' (stereo, backend: {self.speaker_processor.backend_name})"

    def stop_speakers(self) -> None:
        # Restore real headphones/speakers as default before tearing down virtual sink
        _, cur, _ = _run_pactl(["get-default-sink"])
        if cur and VIRTUAL_SPEAKER_SINK_NAME in cur:
            restore = self._prev_default_sink or self.resolve_physical_speaker()
            if restore:
                _run_pactl(["set-default-sink", restore])
        self._spk_stop.set()
        if self._spk_thread is not None and self._spk_thread.is_alive():
            self._spk_thread.join(timeout=2.5)
        self._spk_thread = None
        self._spk_running = False
        self.pw.destroy_virtual_speakers()

    def _run_speaker_stream(self) -> None:
        stereo = HAS_NUMPY
        channels = 2 if stereo else 1
        chunk_bytes = 480 * channels * 2  # 480 frames = 10ms RNNoise frame in stereo or mono
        try:
            while not self._spk_stop.is_set():
                spk_cfg = self.config.data["speaker"]
                rate = int(spk_cfg.get("sample_rate", 48000))
                out_dev = self.resolve_physical_speaker()
                if not out_dev or is_virtual_broadcast_node(out_dev):
                    self._spk_stop.wait(1.0)
                    continue

                # Isolate both capture and playback from WirePlumber auto-routing/reconnection:
                # prevents WirePlumber from EVER moving capture onto the physical speaker's monitor
                # (which causes an infinite echo loop) or moving playback back into nv_broadcast_speakers.
                cap_name = "nvb_spk_cap"
                out_name = "nvb_spk_out"
                rec_cmd = build_isolated_pw_cat_cmd("record", cap_name, rate=rate, channels=channels, latency_ms=20)
                play_cmd = build_isolated_pw_cat_cmd("playback", out_name, rate=rate, channels=channels, latency_ms=20)

                rec_proc: subprocess.Popen[bytes] | None = None
                play_proc: subprocess.Popen[bytes] | None = None
                last_check = time.monotonic()
                cap_linked = False
                out_linked = False
                try:
                    rec_proc = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    play_proc = subprocess.Popen(play_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    assert rec_proc.stdout is not None and play_proc.stdin is not None

                    def _ensure_links() -> tuple[bool, bool]:
                        ok_cap = cap_linked
                        ok_out = out_linked
                        if not ok_cap:
                            ok_cap, _ = link_pw_nodes(
                                VIRTUAL_SPEAKER_SINK_NAME,
                                cap_name,
                                src_prefix="monitor_",
                                dst_prefix="input_",
                                allow_virtual_src=True,
                                allow_virtual_dst=False,
                            )
                        if not ok_out:
                            ok_out, _ = link_pw_nodes(
                                out_name,
                                out_dev,
                                src_prefix="output_",
                                dst_prefix="playback_",
                                allow_virtual_src=False,
                                allow_virtual_dst=False,
                            )
                        return ok_cap, ok_out

                    # Link BEFORE reading because node.autoconnect=false means pw-cat --record
                    # will not emit frames until its input port is linked.
                    deadline = time.monotonic() + 3.0
                    while not self._spk_stop.is_set() and not (cap_linked and out_linked) and time.monotonic() < deadline:
                        cap_linked, out_linked = _ensure_links()
                        if not (cap_linked and out_linked):
                            self._spk_stop.wait(0.05)

                    rec_fd = rec_proc.stdout.fileno()
                    read_buf = bytearray()
                    last_audio_ts = time.monotonic()
                    meter_decayed = False

                    while not self._spk_stop.is_set():
                        raw, timed_out = _read_chunk_nonblocking(rec_fd, read_buf, chunk_bytes, timeout=0.5)
                        now = time.monotonic()
                        if not timed_out:
                            if not raw:
                                break
                            last_audio_ts = now
                            meter_decayed = False
                            if not (cap_linked and out_linked):
                                cap_linked, out_linked = _ensure_links()
                            cur_spk_cfg = self.config.data["speaker"]
                            if stereo and len(raw) >= 4:
                                usable = (len(raw) // 4) * 4
                                interleaved = np.frombuffer(raw[:usable], dtype=np.int16).reshape(-1, 2)
                                l_in = interleaved[:, 0].astype(np.float32) / 32768.0
                                r_in = interleaved[:, 1].astype(np.float32) / 32768.0
                                effs = cur_spk_cfg.get("effects", {})
                                any_eff = bool(
                                    effs.get("speaker_noise_removal", {}).get("enabled")
                                    or effs.get("speaker_echo_removal", {}).get("enabled")
                                )
                                if any_eff:
                                    l_out = self.speaker_processor._process_numpy(l_in, cur_spk_cfg)
                                    r_out = self.speaker_processor_r._process_numpy(r_in, cur_spk_cfg)
                                    stacked = np.column_stack((l_out, r_out))
                                    processed = np.clip(np.round(stacked * 32767.0), -32768, 32767).astype(np.int16).tobytes()
                                else:
                                    in_db, pk_db = compute_rms_db(l_in)
                                    self.speaker_processor.last_meter = AudioMeterReading(
                                        input_rms_db=round(in_db, 1),
                                        output_rms_db=round(in_db, 1),
                                        input_peak_db=round(pk_db, 1),
                                        output_peak_db=round(pk_db, 1),
                                        backend=self.speaker_processor.backend_name,
                                    )
                                    processed = raw[:usable]
                            else:
                                processed = self.speaker_processor.process_pcm16(raw, cur_spk_cfg)
                            try:
                                play_proc.stdin.write(processed)
                                play_proc.stdin.flush()
                            except BrokenPipeError:
                                break
                            meter = self.speaker_processor.last_meter
                            for cb in list(self._speaker_meter_callbacks):
                                try:
                                    cb(meter)
                                except Exception:
                                    pass
                        elif not meter_decayed and (now - last_audio_ts >= 1.0):
                            meter_decayed = True
                            idle_meter = AudioMeterReading(backend=self.speaker_processor.backend_name)
                            self.speaker_processor.last_meter = idle_meter
                            for cb in list(self._speaker_meter_callbacks):
                                try:
                                    cb(idle_meter)
                                except Exception:
                                    pass

                        # Seamless Headphones <-> Speakers hot-plug & suspend/resume self-healing!
                        if now - last_check >= 1.5:
                            last_check = now
                            if rec_proc.poll() is not None or play_proc.poll() is not None:
                                break
                            if self.config.data["speaker"].get("device", "auto") == "auto":
                                new_out = self.resolve_physical_speaker()
                                if new_out and new_out != out_dev:
                                    break
                            links = list_pw_links()
                            if links or timed_out:
                                cap_ok = is_pw_link_active(VIRTUAL_SPEAKER_SINK_NAME, cap_name, links=links)
                                out_ok = is_pw_link_active(out_name, out_dev, links=links)
                                if not cap_ok:
                                    cap_linked = False
                                if not out_ok:
                                    out_linked = False
                                if not (cap_linked and out_linked):
                                    cap_linked, out_linked = _ensure_links()
                                    if not (cap_linked and out_linked) and play_proc.poll() is not None:
                                        break
                finally:
                    for proc in (rec_proc, play_proc):
                        if proc is not None and proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=1.0)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait(timeout=1.0)
                if not self._spk_stop.is_set():
                    time.sleep(0.25)
        finally:
            self._spk_running = False

    def record_mic_test(self, duration_sec: float = 5.0, output_dir: Path | None = None) -> MicTestResult:
        """Record N seconds of real mic audio, create raw + processed WAVs and waveforms."""
        if not shutil.which("pw-cat"):
            raise RuntimeError("pw-cat is required for mic test recording")

        mic_cfg = self.config.data["microphone"]
        rate = int(mic_cfg.get("sample_rate", 48000))
        target = self.resolve_physical_mic()
        out_dir = output_dir or Path(tempfile.gettempdir()) / "nv-broadcast-test"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / "mic_raw.wav"
        proc_path = out_dir / "mic_processed.wav"

        test_node = "nvb_mic_test_cap"
        if target and not is_virtual_broadcast_node(target) and shutil.which("pw-link"):
            cmd = build_isolated_pw_cat_cmd("record", test_node, rate=rate, channels=1, latency_ms=20)
            use_explicit_link = True
        else:
            cmd = ["pw-cat", "--record", "--raw", "--format=s16", f"--rate={rate}", "--channels=1"]
            if target:
                cmd.append(f"--target={target}")
            cmd.append("-")
            use_explicit_link = False

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdout is not None
        target_bytes = int(rate * duration_sec) * 2
        buf = bytearray()
        tmp_buf = bytearray()
        deadline = time.monotonic() + duration_sec + 2.5
        try:
            if use_explicit_link:
                link_deadline = time.monotonic() + 2.0
                linked = False
                while not linked and time.monotonic() < link_deadline:
                    linked, _ = link_pw_nodes(
                        target,
                        test_node,
                        src_prefix="capture_",
                        dst_prefix="input_",
                        allow_virtual_src=False,
                        allow_virtual_dst=False,
                    )
                    if not linked:
                        time.sleep(0.05)
            fd = proc.stdout.fileno()
            while len(buf) < target_bytes and time.monotonic() < deadline:
                want = min(4800, target_bytes - len(buf))
                chunk, timed_out = _read_chunk_nonblocking(fd, tmp_buf, want, timeout=0.25)
                if timed_out:
                    continue
                if not chunk:
                    break
                buf.extend(chunk)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)

        raw_bytes = bytes(buf)
        dsp = AudioProcessor(sample_rate=rate, frame_size=512)
        processed_chunks: list[bytes] = []
        step = 960  # 480 samples = 10ms RNNoise frame
        for off in range(0, len(raw_bytes), step):
            processed_chunks.append(dsp.process_pcm16(raw_bytes[off : off + step], mic_cfg))
        proc_bytes = b"".join(processed_chunks)

        save_pcm16_wav(raw_path, raw_bytes, rate)
        save_pcm16_wav(proc_path, proc_bytes, rate)

        raw_rms, _ = compute_rms_db(_pcm16_to_floats(raw_bytes))
        proc_rms, _ = compute_rms_db(_pcm16_to_floats(proc_bytes))
        return MicTestResult(
            duration_sec=round(len(raw_bytes) / (2.0 * rate), 2),
            raw_wav_path=str(raw_path),
            processed_wav_path=str(proc_path),
            raw_rms_db=round(raw_rms, 1),
            processed_rms_db=round(proc_rms, 1),
            noise_floor_reduction_db=round(max(0.0, raw_rms - proc_rms), 1),
            raw_waveform=compute_waveform_peaks(raw_bytes, 120),
            processed_waveform=compute_waveform_peaks(proc_bytes, 120),
        )

    def build_speaker_sample(self, kind: str = "fan", output_dir: Path | None = None) -> MicTestResult:
        """Generate a raw + neural-cleaned speaker demo clip (fan / typing / echo / fan_echo)."""
        spk_cfg = self.config.data["speaker"]
        rate = int(spk_cfg.get("sample_rate", 48000))
        out_dir = output_dir or Path(tempfile.gettempdir()) / "nv-broadcast-test"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_bytes = generate_sample_clip(kind, sample_rate=rate, duration_sec=4.0)
        # Apply active speaker effects (force noise removal if both are off so demo is informative when toggled)
        demo_cfg = {
            "highpass_enabled": True,
            "highpass_hz": 85,
            "noise_gate_enabled": False,
            "auto_gain_enabled": False,
            "effects": spk_cfg.get("effects", {}),
        }
        dsp = AudioProcessor(sample_rate=rate, frame_size=512)
        proc_chunks = [dsp.process_pcm16(raw_bytes[off : off + 960], demo_cfg) for off in range(0, len(raw_bytes), 960)]
        proc_bytes = b"".join(proc_chunks)

        raw_path = out_dir / f"sample_{kind}_raw.wav"
        proc_path = out_dir / f"sample_{kind}_processed.wav"
        save_pcm16_wav(raw_path, raw_bytes, rate)
        save_pcm16_wav(proc_path, proc_bytes, rate)

        raw_rms, _ = compute_rms_db(_pcm16_to_floats(raw_bytes))
        proc_rms, _ = compute_rms_db(_pcm16_to_floats(proc_bytes))
        return MicTestResult(
            duration_sec=4.0,
            raw_wav_path=str(raw_path),
            processed_wav_path=str(proc_path),
            raw_rms_db=round(raw_rms, 1),
            processed_rms_db=round(proc_rms, 1),
            noise_floor_reduction_db=round(max(0.0, raw_rms - proc_rms), 1),
            raw_waveform=compute_waveform_peaks(raw_bytes, 120),
            processed_waveform=compute_waveform_peaks(proc_bytes, 120),
        )

    def play_wav(self, wav_path: str, target_sink: str | None = None) -> tuple[bool, str]:
        self.stop_playback()
        if not shutil.which("pw-play"):
            return False, "pw-play not found"
        if not Path(wav_path).is_file():
            return False, f"File not found: {wav_path}"
        resolved_sink = target_sink if (target_sink and target_sink != "auto") else self.resolve_physical_speaker()
        cmd = ["pw-play"]
        if resolved_sink and not is_virtual_broadcast_node(resolved_sink):
            cmd.append(f"--target={resolved_sink}")
        cmd.append(wav_path)
        self._play_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, f"Playing {wav_path}"

    def stop_playback(self) -> None:
        if self._play_proc is not None and self._play_proc.poll() is None:
            self._play_proc.terminate()
            try:
                self._play_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._play_proc.kill()
                self._play_proc.wait(timeout=1.0)
        self._play_proc = None

    def stop_all(self) -> None:
        self.stop_playback()
        self.stop_microphone()
        self.stop_speakers()
