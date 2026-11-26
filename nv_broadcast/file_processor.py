"""Offline Audio & Video File Processor (NVIDIA Broadcast v2.3.0 Files tab)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_dsp import AudioProcessor, save_pcm16_wav
from .config import BroadcastConfig
from .video_effects import HAS_CV2, HAS_NUMPY, VideoEffectsPipeline

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


@dataclass
class FileJobState:
    active: bool = False
    status: str = "idle"  # idle | running | completed | cancelled | failed
    progress: float = 0.0
    input_path: str = ""
    output_path: str = ""
    media_kind: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FileProcessor:
    """Applies active Broadcast Audio & Video effects to local media files."""

    def __init__(self, config: BroadcastConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._state = FileJobState()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def _update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)

    def cancel(self) -> None:
        self._cancel.set()

    def start_job(
        self,
        input_path: str,
        output_path: str | None = None,
        apply_audio: bool = True,
        apply_video: bool = True,
    ) -> tuple[bool, str]:
        with self._lock:
            if self._state.active:
                return False, "A file processing job is already running"

        src = Path(input_path).expanduser()
        if not src.is_file():
            return False, f"Input file not found: {src}"
        ext = src.suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            kind = "audio"
        elif ext in VIDEO_EXTENSIONS:
            kind = "video"
        else:
            return False, f"Unsupported media format: {ext}"

        if not shutil.which("ffmpeg"):
            return False, "ffmpeg is required for offline file processing"

        if output_path:
            dst = Path(output_path).expanduser()
        else:
            out_dir = Path.home() / "Videos" / "NVIDIA Broadcast" if kind == "video" else Path.home() / "Music" / "NVIDIA Broadcast"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_ext = ".wav" if kind == "audio" and ext == ".wav" else (".mp3" if kind == "audio" else ".mp4")
            dst = out_dir / f"{src.stem}_broadcast{out_ext}"

        self._cancel.clear()
        self._update(
            active=True,
            status="running",
            progress=0.0,
            input_path=str(src),
            output_path=str(dst),
            media_kind=kind,
            message=f"Processing {src.name}...",
        )
        self._thread = threading.Thread(
            target=self._run_job,
            args=(src, dst, kind, apply_audio, apply_video),
            name="nv-broadcast-fileproc",
            daemon=True,
        )
        self._thread.start()
        return True, f"Started processing {src.name} -> {dst}"

    def process_sync(
        self,
        input_path: str,
        output_path: str,
        apply_audio: bool = True,
        apply_video: bool = True,
    ) -> FileJobState:
        src = Path(input_path).expanduser()
        dst = Path(output_path).expanduser()
        ext = src.suffix.lower()
        kind = "audio" if ext in AUDIO_EXTENSIONS else "video"
        self._cancel.clear()
        self._update(
            active=True,
            status="running",
            progress=0.0,
            input_path=str(src),
            output_path=str(dst),
            media_kind=kind,
            message="Starting...",
        )
        self._run_job(src, dst, kind, apply_audio, apply_video)
        with self._lock:
            return FileJobState(**self._state.to_dict())

    def _run_job(self, src: Path, dst: Path, kind: str, apply_audio: bool, apply_video: bool) -> None:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if kind == "audio":
                self._process_audio_file(src, dst)
            else:
                self._process_video_file(src, dst, apply_audio=apply_audio, apply_video=apply_video)
            if self._cancel.is_set():
                self._update(active=False, status="cancelled", message="Cancelled by user")
            else:
                self._update(active=False, status="completed", progress=100.0, message=f"Saved {dst.name}")
        except Exception as exc:
            self._update(active=False, status="failed", message=str(exc))

    def _denoise_pcm_file(self, src: Path, out_wav: Path, p_start: float = 0.0, p_end: float = 95.0) -> bool:
        """Decode audio stream to 48kHz mono PCM16, run neural AudioProcessor, write clean WAV."""
        dec = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-f",
                "s16le",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if dec.returncode != 0 or not dec.stdout:
            return False

        raw = dec.stdout
        total = len(raw)
        dsp = AudioProcessor(sample_rate=48000, frame_size=512)
        mic_cfg = self.config.data["microphone"]
        chunks: list[bytes] = []
        step = 480 * 2 * 10  # 100 ms blocks
        for off in range(0, total, step):
            if self._cancel.is_set():
                return False
            chunks.append(dsp.process_pcm16(raw[off : off + step], mic_cfg))
            frac = min(1.0, (off + step) / max(1, total))
            self._update(progress=round(p_start + (p_end - p_start) * frac, 1))
        save_pcm16_wav(out_wav, b"".join(chunks), 48000)
        return True

    def _process_audio_file(self, src: Path, dst: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="nvb-audio-") as td:
            clean_wav = Path(td) / "clean.wav"
            ok = self._denoise_pcm_file(src, clean_wav, 0.0, 90.0)
            if self._cancel.is_set():
                return
            if not ok:
                raise RuntimeError("Failed to decode audio stream with ffmpeg")
            if dst.suffix.lower() == ".wav":
                shutil.copy2(clean_wav, dst)
            else:
                enc = subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(clean_wav), str(dst)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if enc.returncode != 0:
                    raise RuntimeError(enc.stderr.strip() or "ffmpeg audio encode failed")
            self._update(progress=100.0)

    def _process_video_file(self, src: Path, dst: Path, apply_audio: bool, apply_video: bool) -> None:
        with tempfile.TemporaryDirectory(prefix="nvb-video-") as td:
            td_path = Path(td)
            clean_wav = td_path / "clean_audio.wav"
            has_clean_audio = False

            if apply_audio:
                self._update(message="Cleaning audio track with neural RNNoise...")
                has_clean_audio = self._denoise_pcm_file(src, clean_wav, 0.0, 25.0)
                if self._cancel.is_set():
                    return

            video_src = src
            if apply_video and HAS_CV2 and HAS_NUMPY:
                self._update(message="Applying neural video effects...")
                cap = cv2.VideoCapture(str(src))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video file {src}")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

                raw_vid = td_path / "processed_video.mp4"
                writer = cv2.VideoWriter(str(raw_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                pipe = VideoEffectsPipeline(segmentation_backend=self.config.data.get("segmentation_backend", "auto"))
                cam_cfg = dict(self.config.data["camera"], split_preview=False)
                idx = 0
                try:
                    while not self._cancel.is_set():
                        ok, bgr = cap.read()
                        if not ok or bgr is None:
                            break
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        _, proc_rgb = pipe.process_frame_np(rgb, cam_cfg)
                        writer.write(cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR))
                        idx += 1
                        if idx % 5 == 0:
                            self._update(progress=round(25.0 + 65.0 * min(1.0, idx / total_frames), 1))
                finally:
                    cap.release()
                    writer.release()
                    pipe.close()

                if self._cancel.is_set():
                    return
                video_src = raw_vid

            self._update(message="Muxing final output...", progress=92.0)
            cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_src)]
            if has_clean_audio:
                cmd += ["-i", str(clean_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(dst)]
            else:
                cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)]
            enc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
            if enc.returncode != 0:
                raise RuntimeError(enc.stderr.strip() or "ffmpeg mux failed")
            self._update(progress=100.0)
