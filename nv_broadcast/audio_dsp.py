"""Real-time Neural Audio AI (RNNoise) + DSP Engine for NVIDIA Broadcast (Linux).

Pipeline (48 kHz mono):
  1. High-pass rumble filter (removes laptop fan / AC sub-bass hum)
  2. Real neural Noise Removal via compiled RNNoise GRU (`pyrnnoise`, 10ms frames, <1% CPU)
     with speech probability (VAD) tracking; falls back to NumPy STFT Wiener if absent
  3. Room Echo Removal (vectorized STFT late-reflection suppression)
  4. Studio Voice broadcast parametric EQ (warm/bright/full/default) + soft-knee compressor
  5. Neural-VAD-gated Auto-Gain (boosts speech cleanly without amplifying silence/fan noise!)
"""

from __future__ import annotations

import array
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

try:
    from scipy import signal as sp_signal  # type: ignore

    HAS_SCIPY = True
except ImportError:
    sp_signal = None  # type: ignore
    HAS_SCIPY = False

try:
    import pyrnnoise  # type: ignore

    HAS_PYRNNOISE = True
except ImportError:
    pyrnnoise = None  # type: ignore
    HAS_PYRNNOISE = False


@dataclass
class AudioMeterReading:
    input_rms_db: float = -90.0
    output_rms_db: float = -90.0
    input_peak_db: float = -90.0
    output_peak_db: float = -90.0
    noise_reduction_db: float = 0.0
    speech_prob: float = 0.0
    gate_open: bool = False
    backend: str = "dsp"


def _pcm16_to_floats(pcm_bytes: bytes) -> list[float]:
    arr = array.array("h")
    arr.frombytes(pcm_bytes[: (len(pcm_bytes) // 2) * 2])
    inv = 1.0 / 32768.0
    return [s * inv for s in arr]


def _floats_to_pcm16(samples: list[float]) -> bytes:
    arr = array.array("h", (max(-32768, min(32767, int(round(s * 32767.0)))) for s in samples))
    return arr.tobytes()


def _radix2_fft(x: list[complex], inverse: bool = False) -> list[complex]:
    n = len(x)
    if n <= 1:
        return list(x)
    out = list(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    sign = 1.0 if inverse else -1.0
    length = 2
    while length <= n:
        half = length >> 1
        angle = sign * 2.0 * math.pi / length
        wlen = complex(math.cos(angle), math.sin(angle))
        for start in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = out[start + k]
                v = out[start + k + half] * w
                out[start + k] = u + v
                out[start + k + half] = u - v
                w *= wlen
        length <<= 1
    if inverse:
        inv_n = 1.0 / n
        return [v * inv_n for v in out]
    return out


def compute_rms_db(samples: list[float] | Any) -> tuple[float, float]:
    """Return (rms_dbfs, peak_dbfs) for float samples in [-1, 1]."""
    if HAS_NUMPY and isinstance(samples, np.ndarray):
        if samples.size == 0:
            return -90.0, -90.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        peak = float(np.max(np.abs(samples)))
    else:
        n = len(samples)
        if n == 0:
            return -90.0, -90.0
        sq = 0.0
        peak = 0.0
        for s in samples:
            a = abs(s)
            sq += s * s
            if a > peak:
                peak = a
        rms = math.sqrt(sq / n)
    rms_db = 20.0 * math.log10(max(rms, 1e-45))
    peak_db = 20.0 * math.log10(max(peak, 1e-45))
    return max(-90.0, rms_db), max(-90.0, peak_db)


def compute_waveform_peaks(pcm_bytes: bytes, buckets: int = 120) -> list[float]:
    """Compute normalized [0..1] amplitude bars for WaveSurfer-style UI display."""
    if not pcm_bytes or buckets <= 0:
        return [0.0] * max(1, buckets)
    if HAS_NUMPY:
        usable = (len(pcm_bytes) // 2) * 2
        if usable == 0:
            return [0.0] * buckets
        arr = np.abs(np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float32) / 32768.0)
        chunks = np.array_split(arr, buckets)
        peaks = [float(np.max(c)) if c.size else 0.0 for c in chunks]
    else:
        floats = _pcm16_to_floats(pcm_bytes)
        if not floats:
            return [0.0] * buckets
        step = max(1, len(floats) // buckets)
        peaks = []
        for i in range(buckets):
            sl = floats[i * step : (i + 1) * step]
            peaks.append(max((abs(v) for v in sl), default=0.0))
    mx = max(peaks) if peaks else 1.0
    if mx > 1e-4:
        # Gentle visual normalization while preserving relative dynamics
        scale = min(3.0, 0.95 / mx)
        peaks = [round(min(1.0, p * scale), 4) for p in peaks]
    return peaks


def generate_sample_clip(kind: str, sample_rate: int = 48000, duration_sec: float = 4.0) -> bytes:
    """Synthesize a clean voice-like harmonic signal mixed with realistic noise for live speaker/mic demos."""
    n = int(sample_rate * duration_sec)
    if HAS_NUMPY:
        rng = np.random.default_rng(42)
        t = np.arange(n, dtype=np.float64) / sample_rate
        # Formant-modulated voice-like cadence (3 spoken phrases with pauses)
        env = (
            np.clip(np.sin(2.0 * np.pi * 0.45 * t), 0.0, 1.0) ** 0.7
            * (0.65 + 0.35 * np.sin(2.0 * np.pi * 3.8 * t))
        )
        f0 = 155.0 + 28.0 * np.sin(2.0 * np.pi * 0.9 * t)
        phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
        voice = env * (
            0.22 * np.sin(phase)
            + 0.14 * np.sin(2.0 * phase)
            + 0.09 * np.sin(3.0 * phase)
            + 0.05 * np.sin(5.0 * phase)
        )
        noise = np.zeros(n, dtype=np.float64)
        if kind in ("fan", "fan_echo", "air_conditioning"):
            # Stationary laptop fan roar + AC rumble
            white = rng.normal(0.0, 1.0, n)
            # Simple low-passed pink-ish fan hum
            b = np.ones(12) / 12.0
            fan = np.convolve(white, b, mode="same") * 0.12
            hum = 0.04 * np.sin(2.0 * np.pi * 120.0 * t) + 0.03 * np.sin(2.0 * np.pi * 240.0 * t)
            noise += fan + hum
        if kind in ("typing", "typing_echo"):
            # Sharp mechanical keyboard clicks every ~0.18s
            for idx in range(int(0.15 * sample_rate), n - 400, int(0.17 * sample_rate)):
                click_len = 360
                decay = np.exp(-np.arange(click_len) / 48.0)
                noise[idx : idx + click_len] += rng.normal(0.0, 0.28, click_len) * decay
        if kind in ("echo", "fan_echo", "typing_echo"):
            # Add room reverb reflections at 45ms, 95ms, 160ms
            reverb = np.zeros(n, dtype=np.float64)
            for delay_ms, gain in ((45, 0.42), (95, 0.28), (160, 0.16)):
                d = int(sample_rate * delay_ms / 1000)
                if d < n:
                    reverb[d:] += voice[:-d] * gain
            voice = voice + reverb
        sig = np.clip(voice + noise, -0.95, 0.95)
        return (sig * 32767.0).astype(np.int16).tobytes()

    # Pure-python fallback
    out = []
    for i in range(n):
        t = i / sample_rate
        env = max(0.0, math.sin(2.0 * math.pi * 0.45 * t))
        v = env * 0.22 * math.sin(2.0 * math.pi * 165.0 * t)
        nz = 0.05 * math.sin(2.0 * math.pi * 120.0 * t)
        out.append(max(-0.95, min(0.95, v + nz)))
    return _floats_to_pcm16(out)


def build_ffmpeg_audio_filter(mic_cfg: dict[str, Any]) -> str:
    """Build an FFmpeg `-af` filter graph mirroring the active mic settings."""
    filters: list[str] = []
    effects = mic_cfg.get("effects", {})

    if mic_cfg.get("highpass_enabled", True):
        hz = int(mic_cfg.get("highpass_hz", 85))
        filters.append(f"highpass=f={hz}:poles=2")

    nr = effects.get("noise_removal", {})
    if nr.get("enabled"):
        strength = float(nr.get("strength", 0.85))
        nr_db = 6.0 + 34.0 * max(0.0, min(1.0, strength))
        filters.append(f"afftdn=nr={nr_db:.1f}:nf=-35:tn=1")

    echo = effects.get("room_echo_removal", {})
    if echo.get("enabled"):
        strength = float(echo.get("strength", 0.5))
        thresh = -48.0 + 14.0 * max(0.0, min(1.0, strength))
        filters.append(f"agate=threshold={thresh:.1f}dB:ratio=3:attack=8:release=140")

    sv = effects.get("studio_voice", {})
    if sv.get("enabled"):
        profile = sv.get("micProfile", "default")
        if profile == "warm":
            filters.append("equalizer=f=160:t=q:w=1.0:g=3.5,equalizer=f=3200:t=q:w=1.2:g=1.5")
        elif profile == "bright":
            filters.append("equalizer=f=4500:t=q:w=1.1:g=4.0,equalizer=f=200:t=q:w=1.0:g=-1.5")
        elif profile == "full":
            filters.append("equalizer=f=140:t=q:w=1.0:g=3.0,equalizer=f=3800:t=q:w=1.2:g=3.0")
        else:
            filters.append("equalizer=f=180:t=q:w=1.0:g=2.0,equalizer=f=3500:t=q:w=1.2:g=2.0")
        filters.append("acompressor=threshold=-20dB:ratio=3:attack=6:release=80:makeup=3")

    if mic_cfg.get("noise_gate_enabled", True) and not echo.get("enabled"):
        gate_db = float(mic_cfg.get("noise_gate_db", -46.0))
        filters.append(f"agate=threshold={gate_db:.1f}dB:ratio=2.5:attack=6:release=120")

    if mic_cfg.get("auto_gain_enabled", True):
        filters.append("dynaudnorm=f=150:g=11:p=0.9:m=6.0")

    return ",".join(filters) if filters else "anull"


class AudioProcessor:
    """Streaming real-time neural audio processor (48 kHz mono)."""

    def __init__(self, sample_rate: int = 48000, frame_size: int = 512) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = frame_size // 2
        self._bins = frame_size // 2 + 1

        self._window = [0.5 * (1.0 - math.cos(2.0 * math.pi * i / frame_size)) for i in range(frame_size)]
        if HAS_NUMPY:
            self._window_np = np.array(self._window, dtype=np.float32)

        # Neural RNNoise engine (480 samples = 10ms @ 48 kHz)
        self._rnnoise: Any = None
        self._rn_in_buf: Any = np.zeros(0, dtype=np.int16) if HAS_NUMPY else []
        self._rn_out_buf: Any = np.zeros(0, dtype=np.float32) if HAS_NUMPY else []
        self._last_speech_prob: float = 0.0
        self._vad_hold_frames: int = 0
        self.backend_name = "dsp"

        if HAS_PYRNNOISE and HAS_NUMPY and sample_rate == 48000:
            try:
                self._rnnoise = pyrnnoise.RNNoise(sample_rate=48000)
                self.backend_name = "pyrnnoise"
            except Exception:
                self._rnnoise = None

        # STFT overlap buffers (for Room Echo Removal & spectral fallback)
        self._in_buf: list[float] = [0.0] * self.hop_size
        self._ola_tail: list[float] = [0.0] * self.hop_size
        self._stft_out_buf: list[float] = []
        self._noise_psd: list[float] = [1e-6] * self._bins
        self._reverb_env: list[float] = [0.0] * self._bins
        self._frames_seen = 0

        # Filter / dynamics state
        self._hp_prev_in = 0.0
        self._hp_prev_out = 0.0
        self._hp_sos: Any = None
        self._hp_zi: Any = None
        self._hp_cutoff_cached: float = -1.0
        self._lp_state = 0.0
        self._eq_b: Any = None
        self._eq_a: Any = None
        self._eq_zi: Any = None
        self._eq_profile_cached: str = ""
        self._gate_gain = 1.0
        self._agc_gain = 1.0

        self.last_meter = AudioMeterReading(backend=self.backend_name)

    def reset(self) -> None:
        if self._rnnoise is not None:
            try:
                self._rnnoise.reset()
            except Exception:
                pass
        if HAS_NUMPY:
            self._rn_in_buf = np.zeros(0, dtype=np.int16)
            self._rn_out_buf = np.zeros(0, dtype=np.float32)
        self._last_speech_prob = 0.0
        self._vad_hold_frames = 0
        self._in_buf = [0.0] * self.hop_size
        self._ola_tail = [0.0] * self.hop_size
        self._stft_out_buf = []
        self._noise_psd = [1e-6] * self._bins
        self._reverb_env = [0.0] * self._bins
        self._frames_seen = 0
        self._hp_prev_in = 0.0
        self._hp_prev_out = 0.0
        self._hp_zi = None
        self._lp_state = 0.0
        self._eq_zi = None
        self._gate_gain = 1.0
        self._agc_gain = 1.0

    def process_pcm16(self, pcm_bytes: bytes, cfg: dict[str, Any]) -> bytes:
        """Process signed 16-bit mono PCM audio (fast vectorized path when NumPy is installed)."""
        if not pcm_bytes:
            return b""
        usable = (len(pcm_bytes) // 2) * 2
        if usable == 0:
            return b""
        if HAS_NUMPY:
            arr = np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float32) / 32768.0
            out = self._process_numpy(arr, cfg)
            return np.clip(np.round(out * 32767.0), -32768, 32767).astype(np.int16).tobytes()
        samples = _pcm16_to_floats(pcm_bytes[:usable])
        processed = self.process_samples(samples, cfg)
        return _floats_to_pcm16(processed)

    def process_samples(self, samples: list[float], cfg: dict[str, Any]) -> list[float]:
        """Process float mono samples in [-1.0, 1.0]."""
        if not samples:
            return []
        if HAS_NUMPY:
            arr = np.asarray(samples, dtype=np.float32)
            return self._process_numpy(arr, cfg).tolist()
        return self._process_pure_python(samples, cfg)

    def _run_rnnoise(self, x: Any, strength: float) -> tuple[Any, float]:
        """Run C-compiled RNNoise GRU on arbitrary-length float32 numpy array."""
        n = x.size
        pcm_in = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
        buf = np.concatenate((self._rn_in_buf, pcm_in))
        n_frames = buf.size // 480
        probs: list[float] = []
        if n_frames > 0:
            consume = n_frames * 480
            feed = buf[:consume]
            self._rn_in_buf = buf[consume:]
            out_chunks = []
            for prob, denoised in self._rnnoise.denoise_chunk(feed):
                p = float(np.mean(prob))
                probs.append(p)
                chunk_f = denoised.reshape(-1).astype(np.float32) / 32768.0
                # Soft neural VAD floor suppression on strong noise settings
                if p > 0.35:
                    self._vad_hold_frames = 18  # ~180ms graceful hang time
                elif self._vad_hold_frames > 0:
                    self._vad_hold_frames -= 1
                if self._vad_hold_frames == 0 and strength > 0.45:
                    # Smoothly attenuate residual non-speech floor (prevents faint hiss)
                    atten = max(0.08, 1.0 - 0.85 * strength * (1.0 - p))
                    chunk_f = chunk_f * atten
                out_chunks.append(chunk_f)
            if out_chunks:
                self._rn_out_buf = np.concatenate((self._rn_out_buf, *out_chunks))
        else:
            self._rn_in_buf = buf

        if probs:
            self._last_speech_prob = float(np.mean(probs))

        if self._rn_out_buf.size >= n:
            denoised_out = self._rn_out_buf[:n]
            self._rn_out_buf = self._rn_out_buf[n:]
        elif self._rn_out_buf.size > 0:
            pad = np.zeros(n - self._rn_out_buf.size, dtype=np.float32)
            denoised_out = np.concatenate((pad, self._rn_out_buf))
            self._rn_out_buf = np.zeros(0, dtype=np.float32)
        else:
            denoised_out = x

        # Wet/dry blend by strength
        s = max(0.0, min(1.0, strength))
        blended = x * (1.0 - s) + denoised_out * s
        return blended.astype(np.float32), self._last_speech_prob

    def _process_numpy(self, samples: Any, cfg: dict[str, Any]) -> Any:
        in_rms_db, in_peak_db = compute_rms_db(samples)

        effects = cfg.get("effects", {})
        nr_cfg = effects.get("noise_removal") or effects.get("speaker_noise_removal") or {}
        echo_cfg = effects.get("room_echo_removal") or effects.get("speaker_echo_removal") or {}
        sv_cfg = effects.get("studio_voice") or {}

        nr_on = bool(nr_cfg.get("enabled", False))
        nr_strength = max(0.0, min(1.0, float(nr_cfg.get("strength", 0.85))))
        echo_on = bool(echo_cfg.get("enabled", False))
        echo_strength = max(0.0, min(1.0, float(echo_cfg.get("strength", 0.5))))
        sv_on = bool(sv_cfg.get("enabled", False))
        sv_profile = str(sv_cfg.get("micProfile", "default"))

        if sv_on:
            nr_on = True
            nr_strength = max(nr_strength, 0.85)
            echo_on = True
            echo_strength = max(echo_strength, 0.35)

        work = samples.astype(np.float32, copy=True)

        # 1. High-pass rumble filter (cuts laptop fan / AC sub-bass <85Hz)
        if cfg.get("highpass_enabled", True):
            cutoff = max(20.0, min(300.0, float(cfg.get("highpass_hz", 85.0))))
            if HAS_SCIPY:
                if self._hp_sos is None or abs(self._hp_cutoff_cached - cutoff) > 0.5:
                    self._hp_sos = sp_signal.butter(2, cutoff / (0.5 * self.sample_rate), btype="highpass", output="sos")
                    self._hp_zi = sp_signal.sosfilt_zi(self._hp_sos) * float(work[0])
                    self._hp_cutoff_cached = cutoff
                elif self._hp_zi is None:
                    self._hp_zi = sp_signal.sosfilt_zi(self._hp_sos) * float(work[0])
                work, self._hp_zi = sp_signal.sosfilt(self._hp_sos, work, zi=self._hp_zi)
                work = work.astype(np.float32)
            else:
                rc = 1.0 / (2.0 * math.pi * cutoff)
                dt = 1.0 / self.sample_rate
                alpha = rc / (rc + dt)
                prev_in = self._hp_prev_in
                prev_out = self._hp_prev_out
                for i in range(work.size):
                    x = float(work[i])
                    y = alpha * (prev_out + x - prev_in)
                    prev_in = x
                    prev_out = y
                    work[i] = y
                self._hp_prev_in = prev_in
                self._hp_prev_out = prev_out

        # 2. Neural Noise Removal (RNNoise GRU) + optional spectral Room Echo Removal
        speech_prob = 0.0
        if nr_on and self._rnnoise is not None:
            work, speech_prob = self._run_rnnoise(work, nr_strength)
            if echo_on:
                work = np.asarray(
                    self._process_spectral(work.tolist(), False, 0.0, True, echo_strength),
                    dtype=np.float32,
                )
        elif nr_on or echo_on:
            work = np.asarray(
                self._process_spectral(work.tolist(), nr_on, nr_strength, echo_on, echo_strength),
                dtype=np.float32,
            )
            cur_rms, _ = compute_rms_db(work)
            speech_prob = 1.0 if cur_rms > -35.0 else 0.1

        # 3. Studio Voice broadcast parametric EQ
        if sv_on:
            work = self._apply_studio_voice_eq_np(work, sv_profile)

        # 4. Noise Gate (smooth attack/release)
        gate_open = True
        if cfg.get("noise_gate_enabled", False):
            gate_thresh_db = float(cfg.get("noise_gate_db", -46.0))
            chunk_rms_db, _ = compute_rms_db(work)
            if self._rnnoise is not None and nr_on:
                gate_open = (self._vad_hold_frames > 0) or (chunk_rms_db >= gate_thresh_db + 6.0)
            else:
                gate_open = chunk_rms_db >= gate_thresh_db
            target_gate = 1.0 if gate_open else 0.04
            coeff = 0.45 if target_gate > self._gate_gain else 0.08
            self._gate_gain += coeff * (target_gate - self._gate_gain)
            work = work * self._gate_gain

        # 5. Speech-gated Voice Auto-Gain (ONLY boosts real speech, NEVER amplifies room noise!)
        if cfg.get("auto_gain_enabled", False) or sv_on:
            target_db = float(cfg.get("auto_gain_target_db", -16.0))
            chunk_rms_db, _ = compute_rms_db(work)
            is_speech = (
                (self._vad_hold_frames > 0 and chunk_rms_db > -40.0)
                if (self._rnnoise is not None and nr_on)
                else (chunk_rms_db > -34.0)
            )
            if is_speech:
                desired_db = max(-6.0, min(12.0, target_db - chunk_rms_db))
                desired_lin = 10.0 ** (desired_db / 20.0)
                self._agc_gain = 0.85 * self._agc_gain + 0.15 * desired_lin
            else:
                # Slowly relax back toward unity during silence so noise floor stays quiet
                self._agc_gain = 0.96 * self._agc_gain + 0.04 * 1.0
            work = work * self._agc_gain
            # Soft-knee tanh limiter
            over = np.abs(work) > 0.85
            if np.any(over):
                work[over] = np.sign(work[over]) * (0.85 + 0.15 * np.tanh((np.abs(work[over]) - 0.85) / 0.15))
            work = np.clip(work, -1.0, 1.0)

        out_rms_db, out_peak_db = compute_rms_db(work)
        self.last_meter = AudioMeterReading(
            input_rms_db=round(in_rms_db, 1),
            output_rms_db=round(out_rms_db, 1),
            input_peak_db=round(in_peak_db, 1),
            output_peak_db=round(out_peak_db, 1),
            noise_reduction_db=round(max(0.0, in_rms_db - out_rms_db), 1),
            speech_prob=round(speech_prob, 2),
            gate_open=gate_open,
            backend=self.backend_name,
        )
        return work

    def _apply_studio_voice_eq_np(self, x: Any, profile: str) -> Any:
        # 2-band tilt + presence boost tuned for laptop microphones
        low_g, high_g = {
            "warm": (1.30, 0.92),
            "bright": (0.88, 1.35),
            "full": (1.24, 1.22),
            "default": (1.12, 1.15),
        }.get(profile, (1.12, 1.15))
        if HAS_SCIPY:
            if self._eq_b is None or self._eq_a is None:
                self._eq_b, self._eq_a = sp_signal.butter(1, 1100.0 / (0.5 * self.sample_rate), btype="lowpass")
                self._eq_zi = sp_signal.lfilter_zi(self._eq_b, self._eq_a) * float(x[0])
            elif self._eq_zi is None:
                self._eq_zi = sp_signal.lfilter_zi(self._eq_b, self._eq_a) * float(x[0])
            low, self._eq_zi = sp_signal.lfilter(self._eq_b, self._eq_a, x, zi=self._eq_zi)
            low = low.astype(np.float32)
            high = x - low
            return np.clip(low * low_g + high * high_g, -1.0, 1.0).astype(np.float32)
        return np.asarray(self._apply_studio_voice_eq(x.tolist(), profile), dtype=np.float32)

    def _process_pure_python(self, samples: list[float], cfg: dict[str, Any]) -> list[float]:
        in_rms_db, in_peak_db = compute_rms_db(samples)
        effects = cfg.get("effects", {})
        nr_cfg = effects.get("noise_removal") or effects.get("speaker_noise_removal") or {}
        echo_cfg = effects.get("room_echo_removal") or effects.get("speaker_echo_removal") or {}
        sv_cfg = effects.get("studio_voice") or {}

        nr_on = bool(nr_cfg.get("enabled", False))
        nr_strength = max(0.0, min(1.0, float(nr_cfg.get("strength", 0.85))))
        echo_on = bool(echo_cfg.get("enabled", False))
        echo_strength = max(0.0, min(1.0, float(echo_cfg.get("strength", 0.5))))
        sv_on = bool(sv_cfg.get("enabled", False))
        sv_profile = str(sv_cfg.get("micProfile", "default"))
        if sv_on:
            nr_on = True
            nr_strength = max(nr_strength, 0.8)

        work = list(samples)
        if cfg.get("highpass_enabled", True):
            cutoff = max(20.0, min(300.0, float(cfg.get("highpass_hz", 85.0))))
            rc = 1.0 / (2.0 * math.pi * cutoff)
            dt = 1.0 / self.sample_rate
            alpha = rc / (rc + dt)
            prev_in, prev_out = self._hp_prev_in, self._hp_prev_out
            for i, x in enumerate(work):
                y = alpha * (prev_out + x - prev_in)
                prev_in, prev_out = x, y
                work[i] = y
            self._hp_prev_in, self._hp_prev_out = prev_in, prev_out

        if nr_on or echo_on:
            work = self._process_spectral(work, nr_on, nr_strength, echo_on, echo_strength)
        if sv_on:
            work = self._apply_studio_voice_eq(work, sv_profile)

        gate_open = True
        if cfg.get("noise_gate_enabled", False):
            gate_thresh_db = float(cfg.get("noise_gate_db", -46.0))
            chunk_rms_db, _ = compute_rms_db(work)
            gate_open = chunk_rms_db >= gate_thresh_db
            target_gate = 1.0 if gate_open else 0.06
            coeff = 0.35 if target_gate > self._gate_gain else 0.06
            self._gate_gain += coeff * (target_gate - self._gate_gain)
            g = self._gate_gain
            work = [x * g for x in work]

        if cfg.get("auto_gain_enabled", False) or sv_on:
            target_db = float(cfg.get("auto_gain_target_db", -16.0))
            chunk_rms_db, _ = compute_rms_db(work)
            if chunk_rms_db > -34.0:
                desired_db = max(-6.0, min(12.0, target_db - chunk_rms_db))
                desired_lin = 10.0 ** (desired_db / 20.0)
                self._agc_gain = 0.85 * self._agc_gain + 0.15 * desired_lin
            else:
                self._agc_gain = 0.96 * self._agc_gain + 0.04 * 1.0
            gain = self._agc_gain
            for i, x in enumerate(work):
                v = x * gain
                if abs(v) > 0.85:
                    v = math.copysign(0.85 + 0.15 * math.tanh((abs(v) - 0.85) / 0.15), v)
                work[i] = max(-1.0, min(1.0, v))

        out_rms_db, out_peak_db = compute_rms_db(work)
        self.last_meter = AudioMeterReading(
            input_rms_db=round(in_rms_db, 1),
            output_rms_db=round(out_rms_db, 1),
            input_peak_db=round(in_peak_db, 1),
            output_peak_db=round(out_peak_db, 1),
            noise_reduction_db=round(max(0.0, in_rms_db - out_rms_db), 1),
            speech_prob=1.0 if out_rms_db > -35.0 else 0.0,
            gate_open=gate_open,
            backend="dsp",
        )
        return work

    def _process_spectral(
        self,
        samples: list[float],
        nr_on: bool,
        nr_strength: float,
        echo_on: bool,
        echo_strength: float,
    ) -> list[float]:
        hop = self.hop_size
        n = self.frame_size
        combined = self._in_buf + samples
        out: list[float] = []
        pos = 0
        total = len(combined)

        while pos + n <= total:
            frame = combined[pos : pos + n]
            synth = self._spectral_frame(frame, nr_on, nr_strength, echo_on, echo_strength)
            for i in range(hop):
                self._stft_out_buf.append(self._ola_tail[i] + synth[i])
            self._ola_tail = synth[hop:]
            pos += hop

        # Keep all unconsumed samples so arbitrary chunk sizes (e.g. 480) align cleanly with hop=256
        self._in_buf = combined[pos:]

        need = len(samples)
        if len(self._stft_out_buf) >= need:
            out = self._stft_out_buf[:need]
            self._stft_out_buf = self._stft_out_buf[need:]
        else:
            pad = need - len(self._stft_out_buf)
            out = ([0.0] * pad) + self._stft_out_buf
            self._stft_out_buf = []
        return out

    def _spectral_frame(
        self,
        frame: list[float],
        nr_on: bool,
        nr_strength: float,
        echo_on: bool,
        echo_strength: float,
    ) -> list[float]:
        n = self.frame_size
        bins = self._bins
        self._frames_seen += 1

        if HAS_NUMPY:
            arr = np.asarray(frame, dtype=np.float32) * self._window_np
            spec = np.fft.rfft(arr)
            mag_sq = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float64)
            noise_psd = np.asarray(self._noise_psd, dtype=np.float64)
            reverb_env = np.asarray(self._reverb_env, dtype=np.float64)

            if self._frames_seen <= 8:
                noise_psd = 0.65 * noise_psd + 0.35 * mag_sq
            else:
                falling = mag_sq < noise_psd
                noise_psd = np.where(
                    falling,
                    0.88 * noise_psd + 0.12 * mag_sq,
                    np.where(mag_sq < noise_psd * 4.5, 0.93 * noise_psd + 0.07 * mag_sq, noise_psd),
                )

            gain = np.ones(bins, dtype=np.float64)
            if nr_on:
                oversub = 1.0 + 3.2 * nr_strength
                floor = max(0.02, 0.25 * (1.0 - 0.88 * nr_strength))
                clean_sq = np.maximum(mag_sq - oversub * noise_psd, 0.0)
                wiener = clean_sq / (mag_sq + 1e-12)
                gain *= np.maximum(floor, wiener)

            if echo_on:
                reverb_env = 0.88 * reverb_env + 0.12 * mag_sq
                late = echo_strength * 0.72 * reverb_env
                derev = np.maximum(mag_sq - late, 0.0) / (mag_sq + 1e-12)
                gain *= np.maximum(0.10, derev)

            self._noise_psd = noise_psd.tolist()
            self._reverb_env = reverb_env.tolist()
            return np.fft.irfft(spec * gain, n=n).tolist()

        windowed = [complex(frame[i] * self._window[i], 0.0) for i in range(n)]
        spec = _radix2_fft(windowed, inverse=False)
        gains = [1.0] * bins
        for k in range(bins):
            c = spec[k]
            mag_sq = c.real * c.real + c.imag * c.imag
            npsd = self._noise_psd[k]
            if self._frames_seen <= 8:
                npsd = 0.65 * npsd + 0.35 * mag_sq
            elif mag_sq < npsd:
                npsd = 0.88 * npsd + 0.12 * mag_sq
            elif mag_sq < npsd * 4.5:
                npsd = 0.93 * npsd + 0.07 * mag_sq
            self._noise_psd[k] = npsd
            g = 1.0
            if nr_on:
                oversub = 1.0 + 3.2 * nr_strength
                floor = max(0.02, 0.25 * (1.0 - 0.88 * nr_strength))
                g *= max(floor, max(0.0, mag_sq - oversub * npsd) / (mag_sq + 1e-12))
            if echo_on:
                renv = 0.88 * self._reverb_env[k] + 0.12 * mag_sq
                self._reverb_env[k] = renv
                g *= max(0.10, max(0.0, mag_sq - echo_strength * 0.72 * renv) / (mag_sq + 1e-12))
            gains[k] = g

        out_spec = [0j] * n
        for k in range(bins):
            out_spec[k] = spec[k] * gains[k]
        for k in range(1, bins - 1):
            out_spec[n - k] = spec[n - k] * gains[k]
        return [v.real for v in _radix2_fft(out_spec, inverse=True)]

    def _apply_studio_voice_eq(self, samples: list[float], profile: str) -> list[float]:
        alpha = 0.14
        low_gain, high_gain = {
            "warm": (1.28, 0.92),
            "bright": (0.90, 1.32),
            "full": (1.22, 1.18),
            "default": (1.10, 1.12),
        }.get(profile, (1.10, 1.12))
        lp = self._lp_state
        out: list[float] = []
        for x in samples:
            lp = lp + alpha * (x - lp)
            hp = x - lp
            out.append(max(-1.0, min(1.0, lp * low_gain + hp * high_gain)))
        self._lp_state = lp
        return out


def save_pcm16_wav(path: Path, pcm_bytes: bytes, sample_rate: int = 48000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return path
