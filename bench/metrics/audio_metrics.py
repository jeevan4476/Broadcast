"""Deterministic audio corpus generation and objective speech-enhancement metrics.

All targets (`baseline`, `python`, `rust`) are evaluated using the exact same
48 kHz mono 16-bit PCM clips and identical metric functions:

- **SNR (dB)** and **Delta-SNR (dB)** against ground-truth clean speech (with
  cross-correlation lag alignment so algorithmic STFT/RNNoise group delay does
  not falsely degrade scores).
- **SI-SDR (dB)** (Scale-Invariant Signal-to-Distortion Ratio, Le Roux et al.
  2019), invariant to auto-gain scaling.
- **SegSNR (dB)** (frame-wise segmental SNR over speech-active frames).
- **Silence noise-floor attenuation (dB)**: output vs input RMS on non-speech
  frames.
- **Speech RMS preservation (dB)**: output vs clean RMS on speech-active frames.
- **Zero-speech AGC safety (dB)**: verifies pure-noise output never rises above
  input RMS.
- **Cross-target parity**: RMS delta (dB), spectral centroid shift (Hz), and
  lag-aligned normalized cross-correlation between Python and Rust outputs.
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

SAMPLE_RATE = 48000
FRAME_SAMPLES = 480
EPS = 1e-12


@dataclass
class AudioScenario:
    name: str
    description: str
    noisy_pcm: bytes
    clean_pcm: bytes | None
    speech_mask: list[bool]
    effect_cfg: dict[str, Any]
    sample_rate: int = SAMPLE_RATE


def pcm16_to_float_list(pcm_bytes: bytes) -> list[float]:
    usable = (len(pcm_bytes) // 2) * 2
    arr = array.array("h")
    arr.frombytes(pcm_bytes[:usable])
    inv = 1.0 / 32768.0
    return [s * inv for s in arr]


def float_list_to_pcm16(samples: list[float]) -> bytes:
    arr = array.array("h", (max(-32768, min(32767, int(round(s * 32767.0)))) for s in samples))
    return arr.tobytes()


def write_wav_mono16(path: Path, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return path


def read_wav_mono16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sampwidth * 8}-bit: {path}")
    if channels == 1:
        return raw, sr
    # Downmix multi-channel to mono
    arr = array.array("h")
    arr.frombytes(raw[: (len(raw) // (2 * channels)) * 2 * channels])
    mono = array.array("h", (int(sum(arr[i : i + channels]) / channels) for i in range(0, len(arr), channels)))
    return mono.tobytes(), sr


def _to_f64_array(pcm_or_samples: bytes | list[float] | Any) -> Any:
    if isinstance(pcm_or_samples, (bytes, bytearray)):
        usable = (len(pcm_or_samples) // 2) * 2
        if HAS_NUMPY:
            return np.frombuffer(pcm_or_samples[:usable], dtype=np.int16).astype(np.float64) / 32768.0
        return pcm16_to_float_list(bytes(pcm_or_samples[:usable]))
    if HAS_NUMPY:
        return np.asarray(pcm_or_samples, dtype=np.float64)
    return [float(x) for x in pcm_or_samples]


def rms_db(samples: Any) -> float:
    if HAS_NUMPY and isinstance(samples, np.ndarray):
        if samples.size == 0:
            return -90.0
        val = float(np.sqrt(np.mean(samples * samples)))
    else:
        n = len(samples)
        if n == 0:
            return -90.0
        val = math.sqrt(sum(x * x for x in samples) / n)
    return max(-90.0, 20.0 * math.log10(max(val, 1e-45)))


def peak_db(samples: Any) -> float:
    if HAS_NUMPY and isinstance(samples, np.ndarray):
        if samples.size == 0:
            return -90.0
        val = float(np.max(np.abs(samples)))
    else:
        if not samples:
            return -90.0
        val = max(abs(x) for x in samples)
    return max(-90.0, 20.0 * math.log10(max(val, 1e-45)))


def _best_lag_align(ref: Any, est: Any, max_lag: int = 1600, max_neg_lag: int = 160) -> tuple[Any, Any, int, float]:
    """Align `est` to `ref` over [-max_neg_lag, +max_lag] maximizing normalized cross-correlation.

    Positive lag means `est` is delayed relative to `ref` (standard causal DSP latency,
    e.g. 960 samples for RNNoise or ~1216 samples for RNNoise + STFT echo removal).
    """
    n = len(ref) if not (HAS_NUMPY and isinstance(ref, np.ndarray)) else int(ref.size)
    m = len(est) if not (HAS_NUMPY and isinstance(est, np.ndarray)) else int(est.size)
    length = min(n, m)
    if length < 64:
        return ref[:length], est[:length], 0, 0.0
    ref_c = ref[:length]
    est_c = est[:length]
    search_pos = min(max_lag, length // 4)
    search_neg = min(max_neg_lag, length // 8)

    if HAS_NUMPY:
        r = np.asarray(ref_c, dtype=np.float64)
        e = np.asarray(est_c, dtype=np.float64)
        r_norm = float(np.linalg.norm(r))
        e_norm = float(np.linalg.norm(e))
        if r_norm < 1e-9 or e_norm < 1e-9:
            return r, e, 0, 0.0
        # FFT-accelerated cross-correlation over [-search_neg, +search_pos]
        nfft = 1 << (2 * length - 1).bit_length()
        cc = np.fft.irfft(np.fft.rfft(e, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)
        pos_slice = cc[: search_pos + 1]
        best_pos_idx = int(np.argmax(pos_slice))
        if search_neg > 0:
            neg_slice = cc[nfft - search_neg :]
            best_neg_idx = int(np.argmax(neg_slice))
            if pos_slice[best_pos_idx] >= neg_slice[best_neg_idx]:
                lag = best_pos_idx
            else:
                lag = best_neg_idx - search_neg
        else:
            lag = best_pos_idx
        if lag > 0:
            r_a, e_a = r[:-lag], e[lag:]
        elif lag < 0:
            k = -lag
            r_a, e_a = r[k:], e[:-k]
        else:
            r_a, e_a = r, e
        denom = float(np.linalg.norm(r_a) * np.linalg.norm(e_a)) + EPS
        corr = float(np.dot(r_a, e_a) / denom)
        return r_a, e_a, lag, corr

    # Pure-Python coarse-to-fine lag search (fast for CI without NumPy)
    r_list = [float(x) for x in ref_c]
    e_list = [float(x) for x in est_c]
    r_pow = sum(x * x for x in r_list)
    e_pow = sum(x * x for x in e_list)
    if r_pow < 1e-12 or e_pow < 1e-12:
        return r_list, e_list, 0, 0.0

    best_lag = 0
    best_corr = -2.0
    step = max(1, (search_pos + search_neg) // 48)
    stride = max(1, length // 4000)

    def _score(lag_val: int, st: int) -> float:
        if lag_val >= 0:
            end = length - lag_val
            dot = sum(r_list[i] * e_list[i + lag_val] for i in range(0, end, st))
            rp = sum(r_list[i] * r_list[i] for i in range(0, end, st))
            ep = sum(e_list[i + lag_val] * e_list[i + lag_val] for i in range(0, end, st))
        else:
            k = -lag_val
            end = length - k
            dot = sum(r_list[i + k] * e_list[i] for i in range(0, end, st))
            rp = sum(r_list[i + k] * r_list[i + k] for i in range(0, end, st))
            ep = sum(e_list[i] * e_list[i] for i in range(0, end, st))
        return dot / (math.sqrt(rp * ep) + EPS)

    for lag_val in range(-search_neg, search_pos + 1, step):
        c = _score(lag_val, stride)
        if c > best_corr:
            best_corr = c
            best_lag = lag_val

    fine_lo = max(-search_neg, best_lag - step)
    fine_hi = min(search_pos, best_lag + step)
    for lag_val in range(fine_lo, fine_hi + 1):
        c = _score(lag_val, max(1, stride // 2))
        if c > best_corr:
            best_corr = c
            best_lag = lag_val

    if best_lag > 0:
        r_a, e_a = r_list[:-best_lag], e_list[best_lag:]
    elif best_lag < 0:
        k = -best_lag
        r_a, e_a = r_list[k:], e_list[:-k]
    else:
        r_a, e_a = r_list, e_list
    dot = sum(a * b for a, b in zip(r_a, e_a))
    rp = sum(a * a for a in r_a)
    ep = sum(b * b for b in e_a)
    corr = dot / (math.sqrt(rp * ep) + EPS)
    return r_a, e_a, best_lag, corr


def compute_snr_db(clean: Any, est: Any, align: bool = True) -> tuple[float, int]:
    """Compute SNR in dB against clean reference (optionally lag-aligned)."""
    lag = 0
    if align:
        r, e, lag, _ = _best_lag_align(clean, est)
    else:
        n = min(len(clean), len(est))
        r, e = clean[:n], est[:n]
    if HAS_NUMPY:
        r_np = np.asarray(r, dtype=np.float64)
        e_np = np.asarray(e, dtype=np.float64)
        sig_pow = float(np.sum(r_np * r_np))
        err_pow = float(np.sum((e_np - r_np) ** 2))
    else:
        sig_pow = sum(a * a for a in r)
        err_pow = sum((b - a) ** 2 for a, b in zip(r, e))
    if sig_pow < EPS:
        return -90.0, lag
    return float(10.0 * math.log10((sig_pow + EPS) / (err_pow + EPS))), lag


def compute_si_sdr_db(clean: Any, est: Any, align: bool = True) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    if align:
        r, e, _, _ = _best_lag_align(clean, est)
    else:
        n = min(len(clean), len(est))
        r, e = clean[:n], est[:n]
    if HAS_NUMPY:
        r_np = np.asarray(r, dtype=np.float64)
        e_np = np.asarray(e, dtype=np.float64)
        r_np = r_np - np.mean(r_np)
        e_np = e_np - np.mean(e_np)
        r_energy = float(np.dot(r_np, r_np))
        if r_energy < EPS:
            return -90.0
        alpha = float(np.dot(e_np, r_np)) / (r_energy + EPS)
        s_target = alpha * r_np
        e_noise = e_np - s_target
        t_pow = float(np.dot(s_target, s_target))
        n_pow = float(np.dot(e_noise, e_noise))
    else:
        n = len(r)
        if n == 0:
            return -90.0
        rm = sum(r) / n
        em = sum(e) / n
        rz = [a - rm for a in r]
        ez = [b - em for b in e]
        r_energy = sum(a * a for a in rz)
        if r_energy < EPS:
            return -90.0
        alpha = sum(a * b for a, b in zip(ez, rz)) / (r_energy + EPS)
        t_pow = sum((alpha * a) ** 2 for a in rz)
        n_pow = sum((b - alpha * a) ** 2 for a, b in zip(rz, ez))
    return float(10.0 * math.log10((t_pow + EPS) / (n_pow + EPS)))


def compute_seg_snr_db(
    clean: Any,
    est: Any,
    speech_mask: list[bool],
    frame_samples: int = FRAME_SAMPLES,
    lag: int = 0,
) -> float:
    """Compute Segmental SNR (clamped [-10, +35] dB per speech-active 10ms frame)."""
    n_clean = len(clean)
    n_est = len(est)
    if lag > 0:
        r = clean[: min(n_clean, n_est - lag)]
        e = est[lag : lag + len(r)]
    elif lag < 0:
        k = -lag
        r = clean[k : k + min(n_clean - k, n_est)]
        e = est[: len(r)]
    else:
        ln = min(n_clean, n_est)
        r, e = clean[:ln], est[:ln]

    seg_vals: list[float] = []
    n_frames = min(len(speech_mask), len(r) // frame_samples)
    for fi in range(n_frames):
        if not speech_mask[fi]:
            continue
        s0 = fi * frame_samples
        s1 = s0 + frame_samples
        rf = r[s0:s1]
        ef = e[s0:s1]
        if HAS_NUMPY:
            rf_np = np.asarray(rf, dtype=np.float64)
            ef_np = np.asarray(ef, dtype=np.float64)
            sp = float(np.sum(rf_np * rf_np))
            ep = float(np.sum((ef_np - rf_np) ** 2))
        else:
            sp = sum(a * a for a in rf)
            ep = sum((b - a) ** 2 for a, b in zip(rf, ef))
        if sp < 1e-8:
            continue
        frame_snr = 10.0 * math.log10((sp + EPS) / (ep + EPS))
        seg_vals.append(max(-10.0, min(35.0, frame_snr)))
    return sum(seg_vals) / len(seg_vals) if seg_vals else 0.0


def compute_spectral_centroid_hz(samples: Any, sample_rate: int = SAMPLE_RATE) -> float:
    """Compute mean power-weighted spectral centroid (Hz)."""
    if HAS_NUMPY:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size < 64:
            return 0.0
        n = min(arr.size, 65536)
        win = np.hanning(n)
        mag = np.abs(np.fft.rfft(arr[:n] * win))
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        denom = float(np.sum(mag))
        if denom < EPS:
            return 0.0
        return float(np.sum(freqs * mag) / denom)

    # Pure-Python lightweight zero-crossing-rate approximation fallback
    n = len(samples)
    if n < 2:
        return 0.0
    crossings = 0
    for i in range(1, n):
        if (samples[i - 1] < 0.0 <= samples[i]) or (samples[i - 1] >= 0.0 > samples[i]):
            crossings += 1
    return 0.5 * sample_rate * (crossings / (n - 1))


def _masked_rms_db(samples: Any, speech_mask: list[bool], want_speech: bool, frame_samples: int = FRAME_SAMPLES) -> float:
    n_frames = min(len(speech_mask), len(samples) // frame_samples)
    sq_sum = 0.0
    count = 0
    for fi in range(n_frames):
        if speech_mask[fi] != want_speech:
            continue
        s0 = fi * frame_samples
        s1 = s0 + frame_samples
        sl = samples[s0:s1]
        if HAS_NUMPY and isinstance(sl, np.ndarray):
            sq_sum += float(np.sum(sl * sl))
        else:
            sq_sum += sum(x * x for x in sl)
        count += frame_samples
    if count == 0:
        return -90.0
    return max(-90.0, 20.0 * math.log10(max(math.sqrt(sq_sum / count), 1e-45)))


def evaluate_audio_clip(
    noisy_pcm: bytes,
    processed_pcm: bytes,
    clean_pcm: bytes | None,
    speech_mask: list[bool],
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """Compute the full objective audio quality metric dictionary for one scenario."""
    noisy = _to_f64_array(noisy_pcm)
    proc = _to_f64_array(processed_pcm)
    clean = _to_f64_array(clean_pcm) if clean_pcm is not None else None

    in_rms = rms_db(noisy)
    out_rms = rms_db(proc)
    in_peak = peak_db(noisy)
    out_peak = peak_db(proc)

    in_silence_rms = _masked_rms_db(noisy, speech_mask, want_speech=False)
    out_silence_rms = _masked_rms_db(proc, speech_mask, want_speech=False)
    silence_atten_db = in_silence_rms - out_silence_rms

    in_speech_rms = _masked_rms_db(noisy, speech_mask, want_speech=True)
    out_speech_rms = _masked_rms_db(proc, speech_mask, want_speech=True)
    centroid_hz = compute_spectral_centroid_hz(proc, sample_rate=sample_rate)

    in_voice_snr = in_speech_rms - in_silence_rms
    out_voice_snr = out_speech_rms - out_silence_rms

    result: dict[str, Any] = {
        "input_rms_db": round(in_rms, 3),
        "output_rms_db": round(out_rms, 3),
        "rms_delta_db": round(out_rms - in_rms, 3),
        "input_peak_db": round(in_peak, 3),
        "output_peak_db": round(out_peak, 3),
        "input_silence_rms_db": round(in_silence_rms, 3),
        "output_silence_rms_db": round(out_silence_rms, 3),
        "silence_attenuation_db": round(silence_atten_db, 3),
        "input_speech_rms_db": round(in_speech_rms, 3),
        "output_speech_rms_db": round(out_speech_rms, 3),
        "input_voice_snr_db": round(in_voice_snr, 3),
        "output_voice_snr_db": round(out_voice_snr, 3),
        "delta_voice_snr_db": round(out_voice_snr - in_voice_snr, 3),
        "spectral_centroid_hz": round(centroid_hz, 2),
    }

    if clean is not None and any(speech_mask):
        in_snr, _ = compute_snr_db(clean, noisy, align=False)
        out_snr, lag = compute_snr_db(clean, proc, align=True)
        in_sisdr = compute_si_sdr_db(clean, noisy, align=False)
        out_sisdr = compute_si_sdr_db(clean, proc, align=True)
        in_segsnr = compute_seg_snr_db(clean, noisy, speech_mask, lag=0)
        out_segsnr = compute_seg_snr_db(clean, proc, speech_mask, lag=lag)
        clean_speech_rms = _masked_rms_db(clean, speech_mask, want_speech=True)

        result.update(
            {
                "input_snr_db": round(in_snr, 3),
                "output_snr_db": round(out_snr, 3),
                "delta_snr_db": round(out_snr - in_snr, 3),
                "input_si_sdr_db": round(in_sisdr, 3),
                "output_si_sdr_db": round(out_sisdr, 3),
                "delta_si_sdr_db": round(out_sisdr - in_sisdr, 3),
                "input_seg_snr_db": round(in_segsnr, 3),
                "output_seg_snr_db": round(out_segsnr, 3),
                "delta_seg_snr_db": round(out_segsnr - in_segsnr, 3),
                "speech_preservation_db": round(out_speech_rms - clean_speech_rms, 3),
                "alignment_lag_samples": int(lag),
            }
        )
    else:
        # Pure-noise or external WAV without clean reference
        result["agc_noise_safety_pass"] = bool((out_rms - in_rms) <= 0.5)

    return result


def compare_target_waveforms(pcm_a: bytes, pcm_b: bytes, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """Compare two target outputs (e.g. Python vs Rust) for DSP waveform parity."""
    a = _to_f64_array(pcm_a)
    b = _to_f64_array(pcm_b)
    _, _, lag, corr = _best_lag_align(a, b, max_lag=1600, max_neg_lag=1600)
    rms_a = rms_db(a)
    rms_b = rms_db(b)
    cent_a = compute_spectral_centroid_hz(a, sample_rate=sample_rate)
    cent_b = compute_spectral_centroid_hz(b, sample_rate=sample_rate)
    return {
        "rms_delta_db": round(rms_b - rms_a, 3),
        "spectral_centroid_shift_hz": round(cent_b - cent_a, 2),
        "max_cross_correlation": round(corr, 4),
        "alignment_lag_samples": int(lag),
    }


def _generate_clean_and_noises(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """Synthesize deterministic clean harmonic speech + individual noise beds."""
    n = int(sample_rate * duration_sec)
    n_frames = n // FRAME_SAMPLES

    if HAS_NUMPY:
        rng = np.random.default_rng(42)
        t = np.arange(n, dtype=np.float64) / sample_rate
        # Spoken cadence with explicit silent gaps so silence floor & speech preservation separate cleanly
        raw_env = np.sin(2.0 * np.pi * 0.45 * t)
        env = np.where(raw_env > 0.12, (np.clip(raw_env, 0.0, 1.0) ** 0.7), 0.0) * (
            0.65 + 0.35 * np.sin(2.0 * np.pi * 3.8 * t)
        )
        f0 = 155.0 + 28.0 * np.sin(2.0 * np.pi * 0.9 * t)
        phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
        clean = env * (
            0.22 * np.sin(phase)
            + 0.14 * np.sin(2.0 * phase)
            + 0.09 * np.sin(3.0 * phase)
            + 0.05 * np.sin(5.0 * phase)
        )
        # Fan noise + 120/240 Hz hum
        white = rng.normal(0.0, 1.0, n)
        b = np.ones(12) / 12.0
        fan = np.convolve(white, b, mode="same") * 0.12
        hum = 0.04 * np.sin(2.0 * np.pi * 120.0 * t) + 0.03 * np.sin(2.0 * np.pi * 240.0 * t)
        fan_noise = fan + hum

        # Mechanical keyboard transients
        typing_noise = np.zeros(n, dtype=np.float64)
        for idx in range(int(0.15 * sample_rate), n - 400, int(0.17 * sample_rate)):
            click_len = 360
            decay = np.exp(-np.arange(click_len) / 48.0)
            typing_noise[idx : idx + click_len] += rng.normal(0.0, 0.28, click_len) * decay

        # Room reflections (45ms, 95ms, 160ms)
        reverb = np.zeros(n, dtype=np.float64)
        for delay_ms, gain in ((45, 0.42), (95, 0.28), (160, 0.16)):
            d = int(sample_rate * delay_ms / 1000)
            if d < n:
                reverb[d:] += clean[:-d] * gain

        speech_mask = [
            bool(float(np.mean(env[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])) > 0.05)
            for i in range(n_frames)
        ]

        def to_pcm(sig: Any) -> bytes:
            return (np.clip(sig, -0.95, 0.95) * 32767.0).astype(np.int16).tobytes()

        return {
            "clean": to_pcm(clean),
            "fan": to_pcm(clean + fan_noise),
            "typing": to_pcm(clean + typing_noise),
            "echo": to_pcm(clean + reverb),
            "fan_echo": to_pcm(clean + reverb + fan_noise),
            "typing_echo": to_pcm(clean + reverb + typing_noise),
            "pure_noise": to_pcm(fan_noise),
            "speech_mask": speech_mask,
            "pure_noise_mask": [False] * n_frames,
        }

    # Pure-Python deterministic fallback (for CI environments without NumPy)
    clean_l = [0.0] * n
    fan_l = [0.0] * n
    typing_l = [0.0] * n
    reverb_l = [0.0] * n
    phase = 0.0
    lcg = 123456789

    def _rand_unit() -> float:
        nonlocal lcg
        lcg = (1103515245 * lcg + 12345) & 0x7FFFFFFF
        return (lcg / 0x3FFFFFFF) - 1.0

    for i in range(n):
        t = i / sample_rate
        raw_env = math.sin(2.0 * math.pi * 0.45 * t)
        env = ((raw_env**0.7) if raw_env > 0.12 else 0.0) * (0.65 + 0.35 * math.sin(2.0 * math.pi * 3.8 * t))
        f0 = 155.0 + 28.0 * math.sin(2.0 * math.pi * 0.9 * t)
        phase += 2.0 * math.pi * f0 / sample_rate
        clean_l[i] = env * (
            0.22 * math.sin(phase) + 0.14 * math.sin(2.0 * phase) + 0.09 * math.sin(3.0 * phase)
        )
        fan_l[i] = (
            0.05 * _rand_unit()
            + 0.04 * math.sin(2.0 * math.pi * 120.0 * t)
            + 0.03 * math.sin(2.0 * math.pi * 240.0 * t)
        )

    for idx in range(int(0.15 * sample_rate), n - 400, int(0.17 * sample_rate)):
        for k in range(360):
            typing_l[idx + k] += 0.28 * _rand_unit() * math.exp(-k / 48.0)

    for delay_ms, gain in ((45, 0.42), (95, 0.28), (160, 0.16)):
        d = int(sample_rate * delay_ms / 1000)
        for i in range(d, n):
            reverb_l[i] += clean_l[i - d] * gain

    speech_mask = []
    for fi in range(n_frames):
        sl = clean_l[fi * FRAME_SAMPLES : (fi + 1) * FRAME_SAMPLES]
        speech_mask.append((sum(x * x for x in sl) / FRAME_SAMPLES) > 1e-4)

    def mix(a: list[float], *others: list[float]) -> bytes:
        out = [max(-0.95, min(0.95, a[i] + sum(o[i] for o in others))) for i in range(n)]
        return float_list_to_pcm16(out)

    return {
        "clean": mix(clean_l),
        "fan": mix(clean_l, fan_l),
        "typing": mix(clean_l, typing_l),
        "echo": mix(clean_l, reverb_l),
        "fan_echo": mix(clean_l, reverb_l, fan_l),
        "typing_echo": mix(clean_l, reverb_l, typing_l),
        "pure_noise": mix(fan_l),
        "speech_mask": speech_mask,
        "pure_noise_mask": [False] * n_frames,
    }


def _mic_cfg(
    *,
    nr: bool = False,
    nr_strength: float = 0.85,
    echo: bool = False,
    echo_strength: float = 0.55,
    studio: bool = False,
    profile: str = "default",
    auto_gain: bool = True,
    noise_gate: bool = True,
) -> dict[str, Any]:
    return {
        "highpass_enabled": True,
        "highpass_hz": 85,
        "noise_gate_enabled": noise_gate,
        "noise_gate_db": -46.0,
        "auto_gain_enabled": auto_gain,
        "auto_gain_target_db": -16.0,
        "effects": {
            "noise_removal": {"enabled": nr, "strength": nr_strength},
            "room_echo_removal": {"enabled": echo, "strength": echo_strength},
            "studio_voice": {"enabled": studio, "micProfile": profile},
        },
    }


def build_audio_corpus(
    profile: str = "quick",
    custom_wav: Path | None = None,
) -> list[AudioScenario]:
    """Build the standardized audio quality evaluation scenarios."""
    duration = 2.0 if profile == "quick" else 4.0
    clips = _generate_clean_and_noises(duration_sec=duration, sample_rate=SAMPLE_RATE)
    smask = clips["speech_mask"]
    pmask = clips["pure_noise_mask"]

    scenarios = [
        AudioScenario(
            name="fan",
            description="Speech + stationary laptop fan roar + 120/240 Hz hum",
            noisy_pcm=clips["fan"],
            clean_pcm=clips["clean"],
            speech_mask=smask,
            effect_cfg=_mic_cfg(nr=True, nr_strength=0.85),
        ),
        AudioScenario(
            name="typing",
            description="Speech + mechanical keyboard transients",
            noisy_pcm=clips["typing"],
            clean_pcm=clips["clean"],
            speech_mask=smask,
            effect_cfg=_mic_cfg(nr=True, nr_strength=0.9),
        ),
        AudioScenario(
            name="echo",
            description="Speech + room reflections (45ms, 95ms, 160ms)",
            noisy_pcm=clips["echo"],
            clean_pcm=clips["clean"],
            speech_mask=smask,
            effect_cfg=_mic_cfg(nr=False, echo=True, echo_strength=0.7),
        ),
        AudioScenario(
            name="fan_echo",
            description="Speech + laptop fan noise + room reverb",
            noisy_pcm=clips["fan_echo"],
            clean_pcm=clips["clean"],
            speech_mask=smask,
            effect_cfg=_mic_cfg(nr=True, nr_strength=0.85, echo=True, echo_strength=0.6),
        ),
        AudioScenario(
            name="pure_noise",
            description="Stationary fan noise with zero speech (AGC pumping safety test)",
            noisy_pcm=clips["pure_noise"],
            clean_pcm=None,
            speech_mask=pmask,
            effect_cfg=_mic_cfg(nr=True, nr_strength=0.85, auto_gain=True),
        ),
        AudioScenario(
            name="studio_voice_warm",
            description="Studio Voice (warm profile) on fan + echo call environment",
            noisy_pcm=clips["fan_echo"],
            clean_pcm=clips["clean"],
            speech_mask=smask,
            effect_cfg=_mic_cfg(studio=True, profile="warm"),
        ),
    ]

    if profile == "full":
        scenarios.extend(
            [
                AudioScenario(
                    name="typing_echo",
                    description="Speech + keyboard clicks + room reverb",
                    noisy_pcm=clips["typing_echo"],
                    clean_pcm=clips["clean"],
                    speech_mask=smask,
                    effect_cfg=_mic_cfg(nr=True, nr_strength=0.9, echo=True, echo_strength=0.6),
                ),
                AudioScenario(
                    name="studio_voice_bright",
                    description="Studio Voice (bright profile) on fan noise",
                    noisy_pcm=clips["fan"],
                    clean_pcm=clips["clean"],
                    speech_mask=smask,
                    effect_cfg=_mic_cfg(studio=True, profile="bright"),
                ),
                AudioScenario(
                    name="studio_voice_full",
                    description="Studio Voice (full profile) on fan noise",
                    noisy_pcm=clips["fan"],
                    clean_pcm=clips["clean"],
                    speech_mask=smask,
                    effect_cfg=_mic_cfg(studio=True, profile="full"),
                ),
            ]
        )

    if custom_wav is not None and custom_wav.is_file():
        pcm, sr = read_wav_mono16(custom_wav)
        n_frames = (len(pcm) // 2) // FRAME_SAMPLES
        # Estimate speech frames by energy threshold for reporting
        vals = _to_f64_array(pcm)
        frame_dbs = [
            rms_db(vals[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(n_frames)
        ]
        thresh = (max(frame_dbs) - 18.0) if frame_dbs else -35.0
        est_mask = [d >= thresh for d in frame_dbs]
        scenarios.append(
            AudioScenario(
                name=f"custom_{custom_wav.stem}",
                description=f"User-supplied recording ({custom_wav.name})",
                noisy_pcm=pcm,
                clean_pcm=None,
                speech_mask=est_mask,
                effect_cfg=_mic_cfg(nr=True, nr_strength=0.85, echo=True, echo_strength=0.4),
                sample_rate=sr,
            )
        )

    return scenarios
