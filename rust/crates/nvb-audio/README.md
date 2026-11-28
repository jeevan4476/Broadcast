# `nvb-audio`

Pure-Rust real-time neural audio DSP, PipeWire virtual audio routing engine, and offline audio/video file processor.

## Modules

| File | Responsibility |
| --- | --- |
| `src/lib.rs` | PCM16 <-> `f32` conversion (`1.64 us` / 480 samples, 113.8x faster than Python), RMS/peak metering, isolated `pw-cat` argument builder (`node.autoconnect = false`, `node.dont-reconnect = true`), `pw-link` parser, and deterministic channel port pairing |
| `src/dsp.rs` | 48 kHz mono real-time DSP pipeline: 2nd-order Butterworth HPF -> pure-Rust RNNoise GRU (`nnnoiseless`, 480 samples) -> `realfft` 512/256 OLA STFT Room Echo Removal -> Studio Voice 2-band tilt EQ -> neural-VAD-aware Noise Gate -> VAD-gated Auto-Gain (AGC) & `tanh` limiter; WAV I/O & synthetic clip generator |
| `src/engine.rs` | `PipeWireVirtualManager` and `AudioEngine` managing `NVIDIA Broadcast (Microphone)` and `NVIDIA Broadcast (Speakers)` virtual PipeWire nodes, live audio streaming threads, and anti-loop link verification |
| `src/file_processor.rs` | Offline `FileProcessor` for `.wav`/`.mp3`/`.m4a`/`.flac`/`.ogg`/`.opus` and `.mp4`/`.mkv`/`.webm`/`.mov`/`.avi`: cleans audio with `AudioProcessor`, stream-copies video (`-c:v copy`) for lossless screen-recording voice cleanup, or spawns isolated `nvb-camera-worker process-video` when visual effects are active |

## Performance Highlights (`--profile full`)

- **10 ms frame latency (`noise_removal`):** `57.8 us` p50 / `71.2 us` p99 / `0` deadline misses (`0.58%` of 1 CPU core).
- **Voice-SNR gain (`fan`):** `+14.8 dB` (vs `+12.0 dB` Python); `echo` parity `corr = 1.000`.
- **Zero-speech AGC safety (`pure_noise`):** `-17.8 dB` (never amplifies background hiss).
- **Peak RSS during WAV processing:** `6.49 MB`.

## Testing

```bash
cargo test -p nvb-audio
```
