# Broadcast Linux — Rust Workspace (`rust/`)

Native Rust backend for NVIDIA Broadcast (Linux), designed for **~3 MB idle RAM**, **0 MiB idle GPU VRAM**, **sub-0.1 ms real-time audio DSP**, and **100% OS memory/VRAM reclamation** after every video session.

## Workspace Crates

| Crate | Type | Responsibility | Documentation |
| --- | --- | --- | --- |
| `nvb-core` | library | Config schema & sanitization, anti-loop routing newtypes, `CameraState` state machine, `/proc` loopback consumer monitor, SHA-256 model registry, device & doctor probes, HTTP wire contracts | [crates/nvb-core/README.md](crates/nvb-core/README.md) |
| `nvb-audio` | library | Pure-Rust RNNoise (`nnnoiseless`) + STFT Room Echo Removal (`realfft`) + Studio Voice EQ + VAD-gated AGC, PipeWire virtual audio engine, and offline `FileProcessor` | [crates/nvb-audio/README.md](crates/nvb-audio/README.md) |
| `nvb-video` | library | Dynamic ONNX Runtime (`ort 2.0`) RVM CUDA/CPU matting + `fast_portrait` fallback, pure-Rust video effects, V4L2 mode selector, YUYV packer | [crates/nvb-video/README.md](crates/nvb-video/README.md) |
| `nvb-camera-worker` | binary | Isolated short-lived camera/video worker (`stream`, `hold`, `bench`, `process-video`) spawned only while video is active | [crates/nvb-camera-worker/README.md](crates/nvb-camera-worker/README.md) |
| `nvb-daemon` | binary | Always-on `axum` HTTP/SSE daemon on a 2-thread Tokio runtime with cached hardware telemetry and drop-in `/api/*` parity | [crates/nvb-daemon/README.md](crates/nvb-daemon/README.md) |
| `nvb-cli` | binary | Drop-in `clap` CLI (`nv-broadcast`) and symmetric benchmark driver | [crates/nvb-cli/README.md](crates/nvb-cli/README.md) |

## Architectural Principles

1. **Process-isolated CUDA:** `nvb-daemon` never loads ONNX Runtime, CUDA, or cuDNN in-process. Active webcam sessions and offline video effects run inside short-lived `nvb-camera-worker` child processes so exiting returns system RAM to ~3–5 MB and GPU VRAM to 0 MiB.
2. **Make invalid states unrepresentable:** `PhysicalDevice` vs `VirtualDevice` newtypes reject audio feedback loops at construction time, and `CameraState` encodes `Off` / `Standby` / `OnDemandActive` / `ManualActive` transitions deterministically.
3. **Golden Python oracle fixtures:** `fixtures/*.json` pins config sanitization, conflict resolution, PCM16/RMS math, PipeWire port pairing, V4L2 mode selection, and HTTP API schemas against the Python reference implementation.
4. **Zero native SDK compile friction:** `ort` uses `load-dynamic` so CI and local builds compile in seconds without requiring CUDA/ORT C++ headers, while dynamically binding to local `libonnxruntime.so*` + CUDA/cuDNN libraries at runtime.

## Build & Test

```bash
# Run all workspace unit & integration tests (install ffmpeg for offline MP4 test)
cargo test --workspace

# Zero-warning Clippy lint across all crates and test targets
cargo clippy --workspace --all-targets -- -D warnings

# Build optimized release binaries used by ../nv-broadcast
cargo build --release -p nvb-cli -p nvb-daemon -p nvb-camera-worker
```
