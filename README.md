# Broadcast (Linux)

Native Rust + GPU-accelerated Linux recreation of NVIDIA Broadcast 2.3.0 for PipeWire audio, V4L2 webcams, and NVIDIA CUDA GPUs — with a Bun-managed Electron desktop UI and an instant Python fallback (`NVB_ENGINE=python`).

## Key Capabilities

- **Virtual Microphone (`NVIDIA Broadcast (Microphone)`)**
  - PipeWire native `Audio/Source/Virtual` visible directly in Discord, Google Meet, Zoom, Chrome, Brave, and OBS.
  - Pure-Rust real-time neural Noise Removal (`nnnoiseless` RNNoise GRU, 48 kHz, 10 ms frames, `0.071 ms p99`, zero dropouts), `realfft` STFT Room Echo Removal, Studio Voice EQ (`default`, `bright`, `full`, `warm`), 85 Hz fan-rumble HPF, Noise Gate, and neural-VAD-gated Auto-Gain.
  - Anti-loop routing invariants prevent virtual nodes from ever looping back into themselves.
- **Virtual Speakers (`NVIDIA Broadcast (Speakers)`)**
  - 48 kHz stereo PipeWire virtual sink routing cleanly to speakers or headphones (3.5mm, USB, Bluetooth) with automatic hot-plug reconnection and Bluetooth HFP/A2DP profile switching.
- **Virtual Camera (`NVIDIA Broadcast Camera`)**
  - Isolated short-lived `nvb-camera-worker` process running **Robust Video Matting (`rvm_mobilenetv3_fp32.onnx`)** via dynamic ONNX Runtime CUDA (`rvm_cuda`), CPU (`rvm_cpu`), or pure-Rust `fast_portrait` fallback.
  - Background Blur (`performance` / `quality`), Background Replacement (custom image or 4 studio presets), Background Removal (chroma key), Auto Frame, edge-preserving Video Noise Removal, Virtual Key Light, Vignette, Mirror, and Split Preview.
  - **100% OS RAM & VRAM reclamation:** exiting the isolated camera worker returns daemon memory to **4.9 MB PSS / 0 MiB VRAM** immediately after a video call ends.
- **Offline File & Screen-Recording Processor (`Files` tab & `process-file` CLI)**
  - Clean audio/video recordings (`.wav`, `.mp3`, `.m4a`, `.flac`, `.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`) using your active Broadcast effects.
  - Supports lossless `-c:v copy` video pass-through (`--no-video`) for ultra-fast voice cleanup on screen recordings, plus isolated GPU video effect rendering (`h264_nvenc` → `libx264` → `libopenh264` → `mpeg4`).
- **Desktop UI & Always-On Service**
  - Frameless dark Electron UI modeled on NVIDIA Broadcast 2.3.0 (62px icon rail, audio meters, waveform preview, live camera stage, system tray), plus `nv-broadcast.service` so virtual devices stay alive after closing the window.

## Benchmark Snapshot (Baseline vs Python vs Rust)

Measured with the unified 7-suite harness on Linux (`--profile full --rust-release`, RTX CUDA) — **11 / 11 acceptance gates PASS**. See [bench/RESULTS.md](bench/RESULTS.md) for the complete 7-suite tables.

| Metric | Baseline | Python | Rust | Result |
| --- | ---: | ---: | ---: | --- |
| Idle daemon RAM (PSS) | 0.0 MB | 99.0 MB | **3.0 MB** | **33x less RAM** |
| Post-CUDA-call idle RAM / VRAM | 0.0 MB / 0 MiB | 1,342.6 MB / 88 MiB | **4.9 MB / 0 MiB** | **274x less RAM, 0 MiB leak** |
| Active CPU video RAM (PSS) | 0.0 MB | 524.9 MB | **101.6 MB** | **5.2x less RAM** |
| Audio 10 ms frame `p99` latency | 0.000 ms | 0.552 ms | **0.071 ms** | **7.8x lower (0 dropouts)** |
| Audio `fan` Voice-SNR gain | 0.0 dB | +12.0 dB | **+14.8 dB** | **+2.8 dB better** |
| CLI cold start (`--version` `p50`) | N/A | 803.1 ms | **1.1 ms** | **705x faster** |
| Daemon cold boot to `READY` | N/A | 806.9 ms | **2.0 ms** | **396x faster** |
| HTTP `GET /api/state` `p95` | N/A | 333.66 ms | **1.37 ms** | **244x lower `p95`** |
| Idle background wakeups | 0.0 /s | 2.1 /s | **0.0 /s** | **Zero idle wakeups** |
| 720p RTX CUDA `video_noise_removal` | 1000 fps | 31.2 fps | **48.0 fps** | **1.54x faster** |
| 720p RTX CUDA `background_blur_quality` | 1000 fps | 57.5 fps | **31.9 fps** | **PASS (`>= 28 fps`)** |
| Reliability & anti-loop invariants | N/A | 5/5 | **5/5** | **100% PASS** |

## Repository & Module Documentation

| Module | Path | Purpose |
| --- | --- | --- |
| **Rust Workspace** | [rust/README.md](rust/README.md) | Native default backend architecture, crate graph, and build instructions |
| `nvb-core` | [rust/crates/nvb-core/README.md](rust/crates/nvb-core/README.md) | Config schema, anti-loop routing types, `CameraState`, `/proc` scanner, models, devices, doctor |
| `nvb-audio` | [rust/crates/nvb-audio/README.md](rust/crates/nvb-audio/README.md) | Pure-Rust RNNoise + STFT DSP, PipeWire audio engine, and offline `FileProcessor` |
| `nvb-video` | [rust/crates/nvb-video/README.md](rust/crates/nvb-video/README.md) | Dynamic ONNX Runtime RVM CUDA/CPU matting and pure-Rust video effects |
| `nvb-camera-worker` | [rust/crates/nvb-camera-worker/README.md](rust/crates/nvb-camera-worker/README.md) | Isolated short-lived camera/video worker (`stream`, `hold`, `bench`, `process-video`) |
| `nvb-daemon` | [rust/crates/nvb-daemon/README.md](rust/crates/nvb-daemon/README.md) | `axum` HTTP/SSE daemon with cached telemetry and worker supervision |
| `nvb-cli` | [rust/crates/nvb-cli/README.md](rust/crates/nvb-cli/README.md) | Drop-in `clap` CLI (`doctor`, `status`, `devices`, `mic`, `speaker`, `camera`, `serve`, `process-file`, `bench`) |
| **Python Backend** | [nv_broadcast/README.md](nv_broadcast/README.md) | Reference Python implementation, golden oracle, and `NVB_ENGINE=python` fallback |
| **Electron UI** | [ui/README.md](ui/README.md) | Broadcast 2.3.0 frameless desktop UI and system tray (Bun-only) |
| **Benchmark Suite** | [bench/README.md](bench/README.md) | Unified 7-suite Baseline vs Python vs Rust harness & [full results](bench/RESULTS.md) |

## Quick Start

### 1. Build the native Rust binaries (recommended default)

```bash
cd rust
cargo build --release -p nvb-cli -p nvb-daemon -p nvb-camera-worker
cd ..
```

`./nv-broadcast` automatically uses the Rust binaries when built. Set `NVB_ENGINE=python` at any time to fall back to the Python engine.

### 2. Optional Python GPU AI runtime & neural models

Required for ONNX Runtime CUDA shared libraries (`libonnxruntime.so*`), downloading `rvm_mobilenetv3_fp32.onnx`, or using the Python fallback:

```bash
./install-deps.sh --venv
```

### 3. Electron desktop UI dependencies (Bun only)

Install [Bun](https://bun.sh), then install UI dependencies (never use `npm`, `yarn`, or `pnpm`):

```bash
cd ui && bun install && cd ..
```

### 4. Virtual webcam (`v4l2loopback`)

On Fedora (with RPM Fusion Free):

```bash
sudo dnf install -y akmod-v4l2loopback v4l2loopback v4l-utils
echo 'v4l2loopback' | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast Camera" exclusive_caps=1' | sudo tee /etc/modprobe.d/98-v4l2loopback.conf
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback
```

### 5. Enable always-on user service (optional)

```bash
mkdir -p ~/.local/bin ~/.config/systemd/user
ln -sf "$(pwd)/nv-broadcast" ~/.local/bin/nv-broadcast
cp nv-broadcast.service ~/.config/systemd/user/nv-broadcast.service
systemctl --user daemon-reload
systemctl --user enable --now nv-broadcast.service
```

### 6. CLI & Offline Screen-Recording Usage

```bash
./nv-broadcast gui
./nv-broadcast doctor
./nv-broadcast status
./nv-broadcast devices

# Clean noisy voice in a screen recording (lossless video stream-copy + RNNoise audio)
./nv-broadcast process-file recording.mp4 -o recording_clean.mp4 --no-video

# Clean both voice and active video effects
./nv-broadcast process-file recording.mp4 -o recording_fx.mp4

# Instant rollback to Python backend if needed
NVB_ENGINE=python ./nv-broadcast status
```

## Running Tests & Benchmarks

```bash
# Rust workspace tests & Clippy
(cd rust && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings)

# Python unit & contract tests
python3 -m unittest discover -s tests -p 'test_*.py' -v

# UI Bun tests
bun test ./tests/*.test.js

# Fast CI-safe benchmark comparison
python3 bench/run_suite.py --targets baseline,python,rust --suites all --profile quick --rust-release
```

## CI/CD

- **CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)):** Runs Python tests, Bun syntax & unit tests, Bun-only lockfile enforcement, `bash -n` + `shellcheck`, `ffmpeg`-backed Rust workspace tests, Clippy `-D warnings`, release builds, benchmark contract smoke checks, and uploads Linux x86_64 binaries.
- **CD ([.github/workflows/release.yml](.github/workflows/release.yml)):** Triggered on `v*` tags (or manual `workflow_dispatch`); re-verifies all gates, builds `--locked --release` binaries, packages `nv-broadcast-<tag>-linux-x86_64.tar.gz` + `SHA256SUMS.txt`, and publishes a GitHub Release.

## License

MIT
