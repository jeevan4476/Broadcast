# Broadcast Linux: Rust Port Architecture & Migration Plan

- **Branch:** `feat/rust-port`
- **Worktree:** Dedicated `feat/rust-port` git worktree (isolated from live `master`)
- **Base commit:** `1ede5a9`
- **Local toolchain verified:** `rustc 1.99.1`, `cargo 1.99.1`, `clang 23.1.5` (`clang-devel` installed); `pipewire-devel` (`libpipewire-0.3.pc`) and `libv4l-devel` not yet installed (needed only when enabling native `pipewire-rs` bindings).

---

## 1. Industry Migration Playbooks Studied (Sources Cited)

Rather than doing an ad-hoc rewrite, this plan applies five proven patterns documented by engineering teams that migrated production systems to Rust:

1. **Anthropic — *Migrating our Messages API to Rust* (Nov 2026, `anthropic.com`)**
   - **Avoid partial Python/Rust FFI half-measures on the hot path:** Anthropic first ported only response parsing to Rust behind a Python boundary and saw only an 8–10% CPU win because payloads crossed the FFI boundary repeatedly. Moving the full request/stream path into a native Rust service yielded ~70% lower CPU, >80% lower memory, and ~20% lower median latency.
   - **Use the legacy Python service as a golden oracle:** Even after 98% of unit tests passed, live shadow traffic exposed subtle parity mismatches (~25% initial match rate) caused by weakened assertions. Anthropic closed the gap by running Python and Rust side-by-side on real traffic and diffing outputs programmatically.
   - **Applied here:** Do not wrap tiny DSP functions in PyO3 while keeping Python's ~98 MB interpreter + NumPy/SciPy runtime. Port the entire always-on daemon (`nvb-daemon`) to Rust while keeping the existing `/api/*` HTTP/SSE/MJPEG contract identical, and run a **differential oracle test suite** comparing Python vs. Rust outputs on config clamping, DSP WAV waveforms/RMS, and HTTP API payloads.

2. **Discord — *Why Discord is switching from Go to Rust* (`https://discord.com/blog/why-discord-is-switching-from-go-to-rust`)**
   - **Port the latency-sensitive, always-on hot path first:** Discord targeted its Read States service where garbage-collection pauses caused periodic tail-latency spikes and high resident memory.
   - **Applied here:** Broadcast's always-on hot path is 10 ms real-time audio frames (480 samples @ 48 kHz) plus PipeWire routing and virtual camera standby. Moving this path from Python + 4 `pw-cat` subprocesses to Rust eliminates GC/GIL jitter and drops idle daemon RAM from ~98 MB (and 894 MB pre-Phase-1) to **~10–20 MB**.

3. **Cloudflare — *How we built Pingora* (`https://blog.cloudflare.com/how-we-built-pingora-the-proxy-that-connects-cloudflare-to-the-internet/`)**
   - **Strangler Fig behind a stable API seam with instant rollback:** Cloudflare kept its internal control boundary stable, migrated one subsystem at a time, and allowed instant fallback via configuration toggles instead of a risky big-bang cutover.
   - **Applied here:** Keep `daemon.json`, `~/.config/nv-broadcast/config.json`, the Electron UI (`ui/`), and all `/api/*` endpoints unchanged. Ship a backend selector (`NVB_ENGINE=rust|python` in `nv-broadcast` / `nv-broadcast.service`) so you can switch back to Python in one command during rollout.

4. **Dropbox — *Rewriting the heart of our sync engine* (`https://dropbox.tech/infrastructure/rewriting-the-heart-of-our-sync-engine`)**
   - **Make invalid states unrepresentable and separate pure state machines from I/O:** Dropbox's Nucleus rewrite encoded invariants in Rust's type system and isolated deterministic control logic from I/O so simulation tests caught race conditions before shipping.
   - **Applied here:** Encode our hardest-won invariants into Rust types:
     - `PhysicalDevice` vs `VirtualDevice` newtypes so an audio playback route can never target `nv_broadcast_speakers` and mic capture can never target `nv_broadcast_mic` or an output monitor.
     - Explicit `CameraState` enum (`Off`, `Standby { device }`, `OnDemandActive { device, consumers }`, `ManualActive { device }`) with deterministic state-machine unit tests.

5. **Astral (`uv` / `Ruff`, `https://astral.sh`) & Linux Rust Multimedia Stack (`pipewire-rs`, `nnnoiseless`, `ort`)**
   - **Drop-in compatibility:** Preserve CLI flags (`nv-broadcast doctor|status|devices|mic|speaker|camera|serve|gui`) and on-disk config formats verbatim.
   - **Verified ecosystem crates:**
     - `nnnoiseless` (pure Rust RNNoise GRU, zero C FFI, 480-sample 48 kHz frames)
     - `realfft` + pure Rust biquad filters (replaces NumPy + SciPy STFT Room Echo Removal & Studio Voice EQ)
     - `pipewire` (`pipewire-rs`, official `libpipewire` bindings — eliminates all 4 `pw-cat` processes and `pw-link` polling via native PipeWire `Registry` hot-plug events)
     - `v4l` (pure Rust V4L2 capture & hardware controls — replaces `v4l2-ctl` subprocesses and OpenCV capture)
     - `ort` (`ort` 2.x with `load-dynamic` + CUDA EP) running in an **isolated `nvb-camera-worker` child process** so cuBLAS/cuDNN/CUDA memory (~580 MB) and VRAM (~146 MiB) are 100% reclaimed by the OS the instant a video call ends.

---

## 2. Current Codebase Audit (Python -> Rust Mapping)

Total codebase today: **11,402 LOC** across Python backend (7,701 LOC), Electron UI (1,961 LOC), and tests (1,740 LOC).

| Current Module | LOC | Keep / Port | Target Rust Crate | RAM Impact |
| --- | ---: | --- | --- | --- |
| `nv_broadcast/config.py` | 367 | **Port (Phase R1)** | `crates/nvb-core` | Part of ~10–15 MB daemon |
| `nv_broadcast/loopback_monitor.py` | 122 | **Port (Phase R1)** | `crates/nvb-core` | Pure `/proc` scan, <1 MB |
| `nv_broadcast/models.py` | 90 | **Port (Phase R1)** | `crates/nvb-core` | Adds SHA-256 verification (#9) |
| `nv_broadcast/audio_dsp.py` | 763 | **Port (Phase R1)** | `crates/nvb-audio` | `nnnoiseless` + `realfft` (~3 MB vs ~65 MB NumPy+SciPy+librnnoise) |
| `nv_broadcast/pipewire.py` | 539 | **Port (Phase R2)** | `crates/nvb-audio` | Native `pipewire-rs` (kills 4x `pw-cat` = -36 MB) |
| `nv_broadcast/audio_engine.py` | 799 | **Port (Phase R2)** | `crates/nvb-audio` | Zero-copy RT audio ring buffer + instant registry hot-plug |
| `nv_broadcast/devices.py` | 505 | **Port (Phase R2)** | `crates/nvb-core` | Direct PipeWire/sysfs/V4L2 queries (no `pactl` subprocess storms) |
| `nv_broadcast/v4l2_ctl.py` | 305 | **Port (Phase R2)** | `crates/nvb-video` | Native V4L2 ioctls via `v4l` crate |
| `nv_broadcast/server.py` | 618 | **Port (Phase R2)** | `crates/nvb-daemon` | `axum` + `tokio` HTTP/SSE/MJPEG (drop-in API parity) |
| `nv_broadcast/cli.py` + `doctor.py` | 756 | **Port (Phase R2)** | `crates/nvb-cli` | `clap` single binary, <10 ms startup |
| `nv_broadcast/video_engine.py` | 856 | **Port (Phase R3)** | `crates/nvb-camera-worker` | Isolated child process; standby writer stays lightweight |
| `nv_broadcast/video_effects.py` | 917 | **Port (Phase R3)** | `crates/nvb-camera-worker` | `ort` (RVM CUDA/CPU) + ONNX face/iris landmarks; exits on idle -> **0 MB retained** |
| `nv_broadcast/file_processor.py` | 273 | **Port (Phase R4)** | `crates/nvb-cli` / worker | Reuses `nvb-audio` + `nvb-camera-worker` + `ffmpeg` NVENC |
| `nv_broadcast/gui.py` | 787 | **Retire** | (Optional) | Legacy GTK4 fallback; Electron UI is primary |
| `ui/*.js`, `ui/index.html`, `ui/styles.css` | 1,961 | **Keep (Phase R1–R3)** | Unchanged (`Bun` + Electron) | Talks to `nvb-daemon` over exact `/api/*` contract; optional Tauri/GTK4 later |

---

## 3. Target Memory Budget (< 100–200 MB Guarantee)

| Operating State | Python (Pre-Phase-1) | Python (Current `master`) | Rust Target (`nvb-daemon` + `nvb-camera-worker`) |
| --- | ---: | ---: | ---: |
| **Daemon idle (Mic + Speakers + Cam Standby)** | 894 MB RAM, 146 MiB VRAM | 98 MB RAM, 0 MiB VRAM | **12–25 MB RAM, 0 MiB VRAM** |
| **After a CUDA video call ends** | 894 MB RAM, 146 MiB VRAM | ~700 MB RAM (dlopen sticky), 0 MiB VRAM | **12–25 MB RAM, 0 MiB VRAM** (worker exits -> OS reclaims 100%) |
| **During active Low-RAM camera call (CPU/TFLite/ONNX CPU)** | ~894 MB RAM | ~250–350 MB RAM | **90–160 MB RAM, 0 MiB VRAM** |
| **During active RTX CUDA camera call (`rvm_cuda`)** | ~1.4 GB peak | ~850 MB peak | **~15 MB daemon + isolated worker (~350–500 MB temporary, reclaimed on exit)** |

*(Note: Electron UI itself uses ~160–220 MB only while the settings window is open; closing the window or running headless with `nv-broadcast.service` keeps total system RAM at 12–25 MB. A later Phase R5 could optionally wrap `ui/` in Tauri/WebKitGTK to cut open-window UI RAM to ~40–60 MB as well.)*

---

## 4. Phased Strangler-Fig Rollout & Oracle Verification

### Phase R0 — Workspace Scaffold & Golden Contract Pinning (Zero Risk)
- Create Cargo workspace at `rust/` (`nvb-core`, `nvb-audio`, `nvb-video`, `nvb-daemon`, `nvb-cli`, `nvb-camera-worker`).
- Record golden test fixtures from the live Python implementation:
  - Config migration, clamping, and mutual-exclusion conflict matrices (`config.json`).
  - Raw + processed 48 kHz PCM WAV fixtures for `fan`, `typing`, `echo`, and `studio_voice` profiles.
  - Exact JSON response schemas for every `/api/*` endpoint (`GET /api/state`, `/api/devices`, `/api/doctor`, `POST /api/effect`, etc.).
- Add Rust CI job (`cargo test --workspace` + `cargo clippy --workspace -- -D warnings`) to `.github/workflows/ci.yml` alongside existing Python + Bun jobs.

### Phase R1 — Pure-Rust Core & DSP (`nvb-core` + `nvb-audio` DSP)
- Port `config.py`, `loopback_monitor.py`, `models.py` (with SHA-256 verification), and `audio_dsp.py` (`nnnoiseless` + `realfft` STFT Room Echo Removal + biquad HPF/EQ + VAD-gated AGC).
- **Oracle gate:** Run differential tests comparing Rust DSP vs Python DSP on identical noisy speech clips (verify SNR improvement, zero AGC noise-floor boost, RMS tolerance within ±1.5 dB, and zero panics on edge frames).
- **Benchmark gate:** Verify single 10 ms frame processing < 0.3 ms on CPU and RSS < 10 MB.

### Phase R2 — Always-On Daemon, Native PipeWire Audio, CLI & HTTP/SSE (`nvb-daemon` + `nvb-cli`)
- Port HTTP/SSE/MJPEG server (`axum`), CLI (`clap`), device enumeration, and audio engine.
- **Two-step PipeWire strategy:**
  1. First pass: isolated `pw-cat` + explicit `pw-link` (zero C-header build dependency, compiles anywhere immediately).
  2. Second pass (once `pipewire-devel` is installed): native `pipewire-rs` `Stream` + `Registry` listener — drops all 4 `pw-cat` processes, gives instant kernel-event hot-plug/suspend recovery, and cuts audio latency from ~20 ms to ~10 ms.
- **Strangler-Fig bridge for Camera:** During Phase R2, `nvb-daemon` runs the lightweight standby slate directly and delegates active GPU camera sessions to the existing Python worker (`python -m nv_broadcast.camera_worker`) as a short-lived child process — immediately achieving **~15–25 MB idle RAM AND 100% RAM/VRAM reclamation after every call** before writing a single line of Rust video inference!
- **Oracle gate:** Run a live differential harness hitting both Python and Rust daemons with every `/api/*` request and diffing JSON keys/types + verifying live PipeWire graph has zero feedback loops + live headphone unplug/replug recovery.

### Phase R3 — Native Rust Camera Worker (`nvb-camera-worker`)
- Port V4L2 capture & controls (`v4l` crate), `v4l2loopback` output, RVM ONNX CUDA/CPU (`ort` crate), and face/iris landmarks (ONNX face mesh/iris model, eliminating MediaPipe Python + OpenCV + Python `.venv` completely!).
- Keep `nvb-camera-worker` as a separate short-lived process spawned by `nvb-daemon` only when camera is active (manual or on-demand consumer wake), exchanging preview JPEG frames over a local Unix socket/pipe.
- **Oracle gate:** Verify >= 27 FPS on RTX 4050 at 1280x720, zero webcam LED when idle, and immediate return to < 25 MB RAM / 0 MiB VRAM when call ends.

### Phase R4 — Offline File Processor & Cutover
- Port `file_processor.py` (reusing `nvb-audio` + `nvb-camera-worker` + `ffmpeg` with optional `h264_nvenc`).
- Make Rust `nvb-daemon` the default in `nv-broadcast` and `nv-broadcast.service`, keeping `NVB_ENGINE=python` fallback until verified across multiple days of real Discord/Meet/headphone/suspend usage.

---

## 5. Key Risks & Decisions for Your Audit

1. **PipeWire native bindings (`pipewire-rs`) vs isolated `pw-cat`:**
   - Native `pipewire-rs` is faster, uses ~36 MB less total RAM across the cgroup, and reacts instantly to hot-plug/suspend events, but requires `sudo dnf install pipewire-devel` at build time (and `libpipewire-0.3-dev` on Ubuntu CI).
   - **Recommendation:** Start Phase R2 with native `pipewire-rs` behind a Cargo feature flag (with `pw-cat` fallback if built without `libpipewire-0.3`).
2. **Face & Iris Landmarks in Rust (Phase R3):**
   - MediaPipe's `face_landmarker.task` has no official Rust crate (its C++ wrapper is Python/C-oriented). In Phase R3 we can either load the extracted TFLite/ONNX face-mesh + iris models via `ort` (pure Rust + ONNX Runtime, zero MediaPipe dependency) OR keep the isolated Python `camera_worker` strictly for video while `nvb-daemon` + audio + CLI are 100% Rust!
   - **Recommendation:** Ship Phase R0 + R1 + R2 first (with isolated Python camera worker in Phase R2) so you get **~15–25 MB idle RAM and 100% post-call RAM reclamation immediately**, then evaluate pure-ONNX Rust face landmarks in Phase R3.
3. **Desktop UI (`ui/`):**
   - Keep the existing Broadcast 2.3.0 Electron UI untouched during R0–R3 (Zero UI regression risk, Bun-only tooling preserved). Tauri/WebKitGTK can be an optional Phase R5 if you also want open-window UI RAM below 100 MB.
