# Broadcast Linux: Unified Benchmark and Comparison Suite Plan

This document specifies a single shared benchmark and verification harness (`bench/`) that runs identical workloads across three targets:

1. **`baseline` (No Broadcast):** System and raw media signals with Broadcast absent or bypassed.
2. **`python` (Python Broadcast):** Existing `nv_broadcast` daemon, DSP, video pipeline, and CLI.
3. **`rust` (Rust Broadcast):** New `nvb-*` workspace (`nvb-core`, `nvb-audio`, `nvb-video`, `nvb-daemon`, `nvb-cli`, `nvb-camera-worker`).

Every suite measures the exact same inputs, collects the same machine-readable schema, and produces side-by-side reports showing both **Broadcast vs No Broadcast** (quality gains and resource cost) and **Rust vs Python** (parity, speedup, memory savings, and tail-latency improvement).

---

## 1. Design Principles

- **One harness, three identical target adapters:** All three targets implement the same tiny JSON-over-stdio / HTTP adapter contract. No target gets custom scoring logic.
- **Works immediately at Phase R0 and fills in automatically through R4:** Unported Rust subsystems report `"status": "pending_phase_rX"` instead of breaking the run. Every completed phase turns pending cells into live verified measurements.
- **Deterministic by default, live hardware optional:**
  - `--profile quick` (~20 to 30 seconds, CI-safe, no webcam/mic hardware required): synthetic audio/video fixtures, frame latency, API/CLI benchmarks, differential parity, and fault-injection tests.
  - `--profile full` (~3 to 5 minutes, local hardware): adds live PipeWire graph inspection (`pw-top`, `pw-link`), live V4L2/v4l2loopback wake tests, `nvidia-smi` GPU power/P-state/VRAM sampling, and an optional `--soak <seconds>` leak run.
- **No root required:** Uses unprivileged Linux `/proc/<pid>/smaps_rollup`, `/proc/<pid>/stat`, `/proc/<pid>/status`, `/proc/<pid>/fd`, `nvidia-smi`, `pw-top`, `pw-link`, `v4l2-ctl`, and `/sys/class/power_supply/BAT*`. Optional root-only counters (such as Intel RAPL `/sys/class/powercap/intel-rapl:0/energy_uj`) degrade gracefully to `unavailable_nonroot`.

---

## 2. Complete Comparison Matrix (7 Suites)

### Suite 1: Audio Signal Quality and Noise Reduction (`audio_quality`)

Measures how much Broadcast improves real speech over `baseline` (raw mic/speaker audio) and verifies Rust DSP matches or beats Python DSP.

- **Identical 48 kHz 16-bit mono PCM test corpus:**
  - `clean_speech`: formant-modulated harmonic speech cadence with natural pauses (ground-truth clean reference).
  - `fan`: clean speech + stationary laptop fan noise + 120 Hz / 240 Hz AC hum.
  - `typing`: clean speech + sharp mechanical keyboard transients every ~170 ms.
  - `echo`: clean speech + room reflections at 45 ms, 95 ms, and 160 ms.
  - `fan_echo` and `typing_echo`: combined realistic call environments.
  - `pure_noise`: stationary fan noise with zero speech (tests AGC noise-floor pumping).
  - `studio_voice` profiles: `default`, `warm`, `bright`, `full`.
  - Optional user-supplied real recording (`--wav path/to/recording.wav`).
- **Metrics (identical across `baseline`, `python`, `rust`):**
  - **SNR (dB) and Delta-SNR (dB)** against clean reference.
  - **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio, dB):** standard speech-enhancement metric invariant to auto-gain scaling.
  - **Segmental SNR (SegSNR, dB):** separates speech-active frames from silence/noise-only frames.
  - **Silence noise-floor attenuation (dB):** how much noise is removed between spoken phrases.
  - **Speech RMS preservation (dB):** ensures speech is not clipped or crushed.
  - **Zero-speech AGC safety (dB):** asserts `pure_noise` output RMS never rises above input RMS.
  - **Python vs Rust parity:** RMS delta (dB), spectral centroid shift (Hz), and frame cross-correlation.
  - **Human A/B/C listening files:** writes `raw.wav`, `python.wav`, and `rust.wav` into the result folder.

### Suite 2: Real-Time Audio Latency, Jitter, and Throughput (`audio_latency`)

At 48 kHz, a 480-sample RNNoise frame has a hard **10.0 ms real-time budget**. GC pauses, GIL contention, or subprocess pipes show up directly in tail latency (`p99`, `p99.9`, `max`).

- **Workload:** 10,000 continuous 10 ms frames (100 seconds of audio) across `noise_removal`, `room_echo_removal`, and `studio_voice`.
- **Metrics (`baseline` passthrough vs `python` vs `rust`):**
  - Per-frame processing latency: **min, p50, p95, p99, p99.9, max** (microseconds).
  - **Real-Time Factor (RTF):** total compute time divided by audio duration.
  - **Budget misses:** count of frames exceeding `2.0 ms` (warning) and `10.0 ms` (hard xrun).
  - **Single-core CPU% at 1x real-time.**
  - **Live PipeWire graph telemetry (`--profile full`):** `pw-top` quantum, busy%, wait time, and xrun error counter comparing No Broadcast vs Python vs Rust.

### Suite 3: Memory (RAM/PSS) and GPU VRAM Lifecycle (`memory`)

Tracks full process-tree memory (daemon + all child `pw-cat`, `ffmpeg`, and `nvb-camera-worker` processes) via `/proc/<pid>/smaps_rollup` (`Rss`, `Pss`, `Private_Dirty`) and `nvidia-smi` per-PID VRAM.

- **5 lifecycle checkpoints (`baseline` = 0 vs `python` vs `rust`):**
  1. **Idle / Standby:** Mic + Speakers + Camera on-demand standby, no active video consumer.
  2. **Active Audio Call:** Mic RNNoise + Speaker NR streaming simultaneously for 30 to 60 s.
  3. **Active Video Call (CPU / Low-RAM mode):** 720p30 portrait blur/replace.
  4. **Active Video Call (RTX CUDA mode):** `rvm_cuda` 720p30 + face/iris landmarks.
  5. **Post-Call Idle (T+15s after video call ends):** Measures RAM and VRAM reclamation after a CUDA session (specifically exposes Python's sticky `dlopen` cuBLAS/cuDNN retention vs Rust's exiting `nvb-camera-worker`).
- **Soak / leak check (`--soak <seconds>`):** Samples RSS, PSS, and open FD count every 5 seconds and reports memory slope (`MB/min`) and FD drift.

### Suite 4: Video Quality, FPS, Frame Jitter, and On-Demand Wake (`video`)

- **Inputs:** Deterministic 1280x720 synthetic portrait frames (face + hair-edge detail + textured background + synthetic low-light noise) and optional live physical webcam (`--live-camera`).
- **Effects tested:** `background_blur` (performance and quality), `background_replace`, `background_remove`, `auto_frame`, `eye_contact`, `video_noise_removal`, `virtual_key_light`, `vignette`, and multi-effect combo (`blur + key_light + auto_frame`).
- **Metrics (`baseline` raw capture vs `python` vs `rust`):**
  - Sustained **FPS** and per-frame latency **p50, p95, p99, max (ms)** against the 33.3 ms (30 fps) budget.
  - **Mask parity (Python vs Rust):** alpha matte **IoU** and **MAE** on identical frames.
  - **Visual parity (Python vs Rust):** **PSNR (dB)** and **SSIM**, plus side-by-side comparison PNG strips.
  - **On-demand webcam wake and sleep (`--profile full`):**
    - External consumer open `/dev/videoN` -> first processed frame latency (ms).
    - External consumer close -> standby restored, physical webcam FD count = 0 (LED off), and VRAM = 0 MiB (seconds).

### Suite 5: Control Plane, CLI Startup, and HTTP/SSE/MJPEG API (`api_cli`)

- **CLI cold startup (`p50`, `p95` ms over 20 runs):** `doctor --json`, `status`, `devices --json`, `config`.
- **Daemon cold boot to `/api/health` ready (ms).**
- **HTTP endpoint latency (`p50`, `p95`, `p99` ms) and throughput (`req/s`):**
  - `GET /api/health`
  - `GET /api/state`
  - `GET /api/devices`
  - `POST /api/effect` (including mutual-exclusion toggle cycles)
- **SSE `/api/events` fan-out overhead:** CPU% and RSS with 1 vs 5 concurrent subscribers.
- **MJPEG `/api/camera/stream.mjpeg`:** effective frame delivery rate and dropped frames.
- **Differential API parity (Anthropic Oracle pattern):** replays the same sequence of HTTP calls against Python and Rust daemons and diffs status codes, auth rejection, JSON keys, and sanitized config values.

### Suite 6: System, Power, and Idle Wakeup Overhead (`system_impact`)

Directly answers: *"What is the real cost of keeping Broadcast running in the background compared to not running it at all?"*

- **Measured over a 30-second steady-state window (`baseline` vs `python` vs `rust`):**
  - **Total CPU time consumed** (`utime + stime` across process tree, in CPU-ms per wall-second and % of 1 core).
  - **Thread count and context switches per second** (`voluntary_ctxt_switches`, `nonvoluntary_ctxt_switches` from `/proc/<pid>/status`) as a direct proxy for laptop battery wakeups.
  - **Subprocess spawn rate** (e.g., periodic `pactl`, `pw-link`, `nvidia-smi` spawns per minute).
  - **GPU idle residency:** `nvidia-smi` P-state (`P8` vs `P0/P2`), GPU power draw (W), and VRAM (MiB) proving idle Broadcast does not keep the RTX GPU awake.
  - **Battery discharge draw (W):** `/sys/class/power_supply/BAT1/power_now` (when unplugged).

### Suite 7: Reliability, Anti-Loop, and Recovery Chaos Matrix (`reliability`)

Same fault-injection test matrix executed against both Python and Rust implementations:

- **PipeWire feedback loop rejection:** routing `nv_broadcast_mic` or `nv_broadcast_speakers` into itself (must fail 100%).
- **Isolated `pw-cat` / stream properties check:** verifies `node.autoconnect = false` and `node.dont-reconnect = true`.
- **Dropped PipeWire link self-healing:** measures detection and re-link recovery time (ms).
- **Virtual camera standby writer drop:** measures self-heal time (ms).
- **Corrupt / truncated `config.json` resilience:** verifies clean default recovery without crashing.

---

## 3. Target Acceptance Gates (Pass/Fail Thresholds)

| Metric | No Broadcast (`baseline`) | Python (`master`) | Rust Target Gate |
| --- | --- | --- | --- |
| Idle daemon RAM (process tree PSS/RSS) | 0 MB | ~98 MB | **<= 25 MB** |
| Post-CUDA-call idle RAM (T+15s) | 0 MB | ~700 MB | **<= 25 MB** (100% worker reclaim) |
| Idle GPU VRAM / P-state | 0 MiB / `P8` | 0 MiB / `P8` | **0 MiB / `P8`** |
| Audio 10 ms frame latency `p99` | ~0 ms | ~1.5 to 4.0 ms | **<= 0.5 ms** (0 misses > 10 ms) |
| Audio `fan` SNR / SI-SDR delta vs raw | 0 dB | +10 to +18 dB | **Within +/-1.5 dB of Python (or better)** |
| Audio `pure_noise` AGC safety | 0 dB | <= 0 dB delta | **<= 0 dB delta** (never amplifies noise) |
| CLI cold start (`status` / `devices`) | N/A | ~120 to 250 ms | **<= 25 ms** |
| HTTP `/api/state` `p95` latency | N/A | ~15 to 40 ms | **<= 5 ms** |
| Idle background wakeups (ctxt switches/s) | 0 | ~200 to 600 /s | **<= 50 /s** |
| 720p30 CUDA video FPS (Phase R3) | 30 fps | 27 to 30 fps | **>= 28 fps** (mask IoU >= 0.92 vs Python) |

---

## 4. Proposed Directory Layout and Shared Adapter Contract

```text
bench/
  README.md                 # Usage guide and metric definitions
  run_suite.py              # Unified orchestrator (--targets, --suites, --profile, --soak)
  compare.py                # Diffs two report.json runs and exits non-zero on regressions
  metrics/
    audio_metrics.py        # SNR, SI-SDR, SegSNR, silence floor, spectral centroid, WAV I/O
    video_metrics.py        # PSNR, SSIM, alpha matte IoU/MAE, comparison strip renderer
    system_metrics.py       # /proc smaps_rollup tree walker, ctxt switches, nvidia-smi, pw-top
  adapters/
    base.py                 # Common TargetAdapter protocol & capability discovery
    baseline_adapter.py     # Identity passthrough + no-Broadcast system baseline
    python_adapter.py       # Runs Python nv_broadcast (auto-detects .venv or --python-bin)
    rust_adapter.py         # Runs Rust nvb-bench / nvb-daemon / nvb-cli binaries
  reports/
    render_markdown.py      # Side-by-side Markdown comparison tables + pass/fail badges
    render_html.py          # Self-contained dark/light HTML dashboard with SVG charts
  results/                  # Gitignored timestamped run outputs (report.json, report.md, report.html, audio_ab/)
```

### Shared Rust Benchmark Driver (`crates/nvb-bench` or subcommand in `nvb-cli`)

To keep the Python and Rust adapters completely symmetric without FFI overhead, Rust exposes a small benchmark runner (`nvb-cli bench <subcommand>` or `crates/nvb-bench`) accepting the exact same JSON commands as `python_adapter.py`:

- `audio-file --in <wav> --out <wav> --config <json>` -> writes processed WAV + prints JSON `{per_frame_us, rtf, peak_rss_kb, meter}`
- `audio-loop --frames 10000 --config <json>` -> prints JSON `{min_us, p50_us, p95_us, p99_us, p999_us, max_us, budget_misses_2ms, budget_misses_10ms, rtf, rss_kb}`
- `config-bench --iterations 10000` -> config load/sanitize/conflict throughput
- `video-frames --in-dir <dir> --out-dir <dir> --config <json>` (activated in Phase R3)

---

## 5. Phased Implementation Order

1. **Step B1 (Now, alongside Phase R0/R1):**
   - Build `bench/run_suite.py`, `bench/compare.py`, `bench/metrics/` (audio + system + video metrics), `bench/adapters/` (`baseline`, `python`, `rust`), and Markdown/HTML report renderers.
   - Run an immediate **Baseline vs Python** capture right now (with Rust cells clearly showing Phase R0 implemented items like config/PCM/routing and `pending_phase_r1..r3` for not-yet-ported modules) so we lock in our ground-truth before-migration numbers.
2. **Step B2 (With Phase R1 Audio DSP):**
   - Wire `nvb-audio` into `rust_adapter.py` and run `audio_quality` + `audio_latency` across `baseline`, `python`, and `rust`. Verify Rust beats the `<= 0.5 ms p99` latency gate and matches Python SNR/SI-SDR within tolerance.
3. **Step B3 (With Phase R2 Daemon + PipeWire + CLI):**
   - Run `memory`, `api_cli`, `system_impact`, and `reliability` across `baseline`, `python`, and `rust`. Verify `<= 25 MB` idle RAM, `0 MiB` idle VRAM, and `100%` post-call RAM reclamation.
4. **Step B4 (With Phase R3 Native Rust Camera Worker):**
   - Enable `video` suite in `rust_adapter.py` and verify 720p30 FPS, mask IoU/MAE, PSNR/SSIM, and on-demand wake latency against both `baseline` and `python`.
