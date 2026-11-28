# Broadcast Linux — Unified Benchmark & Comparison Suite

Single shared harness comparing three targets on identical workloads:

1. **`baseline` (No Broadcast):** Raw media passthrough and zero-process host baseline.
2. **`python` (`nv_broadcast`):** Python daemon, `pyrnnoise`/SciPy DSP, OpenCV/MediaPipe/ORT video pipeline, and CLI.
3. **`rust` (`nvb-*`):** Native Rust workspace (`nvb-core`, `nvb-audio`, `nvb-video`, `nvb-daemon`, `nvb-cli`, `nvb-camera-worker`).

All runs execute in isolated temporary directories on ephemeral ports so an active `nv-broadcast.service` is never disturbed.

## Quick Start

```bash
# Fast CI-safe comparison (~20-30s, synthetic fixtures, no hardware required)
python3 bench/run_suite.py --targets baseline,python,rust --suites all --profile quick --rust-release

# Full local hardware capture (10,000 audio frames, 720p30 CUDA video, pw-top, nvidia-smi, post-CUDA reclaim)
python3 bench/run_suite.py --targets baseline,python,rust --suites all --profile full --rust-release

# Include a real recorded microphone WAV for A/B/C listening and metrics
python3 bench/run_suite.py --targets baseline,python,rust --suites audio_quality --wav /path/to/voice.wav

# Compare two runs and exit non-zero on regressions
python3 bench/compare.py --before bench/results/<run_a>/report.json --after bench/results/<run_b>/report.json
```

## Latest Full-Profile Results (11 / 11 Acceptance Gates PASS)

Captured with `--profile full --rust-release` on Fedora Linux with RTX CUDA (`rvm_cuda`). See **[RESULTS.md](RESULTS.md)** for the complete 7-suite breakdown.

| Acceptance Gate | Baseline | Python | Rust | Target | Status |
| --- | ---: | ---: | ---: | --- | --- |
| Idle daemon RAM (PSS) | 0.0 MB | 99.0 MB | **3.0 MB** | Rust <= 25 MB | **PASS** |
| Post-CUDA-call idle RAM (PSS) | 0.0 MB | 1,342.6 MB | **4.9 MB** | Rust <= 25 MB (100% worker reclaim) | **PASS** |
| Idle GPU VRAM / P-state | 0 MiB | 0 MiB (`P8`) | **0 MiB (`P8`)** | 0 MiB / `P8` | **PASS** |
| Audio 10 ms frame latency (`p99`) | 0.000 ms | 0.552 ms | **0.071 ms** | Rust <= 0.500 ms (0 misses > 10 ms) | **PASS** |
| Audio `fan` Voice-SNR gain vs raw | 0.0 dB | +12.0 dB | **+14.8 dB** | Within +/-1.5 dB of Python (or better) | **PASS** |
| Zero-speech AGC safety (`pure_noise`) | 0.0 dB | SAFE | **SAFE** | Never boosts pure noise (<= +0.5 dB) | **PASS** |
| CLI cold start (`--version` `p50`) | N/A | 803.1 ms | **1.1 ms** | Rust <= 25.0 ms | **PASS** |
| HTTP `/api/state` `p95` latency | N/A | 333.66 ms | **1.37 ms** | Rust <= 5.0 ms | **PASS** |
| Idle background wakeups (ctxt/s) | 0.0 /s | 2.1 /s | **0.0 /s** | Rust <= 50 /s | **PASS** |
| 720p video FPS (`background_blur_quality`) | 30.0 fps | 57.5 fps | **31.9 fps** | Rust >= 28.0 fps | **PASS** |
| Reliability & anti-loop invariants | N/A | 5/5 | **5/5** | 100% checks passing | **PASS** |

## Suites Included

| Suite | Module | What It Measures |
| --- | --- | --- |
| `audio_quality` | `metrics/audio_metrics.py` | SNR, Voice-SNR gain, SI-SDR, SegSNR, silence-floor attenuation, speech preservation, pure-noise AGC safety, and A/B/C WAVs (`audio_ab/`) |
| `audio_latency` | `adapters/*` | Per-frame 10 ms real-time latency (`min/p50/p95/p99/p999/max`), RTF, `>2ms` / `>10ms` misses, PCM16+RMS primitive, and `pw-top` |
| `memory` | `metrics/system_metrics.py` | Process-tree `/proc/<pid>/smaps_rollup` (`PSS`, `RSS`, `Private_Dirty`) + `nvidia-smi` per-PID VRAM across 5 lifecycle states + optional `--soak <sec>` |
| `video` | `metrics/video_metrics.py` | Sustained 720p FPS, frame-time jitter vs 33.3 ms budget, ground-truth matte IoU/MAE, PSNR/SSIM, background smoothness, denoise PSNR gain, and PNG strips |
| `api_cli` | `adapters/*` | Config sanitize/conflict throughput, CLI cold start, daemon boot, HTTP `/api/*` latency & req/s, SSE fan-out (1 vs 5 clients), and differential API schema parity |
| `system_impact` | `metrics/system_metrics.py` | Steady-state CPU-ms/s, % core, thread/FD count, `/proc/<pid>/status` context switches/s (battery wakeups), GPU P-state & watts, battery/RAPL power |
| `reliability` | `adapters/*` | Shared fault-injection invariants: feedback-loop rejection, isolated `pw-cat` flags, dropped link detection, corrupt `config.json` recovery, camera standby state machine |

## Directory Layout

- `run_suite.py` — Unified orchestrator (`--targets`, `--suites`, `--profile`, `--rust-release`, `--soak`, `--wav`)
- `compare.py` — Regression checker diffing two `report.json` files
- `adapters/` — Symmetric target adapters (`base.py`, `baseline_adapter.py`, `python_adapter.py`, `rust_adapter.py`)
- `metrics/` — Audio, video, and Linux `/proc` + `nvidia-smi` + `pw-top` collectors
- `reports/` — Markdown (`render_markdown.py`) and self-contained SVG HTML (`render_html.py`) renderers
- `RESULTS.md` — Committed reference results from the full local hardware run
- `results/` — Gitignored local timestamped outputs (`report.json`, `report.md`, `report.html`, `audio_ab/*.wav`, `video_strips/*.png`)
