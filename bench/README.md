# Broadcast Linux — Unified Benchmark & Comparison Suite

Single shared harness comparing three targets on identical workloads:

1. **`baseline` (No Broadcast):** Raw media passthrough and zero-process host baseline.
2. **`python` (`nv_broadcast`):** Existing Python daemon, RNNoise/STFT DSP, GPU/CPU video pipeline, and CLI.
3. **`rust` (`nvb-*`):** Native Rust port (`nvb-core`, `nvb-audio`, `nvb-video`, `nvb-daemon`, `nvb-cli`, `nvb-camera-worker`), automatically tracking completion from Phase R0 through Phase R4.

## Quick Start

```bash
# Fast CI-safe comparison (~20-30s, synthetic fixtures, does not disturb live services)
python3 bench/run_suite.py --targets baseline,python,rust --suites all --profile quick

# Full local hardware capture (10,000 audio frames, 720p30 video, pw-top, nvidia-smi, 15s post-CUDA reclaim)
python3 bench/run_suite.py --targets baseline,python,rust --suites all --profile full --rust-release

# Include a real recorded microphone WAV for A/B/C listening and metrics
python3 bench/run_suite.py --targets baseline,python,rust --suites audio_quality --wav /path/to/voice.wav

# Compare two runs and exit non-zero on regressions
python3 bench/compare.py --before bench/results/<run_a>/report.json --after bench/results/<run_b>/report.json
```

## Suites Included

| Suite | What It Measures |
| --- | --- |
| `audio_quality` | SNR, Delta-SNR, SI-SDR, SegSNR, silence-floor attenuation, speech preservation, pure-noise AGC safety, and A/B/C WAVs (`audio_ab/`) |
| `audio_latency` | Per-frame 10 ms real-time latency percentiles (`min/p50/p95/p99/p999/max`), RTF, `>2ms` / `>10ms` misses, PCM16+RMS primitive, and `pw-top` |
| `memory` | Process-tree `/proc/<pid>/smaps_rollup` (`PSS`, `RSS`, `Private_Dirty`) + `nvidia-smi` per-PID VRAM across 5 lifecycle states + optional `--soak <sec>` |
| `video` | Sustained FPS, frame-time jitter vs 33.3 ms budget, ground-truth matte IoU/MAE, PSNR/SSIM, background smoothness, denoise PSNR gain, and PNG strips |
| `api_cli` | Config sanitize/conflict throughput, CLI cold start, daemon boot, HTTP `/api/*` latency & req/s, SSE fan-out (1 vs 5 clients), and API schema diff |
| `system_impact` | Steady-state CPU-ms/s, % core, thread/FD count, `/proc/<pid>/status` context switches/s (battery wakeups), GPU P-state & watts, battery/RAPL power |
| `reliability` | Shared fault-injection invariants: feedback-loop rejection, isolated `pw-cat` flags, dropped link detection, corrupt `config.json` recovery, camera standby state machine |

## Output Artifacts

Each run writes a timestamped folder under `bench/results/` (gitignored):

- `report.json` — full machine-readable metrics and acceptance gate status
- `report.md` — side-by-side Markdown tables ready for PRs or docs
- `report.html` — self-contained dark/light interactive dashboard with inline SVG charts
- `audio_ab/*.wav` — raw, Python-processed, and Rust-processed clips for human listening
- `video_strips/*.png` — side-by-side Before vs Processed portrait comparison strips
