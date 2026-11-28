# Broadcast Linux — Reference Benchmark Results (`--profile full --rust-release`)

- **Profile:** `full` (10,000 audio frames per workload, 720p30 RTX CUDA video, 5-state memory/VRAM lifecycle with post-CUDA reclaim, live HTTP/SSE load, 30s idle system impact, and fault-injection chaos matrix)
- **Targets:** `baseline` (No Broadcast), `python` (`nv_broadcast` `.venv`, `pyrnnoise` + OpenCV + MediaPipe + ORT CUDA), `rust` (`nvb-cli` / `nvb-daemon` / `nvb-camera-worker` release builds, `nnnoiseless` + `realfft` + `ort` CUDA)
- **Hardware:** Linux x86_64 host with NVIDIA RTX GPU (CUDA `rvm_cuda`) and PipeWire audio

---

## Target Acceptance Gates (11 / 11 PASS)

| Gate | Baseline | Python | Rust | Target | Status |
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

---

## Suite 1: Audio Signal Quality & Noise Reduction (`audio_quality`)

| Scenario | Baseline SNR / Voice-SNR | Python Voice-SNR Gain (Sil Atten) | Rust Voice-SNR Gain (Sil Atten) | Python vs Rust Parity |
| --- | ---: | ---: | ---: | ---: |
| `fan` | 3.8 dB / 7.5 dB | +12.0 dB (+9.9 dB) | **+14.8 dB (+11.7 dB)** | dRMS +0.97 dB, corr 0.823 |
| `typing` | 14.3 dB / 17.4 dB | +4.2 dB (+0.2 dB) | **+4.5 dB (+0.4 dB)** | dRMS +0.12 dB, corr 0.974 |
| `echo` | 5.5 dB / 22.8 dB | +4.0 dB (+2.1 dB) | **+4.0 dB (+2.1 dB)** | dRMS -0.00 dB, corr 1.000 |
| `fan_echo` | 1.6 dB / 8.4 dB | +17.4 dB (+17.7 dB) | **+21.9 dB (+21.1 dB)** | dRMS +1.10 dB, corr 0.864 |
| `pure_noise` | RMS -26.1 dB | -20.2 dB (SAFE) | **-17.8 dB (SAFE)** | dRMS +2.33 dB, corr 0.697 |
| `studio_voice_warm` | 1.6 dB / 8.4 dB | +16.0 dB (+15.2 dB) | **+22.1 dB (+20.1 dB)** | dRMS +1.23 dB, corr 0.867 |
| `typing_echo` | 5.0 dB / 17.2 dB | +12.0 dB (+10.3 dB) | +5.5 dB (+3.3 dB) | dRMS +0.55 dB, corr 0.945 |
| `studio_voice_bright` | 3.8 dB / 7.5 dB | +16.9 dB (+17.1 dB) | **+19.8 dB (+18.8 dB)** | dRMS +1.23 dB, corr 0.817 |
| `studio_voice_full` | 3.8 dB / 7.5 dB | +16.1 dB (+14.9 dB) | **+20.8 dB (+18.4 dB)** | dRMS +1.22 dB, corr 0.809 |

- **Rust `nvb-audio` peak RSS during WAV processing:** `6.49 MB` (`nnnoiseless`).

---

## Suite 2: Real-Time Audio Latency & Jitter (`audio_latency`, 10 ms budget)

| Workload | Metric | Baseline | Python | Rust |
| --- | --- | ---: | ---: | ---: |
| `noise_removal` | `p50 / p99 / max` (us) | 0.3 / 0.3 / 16.3 | 485.7 / 551.9 / 720.6 | **57.8 / 71.2 / 180.4** |
| `noise_removal` | RTF (1x CPU%) | 0.0000 (0.00%) | 0.0491 (4.91%) | **0.0058 (0.58%)** |
| `noise_removal` | Misses `>2ms / >10ms` | 0 / 0 | 0 / 0 | **0 / 0** |
| `room_echo_removal` | `p50 / p99 / max` (us) | 0.3 / 0.3 / 16.3 | 717.4 / 905.6 / 2345.8 | **60.7 / 74.7 / 244.4** |
| `room_echo_removal` | RTF (1x CPU%) | 0.0000 (0.00%) | 0.0718 (7.18%) | **0.0061 (0.61%)** |
| `room_echo_removal` | Misses `>2ms / >10ms` | 0 / 0 | 2 / 0 | **0 / 0** |
| `studio_voice` | `p50 / p99 / max` (us) | 0.3 / 0.3 / 16.3 | 750.3 / 843.2 / 13289.9 | **66.0 / 78.9 / 277.4** |
| `studio_voice` | RTF (1x CPU%) | 0.0000 (0.00%) | 0.0750 (7.50%) | **0.0065 (0.65%)** |
| `studio_voice` | Misses `>2ms / >10ms` | 0 / 0 | 1 / 1 | **0 / 0** |

### 480-Sample PCM16 <-> Float + RMS Primitive (`pcm_primitive`)

| Target | `p50` (us) | `p95` (us) | `p99` (us) | `max` (us) | Total Wall (ms) | Speedup vs Python |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `python` | 186.59 | 193.97 | 197.88 | 248.38 | 1870.75 | 1.0x |
| `rust` | **1.64** | **1.78** | **1.79** | **5.08** | **16.39** | **113.8x** |

---

## Suite 3: Process-Tree RAM/PSS & GPU VRAM Lifecycle (`memory`)

| Lifecycle Checkpoint | Baseline (PSS / VRAM) | Python (PSS / RSS / VRAM) | Rust (PSS / VRAM) |
| --- | ---: | ---: | ---: |
| `idle_standby` | 0.0 MB / 0 MiB | 99.0 MB / 128.3 MB / 0 MiB | **3.0 MB / 0 MiB** |
| `active_audio_call` | 0.0 MB / 0 MiB | 102.4 MB / 136.4 MB / 0 MiB | **4.9 MB / 0 MiB** |
| `active_video_cpu` | 0.0 MB / 0 MiB | 524.9 MB / 545.8 MB / 0 MiB | **101.6 MB / 0 MiB** |
| `active_video_cuda` | 0.0 MB / 0 MiB | 1,348.8 MB / 1,380.4 MB / 232 MiB | **1,112.0 MB / 232 MiB** |
| `post_call_idle` | 0.0 MB / 0 MiB | 1,342.6 MB / 1,374.7 MB / 88 MiB | **4.9 MB / 0 MiB** |

---

## Suite 4: Video Quality, FPS & Frame Jitter (`video`, 1280x720)

- **Python Video Backends:** `segmentation=rvm_cuda`, `face=mediapipe_face_landmarker` — Matte vs synthetic GT: `IoU=0.3009`, `MAE=0.1699`
- **Rust Video Backends:** `segmentation=rvm_cuda`, `face=ycbcr_skin_tracker` — Matte vs synthetic GT: `IoU=0.3011`, `MAE=0.1690`

| Effect | Baseline FPS | Python FPS (`p50`/`p95` ms) | Rust FPS (`p50`/`p95` ms) | Relative Speed | Rust Quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| `background_blur_perf` | 1000.0 | 66.8 (14.9 / 16.2) | 35.3 (27.9 / 30.0) | 0.53x | +55.7% grad |
| `background_blur_quality` | 1000.0 | 57.5 (17.3 / 18.3) | **31.9 (31.3 / 33.0)** | 0.55x (PASS >= 28) | +57.1% grad |
| `background_replace` | 1000.0 | 64.7 (15.3 / 16.4) | 56.9 (17.5 / 18.3) | 0.88x | +93.9% grad |
| `background_remove` | 1000.0 | 55.1 (18.1 / 19.4) | **57.0 (17.6 / 18.8)** | **1.03x** | +93.7% grad |
| `video_noise_removal` | 1000.0 | 31.2 (31.9 / 34.3) | **48.0 (20.8 / 21.4)** | **1.54x** | +10.91 dB PSNR |
| `combo_blur_keylight_autoframe` | 1000.0 | 23.6 (42.1 / 44.5) | 18.5 (53.7 / 58.6) | 0.78x | +58.5% grad |

---

## Suite 5: Control Plane, CLI Startup & HTTP/SSE API (`api_cli`)

| Metric | Python | Rust | Speedup / Delta |
| --- | ---: | ---: | ---: |
| Config sanitize + conflict toggle throughput | 5,755 ops/s | **53,187 ops/s** | **9.2x faster** |
| Config toggle `p50` latency | 172.9 us | **16.7 us** | **10.3x lower** |
| CLI cold start `--version` (`p50`) | 803.1 ms | **1.1 ms** | **705.1x faster** |
| CLI cold start `status` (`p50` / `p95`) | 856.0 / 1,149.9 ms | **29.5 / 35.2 ms** | **29.0x faster** |
| CLI cold start `devices` (`p50` / `p95`) | 876.5 / 898.0 ms | **76.2 / 81.6 ms** | **11.5x faster** |
| CLI cold start `effects` (`p50` / `p95`) | 820.1 / 914.8 ms | **3.1 / 3.6 ms** | **268.1x faster** |
| CLI cold start `doctor` (`p50` / `p95`) | 1,009.1 / 1,039.2 ms | **105.0 / 119.0 ms** | **9.6x faster** |
| Daemon cold boot to `READY` | 806.9 ms | **2.0 ms** | **395.6x faster** |
| HTTP `GET /api/health` (`p50` / `p95`, req/s) | 1.79 / 2.11 ms (560/s) | **0.67 / 1.33 ms (1,330/s)** | **1.6x lower `p95`** |
| HTTP `GET /api/state` (`p50` / `p95`, req/s) | 254.50 / 333.66 ms (4/s) | **0.41 / 1.37 ms (1,970/s)** | **244.1x lower `p95`** |
| HTTP `GET /api/devices` (`p50` / `p95`, req/s) | 120.68 / 144.08 ms (8/s) | **0.15 / 0.25 ms (5,786/s)** | **567.2x lower `p95`** |
| HTTP `POST /api/effect` (`p50` / `p95`, req/s) | 2.85 / 3.45 ms (370/s) | **0.35 / 1.11 ms (1,849/s)** | **3.1x lower `p95`** |
| SSE `/api/events` (1 client) | 0.0 CPU-ms/s, 1.6 ctxt/s, 129.2 MB | **0.0 CPU-ms/s, 0.0 ctxt/s, 5.6 MB** | **23x less RAM** |
| SSE `/api/events` (5 clients) | 24.5 CPU-ms/s, 1.6 ctxt/s, 129.4 MB | **0.0 CPU-ms/s, 0.0 ctxt/s, 5.6 MB** | **Zero CPU overhead** |

- **Differential API Contract Parity (Python vs Rust):** **EXACT MATCH** (HTTP status codes, auth rejection, JSON keys, and effect conflict resolution identical).

---

## Suite 6: Background System, Power & Idle Wakeup Overhead (`system_impact`)

| Metric | Baseline (No Broadcast) | Python (Idle Daemon) | Rust (Idle Daemon) |
| --- | ---: | ---: | ---: |
| Process tree PSS (MB) | 0.0 MB | 98.8 MB | **2.8 MB** |
| CPU time (ms / wall-sec) | 0.00 ms/s | 0.00 ms/s | **0.00 ms/s** |
| CPU (% of 1 core) | 0.00 % | 0.00 % | **0.00 %** |
| Threads / Open FDs | 0 / 0 | 47 / 4 | **3 / 10** |
| Context switches / sec (wakeups) | 0.0 /s | 2.1 /s | **0.0 /s** |
| GPU P-state / Power (W) | `P8` / 1.91 W | `P8` / 1.89 W | **`P8` / 1.91 W** |
| Process GPU VRAM (MiB) | 0 MiB | 0 MiB | **0 MiB** |

---

## Suite 7: Reliability, Anti-Loop & Recovery Chaos Matrix (`reliability`)

| Check | Baseline | Python | Rust |
| --- | --- | --- | --- |
| `feedback_loop_rejection` | N/A | **PASS** | **PASS** |
| `isolated_stream_properties` | N/A | **PASS** | **PASS** |
| `dropped_link_detection` | N/A | **PASS** | **PASS** |
| `corrupt_config_recovery` | N/A | **PASS** | **PASS** |
| `camera_standby_state_machine` | N/A | **PASS** | **PASS** |
