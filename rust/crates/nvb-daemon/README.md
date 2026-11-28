# `nvb-daemon`

Always-on localhost HTTP/SSE daemon for Broadcast Linux built on `axum 0.8` and a lean 2-thread `tokio` runtime.

## Design Highlights

- **~3.0 MB idle PSS**, **0.0 context-switches/sec** idle wakeups, **3 threads**, **0 MiB GPU VRAM**, and **2.0 ms** cold boot to `READY`.
- **Cached hardware probes:** `/api/state` (`1.37 ms` p95, `1,970 req/s`) and `/api/devices` (`0.25 ms` p95, `5,786 req/s`) serve cached snapshots without spawning `pactl`/`nvidia-smi` per request.
- **Per-session authentication:** Enforces `X-NVB-Token` header or `?token=` query parameter on every endpoint; writes `0600` `~/.config/nv-broadcast/daemon.json` when `--daemon-file` is passed.
- **Exact API contract parity** with `nv_broadcast/server.py` so the Electron UI (`ui/`) works without changes.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check (`{"status": "ok"}`) |
| `GET` | `/api/state` | Full config, effect catalog, incompatible groups, devices, capabilities, virtual nodes, models, and live telemetry |
| `GET` | `/api/devices` | Microphones, speakers, cameras, loopback cameras, Bluetooth cards, and defaults |
| `GET` | `/api/doctor` | Full system diagnostics report |
| `GET` | `/api/events` | Server-Sent Events (SSE) stream pushing live audio meters, camera status, GPU telemetry, and `file_job` progress |
| `POST` | `/api/effect` | Toggle/configure any audio or video effect with automatic mutual-exclusion conflict resolution |
| `POST` | `/api/audio/mic/{start,stop,config}` | Manage virtual microphone stream and DSP settings |
| `POST` | `/api/audio/speaker/{start,stop,config,sample}` | Manage virtual speakers and noisy sample testing |
| `POST` | `/api/audio/{play,stop-play,set-default,bluetooth-profile}` | WAV playback, PipeWire default routing, and Bluetooth HFP/A2DP switching |
| `POST` | `/api/camera/{start,stop,config}` & `GET /api/camera/v4l2` | Spawn/reap isolated `nvb-camera-worker` and configure camera settings |
| `GET` / `POST` | `/api/files/{state,start,cancel}` | Start, monitor, and cancel offline audio/video/screen-recording jobs |

## Usage & Testing

```bash
nvb-daemon --host 127.0.0.1 --port 0 --autostart --daemon-file
cargo test -p nvb-daemon
```
