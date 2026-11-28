# `nvb-camera-worker`

Isolated short-lived GPU/CPU camera and offline video worker binary.

## Why an Isolated Worker?

Loading ONNX Runtime + CUDA + cuDNN allocates ~1.1 GB of process memory and ~232 MiB of GPU VRAM that dynamic loaders (`dlopen`) do not release back to the OS within a long-running process. By running all GPU video inference inside `nvb-camera-worker` and reaping the process when a call or offline job ends, `nvb-daemon` immediately drops back to **4.9 MB PSS and 0 MiB GPU VRAM** (versus `1,342.6 MB / 88 MiB` retained in Python).

## Subcommands

```bash
# Live V4L2 capture -> Rust effects -> v4l2loopback YUYV output
nvb-camera-worker stream --input /dev/video0 --output /dev/video10 --width 1280 --height 720 --fps 30 --backend auto

# Offline video effects processing (emits `PROGRESS <0..1>` lines for daemon SSE)
nvb-camera-worker process-video --in input.mp4 --out processed.mp4 --config config.json --backend auto

# Benchmark all 6 standard video effects + export alpha matte PNG
nvb-camera-worker bench --in-dir /tmp/frames --out-dir /tmp/out --backend auto

# Memory/VRAM lifecycle probe used by bench/run_suite.py
nvb-camera-worker hold --backend auto --width 1280 --height 720 --frames 5
```

## Encoder Auto-Selection (`process-video`)

Automatically probes available `ffmpeg` encoders in priority order:

1. `h264_nvenc` (RTX hardware H.264, automatically retries software if NVENC session fails)
2. `libx264`
3. `libopenh264` (stock Fedora `ffmpeg-free`)
4. `mpeg4` (universal fallback)

## Testing

```bash
cargo test -p nvb-camera-worker
```
