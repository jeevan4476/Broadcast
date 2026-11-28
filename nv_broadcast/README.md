# `nv_broadcast` (Python Reference Backend & Fallback)

Reference Python implementation of NVIDIA Broadcast (Linux), serving three roles:

1. **Instant fallback runtime:** Pass `NVB_ENGINE=python` to `./nv-broadcast` to run the Python CLI or daemon at any time.
2. **GUI launcher & utility commands:** Launches the Electron UI (`nv-broadcast gui`), optional legacy GTK4 UI (`--gtk`), `install-filter`, `test-mic`, and `v4l2` utilities, plus MediaPipe 478-point Eye Contact gaze redirection.
3. **Golden differential oracle:** Generates `rust/fixtures/*.json` via `scripts/generate_oracle_fixtures.py` and serves as the reference target in `bench/run_suite.py`.

## Modules

| File | Responsibility |
| --- | --- |
| `config.py` | Configuration defaults, clamping, effect catalog, and mutual-exclusion conflict rules |
| `audio_dsp.py` | `pyrnnoise` RNNoise GRU, SciPy STFT Room Echo Removal, Studio Voice EQ, Noise Gate, VAD-gated AGC |
| `pipewire.py` & `audio_engine.py` | PipeWire virtual mic/speaker node management, anti-loop checks, and real-time audio workers |
| `video_effects.py` & `video_engine.py` | ONNX Runtime CUDA RVM matting, MediaPipe Face & Iris Landmarker (Eye Contact), OpenCV effects, `v4l2loopback` standby & on-demand wake |
| `loopback_monitor.py` | `/proc` consumer scanner for on-demand virtual camera wake |
| `v4l2_ctl.py` & `devices.py` | V4L2 camera controls, PipeWire/PulseAudio device discovery, and `nvidia-smi` telemetry |
| `file_processor.py` | Offline audio/video file enhancement (`Files` tab & `process-file`) |
| `models.py` | Neural model download & verification |
| `server.py` | Localhost HTTP/SSE/MJPEG API server (`ThreadingHTTPServer`) |
| `cli.py`, `doctor.py`, `gui.py` | CLI entrypoint, system diagnostics, and Electron/GTK4 GUI launcher |

## Testing

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```
