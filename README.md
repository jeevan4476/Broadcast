# Broadcast (Linux)

GPU-accelerated Linux recreation of NVIDIA Broadcast 2.3.0 for PipeWire audio, V4L2 webcams, and NVIDIA CUDA GPUs.

## Features

- **Virtual Microphone (`NVIDIA Broadcast (Microphone)`)**
  - PipeWire native `Audio/Source/Virtual` visible directly in Discord, Google Meet, Zoom, Chrome, Brave, and OBS.
  - Real-time neural Noise Removal (`pyrnnoise` RNNoise GRU, 48 kHz, 10 ms frames) with neural VAD gating so room/fan noise is attenuated rather than amplified.
  - Room Echo Removal (vectorized STFT late-reflection suppression), Studio Voice EQ profiles (`default`, `bright`, `full`, `warm`), 85 Hz fan-rumble high-pass filter, Noise Gate, speech-gated Auto-Gain, 5-second raw-vs-processed Mic Test with waveform display, and one-click `Use as Default Mic`.
- **Virtual Speakers (`NVIDIA Broadcast (Speakers)`)**
  - Full 48 kHz stereo PipeWire virtual sink routing cleanly to your speakers or headphones (3.5mm, USB, or Bluetooth) with automatic hot-plug reconnection.
  - Incoming call Noise Removal and Room Echo Removal, stereo passthrough when effects are off, built-in noisy sample tester, Bluetooth Headset Mic (HFP/mSBC) vs A2DP Stereo switcher, and one-click `Use as Default Output` with auto-restore on exit.
- **Virtual Camera (`NVIDIA Broadcast Camera`)**
  - **Robust Video Matting (`rvm_mobilenetv3_fp32.onnx`)** running on NVIDIA CUDA (`CUDAExecutionProvider`) with MediaPipe `ImageSegmenter` fallback.
  - **MediaPipe 478-point Face & Iris Landmarker (`face_landmarker.task`)** powering **Eye Contact** gaze redirection (with blink detection and look-away guard), smooth damped **Auto Frame**, subject-masked **Virtual Key Light**, and face-tracked **Vignette**.
  - **Background Blur**, **Background Replacement** (custom image or 4 studio presets), **Background Removal** (chroma key), motion-aware **Video Noise Removal**, interactive **Before/After Split Screen**, PNG snapshots, and direct V4L2 hardware controls (brightness, contrast, exposure, white balance, 50/60 Hz anti-flicker).
  - Always-visible `v4l2loopback` standby slate keeps the virtual camera selectable in browsers and Discord while leaving your physical webcam off until enabled.
- **Offline Files Tab**
  - Apply active neural audio and video effects to `.wav`, `.mp3`, `.m4a`, `.flac`, `.mp4`, `.mkv`, and `.webm` files.
- **Desktop UI & Background Service**
  - Frameless dark Electron UI modeled on Broadcast 2.3.0 (62px icon rail, 15-pill audio meters, waveform preview, live camera stage, RTX GPU utilization pill), plus a `systemd --user` service so devices stay visible in calls after closing the window.

## Quick Start

### 1. Python GPU AI environment and neural models

```bash
./install-deps.sh --venv
```

### 2. Electron desktop UI dependencies

```bash
cd ui && npm install && cd ..
```

### 3. Virtual webcam (`v4l2loopback`)

On Fedora (with RPM Fusion Free):

```bash
sudo dnf install -y akmod-v4l2loopback v4l2loopback v4l-utils
echo 'v4l2loopback' | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast Camera" exclusive_caps=1' | sudo tee /etc/modprobe.d/98-v4l2loopback.conf
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback
```

### 4. Enable always-on user service (optional, recommended for Discord / Meet)

```bash
mkdir -p ~/.local/bin ~/.config/systemd/user
ln -sf "$(pwd)/nv-broadcast" ~/.local/bin/nv-broadcast
cp nv-broadcast.service ~/.config/systemd/user/nv-broadcast.service
systemctl --user daemon-reload
systemctl --user enable --now nv-broadcast.service
```

### 5. Launch the UI or CLI

```bash
./nv-broadcast gui
./nv-broadcast doctor
./nv-broadcast devices
```

## Running Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
# Or with the GPU/neural venv:
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## License

MIT
