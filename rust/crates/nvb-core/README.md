# `nvb-core`

Shared domain logic, configuration schema, typed safety invariants, hardware discovery, and wire contracts for Broadcast Linux.

## Modules

| File | Responsibility |
| --- | --- |
| `src/config.rs` | `BroadcastConfig` loading, atomic save, parameter clamping, schema migration, and effect mutual-exclusion conflict resolution (`studio_voice` vs `noise_removal`/`room_echo_removal`, background blur/replace/remove) |
| `src/routing.rs` | `PhysicalDevice`, `VirtualDevice`, and `AudioRoute` newtypes enforcing anti-loop routing invariants at construction time |
| `src/camera_state.rs` | Deterministic `CameraState` enum (`Off`, `Standby`, `OnDemandActive`, `ManualActive`) for zero-LED standby and consumer wake/sleep transitions |
| `src/loopback_monitor.rs` | Pure-Rust `/proc/<pid>/fd` consumer scanner detecting external apps (Discord, Chrome, OBS) holding `/dev/video*` open while excluding Broadcast itself |
| `src/models.rs` | Neural model manifest, local paths (`~/.local/share/nv-broadcast/models`), and SHA-256 integrity verification (`rvm_mobilenetv3_fp32.onnx`, `selfie_segmenter.tflite`, `face_landmarker.task`) |
| `src/devices.rs` | PipeWire/PulseAudio microphones, speakers, Bluetooth cards/profiles, V4L2 cameras, `v4l2loopback` nodes, and live `nvidia-smi` GPU telemetry |
| `src/doctor.rs` | System readiness check (`run_doctor`, `format_doctor_text`) inspecting PipeWire, V4L2, `v4l2loopback`, `ffmpeg`, GPU/CUDA, and neural models |
| `src/api_contract.rs` | Strongly typed serde structs matching the exact JSON wire format consumed by `ui/` and verified against `fixtures/api_contract_oracle.json` |

## Testing

```bash
cargo test -p nvb-core
```
