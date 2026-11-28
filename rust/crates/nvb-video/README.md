# `nvb-video`

Pure-Rust video effects pipeline with dynamic ONNX Runtime CUDA/CPU portrait matting (`ort 2.0`) and zero OpenCV/MediaPipe/Python dependencies.

## Modules

| File | Responsibility |
| --- | --- |
| `src/lib.rs` | V4L2 pixel-format/mode selection (`v4l2_ctl` parity), on-demand idle debounce timer, and re-exports |
| `src/effects.rs` | `RgbFrame`, `to_yuyv()` 4:2:2 packer, `RvmSegmenter` (dynamic `ort` loading `libonnxruntime.so*` with `CUDAExecutionProvider` -> `CPUExecutionProvider` -> pure-Rust `fast_portrait` fallback), and fixed-point effects: `background_blur` (`performance`/`quality`), `background_replace` (image or 4 presets), `background_remove` (chroma key), `auto_frame` (YCbCr skin tracker), `video_noise_removal` (3x3 edge-preserving spatial + motion-aware temporal filter), `virtual_key_light`, `vignette`, `mirror`, and `split_preview` |

## Dynamic ONNX Runtime Loading

`nvb-video` enables `ort` with `features = ["std", "load-dynamic", "cuda", "api-20"]` so the crate compiles cleanly without CUDA or ONNX Runtime SDK headers installed. At runtime, `RvmSegmenter` locates `libonnxruntime.so*` via `ORT_DYLIB_PATH` or standard local/venv paths and runs `rvm_mobilenetv3_fp32.onnx` on RTX CUDA (`rvm_cuda`) with seamless CPU (`rvm_cpu`) and `fast_portrait` fallbacks.

## 720p RTX CUDA Benchmarks

- **Matte parity vs Python RVM CUDA:** identical (`IoU 0.3011` vs `0.3009`, `MAE 0.1690` vs `0.1699`).
- **`background_remove`:** `57.0 fps` (1.03x Python).
- **`background_replace`:** `56.9 fps`.
- **`video_noise_removal`:** `48.0 fps` (**1.54x faster** than Python, `+10.91 dB` PSNR).
- **`background_blur_quality`:** `31.9 fps` (passes `>= 28.0 fps` 720p30 gate).

## Testing

```bash
cargo test -p nvb-video
```
