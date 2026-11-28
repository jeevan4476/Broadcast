# `nvb-cli`

Drop-in `clap 4` CLI (`nv-broadcast`) and symmetric benchmark driver (`1.1 ms` cold start vs `803 ms` in Python).

## Subcommands

```bash
# Diagnostics & inspection
nvb-cli doctor [--json]
nvb-cli status [--json]
nvb-cli devices [--all] [--json]
nvb-cli effects [--json]

# Configure audio & video effects
nvb-cli mic --noise-removal true --noise-strength 0.85 --studio-voice false
nvb-cli speaker --noise-removal true --echo-removal false
nvb-cli camera --blur true --blur-strength 0.7 --blur-mode quality

# Offline audio / video / screen-recording enhancement
nvb-cli process-file screenrec.mp4 -o screenrec_clean.mp4 --no-video
nvb-cli process-file recording.mp4 -o recording_fx.mp4

# Launch the local HTTP/SSE daemon (delegates to sibling nvb-daemon)
nvb-cli serve --host 127.0.0.1 --port 0 --autostart --daemon-file

# Internal benchmark commands invoked by bench/run_suite.py
nvb-cli bench capabilities
nvb-cli bench config-bench --iterations 1000
nvb-cli bench pcm-bench --frames 2000
nvb-cli bench audio-file --in raw.wav --out clean.wav --config '{}'
nvb-cli bench audio-loop --frames 10000
nvb-cli bench reliability
```

## Launcher Integration

The top-level `nv-broadcast` wrapper defaults to `nvb-cli` when built (`NVB_ENGINE=rust`), automatically forwarding `gui` and Python-only helper commands (`install-filter`, `install-deps`, `test-mic`, `v4l2`, `stop`) to Python. Set `NVB_ENGINE=python` to force the Python CLI for any command.

## Testing

```bash
cargo test -p nvb-cli
```
