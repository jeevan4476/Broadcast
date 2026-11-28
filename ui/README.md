# `ui` — Broadcast 2.3.0 Electron Desktop UI

Frameless dark desktop interface modeled on NVIDIA Broadcast 2.3.0, communicating with either `nvb-daemon` (Rust default) or `nv_broadcast/server.py` (Python fallback) over the same localhost HTTP + SSE API.

## Package Manager Policy (Bun Only)

This workspace strictly enforces **Bun** via `packageManager`, `engines`, `preinstall`, and `.npmrc` (`engine-strict=true`). Never use `npm`, `npx`, `yarn`, or `pnpm`.

```bash
cd ui
bun install
```

## Files

| File | Responsibility |
| --- | --- |
| `main.js` | Electron main process: discovers running daemon via `~/.config/nv-broadcast/daemon.json` (or spawns `nv-broadcast serve`), manages frameless `BrowserWindow`, IPC handlers, file picker, and single-instance lock |
| `tray.js` | System tray icon generator (32x32 RGBA PNG), quick-toggle tray menu, and atomic UI preferences (`~/.config/nv-broadcast/ui-prefs.json` for `minimizeToTray` and `startMinimized`) |
| `preload.js` | Context-isolated IPC bridge (`window.nvBroadcast` / `window.nvb`) |
| `index.html` & `styles.css` | Broadcast 2.3.0 dark layout (62px icon rail, Mic / Speakers / Camera / Files / Settings tabs) |
| `renderer.js` | Frontend controller binding sliders/toggles to `/api/effect`, rendering live 15-pill audio meters, waveform preview, camera stage, and Files tab progress from `/api/events` SSE |

## Running & Testing

```bash
# Launch from repository root
./nv-broadcast gui

# Run Bun unit tests & syntax checks from repository root
bun test ./tests/*.test.js
```
