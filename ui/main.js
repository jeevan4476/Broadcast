"use strict";

const { app, BrowserWindow, ipcMain, dialog, shell, clipboard, Menu } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const os = require("os");
const path = require("path");
const fs = require("fs");

const NVB_ROOT = path.resolve(__dirname, "..");
let mainWin = null;
let backendProc = null;
let attachedToDaemon = false;
let backendInfo = { url: "", token: "" };
let initialTab = "mic";

for (const arg of process.argv) {
  if (arg.startsWith("--tab=")) {
    initialTab = arg.slice("--tab=".length);
  }
}
if (process.env.NVB_INITIAL_TAB) {
  initialTab = process.env.NVB_INITIAL_TAB;
}

function resolvePythonBin() {
  if (process.env.NVB_PYTHON_BIN && fs.existsSync(process.env.NVB_PYTHON_BIN)) {
    return process.env.NVB_PYTHON_BIN;
  }
  const localVenv = path.join(NVB_ROOT, ".venv", "bin", "python");
  if (fs.existsSync(localVenv)) return localVenv;
  return "/usr/bin/python3";
}

function startBackend() {
  return new Promise((resolve, reject) => {
    const py = resolvePythonBin();
    const args = ["-m", "nv_broadcast.server", "--host", "127.0.0.1", "--port", "0"];
    if (process.env.NVB_CONFIG_PATH) {
      args.push("--config", process.env.NVB_CONFIG_PATH);
    }
    const env = {
      ...process.env,
      PYTHONPATH: process.env.PYTHONPATH ? `${NVB_ROOT}:${process.env.PYTHONPATH}` : NVB_ROOT,
      PYTHONUNBUFFERED: "1",
    };

    backendProc = spawn(py, args, { env, stdio: ["ignore", "pipe", "pipe"] });
    let stdoutBuf = "";
    let stderrBuf = "";
    let resolved = false;

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        reject(new Error(`Backend startup timed out. stderr: ${stderrBuf}`));
      }
    }, 15000);

    backendProc.stdout.on("data", (chunk) => {
      stdoutBuf += chunk.toString("utf8");
      const nl = stdoutBuf.indexOf("\n");
      if (nl !== -1 && !resolved) {
        const line = stdoutBuf.slice(0, nl).trim();
        const m = line.match(/^READY\s+(http:\/\/\S+)\s+(\S+)$/);
        if (m) {
          resolved = true;
          clearTimeout(timer);
          backendInfo = { url: m[1], token: m[2] };
          resolve(backendInfo);
        }
      }
    });

    backendProc.stderr.on("data", (chunk) => {
      stderrBuf += chunk.toString("utf8");
    });

    backendProc.on("exit", (code) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        reject(new Error(`Backend exited early (${code}): ${stderrBuf}`));
      }
    });
  });
}

function tryAttachExistingDaemon() {
  return new Promise((resolve) => {
    const xdg = process.env.XDG_CONFIG_HOME || path.join(os.homedir(), ".config");
    const stateFile = path.join(xdg, "nv-broadcast", "daemon.json");
    if (!fs.existsSync(stateFile)) return resolve(null);
    let info;
    try {
      info = JSON.parse(fs.readFileSync(stateFile, "utf8"));
    } catch (_) {
      return resolve(null);
    }
    if (!info || !info.url || !info.token) return resolve(null);
    const req = http.get(
      `${info.url}/api/health`,
      { headers: { "X-NVB-Token": info.token }, timeout: 1200 },
      (res) => {
        res.resume();
        if (res.statusCode === 200) {
          resolve({ url: info.url, token: info.token });
        } else {
          resolve(null);
        }
      }
    );
    req.on("error", () => resolve(null));
    req.on("timeout", () => {
      req.destroy();
      resolve(null);
    });
  });
}

function stopBackend() {
  if (attachedToDaemon) return; // Leave background systemd daemon alive for Discord/Meet!
  if (backendProc && !backendProc.killed) {
    try {
      backendProc.kill("SIGTERM");
    } catch (_) {
      // ignore
    }
    backendProc = null;
  }
}

function createWindow() {
  Menu.setApplicationMenu(null);
  mainWin = new BrowserWindow({
    width: 1280,
    height: 780,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#191919",
    frame: false,
    title: "NVIDIA Broadcast",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWin.loadFile(path.join(__dirname, "index.html"));
  mainWin.on("closed", () => {
    mainWin = null;
  });
}

ipcMain.handle("nvb:get-bootstrap", () => ({
  url: backendInfo.url,
  token: backendInfo.token,
  initialTab,
}));

ipcMain.handle("nvb:window-control", (_ev, action) => {
  if (!mainWin) return false;
  if (action === "minimize") mainWin.minimize();
  else if (action === "maximize") {
    if (mainWin.isMaximized()) mainWin.unmaximize();
    else mainWin.maximize();
  } else if (action === "close") mainWin.close();
  return true;
});

ipcMain.handle("nvb:pick-image", async () => {
  if (!mainWin) return "";
  const res = await dialog.showOpenDialog(mainWin, {
    title: "Select Background Image",
    properties: ["openFile"],
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp"] }],
  });
  return res.canceled || !res.filePaths.length ? "" : res.filePaths[0];
});

ipcMain.handle("nvb:pick-media-file", async () => {
  if (!mainWin) return "";
  const res = await dialog.showOpenDialog(mainWin, {
    title: "Select Audio or Video File",
    properties: ["openFile"],
    filters: [
      {
        name: "Audio & Video",
        extensions: ["wav", "mp3", "m4a", "flac", "ogg", "aac", "mp4", "mkv", "webm", "mov", "avi"],
      },
    ],
  });
  return res.canceled || !res.filePaths.length ? "" : res.filePaths[0];
});

ipcMain.handle("nvb:reveal-path", (_ev, targetPath) => {
  if (targetPath && fs.existsSync(targetPath)) {
    shell.showItemInFolder(targetPath);
    return true;
  }
  return false;
});

ipcMain.handle("nvb:copy-text", (_ev, text) => {
  clipboard.writeText(String(text || ""));
  return true;
});

app.whenReady().then(async () => {
  try {
    const existing = await tryAttachExistingDaemon();
    if (existing) {
      backendInfo = existing;
      attachedToDaemon = true;
    } else {
      await startBackend();
    }
    createWindow();
  } catch (err) {
    console.error("Failed to start NVIDIA Broadcast backend:", err);
    app.quit();
  }
});

app.on("before-quit", () => {
  stopBackend();
});

app.on("window-all-closed", () => {
  stopBackend();
  app.quit();
});
