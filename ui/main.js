"use strict";

const { app, BrowserWindow, ipcMain, dialog, shell, clipboard, Menu, Tray, nativeImage } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const os = require("os");
const path = require("path");
const fs = require("fs");
const { loadUiPrefs, saveUiPrefs, parseCliArgs, createTrayController } = require("./tray");

const NVB_ROOT = path.resolve(__dirname, "..");
let mainWin = null;
let backendProc = null;
let attachedToDaemon = false;
let backendInfo = { url: "", token: "" };
let initialTab = "mic";
let isQuitting = false;
let trayController = null;
let uiPrefs = loadUiPrefs();

const cliArgs = parseCliArgs(process.argv);
if (cliArgs.tab) {
  initialTab = cliArgs.tab;
}
if (process.env.NVB_INITIAL_TAB) {
  initialTab = process.env.NVB_INITIAL_TAB;
}
const startMinimizedFlag = cliArgs.minimized;

const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
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

function shouldHideOnClose() {
  return Boolean(!isQuitting && uiPrefs.minimizeToTray && trayController && trayController.isAvailable());
}

function showMainWindow() {
  if (!mainWin || mainWin.isDestroyed()) {
    createWindow({ show: true });
    return;
  }
  if (mainWin.isMinimized()) mainWin.restore();
  mainWin.show();
  mainWin.focus();
  trayController?.rebuildMenu();
}

function sendSwitchTab(tab) {
  if (!tab || !mainWin || mainWin.isDestroyed()) return;
  if (mainWin.webContents.isLoading()) {
    mainWin.webContents.once("did-finish-load", () => {
      if (mainWin && !mainWin.isDestroyed()) {
        mainWin.webContents.send("nvb:switch-tab", tab);
      }
    });
  } else {
    mainWin.webContents.send("nvb:switch-tab", tab);
  }
}

function createWindow({ show = true } = {}) {
  Menu.setApplicationMenu(null);
  mainWin = new BrowserWindow({
    width: 1280,
    height: 780,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#191919",
    frame: false,
    show: Boolean(show),
    title: "NVIDIA Broadcast",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWin.loadFile(path.join(__dirname, "index.html"));
  mainWin.on("close", (e) => {
    if (shouldHideOnClose()) {
      e.preventDefault();
      mainWin.hide();
      trayController?.rebuildMenu();
    }
  });
  mainWin.on("show", () => trayController?.rebuildMenu());
  mainWin.on("hide", () => trayController?.rebuildMenu());
  mainWin.on("minimize", () => trayController?.rebuildMenu());
  mainWin.on("restore", () => trayController?.rebuildMenu());
  mainWin.on("closed", () => {
    mainWin = null;
    trayController?.rebuildMenu();
  });
}

ipcMain.handle("nvb:get-bootstrap", () => ({
  url: backendInfo.url,
  token: backendInfo.token,
  initialTab,
}));

ipcMain.handle("nvb:window-control", (_ev, action) => {
  if (!mainWin || mainWin.isDestroyed()) return false;
  if (action === "minimize") mainWin.minimize();
  else if (action === "maximize") {
    if (mainWin.isMaximized()) mainWin.unmaximize();
    else mainWin.maximize();
  } else if (action === "close") {
    if (shouldHideOnClose()) {
      mainWin.hide();
      trayController?.rebuildMenu();
    } else {
      mainWin.close();
    }
  }
  return true;
});

ipcMain.handle("nvb:get-ui-prefs", () => ({ ...uiPrefs }));

ipcMain.handle("nvb:set-ui-prefs", (_ev, patch) => {
  uiPrefs = saveUiPrefs(patch || {}, { currentPrefs: uiPrefs });
  trayController?.rebuildMenu();
  return { ...uiPrefs };
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

if (gotSingleInstanceLock) {
  app.on("second-instance", (_event, argv) => {
    const parsed = parseCliArgs(argv);
    if (parsed.tab) {
      initialTab = parsed.tab;
    }
    showMainWindow();
    if (parsed.tab) {
      sendSwitchTab(parsed.tab);
    }
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

      trayController = createTrayController({
        Tray,
        Menu,
        nativeImage,
        getWindow: () => mainWin,
        getBackendInfo: () => backendInfo,
        getPrefs: () => uiPrefs,
        updatePrefs: (patch) => {
          uiPrefs = saveUiPrefs(patch || {}, { currentPrefs: uiPrefs });
          return uiPrefs;
        },
        onQuit: () => {
          isQuitting = true;
          app.quit();
        },
        pollIntervalMs: 5000,
      });

      const trayReady = trayController.init();
      const startHidden = Boolean((startMinimizedFlag || uiPrefs.startMinimized) && trayReady);
      createWindow({ show: !startHidden });
      trayController.rebuildMenu();
    } catch (err) {
      console.error("Failed to start NVIDIA Broadcast backend:", err);
      isQuitting = true;
      app.quit();
    }
  });

  app.on("before-quit", () => {
    isQuitting = true;
    trayController?.destroy();
    stopBackend();
  });

  app.on("window-all-closed", () => {
    if (isQuitting || !trayController || !trayController.isAvailable()) {
      stopBackend();
      app.quit();
    }
  });
}
