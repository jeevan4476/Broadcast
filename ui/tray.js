"use strict";

const fs = require("fs");
const http = require("http");
const https = require("https");
const os = require("os");
const path = require("path");
const zlib = require("zlib");

const DEFAULT_UI_PREFS = Object.freeze({
  minimizeToTray: true,
  startMinimized: false,
});

// Precompute standard IEEE 802.3 CRC32 table for PNG chunks
const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    c = CRC32_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function makePngChunk(type, data) {
  const typeBuf = Buffer.from(type, "ascii");
  const payload = Buffer.isBuffer(data) ? data : Buffer.from(data);
  const lenBuf = Buffer.alloc(4);
  lenBuf.writeUInt32BE(payload.length, 0);

  const crcInput = Buffer.concat([typeBuf, payload]);
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(crcInput), 0);

  return Buffer.concat([lenBuf, typeBuf, payload, crcBuf]);
}

function signedDistanceRoundedBox(px, py, halfW, halfH, radius) {
  const qx = Math.abs(px) - halfW + radius;
  const qy = Math.abs(py) - halfH + radius;
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0));
  const inside = Math.min(Math.max(qx, qy), 0);
  return outside + inside - radius;
}

/**
 * Programmatically generate a crisp RGBA PNG tray icon buffer.
 * NVIDIA green (#76b900) rounded badge with a dark inner broadcast eye/wave mark.
 */
function createTrayIconPngBuffer(size = 32) {
  const dim = Math.max(16, Math.min(256, Math.round(Number(size) || 32)));
  const stride = 1 + dim * 4;
  const raw = Buffer.alloc(stride * dim);

  const half = dim / 2;
  const badgeHalf = dim * 0.45;
  const badgeRadius = dim * 0.22;

  // NVIDIA green #76b900 -> RGB(118, 185, 0)
  const bgR = 0x76;
  const bgG = 0xb9;
  const bgB = 0x00;

  // Dark inner mark #141414 -> RGB(20, 20, 20)
  const fgR = 0x14;
  const fgG = 0x14;
  const fgB = 0x14;

  // Concentric broadcast glyph radii (scaled to icon dimension)
  const outerRingR = dim * 0.30;
  const ringHalfWidth = Math.max(1.1, dim * 0.062);
  const midRingR = dim * 0.175;
  const midHalfWidth = Math.max(1.0, dim * 0.052);
  const dotR = dim * 0.075;

  for (let y = 0; y < dim; y++) {
    const rowOff = y * stride;
    raw[rowOff] = 0; // PNG filter type 0 (None)

    for (let x = 0; x < dim; x++) {
      const px = x + 0.5 - half;
      const py = y + 0.5 - half;

      const badgeDist = signedDistanceRoundedBox(px, py, badgeHalf, badgeHalf, badgeRadius);
      // Anti-aliased outer alpha in [0, 1]
      const badgeAlpha = Math.max(0, Math.min(1, 0.5 - badgeDist));

      const pixOff = rowOff + 1 + x * 4;
      if (badgeAlpha <= 0) {
        raw[pixOff] = 0;
        raw[pixOff + 1] = 0;
        raw[pixOff + 2] = 0;
        raw[pixOff + 3] = 0;
        continue;
      }

      const r = Math.hypot(px, py);
      const angle = Math.atan2(py, px); // -PI..PI

      // Outer broadcast ring with an angled gap at bottom-left for stylized NV/broadcast look
      const inOuterGap = angle > 2.15 && angle < 2.85;
      const dOuter = inOuterGap ? 99 : Math.abs(r - outerRingR) - ringHalfWidth;

      // Inner wave arc (right & top sweep)
      const inMidGap = angle > -2.6 && angle < -1.85;
      const dMid = inMidGap ? 99 : Math.abs(r - midRingR) - midHalfWidth;

      // Central core dot
      const dDot = r - dotR;

      const markDist = Math.min(dOuter, dMid, dDot);
      const markCov = Math.max(0, Math.min(1, 0.5 - markDist));

      const outR = Math.round(bgR * (1 - markCov) + fgR * markCov);
      const outG = Math.round(bgG * (1 - markCov) + fgG * markCov);
      const outB = Math.round(bgB * (1 - markCov) + fgB * markCov);
      const outA = Math.round(badgeAlpha * 255);

      raw[pixOff] = outR;
      raw[pixOff + 1] = outG;
      raw[pixOff + 2] = outB;
      raw[pixOff + 3] = outA;
    }
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(dim, 0);
  ihdr.writeUInt32BE(dim, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const idat = zlib.deflateSync(raw, { level: 9 });
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

  return Buffer.concat([
    signature,
    makePngChunk("IHDR", ihdr),
    makePngChunk("IDAT", idat),
    makePngChunk("IEND", Buffer.alloc(0)),
  ]);
}

function getUiPrefsPath(options = {}) {
  if (options.prefsPath) return options.prefsPath;
  if (options.configDir) return path.join(options.configDir, "ui-prefs.json");
  const xdg = options.xdgConfigHome || process.env.XDG_CONFIG_HOME || path.join(os.homedir(), ".config");
  return path.join(xdg, "nv-broadcast", "ui-prefs.json");
}

function normalizeUiPrefs(raw, base = DEFAULT_UI_PREFS) {
  const out = {
    minimizeToTray: typeof base.minimizeToTray === "boolean" ? base.minimizeToTray : DEFAULT_UI_PREFS.minimizeToTray,
    startMinimized: typeof base.startMinimized === "boolean" ? base.startMinimized : DEFAULT_UI_PREFS.startMinimized,
  };
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    if (typeof raw.minimizeToTray === "boolean") out.minimizeToTray = raw.minimizeToTray;
    if (typeof raw.startMinimized === "boolean") out.startMinimized = raw.startMinimized;
  }
  return out;
}

function loadUiPrefs(options = {}) {
  const filePath = getUiPrefsPath(options);
  try {
    if (!fs.existsSync(filePath)) {
      return { ...DEFAULT_UI_PREFS };
    }
    const rawText = fs.readFileSync(filePath, "utf8");
    const parsed = JSON.parse(rawText);
    return normalizeUiPrefs(parsed, DEFAULT_UI_PREFS);
  } catch (_) {
    return { ...DEFAULT_UI_PREFS };
  }
}

function saveUiPrefs(patch, options = {}) {
  const filePath = getUiPrefsPath(options);
  const current = options.currentPrefs ? normalizeUiPrefs(options.currentPrefs) : loadUiPrefs(options);
  const next = normalizeUiPrefs(patch, current);

  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });

  const tmpPath = `${filePath}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
  try {
    fs.writeFileSync(tmpPath, JSON.stringify(next, null, 2) + "\n", { encoding: "utf8", mode: 0o600 });
    fs.renameSync(tmpPath, filePath);
  } catch (err) {
    try {
      if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
    } catch (_) {
      // ignore cleanup error
    }
    throw err;
  }
  return next;
}

function parseCliArgs(argv = []) {
  let tab = null;
  let minimized = false;
  if (!Array.isArray(argv)) return { tab, minimized };
  for (const arg of argv) {
    if (typeof arg !== "string") continue;
    if (arg.startsWith("--tab=")) {
      const val = arg.slice("--tab=".length).trim();
      if (val) tab = val;
    } else if (arg === "--minimized") {
      minimized = true;
    }
  }
  return { tab, minimized };
}

function selectHttpModule(urlStr, override) {
  if (override) return override;
  return String(urlStr).startsWith("https://") ? https : http;
}

function fetchBackendState({ url, token, httpModule, timeoutMs = 2000 } = {}) {
  return new Promise((resolve) => {
    if (!url) return resolve(null);
    const mod = selectHttpModule(url, httpModule);
    const headers = token ? { "X-NVB-Token": token } : {};
    let settled = false;
    const finish = (val) => {
      if (!settled) {
        settled = true;
        resolve(val);
      }
    };

    try {
      const req = mod.get(`${url}/api/state`, { headers, timeout: timeoutMs }, (res) => {
        if (res.statusCode !== 200) {
          res.resume();
          return finish(null);
        }
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          try {
            finish(JSON.parse(body));
          } catch (_) {
            finish(null);
          }
        });
      });
      req.on("error", () => finish(null));
      req.on("timeout", () => {
        req.destroy();
        finish(null);
      });
    } catch (_) {
      finish(null);
    }
  });
}

function postBackendEffect({ url, token, effectId, enabled, httpModule, timeoutMs = 2500 } = {}) {
  return new Promise((resolve) => {
    if (!url || !effectId) return resolve({ ok: false });
    const mod = selectHttpModule(url, httpModule);
    const payload = JSON.stringify({ effect_id: effectId, enabled: Boolean(enabled) });
    let parsed;
    try {
      parsed = new URL(`${url}/api/effect`);
    } catch (_) {
      return resolve({ ok: false });
    }

    let settled = false;
    const finish = (val) => {
      if (!settled) {
        settled = true;
        resolve(val);
      }
    };

    try {
      const req = mod.request(
        {
          hostname: parsed.hostname,
          port: parsed.port || (parsed.protocol === "https:" ? 443 : 80),
          path: parsed.pathname + parsed.search,
          method: "POST",
          timeout: timeoutMs,
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(payload),
            ...(token ? { "X-NVB-Token": token } : {}),
          },
        },
        (res) => {
          let body = "";
          res.setEncoding("utf8");
          res.on("data", (chunk) => {
            body += chunk;
          });
          res.on("end", () => {
            try {
              const data = JSON.parse(body);
              finish(data);
            } catch (_) {
              finish({ ok: res.statusCode === 200 });
            }
          });
        }
      );
      req.on("error", () => finish({ ok: false }));
      req.on("timeout", () => {
        req.destroy();
        finish({ ok: false });
      });
      req.write(payload);
      req.end();
    } catch (_) {
      finish({ ok: false });
    }
  });
}

function readEffectEnabled(state, section, effectId) {
  return Boolean(state && state.config && state.config[section] && state.config[section].effects && state.config[section].effects[effectId] && state.config[section].effects[effectId].enabled);
}

function buildTrayMenuTemplate({
  isWindowVisible = true,
  state = null,
  backendConnected,
  prefs = DEFAULT_UI_PREFS,
  onToggleWindow = () => {},
  onToggleEffect = () => {},
  onUpdatePrefs = () => {},
  onQuit = () => {},
} = {}) {
  const normPrefs = normalizeUiPrefs(prefs);
  const hasBackend = backendConnected !== undefined ? Boolean(backendConnected) : Boolean(state);

  const micNoiseChecked = readEffectEnabled(state, "microphone", "noise_removal");
  const spkNoiseChecked = readEffectEnabled(state, "speaker", "speaker_noise_removal");
  const camBlurChecked = readEffectEnabled(state, "camera", "background_blur");

  return [
    {
      id: "toggle-window",
      label: isWindowVisible ? "Hide NVIDIA Broadcast" : "Show NVIDIA Broadcast",
      click: () => onToggleWindow(),
    },
    { type: "separator" },
    {
      id: "effect-noise_removal",
      label: "Microphone Noise Removal",
      type: "checkbox",
      checked: micNoiseChecked,
      enabled: hasBackend,
      click: (menuItem) => {
        const next = menuItem && typeof menuItem.checked === "boolean" ? menuItem.checked : !micNoiseChecked;
        return onToggleEffect("noise_removal", next);
      },
    },
    {
      id: "effect-speaker_noise_removal",
      label: "Speaker Noise Removal",
      type: "checkbox",
      checked: spkNoiseChecked,
      enabled: hasBackend,
      click: (menuItem) => {
        const next = menuItem && typeof menuItem.checked === "boolean" ? menuItem.checked : !spkNoiseChecked;
        return onToggleEffect("speaker_noise_removal", next);
      },
    },
    {
      id: "effect-background_blur",
      label: "Camera Background Blur",
      type: "checkbox",
      checked: camBlurChecked,
      enabled: hasBackend,
      click: (menuItem) => {
        const next = menuItem && typeof menuItem.checked === "boolean" ? menuItem.checked : !camBlurChecked;
        return onToggleEffect("background_blur", next);
      },
    },
    { type: "separator" },
    {
      id: "pref-minimize-to-tray",
      label: "Minimize to Tray on Close",
      type: "checkbox",
      checked: normPrefs.minimizeToTray,
      click: (menuItem) => {
        const next = menuItem && typeof menuItem.checked === "boolean" ? menuItem.checked : !normPrefs.minimizeToTray;
        return onUpdatePrefs({ minimizeToTray: next });
      },
    },
    {
      id: "pref-start-minimized",
      label: "Start Minimized to Tray",
      type: "checkbox",
      checked: normPrefs.startMinimized,
      click: (menuItem) => {
        const next = menuItem && typeof menuItem.checked === "boolean" ? menuItem.checked : !normPrefs.startMinimized;
        return onUpdatePrefs({ startMinimized: next });
      },
    },
    { type: "separator" },
    {
      id: "quit",
      label: "Quit NVIDIA Broadcast",
      click: () => onQuit(),
    },
  ];
}

function createTrayController({
  Tray,
  Menu,
  nativeImage,
  getWindow = () => null,
  getBackendInfo = () => ({ url: "", token: "" }),
  getPrefs = () => ({ ...DEFAULT_UI_PREFS }),
  updatePrefs = (patch) => normalizeUiPrefs(patch),
  onToggleWindow,
  onEffectToggled = () => {},
  onQuit = () => {},
  fetchState = fetchBackendState,
  toggleEffect = postBackendEffect,
  pollIntervalMs = 5000,
} = {}) {
  let tray = null;
  let latestState = null;
  let pollTimer = null;
  let refreshPromise = null;
  let destroyed = false;

  function isWindowVisible() {
    const win = getWindow();
    if (!win) return false;
    if (typeof win.isDestroyed === "function" && win.isDestroyed()) return false;
    const vis = typeof win.isVisible === "function" ? win.isVisible() : false;
    const min = typeof win.isMinimized === "function" ? win.isMinimized() : false;
    return Boolean(vis && !min);
  }

  function defaultToggleWindow() {
    const win = getWindow();
    if (!win || (typeof win.isDestroyed === "function" && win.isDestroyed())) return;
    if (isWindowVisible()) {
      win.hide();
    } else {
      if (typeof win.isMinimized === "function" && win.isMinimized() && typeof win.restore === "function") {
        win.restore();
      }
      win.show();
      if (typeof win.focus === "function") win.focus();
    }
    rebuildMenu();
  }

  const toggleWindowFn = typeof onToggleWindow === "function" ? onToggleWindow : defaultToggleWindow;

  function isAvailable() {
    if (!tray || destroyed) return false;
    if (typeof tray.isDestroyed === "function" && tray.isDestroyed()) return false;
    return true;
  }

  async function handleToggleEffect(effectId, enabled) {
    const info = getBackendInfo() || {};
    const res = await toggleEffect({
      url: info.url,
      token: info.token,
      effectId,
      enabled,
    });
    if (res && res.config) {
      latestState = { ...(latestState || {}), config: res.config };
      rebuildMenu();
    } else {
      await refreshFromBackend();
    }
    try {
      onEffectToggled(effectId, enabled, res);
    } catch (_) {
      // ignore listener errors
    }
    return res;
  }

  function handleUpdatePrefs(patch) {
    const next = updatePrefs(patch);
    rebuildMenu();
    return next;
  }

  function rebuildMenu() {
    if (!isAvailable()) return null;
    const info = getBackendInfo() || {};
    const template = buildTrayMenuTemplate({
      isWindowVisible: isWindowVisible(),
      state: latestState,
      backendConnected: Boolean(info.url && latestState),
      prefs: getPrefs(),
      onToggleWindow: () => toggleWindowFn(),
      onToggleEffect: (effectId, enabled) => handleToggleEffect(effectId, enabled),
      onUpdatePrefs: (patch) => handleUpdatePrefs(patch),
      onQuit: () => onQuit(),
    });
    const menu = Menu.buildFromTemplate(template);
    tray.setContextMenu(menu);
    return template;
  }

  async function refreshFromBackend() {
    if (destroyed) return latestState;
    if (refreshPromise) return refreshPromise;
    const info = getBackendInfo() || {};
    if (!info.url) return latestState;
    refreshPromise = (async () => {
      try {
        const nextState = await fetchState({ url: info.url, token: info.token });
        if (nextState) {
          latestState = nextState;
        }
        rebuildMenu();
        return latestState;
      } finally {
        refreshPromise = null;
      }
    })();
    return refreshPromise;
  }

  function init() {
    if (destroyed) return false;
    if (tray) return true;
    try {
      const pngBuf = createTrayIconPngBuffer(32);
      const icon = nativeImage.createFromBuffer(pngBuf);
      tray = new Tray(icon);
      if (typeof tray.setToolTip === "function") {
        tray.setToolTip("NVIDIA Broadcast");
      }
      if (typeof tray.on === "function") {
        tray.on("click", () => {
          toggleWindowFn();
          refreshFromBackend();
        });
        tray.on("right-click", () => {
          refreshFromBackend();
        });
      }
      rebuildMenu();
      refreshFromBackend();
      if (pollIntervalMs > 0) {
        pollTimer = setInterval(() => {
          refreshFromBackend();
        }, pollIntervalMs);
        if (pollTimer && typeof pollTimer.unref === "function") {
          pollTimer.unref();
        }
      }
      return true;
    } catch (err) {
      tray = null;
      return false;
    }
  }

  function destroy() {
    destroyed = true;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (tray) {
      try {
        if (typeof tray.isDestroyed !== "function" || !tray.isDestroyed()) {
          tray.destroy();
        }
      } catch (_) {
        // ignore
      }
      tray = null;
    }
  }

  return {
    init,
    destroy,
    isAvailable,
    rebuildMenu,
    refreshFromBackend,
    getLatestState: () => latestState,
  };
}

module.exports = {
  DEFAULT_UI_PREFS,
  crc32,
  createTrayIconPngBuffer,
  getUiPrefsPath,
  normalizeUiPrefs,
  loadUiPrefs,
  saveUiPrefs,
  parseCliArgs,
  fetchBackendState,
  postBackendEffect,
  buildTrayMenuTemplate,
  createTrayController,
};
