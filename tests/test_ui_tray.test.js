"use strict";

const { describe, expect, test, afterEach } = require("bun:test");
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");
const zlib = require("zlib");

const {
  DEFAULT_UI_PREFS,
  crc32,
  createTrayIconPngBuffer,
  getUiPrefsPath,
  loadUiPrefs,
  saveUiPrefs,
  parseCliArgs,
  fetchBackendState,
  postBackendEffect,
  buildTrayMenuTemplate,
  createTrayController,
} = require("../ui/tray");

const tempDirs = [];

function makeTempDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "nvb-tray-test-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  while (tempDirs.length) {
    const dir = tempDirs.pop();
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch (_) {
      // ignore cleanup failures
    }
  }
});

function parsePngChunks(buf) {
  const chunks = [];
  let offset = 8; // skip PNG signature
  while (offset + 12 <= buf.length) {
    const length = buf.readUInt32BE(offset);
    const type = buf.subarray(offset + 4, offset + 8).toString("ascii");
    const data = buf.subarray(offset + 8, offset + 8 + length);
    const storedCrc = buf.readUInt32BE(offset + 8 + length);
    const computedCrc = crc32(buf.subarray(offset + 4, offset + 8 + length));
    chunks.push({ type, length, data, storedCrc, computedCrc });
    offset += 12 + length;
  }
  return chunks;
}

function makeMockBackendState({ mic = true, speaker = false, blur = true } = {}) {
  return {
    config: {
      microphone: {
        effects: {
          noise_removal: { enabled: mic, strength: 0.85 },
        },
      },
      speaker: {
        effects: {
          speaker_noise_removal: { enabled: speaker, strength: 0.75 },
        },
      },
      camera: {
        effects: {
          background_blur: { enabled: blur, strength: 0.7, mode: "quality" },
        },
      },
    },
  };
}

describe("createTrayIconPngBuffer", () => {
  test("produces a valid 32x32 RGBA PNG with valid CRCs and NVIDIA green/dark pixels", () => {
    const png = createTrayIconPngBuffer(32);
    expect(Buffer.isBuffer(png)).toBe(true);
    expect(png.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))).toBe(true);

    const chunks = parsePngChunks(png);
    expect(chunks.map((c) => c.type)).toEqual(["IHDR", "IDAT", "IEND"]);
    for (const chunk of chunks) {
      expect(chunk.storedCrc).toBe(chunk.computedCrc);
    }

    const ihdr = chunks[0].data;
    expect(ihdr.length).toBe(13);
    expect(ihdr.readUInt32BE(0)).toBe(32); // width
    expect(ihdr.readUInt32BE(4)).toBe(32); // height
    expect(ihdr[8]).toBe(8); // bit depth
    expect(ihdr[9]).toBe(6); // RGBA color type

    const raw = zlib.inflateSync(chunks[1].data);
    expect(raw.length).toBe(32 * (1 + 32 * 4));

    // Verify transparent corner pixel (0, 0), NVIDIA green badge pixels, and dark center mark
    expect(raw[0]).toBe(0); // filter byte
    expect(raw[1 + 3]).toBe(0); // corner alpha = 0

    let greenPixels = 0;
    let darkPixels = 0;
    for (let y = 0; y < 32; y++) {
      const rowOff = y * (1 + 32 * 4);
      for (let x = 0; x < 32; x++) {
        const off = rowOff + 1 + x * 4;
        const r = raw[off];
        const g = raw[off + 1];
        const b = raw[off + 2];
        const a = raw[off + 3];
        if (a > 240 && r === 0x76 && g === 0xb9 && b === 0x00) greenPixels++;
        if (a > 240 && r <= 0x20 && g <= 0x20 && b <= 0x20) darkPixels++;
      }
    }
    expect(greenPixels).toBeGreaterThan(100);
    expect(darkPixels).toBeGreaterThan(20);
  });

  test("supports custom dimensions", () => {
    const png = createTrayIconPngBuffer(24);
    const chunks = parsePngChunks(png);
    expect(chunks[0].data.readUInt32BE(0)).toBe(24);
    expect(chunks[0].data.readUInt32BE(4)).toBe(24);
    const raw = zlib.inflateSync(chunks[1].data);
    expect(raw.length).toBe(24 * (1 + 24 * 4));
  });
});

describe("UI preferences persistence", () => {
  test("resolves path from XDG_CONFIG_HOME", () => {
    const p = getUiPrefsPath({ xdgConfigHome: "/tmp/custom-xdg" });
    expect(p).toBe(path.join("/tmp/custom-xdg", "nv-broadcast", "ui-prefs.json"));
  });

  test("returns defaults when prefs file is missing", () => {
    const dir = makeTempDir();
    const prefs = loadUiPrefs({ xdgConfigHome: dir });
    expect(prefs).toEqual({ minimizeToTray: true, startMinimized: false });
  });

  test("saves, reloads, and merges partial patches atomically without leftover tmp files", () => {
    const dir = makeTempDir();
    const first = saveUiPrefs({ startMinimized: true }, { xdgConfigHome: dir });
    expect(first).toEqual({ minimizeToTray: true, startMinimized: true });

    const reloaded = loadUiPrefs({ xdgConfigHome: dir });
    expect(reloaded).toEqual({ minimizeToTray: true, startMinimized: true });

    const second = saveUiPrefs({ minimizeToTray: false }, { xdgConfigHome: dir });
    expect(second).toEqual({ minimizeToTray: false, startMinimized: true });
    expect(loadUiPrefs({ xdgConfigHome: dir })).toEqual({ minimizeToTray: false, startMinimized: true });

    const nvbDir = path.join(dir, "nv-broadcast");
    const files = fs.readdirSync(nvbDir);
    expect(files).toEqual(["ui-prefs.json"]);
  });

  test("falls back safely on corrupt JSON or wrong field types", () => {
    const dir = makeTempDir();
    const prefsFile = getUiPrefsPath({ xdgConfigHome: dir });
    fs.mkdirSync(path.dirname(prefsFile), { recursive: true });

    fs.writeFileSync(prefsFile, "{not valid json!!!", "utf8");
    expect(loadUiPrefs({ xdgConfigHome: dir })).toEqual(DEFAULT_UI_PREFS);

    fs.writeFileSync(prefsFile, JSON.stringify(["array", "not", "object"]), "utf8");
    expect(loadUiPrefs({ xdgConfigHome: dir })).toEqual(DEFAULT_UI_PREFS);

    fs.writeFileSync(prefsFile, JSON.stringify({ minimizeToTray: "yes", startMinimized: true }), "utf8");
    expect(loadUiPrefs({ xdgConfigHome: dir })).toEqual({ minimizeToTray: true, startMinimized: true });
  });
});

describe("parseCliArgs", () => {
  test("extracts --tab=<tab> and --minimized flags", () => {
    expect(parseCliArgs(["electron", ".", "--tab=camera", "--minimized"])).toEqual({
      tab: "camera",
      minimized: true,
    });
    expect(parseCliArgs(["electron", ".", "--tab=settings"])).toEqual({
      tab: "settings",
      minimized: false,
    });
    expect(parseCliArgs(["electron", "."])).toEqual({
      tab: null,
      minimized: false,
    });
  });
});

describe("buildTrayMenuTemplate", () => {
  test("builds full menu structure reflecting visibility, backend state, and prefs", () => {
    const calls = { toggleWin: 0, effects: [], prefs: [], quit: 0 };
    const template = buildTrayMenuTemplate({
      isWindowVisible: true,
      state: makeMockBackendState({ mic: true, speaker: false, blur: true }),
      prefs: { minimizeToTray: true, startMinimized: false },
      onToggleWindow: () => {
        calls.toggleWin += 1;
      },
      onToggleEffect: (effectId, enabled) => {
        calls.effects.push({ effectId, enabled });
      },
      onUpdatePrefs: (patch) => {
        calls.prefs.push(patch);
      },
      onQuit: () => {
        calls.quit += 1;
      },
    });

    expect(template.map((i) => i.type === "separator" ? "---" : i.label)).toEqual([
      "Hide NVIDIA Broadcast",
      "---",
      "Microphone Noise Removal",
      "Speaker Noise Removal",
      "Camera Background Blur",
      "---",
      "Minimize to Tray on Close",
      "Start Minimized to Tray",
      "---",
      "Quit NVIDIA Broadcast",
    ]);

    const byId = Object.fromEntries(template.filter((i) => i.id).map((i) => [i.id, i]));
    expect(byId["effect-noise_removal"].checked).toBe(true);
    expect(byId["effect-speaker_noise_removal"].checked).toBe(false);
    expect(byId["effect-background_blur"].checked).toBe(true);
    expect(byId["pref-minimize-to-tray"].checked).toBe(true);
    expect(byId["pref-start-minimized"].checked).toBe(false);

    byId["toggle-window"].click();
    byId["effect-noise_removal"].click({ checked: false });
    byId["effect-speaker_noise_removal"].click({ checked: true });
    byId["effect-background_blur"].click({ checked: false });
    byId["pref-minimize-to-tray"].click({ checked: false });
    byId["pref-start-minimized"].click({ checked: true });
    byId["quit"].click();

    expect(calls.toggleWin).toBe(1);
    expect(calls.effects).toEqual([
      { effectId: "noise_removal", enabled: false },
      { effectId: "speaker_noise_removal", enabled: true },
      { effectId: "background_blur", enabled: false },
    ]);
    expect(calls.prefs).toEqual([{ minimizeToTray: false }, { startMinimized: true }]);
    expect(calls.quit).toBe(1);
  });

  test("shows 'Show NVIDIA Broadcast' when window is hidden and disables effects without backend", () => {
    const template = buildTrayMenuTemplate({
      isWindowVisible: false,
      state: null,
      prefs: { minimizeToTray: false, startMinimized: true },
    });
    const byId = Object.fromEntries(template.filter((i) => i.id).map((i) => [i.id, i]));
    expect(byId["toggle-window"].label).toBe("Show NVIDIA Broadcast");
    expect(byId["effect-noise_removal"].enabled).toBe(false);
    expect(byId["effect-speaker_noise_removal"].enabled).toBe(false);
    expect(byId["effect-background_blur"].enabled).toBe(false);
  });
});

describe("createTrayController with live mock HTTP backend", () => {
  test("fetches state with X-NVB-Token, toggles effects via POST /api/effect, and toggles window/prefs", async () => {
    const token = "secret-test-token";
    const backendState = makeMockBackendState({ mic: true, speaker: false, blur: false });
    const receivedRequests = [];

    const server = http.createServer((req, res) => {
      const chunks = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const rawBody = Buffer.concat(chunks).toString("utf8");
        receivedRequests.push({
          method: req.method,
          url: req.url,
          token: req.headers["x-nvb-token"],
          body: rawBody ? JSON.parse(rawBody) : null,
        });

        if (req.headers["x-nvb-token"] !== token) {
          res.writeHead(401, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "unauthorized" }));
          return;
        }

        if (req.method === "GET" && req.url === "/api/state") {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(backendState));
          return;
        }

        if (req.method === "POST" && req.url === "/api/effect") {
          const body = JSON.parse(rawBody);
          if (body.effect_id === "noise_removal") {
            backendState.config.microphone.effects.noise_removal.enabled = Boolean(body.enabled);
          } else if (body.effect_id === "speaker_noise_removal") {
            backendState.config.speaker.effects.speaker_noise_removal.enabled = Boolean(body.enabled);
          } else if (body.effect_id === "background_blur") {
            backendState.config.camera.effects.background_blur.enabled = Boolean(body.enabled);
          }
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, config: backendState.config }));
          return;
        }

        res.writeHead(404);
        res.end();
      });
    });

    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = server.address().port;
    const baseUrl = `http://127.0.0.1:${port}`;

    try {
      // Direct HTTP helper checks
      const fetched = await fetchBackendState({ url: baseUrl, token });
      expect(fetched.config.microphone.effects.noise_removal.enabled).toBe(true);

      const postRes = await postBackendEffect({
        url: baseUrl,
        token,
        effectId: "background_blur",
        enabled: true,
      });
      expect(postRes.ok).toBe(true);
      expect(backendState.config.camera.effects.background_blur.enabled).toBe(true);

      // Controller test with mocked Electron Tray / Menu / nativeImage / BrowserWindow
      const dir = makeTempDir();
      let prefs = loadUiPrefs({ xdgConfigHome: dir });
      let winVisible = true;
      let quitCount = 0;
      let latestTemplate = null;
      const trayEvents = {};
      let trayDestroyed = false;

      class MockTray {
        constructor(icon) {
          this.icon = icon;
          this.tooltip = "";
        }
        setToolTip(t) {
          this.tooltip = t;
        }
        setContextMenu(menu) {
          this.menu = menu;
        }
        on(ev, fn) {
          trayEvents[ev] = fn;
        }
        isDestroyed() {
          return trayDestroyed;
        }
        destroy() {
          trayDestroyed = true;
        }
      }

      const MockMenu = {
        buildFromTemplate(tpl) {
          latestTemplate = tpl;
          return { items: tpl };
        },
      };

      const MockNativeImage = {
        createFromBuffer(buf) {
          return { buffer: buf };
        },
      };

      const mockWin = {
        isDestroyed: () => false,
        isVisible: () => winVisible,
        isMinimized: () => false,
        hide: () => {
          winVisible = false;
        },
        show: () => {
          winVisible = true;
        },
        focus: () => {},
        restore: () => {},
      };

      const ctrl = createTrayController({
        Tray: MockTray,
        Menu: MockMenu,
        nativeImage: MockNativeImage,
        getWindow: () => mockWin,
        getBackendInfo: () => ({ url: baseUrl, token }),
        getPrefs: () => prefs,
        updatePrefs: (patch) => {
          prefs = saveUiPrefs(patch, { xdgConfigHome: dir, currentPrefs: prefs });
          return prefs;
        },
        onQuit: () => {
          quitCount += 1;
        },
        pollIntervalMs: 0,
      });

      expect(ctrl.init()).toBe(true);
      expect(ctrl.isAvailable()).toBe(true);
      await ctrl.refreshFromBackend();

      let byId = Object.fromEntries(latestTemplate.filter((i) => i.id).map((i) => [i.id, i]));
      expect(byId["toggle-window"].label).toBe("Hide NVIDIA Broadcast");
      expect(byId["effect-noise_removal"].checked).toBe(true);
      expect(byId["effect-background_blur"].checked).toBe(true);

      // Toggle mic noise removal off via tray menu
      await byId["effect-noise_removal"].click({ checked: false });
      expect(backendState.config.microphone.effects.noise_removal.enabled).toBe(false);
      byId = Object.fromEntries(latestTemplate.filter((i) => i.id).map((i) => [i.id, i]));
      expect(byId["effect-noise_removal"].checked).toBe(false);

      // Clicking tray icon hides window & flips label
      trayEvents.click();
      expect(winVisible).toBe(false);
      byId = Object.fromEntries(latestTemplate.filter((i) => i.id).map((i) => [i.id, i]));
      expect(byId["toggle-window"].label).toBe("Show NVIDIA Broadcast");

      // Toggle pref via tray menu
      byId["pref-start-minimized"].click({ checked: true });
      expect(prefs.startMinimized).toBe(true);
      expect(loadUiPrefs({ xdgConfigHome: dir }).startMinimized).toBe(true);

      // Quit via tray menu
      byId["quit"].click();
      expect(quitCount).toBe(1);

      ctrl.destroy();
      expect(ctrl.isAvailable()).toBe(false);
      expect(trayDestroyed).toBe(true);

      // Verify X-NVB-Token was sent on every backend request
      expect(receivedRequests.length).toBeGreaterThanOrEqual(4);
      for (const req of receivedRequests) {
        expect(req.token).toBe(token);
      }
    } finally {
      await new Promise((resolve) => server.close(resolve));
    }
  });

  test("handles Tray constructor failure gracefully without throwing", () => {
    class FailingTray {
      constructor() {
        throw new Error("No AppIndicator service available");
      }
    }
    const ctrl = createTrayController({
      Tray: FailingTray,
      Menu: { buildFromTemplate: (t) => t },
      nativeImage: { createFromBuffer: (b) => b },
      pollIntervalMs: 0,
    });
    expect(ctrl.init()).toBe(false);
    expect(ctrl.isAvailable()).toBe(false);
    ctrl.destroy();
  });
});
