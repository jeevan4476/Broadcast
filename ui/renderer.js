"use strict";

(() => {
  let API_URL = "";
  let API_TOKEN = "";
  let appState = null;
  let micTestResult = null;
  let spkSampleResult = null;
  let selectedMediaFile = "";
  let lastCompletedOutput = "";
  let lastDeviceSig = "";
  let streamSeq = 0;

  const $ = (id) => document.getElementById(id);

  function showToast(msg) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.classList.remove("show"), 3000);
  }

  async function apiGet(path) {
    const res = await fetch(`${API_URL}${path}`, {
      headers: { "X-NVB-Token": API_TOKEN },
    });
    return res.json();
  }

  async function apiPost(path, body = {}) {
    const res = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-NVB-Token": API_TOKEN,
      },
      body: JSON.stringify(body),
    });
    return res.json();
  }

  // ==================== NAVIGATION ====================
  function setActivePage(pageName) {
    const mapped = pageName === "mic" || pageName === "speakers" ? "audio" : pageName;
    document.querySelectorAll(".sidebar-link").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.page === mapped);
    });
    document.querySelectorAll(".page").forEach((pg) => {
      pg.classList.toggle("active", pg.id === `page-${mapped}`);
    });
    if (mapped === "settings") {
      refreshDoctor();
    }
  }

  // ==================== 15-PILL METERS ====================
  function initMeterPills(containerId) {
    const el = $(containerId);
    el.innerHTML = "";
    for (let i = 0; i < 15; i++) {
      const pill = document.createElement("div");
      pill.className = "audio-meter-pill";
      pill.dataset.active = "false";
      el.appendChild(pill);
    }
  }

  function updateMeterPills(containerId, dbLabelId, rmsDb) {
    const el = $(containerId);
    const pills = el.children;
    const norm = Math.max(0, Math.min(1, (rmsDb + 65) / 60));
    const lit = Math.round(norm * pills.length);
    for (let i = 0; i < pills.length; i++) {
      const on = i < lit;
      pills[i].dataset.active = on ? "true" : "false";
      pills[i].dataset.hot = on && i >= 13 ? "true" : "false";
    }
    $(dbLabelId).textContent = `${rmsDb <= -88 ? "-90" : rmsDb.toFixed(0)} dB`;
  }

  // ==================== WAVEFORM CANVAS ====================
  function drawWaveform(canvasId, peaks, accentColor = "#76b900") {
    const canvas = $(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Subtle center baseline
    ctx.strokeStyle = "#2d2d2d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (!peaks || !peaks.length) {
      ctx.fillStyle = "#666666";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No waveform recorded yet", w / 2, h / 2 + 4);
      return;
    }

    const barW = Math.max(2, w / peaks.length - 1.5);
    const gap = w / peaks.length;
    ctx.fillStyle = accentColor;
    for (let i = 0; i < peaks.length; i++) {
      const amp = Math.max(0.03, Math.min(1, peaks[i]));
      const bh = amp * (h - 16);
      const x = i * gap;
      const y = (h - bh) / 2;
      ctx.fillRect(x, y, barW, bh);
    }
  }

  // ==================== EFFECT ACCORDION BUILDER ====================
  const PRESET_STYLES = {
    studio_dark: "linear-gradient(135deg, #1b1e23, #303640)",
    office_blur: "linear-gradient(135deg, #3c4352, #7a8fa6)",
    neon_green: "linear-gradient(135deg, #0d1a10, #3e7a16)",
    soft_daylight: "linear-gradient(135deg, #c5cdd8, #e6ecf5)",
  };

  function buildEffectsSection(containerId, sectionKey, effectIds) {
    const host = $(containerId);
    host.innerHTML = "";
    const catalog = appState.catalog;
    const effData = appState.config[sectionKey].effects;

    effectIds.forEach((eid) => {
      const meta = catalog[eid];
      const cur = effData[eid];
      if (!meta || !cur) return;

      const card = document.createElement("div");
      card.className = `effect-item${cur.enabled ? " enabled" : ""}`;
      card.dataset.effectId = eid;

      const head = document.createElement("div");
      head.className = "effect-head";
      head.innerHTML = `
        <div class="effect-meta">
          <div class="effect-title-row">
            <span class="effect-name">${meta.label}</span>
            <span class="gpu-load-tag">${meta.gpu_load || "Low"} GPU</span>
          </div>
          <div class="effect-desc">${meta.description}</div>
        </div>
        <label class="nv-switch">
          <input type="checkbox" ${cur.enabled ? "checked" : ""} />
          <span class="nv-switch-track"></span>
        </label>
      `;

      const toggle = head.querySelector("input");
      const body = document.createElement("div");
      body.className = `effect-body${cur.enabled ? "" : " collapsed"}`;

      toggle.addEventListener("change", async () => {
        const res = await apiPost("/api/effect", { effect_id: eid, enabled: toggle.checked });
        if (res.config) {
          appState.config = res.config;
          syncAllEffectCards();
          if (res.disabled_conflicts && res.disabled_conflicts[eid]?.length) {
            showToast(`Disabled conflicting effect: ${res.disabled_conflicts[eid].join(", ")}`);
          }
        }
      });

      // Effect parameters
      const params = meta.params || {};
      Object.entries(params).forEach(([pname, pspec]) => {
        if (pspec.type === "float") {
          const val = Number(cur[pname] ?? pspec.default);
          const pct = Math.round(val * 100);
          const row = document.createElement("div");
          row.className = "slider-row";
          const labelMap = {
            strength: "Strength",
            zoom: "Zoom",
            warmth: "Color Temperature (Cool ↔ Warm)",
            intensity: "Intensity",
          };
          const sliderCls = pname === "warmth" ? "nv-slider nv-slider-warmth" : "nv-slider";
          row.innerHTML = `
            <div class="slider-head">
              <span>${labelMap[pname] || pname}</span>
              <span class="slider-val">${pct}%</span>
            </div>
            <input type="range" class="${sliderCls}" min="0" max="100" value="${pct}" />
          `;
          const inp = row.querySelector("input");
          const valLbl = row.querySelector(".slider-val");
          inp.addEventListener("input", () => {
            valLbl.textContent = `${inp.value}%`;
          });
          inp.addEventListener("change", async () => {
            const res = await apiPost("/api/effect", {
              effect_id: eid,
              params: { [pname]: Number(inp.value) / 100 },
            });
            if (res.config) appState.config = res.config;
          });
          body.appendChild(row);
        } else if (pspec.type === "enum") {
          if (eid === "background_replace" && pname === "preset") {
            const grid = document.createElement("div");
            grid.className = "preset-grid";
            pspec.enum.forEach((pr) => {
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = `preset-thumb${cur.preset === pr && !cur.imagePath ? " selected" : ""}`;
              btn.style.background = PRESET_STYLES[pr] || "#333";
              btn.textContent = pr.replace("_", " ");
              btn.addEventListener("click", async () => {
                const res = await apiPost("/api/effect", {
                  effect_id: eid,
                  params: { preset: pr, imagePath: "" },
                });
                if (res.config) {
                  appState.config = res.config;
                  syncAllEffectCards();
                }
              });
              grid.appendChild(btn);
            });
            body.appendChild(grid);
          } else {
            const row = document.createElement("div");
            row.className = "field-row";
            const labelMap = { micProfile: "Voice Profile", mode: "Mode" };
            row.innerHTML = `<label class="field-label">${labelMap[pname] || pname}</label>`;
            const sel = document.createElement("select");
            sel.className = "nv-select";
            pspec.enum.forEach((opt) => {
              const o = document.createElement("option");
              o.value = opt;
              o.textContent = opt.charAt(0).toUpperCase() + opt.slice(1);
              if (cur[pname] === opt) o.selected = true;
              sel.appendChild(o);
            });
            sel.addEventListener("change", async () => {
              const res = await apiPost("/api/effect", {
                effect_id: eid,
                params: { [pname]: sel.value },
              });
              if (res.config) appState.config = res.config;
            });
            row.appendChild(sel);
            body.appendChild(row);
          }
        } else if (pspec.type === "path" && eid === "background_replace") {
          const row = document.createElement("div");
          row.className = "studio-controls";
          row.style.marginTop = "2px";
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "nv-btn nv-btn-sm";
          btn.textContent = cur.imagePath ? "Change Custom Image..." : "Browse Custom Image...";
          const span = document.createElement("span");
          span.className = "studio-footnote";
          span.style.marginTop = "0";
          span.textContent = cur.imagePath ? cur.imagePath.split("/").pop() : "Using built-in backdrop";
          btn.addEventListener("click", async () => {
            const picked = await window.nvBroadcast.pickImage();
            if (!picked) return;
            const res = await apiPost("/api/effect", {
              effect_id: eid,
              enabled: true,
              params: { imagePath: picked },
            });
            if (res.config) {
              appState.config = res.config;
              syncAllEffectCards();
            }
          });
          row.appendChild(btn);
          row.appendChild(span);
          body.appendChild(row);
        } else if (pspec.type === "bool") {
          const lbl = document.createElement("label");
          lbl.className = "mini-check";
          const txt = pname === "lookAway" ? "Allow natural look-away when looking far off screen" : "Follow face position";
          lbl.innerHTML = `<input type="checkbox" ${cur[pname] ? "checked" : ""} /><span>${txt}</span>`;
          const cb = lbl.querySelector("input");
          cb.addEventListener("change", async () => {
            const res = await apiPost("/api/effect", {
              effect_id: eid,
              params: { [pname]: cb.checked },
            });
            if (res.config) appState.config = res.config;
          });
          body.appendChild(lbl);
        } else if (eid === "background_remove" && pname === "chromaColor") {
          const row = document.createElement("div");
          row.className = "swatch-row";
          ["#00ff00", "#0000ff", "#ff00ff", "#191919"].forEach((hex) => {
            const sw = document.createElement("button");
            sw.type = "button";
            sw.className = `swatch-btn${cur.chromaColor?.toLowerCase() === hex ? " selected" : ""}`;
            sw.style.background = hex;
            sw.title = hex;
            sw.addEventListener("click", async () => {
              const res = await apiPost("/api/effect", {
                effect_id: eid,
                params: { chromaColor: hex },
              });
              if (res.config) {
                appState.config = res.config;
                syncAllEffectCards();
              }
            });
            row.appendChild(sw);
          });
          body.appendChild(row);
        }
      });

      card.appendChild(head);
      card.appendChild(body);
      host.appendChild(card);
    });
  }

  function syncAllEffectCards() {
    buildEffectsSection("mic-effects-list", "microphone", ["noise_removal", "room_echo_removal", "studio_voice"]);
    buildEffectsSection("spk-effects-list", "speaker", ["speaker_noise_removal", "speaker_echo_removal"]);
    buildEffectsSection("cam-effects-list", "camera", [
      "background_blur",
      "background_replace",
      "background_remove",
      "auto_frame",
      "eye_contact",
      "video_noise_removal",
      "virtual_key_light",
      "vignette",
    ]);
  }

  function renderBluetoothBanner() {
    const banner = $("bt-banner");
    const cards = appState?.devices?.bluetooth_cards || [];
    if (!cards.length) {
      banner.style.display = "none";
      banner.innerHTML = "";
      return;
    }
    banner.style.display = "flex";
    banner.innerHTML = "";
    cards.forEach((c) => {
      const isHfp = /headset|handsfree/i.test(c.active_profile || "");
      const row = document.createElement("div");
      row.className = "bt-banner-row";
      row.innerHTML = `
        <span><strong>${c.description || c.name}</strong> (${isHfp ? "Headset Mic + Audio" : "Stereo A2DP, mic off"})</span>
        <button class="chip-btn ${isHfp ? "active" : ""}" type="button">${isHfp ? "Switch to Stereo A2DP" : "Enable Headset Mic"}</button>
      `;
      row.querySelector("button").addEventListener("click", async () => {
        const r = await apiPost("/api/audio/bluetooth-profile", {
          card: c.name,
          enable_mic: !isHfp,
        });
        showToast(r.message || "Updated Bluetooth profile");
        if (r.devices) {
          appState.devices = r.devices;
          populateDevices();
        }
      });
      banner.appendChild(row);
    });
  }

  function syncDefaultButtons(micIsDefault, spkIsDefault) {
    const mb = $("btn-mic-default");
    mb.classList.toggle("active", Boolean(micIsDefault));
    mb.textContent = micIsDefault ? "✓ Default Mic" : "Use as Default Mic";
    const sb = $("btn-spk-default");
    sb.classList.toggle("active", Boolean(spkIsDefault));
    sb.textContent = spkIsDefault ? "✓ Default Output" : "Use as Default Output";
  }

  async function refreshDevicesLive(announce = false) {
    const devs = await apiGet("/api/devices");
    if (!devs) return;
    appState.devices = devs;
    populateDevices();
    if (announce) showToast("Refreshed audio & camera devices");
  }

  // ==================== DEVICES & INITIAL POPULATE ====================
  function populateDevices() {
    const micSel = $("mic-device-select");
    micSel.innerHTML = `<option value="auto">System Default Microphone</option>`;
    (appState.devices.microphones || []).forEach((d) => {
      const o = document.createElement("option");
      o.value = d.name;
      o.textContent = d.description + (d.is_default ? " (Default)" : "");
      micSel.appendChild(o);
    });
    const wantedMic = appState.config.microphone.device || "auto";
    micSel.value = [...micSel.options].some((o) => o.value === wantedMic) ? wantedMic : "auto";

    const spkSel = $("spk-device-select");
    spkSel.innerHTML = `<option value="auto">System Default Speakers / Headphones</option>`;
    (appState.devices.speakers || []).forEach((d) => {
      const o = document.createElement("option");
      o.value = d.name;
      o.textContent = d.description + (d.is_default ? " (Default)" : "");
      spkSel.appendChild(o);
    });
    const wantedSpk = appState.config.speaker.device || "auto";
    spkSel.value = [...spkSel.options].some((o) => o.value === wantedSpk) ? wantedSpk : "auto";

    const camSel = $("cam-device-select");
    camSel.innerHTML = `<option value="auto">Auto-detect Webcam</option>`;
    (appState.devices.cameras || []).forEach((c) => {
      const o = document.createElement("option");
      o.value = c.path;
      o.textContent = `${c.card} (${c.path})`;
      camSel.appendChild(o);
    });
    const wantedCam = appState.config.camera.device || "auto";
    camSel.value = [...camSel.options].some((o) => o.value === wantedCam) ? wantedCam : "auto";

    const camCfg = appState.config.camera;
    const resStr = `${camCfg.width}x${camCfg.height}@${camCfg.fps}`;
    if ([...$("cam-res-select").options].some((o) => o.value === resStr)) {
      $("cam-res-select").value = resStr;
    }

    $("mic-hpf-toggle").checked = Boolean(appState.config.microphone.highpass_enabled);
    $("mic-gate-toggle").checked = Boolean(appState.config.microphone.noise_gate_enabled);
    $("mic-agc-toggle").checked = Boolean(appState.config.microphone.auto_gain_enabled);

    $("btn-cam-mirror").classList.toggle("active", Boolean(camCfg.mirror));
    updateSplitUi(Boolean(camCfg.split_preview), Number(camCfg.split_position ?? 0.5));
    renderBluetoothBanner();
    const defs = appState.devices.defaults || {};
    syncDefaultButtons(defs.mic_is_default, defs.speaker_is_default);
  }

  // ==================== CAMERA PREVIEW & SPLIT CURTAIN ====================
  function syncCameraPreview(running) {
    const img = $("camera-stream-img");
    const empty = $("camera-empty");
    const overlay = $("stage-overlay-top");
    $("cam-master-toggle").checked = running;
    if (running) {
      empty.style.display = "none";
      overlay.style.display = "flex";
      img.style.display = "block";
      if (!img.dataset.live) {
        streamSeq += 1;
        img.dataset.live = "1";
        img.src = `${API_URL}/api/camera/stream.mjpeg?token=${encodeURIComponent(API_TOKEN)}&s=${streamSeq}`;
      }
      updateSplitUi(Boolean(appState.config.camera.split_preview), Number(appState.config.camera.split_position ?? 0.5));
    } else {
      img.removeAttribute("src");
      delete img.dataset.live;
      img.style.display = "none";
      overlay.style.display = "none";
      $("split-overlay").style.display = "none";
      empty.style.display = "flex";
    }
  }

  function updateSplitUi(enabled, pos) {
    $("btn-cam-split").classList.toggle("active", enabled);
    const camOn = Boolean($("cam-master-toggle").checked);
    $("split-overlay").style.display = enabled && camOn ? "block" : "none";
    $("split-handle").style.left = `${Math.round(pos * 100)}%`;
  }

  function setupSplitDrag() {
    const handle = $("split-handle");
    const stage = $("camera-stage");
    let dragging = false;
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      dragging = true;
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const rect = stage.getBoundingClientRect();
      const pos = Math.max(0.08, Math.min(0.92, (e.clientX - rect.left) / Math.max(1, rect.width)));
      handle.style.left = `${Math.round(pos * 100)}%`;
      appState.config.camera.split_position = pos;
      apiPost("/api/camera/config", { split_position: pos });
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });
  }

  // ==================== V4L2 HARDWARE CONTROLS ====================
  async function loadV4l2Controls() {
    const dev = $("cam-device-select").value || "auto";
    const data = await apiGet(`/api/camera/v4l2?device=${encodeURIComponent(dev)}`);
    const grid = $("v4l2-controls-grid");
    grid.innerHTML = "";
    const ctrls = data.controls || [];
    if (!ctrls.length) {
      grid.innerHTML = `<div class="studio-footnote">No adjustable V4L2 controls found on ${data.device}.</div>`;
      return;
    }
    ctrls.forEach((c) => {
      const row = document.createElement("div");
      row.className = "slider-row";
      row.innerHTML = `
        <div class="slider-head">
          <span>${c.name}</span>
          <span class="slider-val">${c.value}</span>
        </div>
        <input type="range" class="nv-slider" min="${c.minimum}" max="${c.maximum}" step="${c.step || 1}" value="${c.value}" />
      `;
      const inp = row.querySelector("input");
      const valEl = row.querySelector(".slider-val");
      inp.addEventListener("input", () => {
        valEl.textContent = inp.value;
      });
      inp.addEventListener("change", async () => {
        await apiPost("/api/camera/v4l2", {
          device: data.device,
          control: c.name,
          value: Number(inp.value),
        });
      });
      grid.appendChild(row);
    });
  }

  // ==================== DOCTOR / SETTINGS ====================
  async function refreshDoctor() {
    const doc = await apiGet("/api/doctor");
    const host = $("doctor-list");
    host.innerHTML = "";
    (doc.checks || []).forEach((chk) => {
      const row = document.createElement("div");
      row.className = "doctor-row";
      row.innerHTML = `
        <span class="doc-badge ${chk.status}">${chk.status}</span>
        <div>
          <div class="doc-name">${chk.name.replace(/_/g, " ")}</div>
          <div class="doc-detail">${chk.detail}</div>
        </div>
      `;
      host.appendChild(row);
    });
  }

  // ==================== LIVE SSE TELEMETRY ====================
  function connectEvents() {
    const es = new EventSource(`${API_URL}/api/events?token=${encodeURIComponent(API_TOKEN)}`);
    es.onmessage = (ev) => {
      try {
        const t = JSON.parse(ev.data);
        // GPU pill
        if (t.gpu && t.gpu.available) {
          $("gpu-pill-text").textContent = `${t.gpu.name} • ${t.gpu.utilization_pct}% GPU • ${t.gpu.temp_c}°C`;
        } else {
          $("gpu-pill-text").textContent = "No NVIDIA GPU";
        }

        // Hot-plug detection (Bluetooth, USB, 3.5mm headphones connected/disconnected live)
        if (t.device_sig && t.device_sig !== lastDeviceSig) {
          const hadSig = Boolean(lastDeviceSig);
          lastDeviceSig = t.device_sig;
          if (hadSig) {
            refreshDevicesLive(false);
          }
        }
        syncDefaultButtons(t.mic?.is_default, t.speaker?.is_default);

        // Mic meter & state
        $("mic-master-toggle").checked = Boolean(t.mic.running);
        const micSub = $("mic-engine-status");
        micSub.textContent = t.mic.running
          ? `Active (${t.mic.backend}) • Select "NVIDIA Broadcast (Microphone)"`
          : "Virtual mic inactive";
        micSub.classList.toggle("active", Boolean(t.mic.running));
        updateMeterPills("mic-meter-pills", "mic-meter-db", t.mic.meter?.output_rms_db ?? -90);

        // Speaker meter & state
        $("spk-master-toggle").checked = Boolean(t.speaker.running);
        const spkSub = $("spk-engine-status");
        spkSub.textContent = t.speaker.running
          ? `Active (${t.speaker.backend}) • Route apps to "NVIDIA Broadcast (Speakers)"`
          : "Virtual speakers inactive";
        spkSub.classList.toggle("active", Boolean(t.speaker.running));
        updateMeterPills("spk-meter-pills", "spk-meter-db", t.speaker.meter?.output_rms_db ?? -90);

        // Camera state & overlay chips
        const camSub = $("cam-engine-status");
        camSub.textContent = t.camera.running
          ? `${t.camera.resolution || "Active"} • ${t.camera.output ? t.camera.output : "Live Preview"}`
          : "Camera inactive";
        camSub.classList.toggle("active", Boolean(t.camera.running));
        if (t.camera.running !== Boolean($("camera-stream-img").dataset.live)) {
          syncCameraPreview(Boolean(t.camera.running));
        }
        if (t.camera.running) {
          $("cam-fps-chip").textContent = `${Number(t.camera.fps || 0).toFixed(1)} FPS`;
          const segLabels = {
            rvm_cuda: "RVM CUDA (RTX)",
            mediapipe: "MediaPipe Neural",
            rvm_cpu: "RVM Neural CPU",
            fast_portrait: "Fallback",
          };
          $("cam-backend-chip").textContent = segLabels[t.camera.segmentation_backend] || t.camera.segmentation_backend;
          $("cam-vcam-chip").textContent = t.camera.output ? `Virtual Cam: ${t.camera.output}` : "Preview Only (no v4l2loopback)";
        }

        // File job state
        if (t.file_job) {
          const fj = t.file_job;
          $("file-status-badge").textContent = fj.status.toUpperCase();
          $("file-status-badge").classList.toggle("ok", fj.status === "completed");
          if (fj.active || fj.status === "completed" || fj.status === "failed") {
            $("file-progress-wrap").style.display = "block";
            $("file-progress-fill").style.width = `${fj.progress || 0}%`;
            $("file-progress-pct").textContent = `${Math.round(fj.progress || 0)}%`;
            $("file-progress-msg").textContent = fj.message || fj.status;
          }
          $("btn-file-cancel").style.display = fj.active ? "inline-block" : "none";
          if (fj.status === "completed" && fj.output_path) {
            lastCompletedOutput = fj.output_path;
            $("btn-file-reveal").style.display = "inline-block";
          }
        }
      } catch (_) {
        // ignore parse errors
      }
    };
  }

  // ==================== WIRE EVENTS ====================
  function wireUiHandlers() {
    $("btn-min").addEventListener("click", () => window.nvBroadcast.windowControl("minimize"));
    $("btn-max").addEventListener("click", () => window.nvBroadcast.windowControl("maximize"));
    $("btn-close").addEventListener("click", () => window.nvBroadcast.windowControl("close"));

    document.querySelectorAll(".sidebar-link").forEach((btn) => {
      btn.addEventListener("click", () => setActivePage(btn.dataset.page));
    });

    // Mic master & settings
    $("mic-master-toggle").addEventListener("change", async (e) => {
      const r = await apiPost(e.target.checked ? "/api/audio/mic/start" : "/api/audio/mic/stop");
      showToast(r.message || (e.target.checked ? "Virtual microphone started" : "Virtual microphone stopped"));
    });

    $("mic-device-select").addEventListener("change", (e) => {
      apiPost("/api/audio/mic/config", { device: e.target.value });
    });
    $("btn-refresh-audio").addEventListener("click", () => refreshDevicesLive(true));
    $("btn-mic-default").addEventListener("click", async () => {
      const enable = !$("btn-mic-default").classList.contains("active");
      const r = await apiPost("/api/audio/set-default", { target: "mic", enable });
      showToast(r.message || "Updated default microphone");
      if (r.devices) {
        appState.devices = r.devices;
        populateDevices();
      }
    });
    $("btn-spk-default").addEventListener("click", async () => {
      const enable = !$("btn-spk-default").classList.contains("active");
      const r = await apiPost("/api/audio/set-default", { target: "speaker", enable });
      showToast(r.message || "Updated default output");
      if (r.devices) {
        appState.devices = r.devices;
        populateDevices();
      }
    });
    $("mic-hpf-toggle").addEventListener("change", (e) => {
      apiPost("/api/audio/mic/config", { highpass_enabled: e.target.checked });
    });
    $("mic-gate-toggle").addEventListener("change", (e) => {
      apiPost("/api/audio/mic/config", { noise_gate_enabled: e.target.checked });
    });
    $("mic-agc-toggle").addEventListener("change", (e) => {
      apiPost("/api/audio/mic/config", { auto_gain_enabled: e.target.checked });
    });

    // Speaker master & device
    $("spk-master-toggle").addEventListener("change", async (e) => {
      const r = await apiPost(e.target.checked ? "/api/audio/speaker/start" : "/api/audio/speaker/stop");
      showToast(r.message || (e.target.checked ? "Virtual speakers started" : "Virtual speakers stopped"));
    });
    $("spk-device-select").addEventListener("change", (e) => {
      apiPost("/api/audio/speaker/config", { device: e.target.value });
    });

    // 5s Mic Test
    $("btn-mic-record").addEventListener("click", async () => {
      const btn = $("btn-mic-record");
      btn.disabled = true;
      btn.textContent = "Recording (5s)...";
      $("mic-test-badge").textContent = "Recording";
      const res = await apiPost("/api/audio/mic/test", { seconds: 5.0 });
      btn.disabled = false;
      btn.textContent = "Record 5s";
      if (!res.ok) {
        $("mic-test-badge").textContent = "Error";
        showToast(res.error || "Mic test failed");
        return;
      }
      micTestResult = res;
      $("btn-mic-play").disabled = false;
      $("btn-mic-stop").disabled = false;
      $("mic-test-badge").textContent = `-${res.noise_floor_reduction_db} dB noise`;
      $("mic-test-badge").classList.add("ok");
      $("mic-test-info").textContent = `Raw: ${res.raw_rms_db} dBFS  |  Broadcast: ${res.processed_rms_db} dBFS (${res.duration_sec}s)`;
      const useProc = $("mic-test-ab-toggle").checked;
      drawWaveform("mic-waveform-canvas", useProc ? res.processed_waveform : res.raw_waveform, useProc ? "#76b900" : "#acadad");
    });

    $("mic-test-ab-toggle").addEventListener("change", () => {
      if (!micTestResult) return;
      const useProc = $("mic-test-ab-toggle").checked;
      drawWaveform(
        "mic-waveform-canvas",
        useProc ? micTestResult.processed_waveform : micTestResult.raw_waveform,
        useProc ? "#76b900" : "#acadad"
      );
    });

    $("btn-mic-play").addEventListener("click", () => {
      if (!micTestResult) return;
      const useProc = $("mic-test-ab-toggle").checked;
      apiPost("/api/audio/play", { path: useProc ? micTestResult.processed_wav_path : micTestResult.raw_wav_path });
    });
    $("btn-mic-stop").addEventListener("click", () => apiPost("/api/audio/stop-play"));

    // Speaker Sample Test
    async function loadAndPlaySpeakerSample(playNow) {
      const kind = $("spk-sample-select").value;
      const res = await apiPost("/api/audio/speaker/sample", { kind });
      if (!res.ok) return;
      spkSampleResult = res;
      const useProc = $("spk-test-ab-toggle").checked;
      drawWaveform("spk-waveform-canvas", useProc ? res.processed_waveform : res.raw_waveform, useProc ? "#76b900" : "#acadad");
      $("spk-sample-info").textContent = `Raw: ${res.raw_rms_db} dBFS | Cleaned: ${res.processed_rms_db} dBFS (-${res.noise_floor_reduction_db} dB)`;
      if (playNow) {
        apiPost("/api/audio/play", { path: useProc ? res.processed_wav_path : res.raw_wav_path });
      }
    }

    $("btn-spk-play").addEventListener("click", () => loadAndPlaySpeakerSample(true));
    $("btn-spk-stop").addEventListener("click", () => apiPost("/api/audio/stop-play"));
    $("spk-test-ab-toggle").addEventListener("change", () => {
      if (!spkSampleResult) return;
      const useProc = $("spk-test-ab-toggle").checked;
      drawWaveform(
        "spk-waveform-canvas",
        useProc ? spkSampleResult.processed_waveform : spkSampleResult.raw_waveform,
        useProc ? "#76b900" : "#acadad"
      );
    });

    // Camera controls
    async function toggleCamera(on) {
      const r = await apiPost(on ? "/api/camera/start" : "/api/camera/stop");
      syncCameraPreview(on && r.ok);
      showToast(r.message || (on ? "Camera started" : "Camera stopped"));
    }

    $("cam-master-toggle").addEventListener("change", (e) => toggleCamera(e.target.checked));
    $("btn-cam-start-stage").addEventListener("click", () => toggleCamera(true));

    $("cam-device-select").addEventListener("change", (e) => {
      apiPost("/api/camera/config", { device: e.target.value });
      if ($("v4l2-drawer").style.display !== "none") loadV4l2Controls();
    });
    $("cam-res-select").addEventListener("change", (e) => {
      const m = e.target.value.match(/^(\d+)x(\d+)@(\d+)$/);
      if (m) {
        apiPost("/api/camera/config", { width: Number(m[1]), height: Number(m[2]), fps: Number(m[3]) });
      }
    });

    $("btn-cam-split").addEventListener("click", async () => {
      const next = !Boolean(appState.config.camera.split_preview);
      appState.config.camera.split_preview = next;
      await apiPost("/api/camera/config", { split_preview: next });
      updateSplitUi(next, Number(appState.config.camera.split_position ?? 0.5));
    });

    $("btn-cam-mirror").addEventListener("click", async () => {
      const next = !Boolean(appState.config.camera.mirror);
      appState.config.camera.mirror = next;
      $("btn-cam-mirror").classList.toggle("active", next);
      await apiPost("/api/camera/config", { mirror: next });
    });

    $("btn-cam-snapshot").addEventListener("click", async () => {
      const r = await apiPost("/api/camera/snapshot");
      if (r.ok) {
        showToast(`Saved snapshot: ${r.path.split("/").pop()}`);
      } else {
        showToast(r.error || "Start the camera before taking a snapshot");
      }
    });

    $("btn-cam-v4l2").addEventListener("click", () => {
      const drawer = $("v4l2-drawer");
      const open = drawer.style.display === "none";
      drawer.style.display = open ? "block" : "none";
      $("btn-cam-v4l2").classList.toggle("active", open);
      if (open) loadV4l2Controls();
    });
    $("btn-v4l2-refresh").addEventListener("click", loadV4l2Controls);

    // Files page
    $("btn-pick-file").addEventListener("click", async () => {
      const picked = await window.nvBroadcast.pickMediaFile();
      if (!picked) return;
      selectedMediaFile = picked;
      $("file-selected-label").textContent = picked;
      $("btn-file-start").disabled = false;
    });
    $("btn-file-start").addEventListener("click", async () => {
      if (!selectedMediaFile) return;
      $("btn-file-reveal").style.display = "none";
      const r = await apiPost("/api/files/start", {
        input_path: selectedMediaFile,
        apply_audio: $("file-apply-audio").checked,
        apply_video: $("file-apply-video").checked,
      });
      showToast(r.message || "File processing started");
    });
    $("btn-file-cancel").addEventListener("click", () => apiPost("/api/files/cancel"));
    $("btn-file-reveal").addEventListener("click", () => {
      if (lastCompletedOutput) window.nvBroadcast.revealPath(lastCompletedOutput);
    });

    // Settings / Doctor page
    $("btn-doctor-refresh").addEventListener("click", refreshDoctor);
    $("btn-download-models").addEventListener("click", async () => {
      showToast("Verifying neural models...");
      const r = await apiPost("/api/models/download");
      showToast(r.ok ? "All neural models verified!" : "Some model downloads failed");
      refreshDoctor();
    });
    $("btn-install-pw-filter").addEventListener("click", async () => {
      const r = await apiPost("/api/audio/install-filter");
      showToast(r.ok ? `Installed PipeWire ${r.backend} filter to ${r.path}` : "Filter install failed");
    });
    $("btn-copy-vcam-cmd").addEventListener("click", async () => {
      await window.nvBroadcast.copyText($("vcam-setup-cmd").textContent);
      showToast("Copied Fedora virtual camera command");
    });

    setupSplitDrag();
  }

  async function init() {
    initMeterPills("mic-meter-pills");
    initMeterPills("spk-meter-pills");
    drawWaveform("mic-waveform-canvas", []);
    drawWaveform("spk-waveform-canvas", []);

    const boot = await window.nvBroadcast.getBootstrap();
    API_URL = boot.url;
    API_TOKEN = boot.token;
    appState = await apiGet("/api/state");

    populateDevices();
    syncAllEffectCards();
    wireUiHandlers();
    setActivePage(boot.initialTab || "audio");
    connectEvents();

    // Generate default speaker waveform preview without playing
    const s = await apiPost("/api/audio/speaker/sample", { kind: "fan" });
    if (s.ok) {
      spkSampleResult = s;
      drawWaveform("spk-waveform-canvas", s.processed_waveform, "#76b900");
    }
  }

  init().catch((err) => {
    console.error("NVIDIA Broadcast UI init failed:", err);
  });
})();
