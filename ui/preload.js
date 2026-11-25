"use strict";

const { contextBridge, ipcRenderer } = require("electron");

const api = {
  getBootstrap: () => ipcRenderer.invoke("nvb:get-bootstrap"),
  windowControl: (action) => ipcRenderer.invoke("nvb:window-control", action),
  pickImage: () => ipcRenderer.invoke("nvb:pick-image"),
  pickMediaFile: () => ipcRenderer.invoke("nvb:pick-media-file"),
  revealPath: (filePath) => ipcRenderer.invoke("nvb:reveal-path", filePath),
  copyText: (text) => ipcRenderer.invoke("nvb:copy-text", text),
  getUiPrefs: () => ipcRenderer.invoke("nvb:get-ui-prefs"),
  setUiPrefs: (patch) => ipcRenderer.invoke("nvb:set-ui-prefs", patch),
  onSwitchTab: (cb) => {
    if (typeof cb !== "function") return () => {};
    const handler = (_ev, tab) => cb(tab);
    ipcRenderer.on("nvb:switch-tab", handler);
    return () => ipcRenderer.removeListener("nvb:switch-tab", handler);
  },
};

contextBridge.exposeInMainWorld("nvBroadcast", api);
contextBridge.exposeInMainWorld("nvb", api);
