"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("nvBroadcast", {
  getBootstrap: () => ipcRenderer.invoke("nvb:get-bootstrap"),
  windowControl: (action) => ipcRenderer.invoke("nvb:window-control", action),
  pickImage: () => ipcRenderer.invoke("nvb:pick-image"),
  pickMediaFile: () => ipcRenderer.invoke("nvb:pick-media-file"),
  revealPath: (filePath) => ipcRenderer.invoke("nvb:reveal-path", filePath),
  copyText: (text) => ipcRenderer.invoke("nvb:copy-text", text),
});
