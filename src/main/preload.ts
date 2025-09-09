const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("minml", {
  getMetrics: () => ipcRenderer.invoke("minml:getMetrics"),
  getActive: () => ipcRenderer.invoke("minml:getActive"),
  setActive: (v: boolean) => ipcRenderer.invoke("minml:setActive", v),
  onActiveChanged: (cb: (active: boolean) => void) => ipcRenderer.on("minml:activeChanged", (_e: any, a: any) => cb(a)),
  onMetrics: (cb: (m: any) => void) => ipcRenderer.on("minml:metrics", (_e: any, m: any) => cb(m))
});
