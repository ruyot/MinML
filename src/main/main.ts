import { app, BrowserWindow, ipcMain, nativeTheme, Tray, Menu } from "electron";
import path from "node:path";
import { startProxy, stopProxy, setActive, getActive } from "./proxy.js";
import { getMetricsSnapshot, resetToday } from "./metrics.js";

process.env.ELECTRON_DISABLE_SECURITY_WARNINGS = "true";
nativeTheme.themeSource = "dark";

// Fix for black screen issues - disable GPU acceleration if needed
if (process.argv.includes('--disable-gpu')) {
  app.disableHardwareAcceleration();
}

let win: BrowserWindow | null = null;
let tray: Tray | null = null;

function createWindow() {
  win = new BrowserWindow({
    width: 880,
    height: 560,
    title: "MinML",
    backgroundColor: "#000000",
    webPreferences: {
      preload: path.join(app.getAppPath(), "dist", "main", "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  const isDev = process.env.NODE_ENV === 'development';
  const url = isDev ? 'http://localhost:5173' : `file://${path.join(app.getAppPath(), "dist", "renderer", "index.html")}`;
  win.loadURL(url);
}

app.whenReady().then(async () => {
  await startProxy(); // starts inactive, can still forward
  createWindow();
  // Skip tray for now to avoid icon loading issues
  // tray = new Tray(process.platform === "darwin" ? 
  //   path.join(currentDir, "../../assets/trayTemplate.png") : 
  //   path.join(currentDir, "../../assets/tray.ico"));
  // Skip tray menu for now
  // const menu = Menu.buildFromTemplate([
  //   { label: "Open MinML", click: () => win?.show() },
  //   { type: "separator" },
  //   { label: getActive() ? "Deactivate Compression" : "Activate Compression", click: () => { setActive(!getActive()); updateTray(); win?.webContents.send("minml:activeChanged", getActive()); } },
  //   { label: "Reset Today", click: () => { resetToday(); win?.webContents.send("minml:metrics", getMetricsSnapshot()); } },
  //   { type: "separator" },
  //   { label: "Quit", role: "quit" }
  // ]);
  // tray.setContextMenu(menu);
  // updateTray();
});

function updateTray() {
  // Skip tray for now
  // tray?.setToolTip(`MinML — ${getActive() ? "Active" : "Inactive"}`);
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("minml:getMetrics", () => getMetricsSnapshot());
ipcMain.handle("minml:getActive", () => getActive());
ipcMain.handle("minml:setActive", (_e: any, v: boolean) => { setActive(v); return getActive(); });
