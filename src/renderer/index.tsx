import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

try {
  const rootElement = document.getElementById("root");
  if (!rootElement) {
    throw new Error("Root element not found");
  }
  
  console.log("MinML: Starting React app...");
  const root = createRoot(rootElement);
  root.render(<App />);
  console.log("MinML: React app rendered successfully");
} catch (error) {
  console.error("MinML: Failed to start React app:", error);
  // Fallback: show error message
  const rootElement = document.getElementById("root");
  if (rootElement) {
    rootElement.innerHTML = `
      <div style="color: red; padding: 20px; font-family: Arial;">
        <h1>MinML - Error Loading React App</h1>
        <p>Error: ${error}</p>
        <p>Check the console for more details.</p>
      </div>
    `;
  }
}
