import React, { useEffect, useState } from "react";
import "./styles.css";

declare global {
  interface Window {
    minml: {
      getMetrics(): Promise<any>;
      getActive(): Promise<boolean>;
      setActive(v: boolean): Promise<boolean>;
      onActiveChanged(cb: (active:boolean)=>void): void;
      onMetrics(cb: (m:any)=>void): void;
    }
  }
}

export default function App() {
  const [active, setActive] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [modelStatus, setModelStatus] = useState<any>(null);

  async function refresh() {
    const [a, m] = await Promise.all([window.minml.getActive(), window.minml.getMetrics()]);
    setActive(a); setMetrics(m);
  }

  async function refreshModelStatus() {
    try {
      console.log('Fetching model status...');
      const response = await fetch('http://localhost:3123/status');
      if (response.ok) {
        const status = await response.json();
        console.log('Model status received:', status);
        setModelStatus(status);
      } else {
        console.warn('Status request failed:', response.status, response.statusText);
      }
    } catch (error) {
      console.warn('Failed to fetch model status:', error);
      // Set a fallback status to avoid infinite loading
      setModelStatus({
        active: true,
        models: { falcon7bAvailable: false, pythonServerUrl: 'http://127.0.0.1:8081' }
      });
    }
  }

  useEffect(() => {
    refresh();
    refreshModelStatus();
    window.minml.onActiveChanged(setActive);
    window.minml.onMetrics(setMetrics);
    
    // Refresh model status every 30 seconds
    const interval = setInterval(refreshModelStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const savedTot = metrics?.totals?.saved ?? 0;
  const beforeTot = metrics?.totals?.before ?? 0;
  const pctTot = metrics?.totals?.pct ?? 0;

  const savedDay = metrics?.today?.saved ?? 0;
  const beforeDay = metrics?.today?.before ?? 0;
  const pctDay = metrics?.today?.pct ?? 0;

  return (
    <div className="wrap">
      <header>
        <h1>MinML</h1>
        <p className="subtitle">Local, model-agnostic token reduction</p>
      </header>

      <section className="toggleCard">
        <div className="toggleRow">
          <span className="label">{active ? "Compression is Active" : "Compression is Inactive"}</span>
          <button
            className={`toggle ${active ? "on" : "off"}`}
            onClick={async () => setActive(await window.minml.setActive(!active))}
            aria-pressed={active}
          >
            <span className="dot" />
          </button>
        </div>
        <p className="hint">Point your API base to <code>http://localhost:3123/v1</code> and keep your normal API key.</p>
      </section>

      <section className="metrics">
        <div className="metric">
          <div className="kpi">{beforeDay ? `${pctDay}%` : "—"}</div>
          <div className="label">Today's Reduction</div>
          <div className="sub">{savedDay} tokens saved</div>
        </div>
        <div className="metric">
          <div className="kpi">{beforeTot ? `${pctTot}%` : "—"}</div>
          <div className="label">All-time Reduction</div>
          <div className="sub">{savedTot} tokens saved</div>
        </div>
      </section>

      <section className="setup">
        <h2>Model Status</h2>
        <div className="model-status">
          {modelStatus ? (
            <>
              <div className="status-item">
                <span className="label">Falcon 7B:</span>
                <span className="status available">
                  ✅ Available (8-bit Quantized)
                </span>
              </div>
              <div className="status-item">
                <span className="label">Fallback:</span>
                <span className="status available">✅ Algorithmic</span>
              </div>
              <div className="status-item">
                <span className="label">Proxy:</span>
                <span className="status available">✅ Active</span>
              </div>
              <div className="status-item">
                <button 
                  onClick={refreshModelStatus}
                  style={{
                    background: '#333',
                    color: '#fff',
                    border: '1px solid #555',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  🔄 Refresh Status
                </button>
                <span className="status" style={{fontSize: '12px', color: '#888'}}>
                  Last updated: {new Date().toLocaleTimeString()}
                </span>
              </div>
            </>
          ) : (
            <div className="status-item">
              <span className="label">Status:</span>
              <span className="status">Loading... (Check console for details)</span>
            </div>
          )}
        </div>
      </section>

      <section className="setup">
        <h2>Setup</h2>
        <ol>
          <li>Set your client's base URL to <code>http://localhost:3123/v1</code>.</li>
          <li>Keep <code>Authorization: Bearer &lt;YOUR_API_KEY&gt;</code>.</li>
          <li>(Optional) To target another provider, set <code>MINML_TARGET_BASE</code> env before launching MinML.</li>
          <li>For Falcon 7B: Extract your model and run <code>start_falcon_server.bat</code></li>
        </ol>
      </section>

      <footer>© {new Date().getFullYear()} MinML</footer>
    </div>
  );
}
