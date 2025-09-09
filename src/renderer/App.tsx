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

  async function refresh() {
    const [a, m] = await Promise.all([window.minml.getActive(), window.minml.getMetrics()]);
    setActive(a); setMetrics(m);
  }

  useEffect(() => {
    refresh();
    window.minml.onActiveChanged(setActive);
    window.minml.onMetrics(setMetrics);
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
        <h2>Setup</h2>
        <ol>
          <li>Set your client's base URL to <code>http://localhost:3123/v1</code>.</li>
          <li>Keep <code>Authorization: Bearer &lt;YOUR_API_KEY&gt;</code>.</li>
          <li>(Optional) To target another provider, set <code>MINML_TARGET_BASE</code> env before launching MinML.</li>
        </ol>
      </section>

      <footer>© {new Date().getFullYear()} MinML</footer>
    </div>
  );
}
