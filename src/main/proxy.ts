import express from "express";
import fetch from "node-fetch";
import { z } from "zod";
import { recordRequest } from "./metrics.js";
import { compressPrompt } from "./compressor.js";
import { PORT, TARGET_BASE } from "./env.js";

let ACTIVE = true;
export const getActive = () => ACTIVE;
export const setActive = (v: boolean) => { ACTIVE = v; };

const ChatSchema = z.object({
  model: z.string(),
  messages: z.array(z.object({ role: z.string(), content: z.any() })), // content can be string or array in new specs
}).passthrough();

const app = express();
app.use(express.json({ limit: "2mb" }));

app.post("/v1/chat/completions", async (req, res) => {
  const auth = req.get("authorization");
  if (!auth) return res.status(401).json({ error: { message: "Missing Authorization header" }});

  const parsed = ChatSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: { message: "Bad request" }});

  let beforeTokens = 0;
  let afterTokens = 0;

  const payload = structuredClone(req.body);
  if (ACTIVE && Array.isArray(payload.messages)) {
    payload.messages = payload.messages.map((m: any) => {
      if (m.role === "user") {
        const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
        const { compressed, stats } = compressPrompt(content);
        beforeTokens += stats.beforeTokens;
        afterTokens += stats.afterTokens;
        return { ...m, content: compressed };
      }
      return m;
    });
  }

  try {
    const r = await fetch(`${TARGET_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": auth
      },
      body: JSON.stringify(payload)
    });

    const isStream = r.headers.get("content-type")?.includes("text/event-stream");
    if (isStream) {
      res.setHeader("content-type", "text/event-stream");
      r.body?.pipe(res);
    } else {
      const data = await r.text();
      if (beforeTokens || afterTokens) recordRequest(beforeTokens, afterTokens);
      res.status(r.status).send(data);
    }
  } catch (e: any) {
    res.status(502).json({ error: { message: `Upstream error: ${e?.message ?? e}` }});
  }
});

// Pass-through for everything else under /v1/*
app.use("/v1", async (req, res) => {
  const auth = req.get("authorization");
  try {
    const r = await fetch(`${TARGET_BASE}${req.originalUrl}`, {
      method: req.method,
      headers: { "content-type": req.get("content-type") ?? "application/json", "authorization": auth ?? "" },
      body: ["GET","HEAD"].includes(req.method) ? undefined : (req as any).rawBody ?? JSON.stringify(req.body)
    });
    const buf = await r.arrayBuffer();
    res.status(r.status);
    r.headers.forEach((v,k) => res.setHeader(k, v));
    res.send(Buffer.from(buf));
  } catch (e: any) {
    res.status(502).json({ error: { message: `Upstream error: ${e?.message ?? e}` }});
  }
});

let server: any = null;

export async function startProxy() {
  if (server) return;
  server = app.listen(PORT, () => console.log(`MinML proxy on http://127.0.0.1:${PORT}`));
}
export async function stopProxy() { if (!server) return; await new Promise(r => server.close(r)); server = null; }
