import { encoding_for_model } from "tiktoken";
import { CompressionResult } from "./modelProvider.js";

const STOPWORDS = new Set([
  "please","kindly","basically","actually","just","really","very","like",
  "i need","i want","can you","could you","would you","that","those","these","some","kind of","sort of"
]);

const AUDIENCE = /(beginner|novice|intermediate|advanced|expert|from the ground up|step by step)/gi;

function approxTokens(s: string): number {
  try {
    const enc = encoding_for_model("gpt-3.5-turbo");
    const n = enc.encode(s).length;
    enc.free();
    return n;
  } catch {
    return Math.ceil(s.length / 4);
  }
}

function protectSpans(input: string): string[] {
  const spans = new Set<string>();
  // quoted phrases
  for (const m of input.matchAll(/"([^"]{2,120})"/g)) spans.add(m[1]);
  // audience/level phrases
  for (const m of input.matchAll(AUDIENCE)) spans.add(m[0]);
  // numbers and units
  for (const m of input.matchAll(/\b\d[\d.,%]*\b/g)) spans.add(m[0]);
  // section headers
  for (const m of input.matchAll(/\b(Format|Constraints|Style|Audience|Output|Length)\s*:/gi)) spans.add(m[0]);
  return [...spans].slice(0, 24);
}

function removeFillers(input: string, keep: Set<string>): string {
  // sentence-level dedupe
  const seen = new Set<string>();
  const sentences = input
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean)
    .filter(s => { const k = s.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });

  let text = sentences.join(" ");

  // word-level pruning except inside protected spans
  for (const sw of STOPWORDS) {
    const re = new RegExp(`\\b${sw}\\b`, "gi");
    text = text.replace(re, (m, offset) => {
      // keep if inside a protected span
      for (const p of keep) {
        const idx = text.indexOf(p);
        if (idx >= 0 && offset >= idx && offset <= idx + p.length) return m;
      }
      return "";
    });
  }
  // collapse spaces
  text = text.replace(/\s{2,}/g, " ").trim();
  return text;
}

export function compressPrompt(input: string): CompressionResult {
  const startTime = Date.now();
  const beforeTokens = approxTokens(input);
  const spans = protectSpans(input);
  const keep = new Set(spans);

  // mask spans
  let masked = input;
  const markers: Record<string,string> = {};
  spans.forEach((s, i) => {
    const key = `«${i.toString(36)}»`;
    markers[key] = s;
    masked = masked.replaceAll(s, key);
  });

  // prune + normalize
  let pruned = removeFillers(masked, keep);

  // light keyword prioritization (verbs, nouns by simple heuristic)
  pruned = pruned
    .split(/\s+/)
    .filter((w) => {
      if (/^«[0-9a-z]+»$/.test(w)) return true; // keep markers
      if (/^[A-Z][a-z]{2,}$/.test(w)) return true; // Proper nouns
      if (/^\w{4,}$/.test(w)) return true; // length heuristic
      return false;
    })
    .join(" ");

  // unmask spans
  for (const [k,v] of Object.entries(markers)) pruned = pruned.replaceAll(k, v);

  // tidy punctuation
  pruned = pruned.replace(/\s+([,.:;!?])/g, "$1").replace(/\s{2,}/g, " ").trim();

  const afterTokens = approxTokens(pruned);
  const saved = Math.max(0, beforeTokens - afterTokens);
  const pct = beforeTokens ? Math.round((saved / beforeTokens) * 100) : 0;
  const processingTime = Date.now() - startTime;

  return {
    compressed: pruned,
    protectedSpans: spans,
    stats: { beforeTokens, afterTokens, saved, pct },
    source: 'algorithmic',
    processingTime
  };
}
