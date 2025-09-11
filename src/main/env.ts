export const PORT = Number(process.env.MINML_PORT ?? 3123);
export const TARGET_BASE = process.env.MINML_TARGET_BASE ?? "https://api.openai.com";
export const SAVE_INTERVAL_MS = 10_000;
