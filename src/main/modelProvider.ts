export type Analysis = {
  slots?: { task?: string; topic?: string; audience?: string; constraints?: string[]; format?: string };
  protectedSpans?: string[];
};

export interface ModelProvider {
  analyze(input: string): Promise<Analysis | null>;
}

export class HeuristicProvider implements ModelProvider {
  async analyze(_input: string): Promise<Analysis | null> {
    // placeholder — we rely on compressor's internal protectSpans for now
    return null;
  }
}

// Example future provider:
// export class LlamaCppProvider implements ModelProvider {
//   constructor(private base = "http://127.0.0.1:8080") {}
//   async analyze(input: string): Promise<Analysis | null> {
//     const res = await fetch(this.base + "/analyze", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ input }) });
//     if (!res.ok) return null;
//     return await res.json();
//   }
// }
