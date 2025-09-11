export type Analysis = {
  slots?: { task?: string; topic?: string; audience?: string; constraints?: string[]; format?: string };
  protectedSpans?: string[];
};

export type CompressionResult = {
  compressed: string;
  protectedSpans: string[];
  stats: { beforeTokens: number; afterTokens: number; saved: number; pct: number };
  source: 'falcon7b' | 'algorithmic';
  processingTime: number;
};

export interface ModelProvider {
  analyze(input: string): Promise<Analysis | null>;
  compress?(input: string): Promise<CompressionResult | null>;
}

export class HeuristicProvider implements ModelProvider {
  async analyze(_input: string): Promise<Analysis | null> {
    // placeholder — we rely on compressor's internal protectSpans for now
    return null;
  }
}

export class Falcon7BProvider implements ModelProvider {
  private pythonServerUrl: string;
  private timeout: number;
  private minReductionPercent: number;
  
  constructor(
    pythonServerUrl = "http://127.0.0.1:8081",
    timeout = 60000, // 60 seconds
    minReductionPercent = 30
  ) {
    this.pythonServerUrl = pythonServerUrl;
    this.timeout = timeout;
    this.minReductionPercent = minReductionPercent;
  }

  async analyze(input: string): Promise<Analysis | null> {
    try {
      const response = await this.makeRequest('/analyze', { input });
      return response || null;
    } catch (error) {
      console.warn('Falcon7B analyze failed:', error);
      return null;
    }
  }

  async compress(input: string): Promise<CompressionResult | null> {
    const startTime = Date.now();
    
    try {
      const response = await this.makeRequest('/compress', { input });
      const processingTime = Date.now() - startTime;
      
      if (response && response.compressed) {
        const result: CompressionResult = {
          compressed: response.compressed,
          protectedSpans: response.protectedSpans || [],
          stats: response.stats || { beforeTokens: 0, afterTokens: 0, saved: 0, pct: 0 },
          source: 'falcon7b',
          processingTime
        };
        
        // Check if compression meets minimum requirements
        if (result.stats.pct >= this.minReductionPercent) {
          return result;
        } else {
          console.warn(`Falcon7B compression too low: ${result.stats.pct}% < ${this.minReductionPercent}%`);
          return null;
        }
      }
      
      return null;
    } catch (error) {
      const processingTime = Date.now() - startTime;
      console.warn(`Falcon7B compression failed after ${processingTime}ms:`, error);
      return null;
    }
  }

  private async makeRequest(endpoint: string, data: any): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    
    try {
      const response = await fetch(`${this.pythonServerUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error?.name === 'AbortError') {
        throw new Error(`Request timed out after ${this.timeout}ms`);
      }
      throw error;
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.pythonServerUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000) // 5 second health check timeout
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}
