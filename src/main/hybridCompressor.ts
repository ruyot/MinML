import { Falcon7BProvider, CompressionResult } from "./modelProvider.js";
import { compressPrompt as algorithmicCompress } from "./compressor.js";

export class HybridCompressor {
  private falcon7bProvider: Falcon7BProvider;
  private isModelAvailable: boolean = false;

  constructor() {
    this.falcon7bProvider = new Falcon7BProvider();
    this.checkModelAvailability();
  }

  private async checkModelAvailability() {
    try {
      this.isModelAvailable = await this.falcon7bProvider.isAvailable();
      console.log(`Falcon7B model ${this.isModelAvailable ? 'is available' : 'is not available'}`);
    } catch (error) {
      console.warn('Failed to check Falcon7B availability:', error);
      this.isModelAvailable = false;
    }
  }

  async compressPrompt(input: string): Promise<CompressionResult> {
    // If model is not available, use algorithmic compression directly
    if (!this.isModelAvailable) {
      console.log('Using algorithmic compression (model not available)');
      return algorithmicCompress(input);
    }

    try {
      // Try Falcon 7B first
      console.log('Attempting Falcon7B compression...');
      const falcon7bResult = await this.falcon7bProvider.compress(input);
      
      if (falcon7bResult) {
        console.log(`Falcon7B compression successful: ${falcon7bResult.stats.pct}% reduction in ${falcon7bResult.processingTime}ms`);
        return falcon7bResult;
      } else {
        console.log('Falcon7B compression failed or insufficient, falling back to algorithmic');
      }
    } catch (error) {
      console.warn('Falcon7B compression error, falling back to algorithmic:', error);
    }

    // Fallback to algorithmic compression
    const algorithmicResult = algorithmicCompress(input);
    console.log(`Algorithmic compression: ${algorithmicResult.stats.pct}% reduction in ${algorithmicResult.processingTime}ms`);
    return algorithmicResult;
  }

  async refreshModelAvailability() {
    await this.checkModelAvailability();
  }

  getModelStatus() {
    return {
      falcon7bAvailable: this.isModelAvailable,
      pythonServerUrl: "http://127.0.0.1:8081"
    };
  }
}
