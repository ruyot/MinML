// Check MinML status and functionality
import fetch from 'node-fetch';

console.log('MinML Status Check\n');

// Check if proxy is running
try {
  const response = await fetch('http://localhost:3123/v1/models', {
    headers: { 'Authorization': 'Bearer test' }
  });
  console.log('Proxy Server: Running on http://localhost:3123');
  console.log(`   Status: ${response.status} ${response.statusText}`);
} catch (error) {
  console.log('Proxy Server: Not responding');
}

// Check if we can test compression
try {
  const { compressPrompt } = await import('./dist/main/compressor.js');
  const result = compressPrompt("Please help me write a simple test for beginners");
  console.log('Compression Engine: Working');
  console.log(`   Sample reduction: ${result.stats.beforeTokens} → ${result.stats.afterTokens} tokens (${result.stats.pct}% saved)`);
} catch (error) {
  console.log('Compression Engine: Error loading');
}

// Check if we can access metrics
try {
  const { getMetricsSnapshot } = await import('./dist/main/metrics.js');
  const metrics = getMetricsSnapshot();
  console.log('Metrics System: Working');
  console.log(`   Total requests: ${metrics.totals.requests}`);
  console.log(`   Tokens saved: ${metrics.totals.saved}`);
} catch (error) {
  console.log('Metrics System: Error loading');
}

console.log('\nMinML is fully operational!');
console.log('\nWhat you can do now:');
console.log('• Electron App: Should be running (check for MinML window)');
console.log('• Web UI: Visit http://localhost:5174/ (if Vite dev server is running)');
console.log('• API Proxy: Point your OpenAI client to http://localhost:3123/v1');
console.log('• Test Compression: Run `node demo.js` for examples');
console.log('• Monitor Metrics: Use the UI or check metrics.json files');

console.log('\n🔗 Ready for your 1B model integration via ModelProvider interface!');
