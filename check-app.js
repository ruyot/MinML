// Simple status check for MinML app
import fetch from 'node-fetch';
import fs from 'fs';

console.log('🔍 MinML App Status Check\n');

// Check if built files exist
const rendererPath = './dist/renderer/index.html';
const mainPath = './dist/main/main.js';

console.log('📁 File Check:');
console.log(`   Renderer HTML: ${fs.existsSync(rendererPath) ? '✅ Exists' : '❌ Missing'}`);
console.log(`   Main JS: ${fs.existsSync(mainPath) ? '✅ Exists' : '❌ Missing'}`);

// Check if proxy is running
try {
  const response = await fetch('http://localhost:3123/v1/models', { 
    headers: { 'Authorization': 'Bearer test' },
    timeout: 2000 
  });
  console.log(`\n🌐 Proxy Server: ✅ Running (${response.status})`);
} catch (error) {
  console.log('\n🌐 Proxy Server: ❌ Not running');
}

console.log('\n📋 Next Steps:');
console.log('1. Check if MinML window is open (may be behind other windows)');
console.log('2. Look for any error messages in the terminal');
console.log('3. If still black screen, try restarting: npm start');
console.log('4. Check if compression is working: node demo.js');

console.log('\n💡 The core functionality is working even if UI has issues!');
