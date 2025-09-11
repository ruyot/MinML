// Test the compression functionality directly
import { compressPrompt } from './dist/main/compressor.js';

console.log('🧪 Testing MinML Compression\n');

const testPrompts = [
  "Please write a beginner-friendly tutorial on React. I really need you to explain it step by step, like really thoroughly. Can you please make sure to include examples and actually show me how to create components? I want to understand hooks too, please.",
  
  "Format: JSON\nConstraints: Under 500 words\nAudience: Expert developers\nPlease create a comprehensive API documentation for a user authentication system. I need you to basically cover all the endpoints like login, register, refresh tokens, and logout. Could you actually include example requests and responses? This is really important for our project.",
  
  'Explain "machine learning" to a novice. Please be very thorough and really explain everything step by step. I want to understand neural networks, training data, and algorithms. Can you please make it easy to understand?'
];

testPrompts.forEach((prompt, i) => {
  console.log(`\n${'='.repeat(50)}`);
  console.log(`Test ${i + 1}:`);
  console.log(`${'='.repeat(50)}`);
  
  console.log('\n📝 Original prompt:');
  console.log(`"${prompt}"`);
  console.log(`📏 Length: ${prompt.length} characters`);
  
  const result = compressPrompt(prompt);
  
  console.log('\n⚡ Compressed prompt:');
  console.log(`"${result.compressed}"`);
  console.log(`📏 Length: ${result.compressed.length} characters`);
  
  console.log('\n📊 Compression Stats:');
  console.log(`• Before: ${result.stats.beforeTokens} tokens`);
  console.log(`• After: ${result.stats.afterTokens} tokens`);
  console.log(`• Saved: ${result.stats.saved} tokens (${result.stats.pct}%)`);
  
  console.log('\n🛡️ Protected spans:');
  result.protectedSpans.forEach(span => console.log(`  • "${span}"`));
});

console.log('\n✨ Compression test complete!');
console.log('\n💡 The proxy server applies this compression automatically to user messages when active.');
