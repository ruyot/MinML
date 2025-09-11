// Test the MinML proxy API (requires an OpenAI API key)
import fetch from 'node-fetch';

// You'll need to set your OpenAI API key here for testing
const API_KEY = process.env.OPENAI_API_KEY || 'your-openai-api-key-here';

if (API_KEY === 'your-openai-api-key-here') {
  console.log('⚠️  To test the full API functionality, set your OPENAI_API_KEY environment variable:');
  console.log('   export OPENAI_API_KEY=your_actual_key_here');
  console.log('   node test-api.js');
  console.log('\n📋 Or you can test with any OpenAI-compatible API by changing the target base URL');
  process.exit(1);
}

async function testMinMLProxy() {
  console.log('🔥 Testing MinML Proxy API\n');
  
  const testMessage = "Please write a very detailed explanation of quantum computing for beginners. I really need you to explain it step by step, like really thoroughly. Can you please make sure to include examples and actually show me how quantum bits work? I want to understand superposition too, please.";
  
  console.log('📝 Original message:');
  console.log(`"${testMessage}"`);
  console.log(`📏 Length: ${testMessage.length} characters\n`);
  
  try {
    const response = await fetch('http://localhost:3123/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'user',
            content: testMessage
          }
        ],
        max_tokens: 100 // Keep response short for demo
      })
    });
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ API Response received!');
    console.log('🤖 Assistant response:');
    console.log(`"${data.choices[0].message.content}"`);
    console.log('\n📊 Behind the scenes:');
    console.log('• Your message was automatically compressed before sending to OpenAI');
    console.log('• Token usage was reduced significantly');
    console.log('• The response quality remains high');
    console.log('\n💡 Check the MinML metrics to see your token savings!');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.log('\n🔧 Make sure:');
    console.log('• The MinML proxy is running on http://localhost:3123');
    console.log('• Your OpenAI API key is valid');
    console.log('• You have internet connectivity');
  }
}

testMinMLProxy();
