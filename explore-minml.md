# 🧪 Exploring MinML - What You Can Try

## 1. 🧠 Test the Compression Engine
```bash
node test-compression.js
```
**What it shows:** Real-time compression with protected spans and token savings

## 2. 🔍 Examine the Source Code
Explore these key files to understand how MinML works:

### Core Algorithm
- `src/main/compressor.ts` - The heart of MinML's compression logic
- `src/main/metrics.ts` - Persistent token tracking and statistics
- `src/main/proxy.ts` - Express server that intercepts OpenAI API calls

### UI Components  
- `src/renderer/App.tsx` - React interface with the main toggle
- `src/renderer/styles.css` - Monochrome design system
- Open http://localhost:5174/ to see the UI (Vite dev server)

## 3. 📊 Analyze the Compression Strategy
Look at `compressor.ts` to see how MinML:
- Protects important content (quotes, numbers, technical terms)
- Removes filler words ("please", "really", "actually")
- Deduplicates repeated sentences
- Preserves context with smart keyword prioritization

## 4. 🔧 Modify and Experiment
Try editing these files and see changes in real-time:

### Compression Tweaks
- Add new protected patterns in `compressor.ts`
- Modify the stopwords list
- Adjust the compression aggressiveness

### UI Changes
- Edit `App.tsx` to add new features
- Modify `styles.css` for different themes
- Changes appear instantly at http://localhost:5174/

## 5. 🌐 Test the Full Proxy (Once Running)
When the proxy works, you can:
```javascript
// Point any OpenAI client to MinML
const openai = new OpenAI({
    baseURL: 'http://localhost:3123/v1',  // MinML proxy
    apiKey: 'your-openai-api-key'
});
```

## 6. 📱 Alternative Ways to Run
If Electron has issues:
- **Web Version**: Use the Vite dev server (http://localhost:5174/)
- **Node.js Only**: Run `node test-proxy.js` for backend-only mode
- **Docker**: Package as a web service
- **Manual Build**: Try different Electron installation methods

## 🎯 What Makes MinML Unique
- **Model Agnostic**: Works with OpenAI, Anthropic, or any compatible API
- **Intelligent Compression**: Not just removal, but smart preservation
- **Real-time Metrics**: Track savings across time
- **Zero Code Changes**: Drop-in replacement for existing OpenAI clients
- **Extensible**: Ready for local model integration (llama.cpp)

## 🔮 Future Enhancements Ready
- `modelProvider.ts` - Interface for local LLM integration  
- Slot-based extraction for more sophisticated compression
- Multi-provider support (Anthropic, Cohere, etc.)
- Advanced UI with settings and configuration
