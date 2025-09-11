# MinML

MinML is a local, model-agnostic token-reduction proxy with an Electron UI that helps reduce the token cost of your OpenAI API calls through intelligent compression.

## Features

- **Local Proxy**: Runs a local proxy server that mirrors OpenAI's `/v1/*` API
- **Smart Compression**: Heuristic compression pipeline with protected spans for quotes, numbers, and key instructions
- **Metrics Tracking**: Persistent tracking of tokens saved (total + daily) and compression percentages
- **Monochrome UI**: Clean, minimal black & white interface with a single toggle to activate compression
- **Cross-Platform**: Builds for macOS and Windows using Electron
- **API Compatible**: Drop-in replacement - just change your base URL to `http://localhost:3123/v1`

## Quick Start

### Installation

```bash
pnpm install
```

### Development

Run in development mode (two terminals):

```bash
# Terminal 1: Compile TypeScript and start Vite
pnpm dev

# Terminal 2: Launch Electron (once dist/ exists)
pnpm start
```

Or use the combined command:
```bash
pnpm dev  # Runs both TypeScript compilation and Vite dev server
```

Then in another terminal:
```bash
pnpm start  # Launch Electron
```

### Usage

1. Launch the MinML app
2. The proxy automatically starts on `http://localhost:3123`
3. In your application, change the OpenAI base URL:
   ```javascript
   // Instead of: https://api.openai.com/v1
   // Use: http://localhost:3123/v1
   ```
4. Keep your `Authorization: Bearer <YOUR_API_KEY>` header unchanged
5. Toggle compression on/off in the MinML UI
6. Monitor your token savings in real-time

### API Configuration

Point your OpenAI client to use MinML as a proxy:

```python
# Python example
import openai

client = openai.OpenAI(
    base_url="http://localhost:3123/v1",  # MinML proxy
    api_key="your-openai-api-key"
)
```

```javascript
// Node.js example
import OpenAI from 'openai';

const openai = new OpenAI({
    baseURL: 'http://localhost:3123/v1',  // MinML proxy
    apiKey: 'your-openai-api-key'
});
```

### Environment Variables

- `MINML_PORT`: Proxy server port (default: 3123)
- `MINML_TARGET_BASE`: Target API base URL (default: https://api.openai.com)

Example for using with other providers:
```bash
MINML_TARGET_BASE=https://api.anthropic.com pnpm start
```

## Building for Distribution

### macOS
```bash
pnpm build:mac
```
Produces `.dmg` and `.zip` files in the `dist/` directory.

### Windows
```bash
pnpm build:win
```
Produces `.exe` installer and `.zip` files in the `dist/` directory.

### Full Build
```bash
pnpm build
```
Compiles TypeScript, builds Vite, and runs electron-builder.

## How It Works

### Compression Algorithm

MinML uses a heuristic compression pipeline that:

1. **Protects Important Content**: Preserves quoted phrases, numbers, audience levels, and section headers
2. **Removes Filler Words**: Eliminates common stopwords and redundant phrases
3. **Deduplicates**: Removes repeated sentences
4. **Keyword Prioritization**: Keeps longer words and proper nouns using heuristics

### Protected Spans

The following content is automatically preserved during compression:
- Quoted phrases: `"important text"`
- Numbers and units: `42`, `3.14%`, `$100`
- Audience indicators: `beginner`, `expert`, `step by step`
- Section headers: `Format:`, `Constraints:`, `Style:`

### API Compatibility

- **Full Support**: `/v1/chat/completions` with compression
- **Pass-through**: All other `/v1/*` endpoints forwarded unchanged
- **Streaming**: Supports both regular and streaming responses
- **Headers**: Preserves all headers including authorization

## Architecture

```
src/
  main/                 # Electron main process
    main.ts            # App entry, window management, system tray
    proxy.ts           # Express server, OpenAI API proxy
    compressor.ts      # Token compression algorithms
    metrics.ts         # Persistent metrics storage
    modelProvider.ts   # Interface for future local models
    env.ts             # Environment configuration
    preload.ts         # Secure IPC bridge
  renderer/            # React frontend
    App.tsx            # Main UI component
    index.tsx          # React entry point
    index.html         # HTML template
    styles.css         # Monochrome styling
```

## Future Enhancements

The codebase is designed for easy extension:

- **Local Model Support**: `ModelProvider` interface ready for llama.cpp integration
- **Advanced Compression**: Slot-based extraction for more sophisticated compression
- **Provider Support**: Easy to add Anthropic, Cohere, or other API providers
- **Settings UI**: Modal for configuring target base, auto-start, etc.

## License

© 2024 MinML
