# Falcon 7B Integration Setup Guide

This guide will help you integrate your Falcon 7B model with the MinML application.

## Prerequisites

- Python 3.8 or higher
- At least 16GB RAM (32GB recommended)
- GPU with 8GB+ VRAM (optional but recommended)
- Your Falcon 7B model tar file

## Setup Steps

### 1. Extract Your Falcon 7B Model

1. Create a `models` directory in your MinML application folder:
   ```
   MinML-application/
   ├── models/
   │   └── falcon-7b/     <- Extract your tar file here
   ├── src/
   ├── package.json
   └── ...
   ```

2. Extract your Falcon 7B tar file to `models/falcon-7b/`

3. Verify the structure looks like:
   ```
   models/falcon-7b/
   ├── config.json
   ├── tokenizer.json
   ├── tokenizer_config.json
   ├── pytorch_model.bin (or multiple .bin files)
   ├── generation_config.json
   └── other model files...
   ```

### 2. Install Python Dependencies

#### Option A: Using the provided batch file (Windows)
1. Double-click `start_falcon_server.bat`
2. It will automatically create a virtual environment and install dependencies

#### Option B: Manual setup
1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Start the Falcon 7B Server

#### Option A: Using batch file (Windows)
```bash
start_falcon_server.bat
```

#### Option B: Manual start
```bash
# Activate virtual environment first
python falcon_server.py --host 127.0.0.1 --port 8081
```

The server will start on `http://127.0.0.1:8081`

### 4. Start MinML Application

1. In a separate terminal/command prompt, navigate to your MinML application directory

2. Start the Electron application:
   ```bash
   npm start
   ```

3. The application will automatically detect if the Falcon 7B server is running

## How It Works

### Compression Flow

1. **Primary**: MinML tries to use Falcon 7B for compression
   - Sends prompt to Python server at `http://127.0.0.1:8081/compress`
   - Falcon 7B processes the text using AI-based compression
   - Returns compressed text with statistics

2. **Fallback**: If Falcon 7B fails or is too slow, MinML falls back to algorithmic compression
   - Timeout: >60 seconds
   - Poor performance: <30% reduction
   - Server unavailable

### Performance Criteria

- **Timeout**: 60 seconds maximum
- **Minimum reduction**: 30% token reduction required
- **Fallback**: Automatic switch to algorithmic compression if criteria not met

## Monitoring

### Check Status
Visit `http://127.0.0.1:3123/status` to see:
- Proxy status
- Model availability
- Current configuration

### Logs
- **Python server**: Console output shows compression statistics
- **Electron app**: Check console for compression method used

## Troubleshooting

### Model Not Loading
1. Verify model files are in `models/falcon-7b/`
2. Check Python server logs for specific errors
3. Ensure you have enough RAM/VRAM

### Server Connection Failed
1. Ensure Python server is running on port 8081
2. Check firewall settings
3. Verify no other applications are using port 8081

### Poor Performance
1. **GPU acceleration**: Install CUDA-compatible PyTorch
2. **Memory optimization**: Consider using model quantization
3. **Hardware**: Ensure sufficient RAM/VRAM

### Fallback to Algorithmic
This is normal behavior when:
- Model server is starting up
- Processing takes >60 seconds
- Compression achieves <30% reduction
- Any errors occur with the model

## Configuration

You can modify the behavior by editing `src/main/hybridCompressor.ts`:

```typescript
// Change timeout (default: 60000ms)
this.falcon7bProvider = new Falcon7BProvider("http://127.0.0.1:8081", 30000, 30);

// Change minimum reduction percentage (default: 30%)
this.falcon7bProvider = new Falcon7BProvider("http://127.0.0.1:8081", 60000, 40);
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA is properly installed for GPU acceleration
2. **Model Quantization**: Consider using 8-bit quantization for lower memory usage
3. **Batch Size**: The server processes one request at a time for simplicity
4. **Memory Management**: Close other applications when running the model

## Testing

1. Start both servers (Python + Electron)
2. Check status: `http://127.0.0.1:3123/status`
3. Test with a sample API call to see compression in action
4. Monitor logs to see which compression method is used
