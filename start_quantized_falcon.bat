@echo off
echo Starting Quantized Falcon 7B Server...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if model directory exists
if not exist "models\falcon-7b" (
    echo.
    echo WARNING: Falcon 7B model not found at models\falcon-7b\
    echo Please make sure the model is extracted correctly.
    echo.
    pause
)

echo.
echo 🔢 Starting QUANTIZED Falcon 7B Server
echo =====================================
echo.
echo Optimizations enabled:
echo  ✓ 8-bit/4-bit quantization (uses ~3-6GB VRAM instead of 13GB)
echo  ✓ GPU+CPU hybrid memory management  
echo  ✓ Automatic fallback strategies
echo  ✓ Memory cleanup after each request
echo.
echo Your system: RTX 3060 (12GB VRAM) + 16GB RAM = Perfect for quantization!
echo.
echo Loading strategies (tries in order):
echo  1. 8-bit quantization (~3-4GB VRAM)
echo  2. 4-bit quantization (~2-3GB VRAM) 
echo  3. GPU+CPU hybrid (8GB GPU + 12GB CPU)
echo  4. CPU-only fallback
echo.
echo Server will start immediately, model loads in background.
echo Check http://127.0.0.1:8081/health for loading status.
echo.
echo Press Ctrl+C to stop the server
echo.

python quantized_falcon_server.py --host 127.0.0.1 --port 8081

echo.
echo Server stopped. Press any key to exit.
pause
