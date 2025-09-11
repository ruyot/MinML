@echo off
echo 🚀 MinML Compression Demo
echo ========================
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found!
    echo.
    echo Please create a .env file with your OpenAI API key:
    echo echo OPENAI_API_KEY=your_key_here ^> .env
    echo.
    pause
    exit /b 1
)

REM Check if MinML is running
echo 🔍 Checking MinML connection...
python -c "import requests; requests.get('http://localhost:3123/status', timeout=2)" 2>nul
if errorlevel 1 (
    echo ❌ MinML proxy not running!
    echo.
    echo Please start the MinML Electron app first:
    echo npm start
    echo.
    pause
    exit /b 1
)

echo ✅ MinML is running!
echo.
echo 🎬 Starting Interactive Demo...
echo.

REM Run the demo
python demo_compression.py

pause
