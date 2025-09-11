@echo off
echo Starting Falcon 7B Model Server...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if model directory exists
if not exist "models\falcon-7b" (
    echo.
    echo WARNING: Falcon 7B model not found at models\falcon-7b\
    echo Please extract your Falcon 7B tar file to the models\falcon-7b\ directory
    echo.
    echo Expected structure:
    echo   models\falcon-7b\
    echo     ├── config.json
    echo     ├── tokenizer.json
    echo     ├── pytorch_model.bin (or model files)
    echo     └── ...
    echo.
    echo Press any key to continue anyway (server will start but model won't load)
    pause
)

REM Start the server
echo.
echo Starting Falcon 7B server on http://127.0.0.1:8081
echo This may take 5-10 minutes to load the model...
echo Press Ctrl+C to stop the server
echo.
echo NOTE: If loading stops at 67%%, close this window and try again.
echo The model will fallback to CPU if GPU memory is insufficient.
echo.
python falcon_server.py --host 127.0.0.1 --port 8081

echo.
echo Server stopped. Press any key to exit.
pause
