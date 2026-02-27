@echo off
REM OCTA Backend Startup Script for Windows
REM Activates virtual environment and starts FastAPI server

echo.
echo ╔════════════════════════════════════════════════╗
echo ║   OCTA Image Segmentation Backend Server        ║
echo ║   Starting with CPU mode (fixed configuration)  ║
echo ╚════════════════════════════════════════════════╝
echo.

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call ..\octa_env\Scripts\activate.bat

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo.

REM Start FastAPI server
echo [2/2] Starting FastAPI server...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Server startup failed
    pause
    exit /b 1
)

pause
