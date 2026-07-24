@echo off
chcp 65001 >nul
title LexAI Eval

echo ========================================
echo   LexAI Evaluation - All Modes
echo ========================================
echo.

set VENV_PYTHON=%~dp0..\.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual env python not found: %VENV_PYTHON%
    pause
    exit /b 1
)

echo Checking backend connection...
curl -s http://localhost:8006/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Backend not running on port 8006.
    echo   Start it first: .\.venv\Scripts\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8006 --reload
    pause
    exit /b 1
)
echo [OK] Backend connected
echo.

echo [1/2] Running Baseline mode (w/o Rerank)...
"%VENV_PYTHON%" -m eval.run_eval --mode baseline
echo.

echo [2/2] Running Rerank mode...
"%VENV_PYTHON%" -m eval.run_eval --mode rerank
echo.

echo ========================================
echo   Done!
echo   Results:
echo     eval\results\baseline_metrics.json
echo     eval\results\rerank_metrics.json
echo.
echo   Run comparison: eval\run_comparison.bat
echo ========================================
pause
