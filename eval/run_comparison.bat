@echo off
chcp 65001 >nul
title LexAI Comparison

echo ========================================
echo   LexAI A/B Comparison Report
echo ========================================
echo.

set VENV_PYTHON=%~dp0..\.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual env python not found: %VENV_PYTHON%
    pause
    exit /b 1
)

if not exist "%~dp0results\baseline_metrics.json" (
    echo [WARN] Baseline result not found. Run run_all_eval.bat first.
    pause
    exit /b 1
)
if not exist "%~dp0results\rerank_metrics.json" (
    echo [WARN] Rerank result not found. Run run_all_eval.bat first.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m eval.comparison
echo.
echo Report saved: eval\results\comparison_report.json
pause
