@echo off
cd /d %~dp0

set PY=.venv\Scripts\python.exe
set BACKEND_PORT=8006
set FRONTEND_PORT=8506

if not exist "%PY%" (
    echo [ERROR] .venv not found at: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)

echo ================================================
echo     JinYu Financial Intelligence
echo ================================================
echo.

REM ---- Kill old processes on our ports ----
echo [0/4] Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%BACKEND_PORT% " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%FRONTEND_PORT% " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo    Done.

REM ---- Start Backend ----
echo.
echo [1/4] Starting Backend on port %BACKEND_PORT%...
start "JinYu Backend" "%PY%" -m uvicorn backend.main:app --host 0.0.0.0 --port %BACKEND_PORT% --reload

REM ---- Wait for Backend to be ready ----
echo [2/4] Waiting for Backend to be ready...
set /a RETRY=0
:wait_backend
timeout /t 2 /nobreak >nul
set /a RETRY+=1
"%PY%" -c "import urllib.request; urllib.request.urlopen('http://localhost:%BACKEND_PORT%/docs', timeout=2)" >nul 2>&1
if errorlevel 1 (
    if %RETRY% lss 20 (
        echo    Waiting... ^(attempt %RETRY%/20^)
        goto wait_backend
    )
    echo [WARNING] Backend may not be ready after %RETRY% attempts.
    echo           Check the "JinYu Backend" window for errors.
) else (
    echo    Backend is ready!
)

REM ---- Start Frontend ----
echo.
echo [3/4] Starting Frontend on port %FRONTEND_PORT%...
start "JinYu Frontend" "%PY%" -m streamlit run frontend/app_jinyu.py --server.port %FRONTEND_PORT% --server.headless true --browser.gatherUsageStats false

REM ---- Wait for Frontend ----
echo [4/4] Waiting for Frontend...
timeout /t 5 /nobreak >nul

REM ---- Open Browser ----
start http://localhost:%FRONTEND_PORT%

echo.
echo ================================================
echo    Backend  : http://localhost:%BACKEND_PORT%
echo    Frontend : http://localhost:%FRONTEND_PORT%
echo    API Docs : http://localhost:%BACKEND_PORT%/docs
echo ================================================
echo.
echo All services started. You can close this window.
echo Check the "JinYu Backend" and "JinYu Frontend" windows for logs.
echo.
pause
