@echo off
title Ceramic Defect Inspection System
echo ============================================
echo  Ceramic Defect Inspection - Startup
echo ============================================
echo.

cd /d "%~dp0"

echo [1/4] Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo       (waiting for Docker engine - up to 3 minutes)

set /a waited=0
:waitdocker
docker ps >nul 2>&1
if %errorlevel%==0 goto dockerready
timeout /t 10 /nobreak >nul
set /a waited+=10
if %waited% lss 180 goto waitdocker
echo       WARNING: Docker not ready yet - continuing anyway (app will use SQLite)
goto startapp

:dockerready
echo       Docker engine ready.

echo [2/4] Starting SQL Server container...
docker start screw-inspection-sql >nul 2>&1
if %errorlevel%==0 (
    echo       Container started - waiting for SQL Server to accept connections...
    timeout /t 25 /nobreak >nul
) else (
    echo       Container not found or already running - continuing.
)

echo [3/4] Setting environment...
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v GROQ_API_KEY 2^>nul') do set "GROQ_API_KEY=%%b"
if defined GROQ_API_KEY (echo       GROQ_API_KEY loaded.) else (echo       WARNING: GROQ_API_KEY not set - LLM explanations will use fallback template.)

:startapp
echo [4/4] Starting the inspection API...
echo.
echo  Dashboard will open at: http://127.0.0.1:8901
echo  API docs at:            http://127.0.0.1:8901/docs
echo.
echo  Keep this window OPEN. Close it to stop the system.
echo.
start "" http://127.0.0.1:8901
python -m uvicorn app.main:app --port 8901
pause
