@echo off
title Axiom Trading - InvestPro Dashboard ^& Copilot
echo ========================================================
echo   Iniciando Axiom Dashboard + Copilot IA (RTX 4060)
echo ========================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Error: No se encontro el entorno virtual en .venv
    pause
    exit /b 1
)
echo Servidor activo en:
echo   Localhost: http://localhost:8000
echo   IP Local : http://127.0.0.1:8000
echo.
echo Presiona Ctrl+C para detener el servidor.
echo ========================================================
".venv\Scripts\python.exe" -m uvicorn api.server:app --host 0.0.0.0 --port 8000
pause
