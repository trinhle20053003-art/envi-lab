@echo off
title EnVi Lab - Public Web Launcher
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_public_web.ps1"
pause
