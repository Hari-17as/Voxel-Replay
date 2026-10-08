@echo off
cd /d "%~dp0"
echo === INSTALLING 3-VIDEO UPLOAD + 4D VIEWER ===
.venv\Scripts\python.exe reconstruction\patch_upload_ui.py
if errorlevel 1 pause & exit /b 1
echo.
echo DONE. Keep Vite running in the other CMD.
echo Start upload server with: run_upload_server.bat
pause
