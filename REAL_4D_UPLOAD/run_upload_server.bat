@echo off
cd /d "%~dp0"
echo === Voxel Replay Upload Server ===
.venv\Scripts\python.exe reconstruction\upload_server.py
pause
