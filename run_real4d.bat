@echo off
cd /d "%~dp0"
echo === Voxel Replay REAL 4D HUMAN ===
.venv\Scripts\python.exe reconstruction\real4d_pipeline.py
if errorlevel 1 pause & exit /b 1
.venv\Scripts\python.exe reconstruction\patch_player.py
echo.
echo DONE. Start player with:
echo cd player
echo npm run dev
pause
