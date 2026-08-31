@echo off
setlocal

title AudioBuddy Scan
pushd "%~dp0"
echo AudioBuddy is scanning for new audio files...
echo Keep this window open until the batch result appears.
echo.
".venv\Scripts\python.exe" "summarize_audio.py" --scan
set "audiobuddy_exit_code=%errorlevel%"

echo.
if not "%audiobuddy_exit_code%"=="0" (
    echo AudioBuddy finished with errors. Exit code: %audiobuddy_exit_code%
) else (
    echo AudioBuddy scan finished successfully.
)

pause
popd
exit /b %audiobuddy_exit_code%
