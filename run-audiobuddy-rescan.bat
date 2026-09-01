@echo off
setlocal

title AudioBuddy Rescan
pushd "%~dp0"
echo AudioBuddy is preparing a full rescan using the current prompt.md...
echo Review the selected count, then confirm only when you want to use the Gemini API.
echo.
".venv\Scripts\python.exe" "summarize_audio.py" --rescan
set "audiobuddy_exit_code=%errorlevel%"

echo.
if not "%audiobuddy_exit_code%"=="0" (
    echo AudioBuddy rescan finished with errors. Exit code: %audiobuddy_exit_code%
) else (
    echo AudioBuddy rescan finished successfully or was cancelled.
)

pause
popd
exit /b %audiobuddy_exit_code%
