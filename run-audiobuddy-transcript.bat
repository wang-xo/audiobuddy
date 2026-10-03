@echo off
setlocal

title AudioBuddy Transcript
pushd "%~dp0"
set "input_dir=D:\Documents\Personal\audio\in"

echo AudioBuddy Transcript
echo.
set /p "audio_file=Enter the audio filename from the in folder: "

if "%audio_file%"=="" (
    echo.
    echo No filename was provided. Nothing was processed.
    set "audiobuddy_exit_code=1"
    goto :finish
)

set "audio_path=%input_dir%\%audio_file%"
if not exist "%audio_path%" (
    echo.
    echo Audio file was not found: "%audio_path%"
    set "audiobuddy_exit_code=1"
    goto :finish
)

echo.
echo Creating a transcript for "%audio_file%"...
echo Keep this window open until the final result appears.
echo.
".venv\Scripts\python.exe" "transcribe_audio.py" "%audio_path%"
set "audiobuddy_exit_code=%errorlevel%"

:finish
echo.
if not "%audiobuddy_exit_code%"=="0" (
    echo AudioBuddy transcript finished with errors. Exit code: %audiobuddy_exit_code%
) else (
    echo AudioBuddy transcript finished successfully.
)

pause
popd
exit /b %audiobuddy_exit_code%
