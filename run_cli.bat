@echo off
setlocal
if "%~1"=="" goto usage

set "COMMAND=%~1"
shift

if /I "%COMMAND%"=="track" (
    py -3.12 "%~dp0analysis_tools\track_particles.py" %*
    goto end
)

if /I "%COMMAND%"=="stiffness" (
    py -3.12 "%~dp0analysis_tools\estimate_stiffness.py" %*
    goto end
)

:usage
echo Usage:
echo   run_cli.bat track ^<video^> [options]
echo   run_cli.bat stiffness ^<dat_file^> [options]
exit /b 1

:end
