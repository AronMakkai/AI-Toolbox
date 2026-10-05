@echo off
setlocal enabledelayedexpansion
:: AI-Toolbox launcher for Windows
::
:: The same philosophy as the Mac build: no PyInstaller freeze, just a
:: launcher that finds a Python already carrying this app's real
:: dependencies (torch, cv2, numpy, PIL) and runs ai_toolbox.py
:: directly from wherever this .bat file sits. Place this file in the
:: SAME folder as ai_toolbox.py.
::
:: Double-click to run, or:  AI-Toolbox.bat

set "DIR=%~dp0"
set "SCRIPT=%DIR%ai_toolbox.py"

if not exist "%SCRIPT%" (
    echo ERROR: ai_toolbox.py not found next to this file.
    echo Expected: %SCRIPT%
    pause
    exit /b 1
)

set "PYTHON="

:: Try, in order: an active conda env, then common install locations,
:: then whatever "python" resolves to on PATH -- checking each one
:: actually has the real dependencies installed before accepting it,
:: not just that a python.exe exists.
if defined CONDA_PREFIX (
    call :try_python "%CONDA_PREFIX%\python.exe"
)
if not defined PYTHON call :try_python "%USERPROFILE%\miniconda3\python.exe"
if not defined PYTHON call :try_python "%USERPROFILE%\anaconda3\python.exe"
if not defined PYTHON call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON call :try_python "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON call :try_python "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PYTHON (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON call :try_python "%%P"
    )
)

if not defined PYTHON (
    echo.
    echo AI-Toolbox could not find a Python install with torch,
    echo opencv-python, numpy and Pillow already installed.
    echo.
    echo Activate the conda environment this app was set up with,
    echo then run this file again from that same Command Prompt:
    echo.
    echo     conda activate ^<your-env^>
    echo     AI-Toolbox.bat
    echo.
    pause
    exit /b 1
)

"%PYTHON%" "%SCRIPT%"
exit /b %ERRORLEVEL%

:try_python
set "CANDIDATE=%~1"
if not exist "%CANDIDATE%" goto :eof
"%CANDIDATE%" -c "import torch, cv2, numpy, PIL" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON=%CANDIDATE%"
)
goto :eof
