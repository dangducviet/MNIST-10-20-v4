@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "ERRORLEVEL="
chcp 65001 >nul
set "PYTHONUTF8=1"
pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 2
echo MNIST 10-20 - RESULT VERIFICATION
echo Default: cached validation and historical submission replay.
echo.
if not exist "%~dp0run_verification.py" goto missing_script
if not exist "%~dp0.venv-verification\Scripts\python.exe" goto find_python
"%~dp0.venv-verification\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info < (3,10))" >nul 2>&1
if "%ERRORLEVEL%"=="0" goto use_env
:find_python
python -c "import sys; sys.exit(sys.version_info < (3,10))" >nul 2>&1
if "%ERRORLEVEL%"=="0" goto use_python
py -3 -c "import sys; sys.exit(sys.version_info < (3,10))" >nul 2>&1
if "%ERRORLEVEL%"=="0" goto use_py
echo [ERROR] No working Python 3.10+ runtime was found.
echo Install Python from https://www.python.org/downloads/ then run this file again.
set "VERIFY_EXIT=2"
goto finished
:use_env
"%~dp0.venv-verification\Scripts\python.exe" -u "%~dp0run_verification.py" --open-report %*
set "VERIFY_EXIT=%ERRORLEVEL%"
goto finished
:use_py
py -3 -u "%~dp0run_verification.py" --open-report %*
set "VERIFY_EXIT=%ERRORLEVEL%"
goto finished
:use_python
python -u "%~dp0run_verification.py" --open-report %*
set "VERIFY_EXIT=%ERRORLEVEL%"
goto finished
:missing_script
echo [ERROR] Extract all ZIP contents next to the MNIST_data folder first.
set "VERIFY_EXIT=2"
:finished
echo.
echo Exit code: %VERIFY_EXIT%  (0=completed, 1=mismatch/incomplete, 2=error)
echo Report: verified_submissions\verification_report.html
echo.
pause
popd
exit /b %VERIFY_EXIT%
