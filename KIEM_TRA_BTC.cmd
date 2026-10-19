@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "ERRORLEVEL="
chcp 65001 >nul
set "PYTHONUTF8=1"
pushd "%~dp0"
if not "%ERRORLEVEL%"=="0" exit /b 2
echo MNIST 10-20 - RESULT VERIFICATION
echo.
if not exist "%~dp0run_verification.py" goto missing_script
set "VERIFY_MODE_ARGS="
if not "%~1"=="" goto select_runtime
echo [1] FAST     - Replay caches and compare the 5 submissions
echo [2] WEIGHTS  - Run 28 v4 models; v12 and validation use caches
echo [3] CHECK    - Check TRAIN input files only; no training or installs
echo [4] TRAIN    - Run legacy pipeline; needs CUDA and prepared inputs
echo [5] PREDICT  - Model-only argmax from 28 v4 weights; no prediction caches
echo [0] EXIT
echo.
choice /c 123450 /n /m "Choose 1, 2, 3, 4, 5 or 0: "
set "VERIFY_MENU_CHOICE=%ERRORLEVEL%"
if "%VERIFY_MENU_CHOICE%"=="1" goto menu_fast
if "%VERIFY_MENU_CHOICE%"=="2" goto menu_weights
if "%VERIFY_MENU_CHOICE%"=="3" goto menu_check_train
if "%VERIFY_MENU_CHOICE%"=="4" goto menu_train
if "%VERIFY_MENU_CHOICE%"=="5" goto menu_predict
if "%VERIFY_MENU_CHOICE%"=="6" goto menu_exit
set "VERIFY_EXIT=2"
goto finished
:menu_fast
set "VERIFY_MODE_ARGS=--mode fast"
goto select_runtime
:menu_weights
set "VERIFY_MODE_ARGS=--mode weights"
goto select_runtime
:menu_check_train
set "VERIFY_MODE_ARGS=--mode train --check-only"
goto select_runtime
:menu_train
set "VERIFY_MODE_ARGS=--mode train"
goto select_runtime
:menu_predict
set "VERIFY_MODE_ARGS=--mode predict"
goto select_runtime
:menu_exit
popd
exit /b 0
:select_runtime
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
"%~dp0.venv-verification\Scripts\python.exe" -u "%~dp0run_verification.py" --open-report %VERIFY_MODE_ARGS% %*
set "VERIFY_EXIT=%ERRORLEVEL%"
goto finished
:use_py
py -3 -u "%~dp0run_verification.py" --open-report %VERIFY_MODE_ARGS% %*
set "VERIFY_EXIT=%ERRORLEVEL%"
goto finished
:use_python
python -u "%~dp0run_verification.py" --open-report %VERIFY_MODE_ARGS% %*
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
