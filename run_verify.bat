@echo off
REM ===========================================================================
REM  run_verify.bat
REM  用途：运行 verify_env.py 环境自检，并清理搭建过程中产生的临时文件
REM  环境：conda env "koopman_robot"  ->  F:\conda_envs\koopman_robot  (Python 3.10)
REM ===========================================================================

REM 记录当前控制台代码页，稍后恢复
for /f "tokens=2 delims=:" %%a in ('chcp') do set "OLDPAGE=%%a"
chcp 65001 >nul

set "PY=F:\conda_envs\koopman_robot\python.exe"
set "SCRIPT=F:\control_eng\code\verify_env.py"
set "LOG=F:\control_eng\code\verify_env.log"

echo ================ RUN verify_env.py ================
if not exist "%PY%" (
  echo [ERROR] 找不到解释器: %PY%
  goto :end
)
"%PY%" "%SCRIPT%" > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
echo VERIFY_EXITCODE=%RC%

echo.
echo ================ CLEANUP TEMP FILES ================
del /q "F:\control_eng\code\_*.bat"  2>nul
del /q "F:\control_eng\code\_*.log"  2>nul
del /q "F:\control_eng\code\_*.txt"  2>nul
if exist "F:\control_eng\code\__pycache__" rd /s /q "F:\control_eng\code\__pycache__" 2>nul

echo ================ REMAINING FILES ================
dir /B "F:\control_eng\code"

:end
echo.
echo VERIFY_EXITCODE=%RC%
echo ================ DONE ================
chcp %OLDPAGE% >nul
