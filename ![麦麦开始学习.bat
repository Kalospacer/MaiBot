@echo off
CHCP 65001 > nul
setlocal enabledelayedexpansion

echo 你需要选择启动方式，输入字母来选择:
echo   V = 不知道什么意思就输入 V
echo   C = 输入 C 使用 Conda 环境
echo.
choice /C CV /N /M "不知道什么意思就输入 V (C/V)?" /T 10 /D V

set "ENV_TYPE="
if %ERRORLEVEL% == 1 set "ENV_TYPE=CONDA"
if %ERRORLEVEL% == 2 set "ENV_TYPE=VENV"

if "%ENV_TYPE%" == "CONDA" goto activate_conda
if "%ENV_TYPE%" == "VENV" goto activate_venv

REM 如果 choice 超时或返回意外值，默认使用 venv
echo 警告: 无效选择或选择超时,默认使用 VENV.
set "ENV_TYPE=VENV"
goto activate_venv

:activate_conda
    set /p CONDA_ENV_NAME="请输入要使用的 Conda 环境名称: "
    if not defined CONDA_ENV_NAME (
        echo 错误: 未输入 Conda 环境名称.
        pause
        exit /b 1
    )
    echo 选择: Conda '!CONDA_ENV_NAME!'
    REM 激活Conda环境
    call conda activate !CONDA_ENV_NAME!
    if !ERRORLEVEL! neq 0 (
        echo 错误: Conda环境 '!CONDA_ENV_NAME!' 激活失败. 请确保Conda已安装并正确配置, 且 '!CONDA_ENV_NAME!' 环境存在.
        pause
        exit /b 1
    )
    goto env_activated

:activate_venv
    echo 已选择: venv (默认或已选择).
    REM 查找venv虚拟环境
    set "venv_path=%~dp0venv\Scripts\activate.bat"
    if not exist "%venv_path%" (
        echo 错误: 未找到 venv.请确保 venv 目录与脚本位于同一路径下.
        pause
        exit /b 1
    )
    REM 激活虚拟环境
    call "%venv_path%"
    if %ERRORLEVEL% neq 0 (
        echo 错误: 激活 venv 虚拟环境失败.
        pause
        exit /b 1
    )
    goto env_activated

:env_activated
echo 环境已成功激活！.

REM --- 后续脚本执行 ---

REM 运行预处理脚本
python "%~dp0scripts\raw_data_preprocessor.py"
if %ERRORLEVEL% neq 0 (
    echo 错误: raw_data_preprocessor.py 执行失败.
    pause
    exit /b 1
)

REM 运行信息提取脚本
python "%~dp0scripts\info_extraction.py"
if %ERRORLEVEL% neq 0 (
    echo 错误: info_extraction.py 执行失败.
    pause
    exit /b 1
)

REM 运行OpenIE导入脚本
python "%~dp0scripts\import_openie.py"
if %ERRORLEVEL% neq 0 (
    echo 错误: import_openie.py 执行失败.
    pause
    exit /b 1
)

echo 所有处理步骤已完成！
pause