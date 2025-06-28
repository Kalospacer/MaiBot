@echo off
CHCP 65001

echo 正在启动 MaiBot Core...
echo 正在启动主程序...
REM 检查虚拟环境是否存在且可用
if exist "venv" (
    echo 检测到现有虚拟环境，正在检查是否可用...
    call venv\Scripts\activate.bat
    python -c "import sys; exit(0) if sys.version_info[0] == 3 and sys.version_info[1] == 12 else exit(1)" && (
        echo 虚拟环境可用
        goto :start
    )
    echo 虚拟环境不可用，将创建新的虚拟环境
    rmdir /s /q venv
)

echo 正在创建虚拟环境...
"%PYTHON_PATH%" -m venv venv
call venv\Scripts\activate.bat

:start
python start.py
pause