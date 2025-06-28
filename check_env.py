import sys
import aiohttp
import os

print("--- Python 环境诊断报告 ---")
print(f"[*] 正在使用的Python解释器路径: {sys.executable}")
print("-" * 30)
print(f"[*] aiohttp 库的版本: {aiohttp.__version__}")
print(f"[*] aiohttp 库的安装位置: {aiohttp.__file__}")
print("-" * 30)

print("\n--- 结论 ---")
# 检查解释器路径是否在venv内
is_venv_python = 'venv' in sys.executable.lower() and 'maibot' in sys.executable.lower()
# 检查aiohttp版本是否为3.x或更高
is_aiohttp_new = int(aiohttp.__version__.split('.')[0]) >= 3

if is_venv_python and is_aiohttp_new:
    print("✅ 诊断通过！环境看起来是正确的。")
    print("   这意味着 bot.py 内部可能有非常复杂的机制（如子进程）导致了问题。")
else:
    print("❌ 诊断失败！检测到环境不匹配！")
    if not is_venv_python:
        print(f"   -> 原因: 当前脚本不是由您的 venv 环境中的 Python 运行的。")
        print(f"      它使用的是全局路径: {sys.executable}")
    if not is_aiohttp_new:
        print(f"   -> 原因: 脚本加载的 aiohttp 库版本 ({aiohttp.__version__}) 过旧。")
        print(f"      它加载的库位于: {aiohttp.__file__}")

print("\n--- Python 搜索路径 (sys.path) ---")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print("\n--- 报告结束 ---")