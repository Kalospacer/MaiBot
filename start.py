import os
import subprocess
import tomlkit  # 替换 tomli
from typing import Optional
import re
from init_napcat import create_napcat_config, create_onebot_config
try:
    from src.common.logger_manager import get_logger
except ImportError:
    from loguru import logger

logger = get_logger("init")

def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def read_qq_from_config() -> Optional[str]:
    config_path = get_absolute_path('config/bot_config.toml')
    try:
        if not os.path.exists(config_path):
            logger.error(f"错误：找不到配置文件 {config_path}")
            return None
        with open(config_path, 'r', encoding='utf-8') as f:  # 修改为 'r' 和 utf-8 编码
            config = tomlkit.load(f)  # 使用 tomlkit.load
        if 'bot' not in config or 'qq_account' not in config['bot']:
            logger.error("错误：配置文件格式不正确，缺少 bot.qq_account 配置项")
            return None
        return str(config['bot']['qq_account'])  # 确保返回字符串
    except tomlkit.exceptions.TOMLKitError as e:  # 修改异常类型
        error_message = str(e)
        error_message_zh = f"配置文件解析失败: {error_message}"  # 默认错误信息

        line_num, col_num = None, None
        # 尝试从错误信息中提取行列号
        if " at line " in error_message and " col " in error_message:
            try:
                loc_part = error_message.split(" at line ")[-1]
                parts = loc_part.strip().split(" col ")
                line_num = parts[0].strip()
                if len(parts) > 1:
                    col_num = parts[1].strip().split()[0]  # 获取列号，忽略后续可能的文本
            except IndexError:
                pass # 解析行列号失败，保持为 None

        # 根据具体的错误类型生成汉化信息
        if "Unexpected character" in error_message and line_num and col_num:
            char_info = "未知"
            try:
                char_info = error_message.split("'")[1]
            except IndexError:
                pass
            error_message_zh = f"配置文件语法错误：在第 {line_num} 行，第 {col_num} 列遇到了意外的字符 '{char_info}'。"
        elif "Unclosed string" in error_message and line_num and col_num:
            error_message_zh = f"配置文件语法错误：在第 {line_num} 行，第 {col_num} 列存在未闭合的字符串。"
        elif "Expected a key" in error_message and line_num and col_num:
            error_message_zh = f"配置文件语法错误：在第 {line_num} 行，第 {col_num} 列期望一个键（key）。"
        elif "Duplicate key" in error_message: # 此错误类型通常不直接包含行列号
            key_name = "未知"
            try:
                key_name = error_message.split("'")[1]
            except IndexError:
                pass
            error_message_zh = f"配置文件错误：存在重复的键 '{key_name}'。"
            if line_num and col_num: # 如果错误信息中碰巧有行列号
                error_message_zh += f" (大致位置在第 {line_num} 行，第 {col_num} 列附近)"
        elif "Invalid escape sequence" in error_message and line_num and col_num:
            error_message_zh = f"配置文件语法错误：在第 {line_num} 行，第 {col_num} 列存在无效的转义序列。"
        elif "Expected newline or end of file" in error_message and line_num and col_num:
            error_message_zh = f"配置文件语法错误：在第 {line_num} 行，第 {col_num} 列处，期望换行或文件结束。"
        
        logger.error(error_message_zh)
        return None
    except Exception as e:
        logger.error(f"错误：读取配置文件时出现异常：{str(e)}")
        return None

def create_cmd_window(cwd: str, command: str, use_venv: bool = False) -> bool:
    try:
        if not os.path.exists(cwd):
            logger.error(f"错误：目录不存在 {cwd}")
            return False
            
        venv_activate = ''
        if use_venv:
            venv_path = get_absolute_path('venv/Scripts/activate')
            if not os.path.exists(venv_path):
                logger.error(f"错误：虚拟环境不存在 {venv_path}")
                return False
            venv_activate = f'call "{venv_path}" && '
        
        full_command = f'start cmd /k "cd /d "{cwd}" && {venv_activate}{command}"'
        subprocess.run(full_command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"错误：命令执行失败：{str(e)}")
        return False
    except Exception as e:
        logger.error(f"错误：启动进程时出现异常：{str(e)}")
        return False

def check_napcat() -> bool:
    napcat_path = get_absolute_path('napcat')
    napcat_exe = os.path.join(napcat_path, 'NapCatWinBootMain.exe')
    if not os.path.exists(napcat_exe):
        logger.error(f"错误：找不到NapCat可执行文件 {napcat_exe}")
        return False
    return True

def add_qq_number():
    config_path = get_absolute_path('config/bot_config.toml')
    try:
        while True:
            qq = input("请输入要添加/修改的QQ号：").strip()
            if not re.match(r'^\d+$', qq):
                logger.error("错误：QQ号必须为纯数字")
                continue

            # 更新主配置
            update_qq_in_config(config_path, qq)
            
            # 创建NapCat相关配置
            create_napcat_config(qq)
            create_onebot_config(qq)
            
            logger.success(f"QQ号 {qq} 配置已更新并创建必要文件！")
            return
    except Exception as e:
        logger.error(f"保存配置失败：{str(e)}")

def install_vc_redist():
    """静默安装VC运行库"""
    vc_path = get_absolute_path('onepackdata/vc_redist.x64.exe')
    if not os.path.exists(vc_path):
        logger.warning(f"警告：未找到VC运行库安装包 {vc_path}")
        return
    try:
        # /install /quiet /norestart 静默安装
        subprocess.run([vc_path, '/install', '/quiet', '/norestart'], check=True)
        logger.info("VC运行库已检测并安装（如已安装则自动跳过）")
    except subprocess.CalledProcessError:
        logger.warning("警告：VC运行库安装失败，可能已安装或权限不足")
        print(f"请手动运行以下文件进行安装：\n{vc_path}")
    except Exception as e:
        logger.warning(f"警告：VC运行库安装异常：{str(e)}")
        print(f"请手动运行以下文件进行安装：\n{vc_path}")

def launch_napcat(qq_number=None, headed_mode: bool = False):
    if not qq_number:
        qq_number = read_qq_from_config()
        if not qq_number:
            return False

    if headed_mode:
        napcat_dir = get_absolute_path('./napcatframework')
        napcat_exe_path = os.path.join(napcat_dir, 'NapCatWinBootMain.exe')
        if not os.path.exists(napcat_exe_path):
            logger.error(f"错误：找不到有头模式 NapCat 可执行文件 {napcat_exe_path}")
            return False
        cwd = napcat_dir
        command = f'CHCP 65001 & start http://127.0.0.1:6099/webui/web_login?token=napcat & NapCatWinBootMain.exe {qq_number}'
        logger.info(f"尝试以有头模式启动 NapCat (QQ: {qq_number})")
    else: # Headless mode (default)
        if not check_napcat(): # check_napcat 检查 napcat/NapCatWinBootMain.exe
            return False # check_napcat() 会记录错误
        cwd = get_absolute_path('napcat')
        command = f'CHCP 65001 & start http://127.0.0.1:6099/webui/web_login?token=napcat & NapCatWinBootMain.exe {qq_number}'
        logger.info(f"尝试以无头模式启动 NapCat (QQ: {qq_number})")

    return create_cmd_window(cwd, command)

def launch_adapter():
    adapter_path = get_absolute_path('MaiBot-Napcat-Adapter')
    return create_cmd_window(adapter_path, 'python main.py', use_venv=True)

def launch_main_bot():
    main_path = os.path.dirname(os.path.abspath(__file__))
    return create_cmd_window(main_path, 'python bot.py', use_venv=True)

def update_qq_in_config(config_path: str, qq_number: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            doc = tomlkit.parse(f.read())
        
        if 'bot' not in doc:
            doc['bot'] = tomlkit.table()  # 如果 bot 表不存在则创建
        
        doc['bot']['qq'] = qq_number  # 直接赋值，tomlkit 会处理类型
        
        with open(config_path, 'w', encoding='utf-8') as f:
            tomlkit.dump(doc, f)
            
    except Exception as e:
        logger.error(f"更新配置文件失败：{str(e)}")
        raise

def launch_config_manager():
    config_path = os.path.dirname(os.path.abspath(__file__))
    return create_cmd_window(config_path, 'python config_manager.py', use_venv=True)

def launch_venv_cmd():
    """启动一个激活了虚拟环境的CMD窗口"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return create_cmd_window(script_dir, "echo Virtual environment activated. You can now install packages or run scripts. Type 'exit' to close.", use_venv=True)

def launch_sqlite_studio():
    """启动SQLiteStudio可视化数据库管理工具"""
    sqlite_studio_path = get_absolute_path('SQLiteStudio/SQLiteStudio.exe')
    if not os.path.exists(sqlite_studio_path):
        logger.error(f"错误：找不到SQLiteStudio可执行文件 {sqlite_studio_path}")
        return False
    try:
        subprocess.Popen([sqlite_studio_path], cwd=get_absolute_path('SQLiteStudio'))
        logger.info("SQLiteStudio 已启动")
        return True
    except Exception as e:
        logger.error(f"错误：启动SQLiteStudio时出现异常：{str(e)}")
        return False

def delete_maibot_memory():
    """删除MaiBot的所有记忆（删除数据库文件）"""
    db_path = get_absolute_path('data/MaiBot.db')
    if not os.path.exists(db_path):
        logger.warning("数据库文件不存在，麦麦原本就没有记忆")
        return True
    
    try:
        # 确认删除
        confirm = input("⚠️  警告：此操作将删除麦麦的所有记忆，包括聊天记录、用户数据等，无法恢复！\n确定要继续吗？(输入 'YES' 确认): ").strip()
        if confirm != 'YES':
            logger.info("操作已取消")
            return False
        
        os.remove(db_path)
        logger.success("麦麦的所有记忆已删除成功！")
        return True
    except Exception as e:
        logger.error(f"错误：删除数据库文件时出现异常：{str(e)}")
        return False

def migrate_database_from_old_version():
    """从旧版本(0.6.x)迁移数据库到0.7.x版本"""
    migration_script = get_absolute_path('scripts/mongodb_to_sqlite.py')
    if not os.path.exists(migration_script):
        logger.error(f"错误：找不到迁移脚本 {migration_script}")
        return False
    
    try:
        logger.info("正在从旧版本迁移数据库...")
        logger.info("请在弹出的命令行窗口中查看迁移进度")
        return create_cmd_window(
            get_absolute_path('scripts'), 
            'python mongodb_to_sqlite.py', 
            use_venv=True
        )
    except Exception as e:
        logger.error(f"错误：启动数据库迁移时出现异常：{str(e)}")
        return False

def delete_knowledge_base():
    """删除麦麦的知识库"""
    rag_path = get_absolute_path('data/rag')
    embedding_path = get_absolute_path('data/embedding')
    
    # 检查是否存在知识库文件夹
    rag_exists = os.path.exists(rag_path)
    embedding_exists = os.path.exists(embedding_path)
    
    if not rag_exists and not embedding_exists:
        logger.warning("知识库原本就是空的，没有需要删除的内容")
        return True
    
    try:
        # 确认删除
        confirm = input("⚠️  警告：此操作将删除麦麦的所有知识库，包括RAG数据和向量数据，无法恢复！\n确定要继续吗？(输入 'YES' 确认): ").strip()
        if confirm != 'YES':
            logger.info("操作已取消")
            return False
        
        import shutil
        deleted_items = []
        
        if rag_exists:
            shutil.rmtree(rag_path)
            deleted_items.append("RAG数据")
        
        if embedding_exists:
            shutil.rmtree(embedding_path)
            deleted_items.append("向量数据")
        
        if deleted_items:
            logger.success(f"知识库删除成功！已删除：{', '.join(deleted_items)}")
        
        return True
    except Exception as e:
        logger.error(f"错误：删除知识库时出现异常：{str(e)}")
        return False

def import_openie_file():
    """导入其他人的OpenIE文件"""
    import_script = get_absolute_path('scripts/import_openie.py')
    if not os.path.exists(import_script):
        logger.error(f"错误：找不到导入脚本 {import_script}")
        return False
    
    try:
        logger.info("正在启动OpenIE文件导入工具...")
        logger.info("请在弹出的命令行窗口中按照提示选择要导入的文件")
        return create_cmd_window(
            get_absolute_path('scripts'), 
            'python import_openie.py', 
            use_venv=True
        )
    except Exception as e:
        logger.error(f"错误：启动OpenIE导入工具时出现异常：{str(e)}")
        return False

def start_maibot_learning():
    """麦麦开始学习（完整学习流程）"""
    scripts_dir = get_absolute_path('scripts')
    
    # 检查所需脚本是否存在
    required_scripts = [
        'raw_data_preprocessor.py',
        'info_extraction.py', 
        'import_openie.py'
    ]
    
    for script in required_scripts:
        script_path = os.path.join(scripts_dir, script)
        if not os.path.exists(script_path):
            logger.error(f"错误：找不到学习脚本 {script_path}")
            return False
    
    try:
        logger.info("开始麦麦学习流程...")
        logger.info("这将依次执行：数据预处理 → 信息提取 → OpenIE导入")
        
        # 构建批处理命令，依次执行三个脚本
        learning_command = (
            'python raw_data_preprocessor.py && '
            'python info_extraction.py && '
            'python import_openie.py && '
            'echo. && echo 🎉 麦麦学习流程已完成！ && pause'
        )
        
        logger.info("请在弹出的命令行窗口中查看学习进度")
        return create_cmd_window(scripts_dir, learning_command, use_venv=True)
        
    except Exception as e:
        logger.error(f"错误：启动麦麦学习流程时出现异常：{str(e)}")
        return False

def show_menu():
    print("\n=== MaiBot 控制菜单 ===")
    print("简易麦麦控制台 ")
    print("制作By MaiBot Team @MotricSeven")
    print("======================")
    print("1. 启动所有服务")
    print("2. 单独启动 NapCat")
    print("3. 单独启动 Adapter")
    print("4. 单独启动 麦麦主程序")
    print("5. 添加/修改QQ号")
    print("6. 麦麦基础配置")
    print("7. 安装VC运行库")
    print("8. 启动可视化数据库管理")
    print("9. 启动虚拟环境命令行")
    print("======================")
    print("数据管理功能：")
    print("10. 麦麦删除所有记忆（删库）")
    print("11. 从旧版(0.6.x)迁移数据库到0.7.x")
    print("12. 麦麦知识忘光光（删除知识库）")
    print("13. 导入其他人的OpenIE文件")
    print("14. 麦麦开始学习")
    print("======================")
    print("0. 退出程序")
    print("======================")
    return input("请输入选项：").strip()

def main():
    try:
        while True:
            choice = show_menu()

            if choice == '0':
                logger.info("程序已退出")
                break

            elif choice == '1':
                qq_number = read_qq_from_config()
                if not qq_number:
                    logger.error("请先配置QQ号（选项5）")
                    continue

                # 新增 NapCat 启动模式选择
                print("=== 选择 NapCat 启动模式 ===")
                print(" 1: 无头模式 (默认) : 只有命令行窗口，没有图形界面")
                print(" 2: 有头模式 : 带QQ电脑版图形界面")
                napcat_launch_choice = input("选择 NapCat 启动模式: ").strip()
                headed_mode_napcat = False
                if napcat_launch_choice == '2':
                    headed_mode_napcat = True
                    logger.info("NapCat 将以有头模式启动。")
                else:
                    if napcat_launch_choice not in ['1', '']: # 如果不是 '1' 也不是空 (默认)
                        logger.warning("无效的 NapCat 启动模式选择，将使用默认无头模式。")
                    logger.info("NapCat 将以无头模式启动。")
                
                if not all([
                    launch_napcat(qq_number, headed_mode=headed_mode_napcat),
                    launch_adapter(),
                    launch_main_bot()
                ]):
                    logger.error("部分服务启动失败")
                else:
                    logger.success("所有组件启动成功！")

            elif choice == '2':
                qq = read_qq_from_config()
                if qq:
                    # 新增 NapCat 启动模式选择
                    print("=== 选择 NapCat 启动模式 ===")
                    print(" 1: 无头模式 (默认) : 只有命令行窗口，没有图形界面")
                    print(" 2: 有头模式 : 带QQ电脑版图形界面")
                    napcat_launch_choice = input("选择 NapCat 启动模式: ").strip()
                    headed_mode_napcat = False
                    if napcat_launch_choice == '2':
                        headed_mode_napcat = True
                        logger.info("NapCat 将以有头模式启动。")
                    else:
                        if napcat_launch_choice not in ['1', '']: # 如果不是 '1' 也不是空 (默认)
                            logger.warning("无效的 NapCat 启动模式选择，将使用默认无头模式。")
                        logger.info("NapCat 将以无头模式启动。")
                    
                    logger.info("正在启动 NapCat..." + ("成功" if launch_napcat(qq, headed_mode=headed_mode_napcat) else "失败"))
                else:
                    logger.error("请先配置QQ号（选项5）")
                    
            elif choice == '3':
                logger.info("正在启动 Adapter..." + ("成功" if launch_adapter() else "失败"))
                
            elif choice == '4':
                logger.info("正在启动主程序..." + ("成功" if launch_main_bot() else "失败"))
                
            elif choice == '5':
                add_qq_number()
            elif choice == '6':
                logger.info("正在启动配置管理..." + ("成功" if launch_config_manager() else "失败"))
                
            elif choice == '7':
                install_vc_redist()
            elif choice == '8':
                logger.info("正在启动SQLiteStudio..." + ("成功" if launch_sqlite_studio() else "失败"))
                
            elif choice == '9':
                logger.info("正在启动虚拟环境命令行..." + ("成功" if launch_venv_cmd() else "失败"))
                
            elif choice == '10':
                logger.info("正在删除麦麦所有记忆..." + ("成功" if delete_maibot_memory() else "失败"))
                
            elif choice == '11':
                logger.info("正在启动数据库迁移..." + ("成功" if migrate_database_from_old_version() else "失败"))
            elif choice == '12':
                logger.info("正在删除麦麦知识库..." + ("成功" if delete_knowledge_base() else "失败"))
                
            elif choice == '13':
                logger.info("正在启动OpenIE文件导入工具..." + ("成功" if import_openie_file() else "失败"))
            elif choice == '14':
                logger.info("正在启动麦麦学习流程..." + ("成功" if start_maibot_learning() else "失败"))

            else:
                logger.error("无效选项，请重新输入")

    except KeyboardInterrupt:
        logger.info("\n程序已被用户中断")
        

if __name__ == '__main__':
    main()
