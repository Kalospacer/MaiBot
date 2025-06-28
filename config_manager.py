import os
import shutil
import tomlkit  # 替换 tomli 和 tomli_w
from dotenv import dotenv_values
try:
    from src.common.logger_manager import get_logger
except ImportError:
    from loguru import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "bot_config.toml")
BACKUP_PATH = CONFIG_PATH + ".bak"
LPMM_CONFIG_PATH = os.path.join(BASE_DIR, "config", "lpmm_config.toml")
LPMM_BACKUP_PATH = LPMM_CONFIG_PATH + ".bak"

logger = get_logger("init")

def backup_config():
    try:
        if not os.path.exists(BACKUP_PATH) and os.path.exists(CONFIG_PATH):
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            shutil.copy(CONFIG_PATH, BACKUP_PATH)
            logger.info(f"已创建配置文件备份：{BACKUP_PATH}")
        else:
            logger.info("主配置备份文件已存在，跳过备份")
        # 新增lpmm_config.toml的备份
        if os.path.exists(LPMM_CONFIG_PATH) and not os.path.exists(LPMM_BACKUP_PATH):
            os.makedirs(os.path.dirname(LPMM_CONFIG_PATH), exist_ok=True)
            shutil.copy(LPMM_CONFIG_PATH, LPMM_BACKUP_PATH)
            logger.info(f"已创建lpmm配置文件备份：{LPMM_BACKUP_PATH}")
        elif os.path.exists(LPMM_BACKUP_PATH):
            logger.info("lpmm配置备份文件已存在，跳过备份")
    except Exception as e:
        logger.error(f"备份失败: {str(e)}")
        raise

def load_config():
    try:
        if not os.path.exists(CONFIG_PATH):
            logger.error(f"错误：找不到配置文件 {CONFIG_PATH}")
            raise FileNotFoundError(f"配置文件 {CONFIG_PATH} 未找到")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_config = tomlkit.load(f)
            return user_config
    except tomlkit.exceptions.TOMLKitError as e:
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
        raise Exception(error_message_zh) # 抛出汉化后的异常信息 #noqa
    except Exception as e:
        logger.error(f"读取配置失败: {str(e)}")
        raise

def save_config(config):
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:  # 修改打开模式和编码
            tomlkit.dump(config, f)  # 使用 tomlkit.dump
        logger.success("配置文件已保存")
    except Exception as e:
        logger.error(f"保存配置失败: {str(e)}")
        raise

def step_bot(config):
    print("\n=== 配置昵称 ===")
    try:
        nickname = input(f"请输入机器人的昵称（当前：{config['bot']['nickname']}）：").strip()
        if nickname.replace(" ", "") != "":
            config['bot']['nickname'] = nickname
    except Exception as e:
        logger.error(f"昵称配置异常: {str(e)}")
    return config

def step_groups(config):
    print("\n=== 配置可发消息群聊 ===")
    groups = []
    while True:
        try:
            group = input(f"请输入第{len(groups)+1}个群号（留空结束）：").strip()
            if not group or group.isspace():
                break
            if group.isdigit():
                groups.append(int(group))
            else:
                logger.warning("错误：请输入有效的数字群号！")
        except ValueError:
            logger.warning("错误：无效的数字格式！")
        except KeyboardInterrupt:
            print("\n输入已取消")
            break
        except Exception as e:
            logger.error(f"群聊配置异常: {str(e)}")
            break
    if groups:
        # 仅更新MaiBot-Napcat-Adapter的配置
        try:
            napcat_config_path = os.path.join(BASE_DIR, "MaiBot-Napcat-Adapter", "config.toml")
            if os.path.exists(napcat_config_path):
                with open(napcat_config_path, "r", encoding="utf-8") as f:
                    napcat_config = tomlkit.load(f)
                
                # 更新群组列表
                if 'Chat' not in napcat_config:
                    napcat_config['Chat'] = {}
                napcat_config['Chat']['group_list'] = groups
                
                with open(napcat_config_path, "w", encoding="utf-8") as f:
                    tomlkit.dump(napcat_config, f)
                logger.success("已配置群组到MaiBot-Napcat-Adapter")
            else:
                logger.warning(f"未找到MaiBot-Napcat-Adapter配置文件: {napcat_config_path}")
        except Exception as e:
            logger.error(f"配置MaiBot-Napcat-Adapter失败: {str(e)}")
    return config

def step_personality(config):
    print("\n=== 配置人格 ===")
    try:
        personality = config.get('personality', {})
        
        # 配置主人格描述
        current_core = personality.get('personality_core', '是一个积极向上的女大学生')
        core = input(f"请输入主人格描述（建议200字以内，当前：{current_core}）：").strip()
        if core:
            personality['personality_core'] = core
        else:
            personality['personality_core'] = current_core
            
        print("\n--- 配置人格细节 ---")
        print("请输入人格的详细描述，每条用一句话或几句话描述人格的一些细节")
        
        # 显示当前的人格细节
        current_sides = personality.get('personality_sides', [])
        if current_sides:
            print("当前人格细节：")
            for i, side in enumerate(current_sides, 1):
                print(f"  {i}. {side}")
        
        # 询问是否重新配置
        choice = input("是否重新配置人格细节？[是/否]（默认：否）：").strip().lower()
        if choice in ['y', 'yes', '是']:
            sides = []
            while True:
                try:
                    side = input(f"请输入第{len(sides)+1}条人格细节（留空结束）：").strip()
                    if not side or side.isspace():
                        break
                    sides.append(side)
                except KeyboardInterrupt:
                    print("\n输入已取消")
                    break
                except Exception as e:
                    logger.error(f"人格细节配置异常: {str(e)}")
                    break
            
            if sides:
                personality['personality_sides'] = sides
            else:
                logger.warning("未输入任何人格细节，保持原有配置")
        else:
            # 保持原有配置
            if not current_sides:
                personality['personality_sides'] = [
                    "用一句话或几句话描述人格的一些细节",
                    "用一句话或几句话描述人格的一些细节",
                    "用一句话或几句话描述人格的一些细节",
                ]
        
        config['personality'] = personality
    except Exception as e:
        logger.error(f"人格配置异常: {str(e)}")
    return config

def step_identity(config):
    print("\n=== 配置身份特征 ===")
    print("请输入身份特征，可以描述外貌、性别、身高、职业、属性等等")
    try:
        identity = config.get('identity', {})
        
        # 显示当前的身份特征
        current_details = identity.get('identity_detail', [])
        if current_details:
            print("当前身份特征：")
            for i, detail in enumerate(current_details, 1):
                print(f"  {i}. {detail}")
        
        # 询问是否重新配置
        choice = input("是否重新配置身份特征？[是/否]（默认：否）：").strip().lower()
        if choice in ['y', 'yes', '是']:
            details = []
            while True:
                try:
                    detail = input(f"请输入第{len(details)+1}条身份特征（留空结束）：").strip()
                    if not detail or detail.isspace():
                        break
                    details.append(detail)
                except KeyboardInterrupt:
                    print("\n输入已取消")
                    break
                except Exception as e:
                    logger.error(f"身份特征配置异常: {str(e)}")
                    break
            
            if details:
                identity['identity_detail'] = details
            else:
                logger.warning("未输入任何身份特征，保持原有配置")
        else:
            # 保持原有配置
            if not current_details:
                identity['identity_detail'] = [
                    "年龄为19岁",
                    "是女孩子", 
                    "身高为160cm",
                    "有橙色的短发",
                ]
        
        config['identity'] = identity
    except Exception as e:
        logger.error(f"身份特征配置异常: {str(e)}")
    return config

def step_expression(config):
    print("\n=== 配置语言风格 ===")
    try:
        # 配置表达方式
        expression = config.get('expression', {})
        
        print("\n--- 配置表达风格 ---")
        current_style = expression.get('expression_style', '描述麦麦说话的表达风格，表达习惯，例如：(回复尽量简短一些。可以参考贴吧，知乎和微博的回复风格，回复不要浮夸，不要用夸张修辞，平淡一些。不要有额外的符号，尽量简单简短)')
        style = input(f"请输入表达风格描述（当前：{current_style}）：").strip()
        if style:
            expression['expression_style'] = style
        else:
            expression['expression_style'] = current_style
            enable_learning = input(f"是否启用表达学习？[是/否]（当前：{'是' if expression.get('enable_expression_learning', False) else '否'}）：").strip().lower()
        if enable_learning in ['y', 'yes', '是']:
            expression['enable_expression_learning'] = True
            # 只有启用表达学习时才配置学习间隔
            interval = input(f"学习间隔（秒，当前：{expression.get('learning_interval', 600)}）：").strip()
            if interval.isdigit():
                expression['learning_interval'] = int(interval)
            elif interval:
                logger.warning("输入非法，已使用默认值。")
                expression['learning_interval'] = 600
            else:
                expression['learning_interval'] = expression.get('learning_interval', 600)
        elif enable_learning in ['n', 'no', '否']:
            expression['enable_expression_learning'] = False
        elif enable_learning:
            logger.warning("输入非法，已使用默认值。")
            
        config['expression'] = expression
        
        # 配置关系设置
        relationship = config.get('relationship', {})
        
        print("\n--- 配置关系设置 ---")
        give_name = input(f"麦麦是否给其他人取名？[是/否]（当前：{'是' if relationship.get('give_name', True) else '否'}）：").strip().lower()
        if give_name in ['y', 'yes', '是']:
            relationship['give_name'] = True
        elif give_name in ['n', 'no', '否']:
            relationship['give_name'] = False
        elif give_name:
            logger.warning("输入非法，已使用默认值。")
            
        config['relationship'] = relationship
        
    except Exception as e:
        logger.error(f"语言风格配置异常: {str(e)}")
    return config

def step_response(config):
    print("\n=== 配置聊天模式 ===")
    print("聊天模式说明：")
    print("1. normal（普通模式）：针对感兴趣的消息进行回复，token消耗量较低")
    print("2. focus（专注模式）：进行主动的观察和回复，token消耗量较高")
    print("3. auto（自动模式）：根据消息内容自动切换模式")
    try:
        chat = config.get('chat', {})
        
        # 配置聊天模式
        current_mode = chat.get('chat_mode', 'normal')
        mode = input(f"请选择聊天模式（normal/focus/auto，当前：{current_mode}）：").strip()
        if mode in ['normal', 'focus', 'auto']:
            chat['chat_mode'] = mode
        elif mode:
            logger.warning("输入非法，已使用默认值。")
        else:
            chat['chat_mode'] = current_mode
            
        # 如果选择了auto模式，配置阈值
        if chat.get('chat_mode') == 'auto':
            print("\n--- 配置自动切换阈值 ---")
            
            # 自动进入专注模式阈值
            current_auto_focus = chat.get('auto_focus_threshold', 1)
            auto_focus = input(f"自动切换到专注聊天的阈值（越低越容易进入，当前：{current_auto_focus}）：").strip()
            if auto_focus:
                try:
                    chat['auto_focus_threshold'] = float(auto_focus)
                except ValueError:
                    logger.warning("输入非法，已使用默认值。")
                    chat['auto_focus_threshold'] = current_auto_focus
            else:
                chat['auto_focus_threshold'] = current_auto_focus
                
            # 自动退出专注模式阈值
            current_exit_focus = chat.get('exit_focus_threshold', 1)
            exit_focus = input(f"自动退出专注聊天的阈值（越低越容易退出，当前：{current_exit_focus}）：").strip()
            if exit_focus:
                try:
                    chat['exit_focus_threshold'] = float(exit_focus)
                except ValueError:
                    logger.warning("输入非法，已使用默认值。")
                    chat['exit_focus_threshold'] = current_exit_focus
            else:
                chat['exit_focus_threshold'] = current_exit_focus
        
        config['chat'] = chat
    except Exception as e:
        logger.error(f"聊天模式配置异常: {str(e)}")
    return config


def step_emoji(config):
    print("\n=== 配置表情包 ===")
    try:
        emoji = config.get('emoji', {})
        
        # 表情包最大注册数量
        max_reg_num = input(f"表情包最大注册数量（当前：{emoji.get('max_reg_num', 40)}）：").strip()
        if max_reg_num.isdigit():
            emoji['max_reg_num'] = int(max_reg_num)
        elif max_reg_num:
            logger.warning("输入非法，已使用默认值。")
            
        # 是否替换表情包
        do_replace = input(f"达到最大数量时删除（替换）表情包？[是/否]（当前：{'是' if emoji.get('do_replace', True) else '否'}）：").strip().lower()
        if do_replace in ['y', 'yes', '是']:
            emoji['do_replace'] = True
        elif do_replace in ['n', 'no', '否']:
            emoji['do_replace'] = False
        elif do_replace:
            logger.warning("输入非法，已使用默认值。")
            
        # 检查表情包间隔
        check_interval = input(f"检查表情包间隔（分钟，当前：{emoji.get('check_interval', 120)}）：").strip()
        if check_interval.isdigit():
            emoji['check_interval'] = int(check_interval)
        elif check_interval:
            logger.warning("输入非法，已使用默认值。")
            
        # 是否保存图片
        save_pic = input(f"是否保存图片？[是/否]（当前：{'是' if emoji.get('save_pic', True) else '否'}）：").strip().lower()
        if save_pic in ['y', 'yes', '是']:
            emoji['save_pic'] = True
        elif save_pic in ['n', 'no', '否']:
            emoji['save_pic'] = False
        elif save_pic:
            logger.warning("输入非法，已使用默认值。")
            
        # 是否缓存表情包
        cache_emoji = input(f"是否缓存表情包？[是/否]（当前：{'是' if emoji.get('cache_emoji', True) else '否'}）：").strip().lower()
        if cache_emoji in ['y', 'yes', '是']:
            emoji['cache_emoji'] = True
        elif cache_emoji in ['n', 'no', '否']:
            emoji['cache_emoji'] = False
        elif cache_emoji:
            logger.warning("输入非法，已使用默认值。")
            
        # 是否偷取表情包
        steal_emoji = input(f"是否偷取表情包？[是/否]（当前：{'是' if emoji.get('steal_emoji', True) else '否'}）：").strip().lower()
        if steal_emoji in ['y', 'yes', '是']:
            emoji['steal_emoji'] = True
        elif steal_emoji in ['n', 'no', '否']:
            emoji['steal_emoji'] = False
        elif steal_emoji:
            logger.warning("输入非法，已使用默认值。")
            
        # 表情包内容过滤
        content_filtration = input(f"启用表情包内容过滤？[是/否]（当前：{'是' if emoji.get('content_filtration', False) else '否'}）：").strip().lower()
        if content_filtration in ['y', 'yes', '是']:
            emoji['content_filtration'] = True
            filtration_prompt = input(f"表情包过滤要求（当前：{emoji.get('filtration_prompt', '符合公序良俗')}）：").strip()
            if filtration_prompt:
                emoji['filtration_prompt'] = filtration_prompt
            else:
                emoji['filtration_prompt'] = '符合公序良俗'
        elif content_filtration in ['n', 'no', '否']:
            emoji['content_filtration'] = False
        elif content_filtration:
            logger.warning("输入非法，已使用默认值。")
            
        config['emoji'] = emoji
    except Exception as e:
        logger.error(f"表情包配置异常: {str(e)}")
    return config

def step_chinese_typo(config):
    print("\n=== 配置中文错别字生成器 ===")
    try:
        typo = config.get('chinese_typo', {})
        enable = input(f"启用中文错别字生成器？[是/否]（当前：{'是' if typo.get('enable', True) else '否'}）：").strip().lower()
        if enable in ['y', 'yes', '是']:
            typo['enable'] = True
        elif enable in ['n', 'no', '否']:
            typo['enable'] = False
        elif enable:
            logger.warning("输入非法，已使用默认值。")
        config['chinese_typo'] = typo
    except Exception as e:
        logger.error(f"中文错别字配置异常: {str(e)}")
    return config

def step_response_splitter(config):
    print("\n=== 配置回复分割器 ===")
    try:
        splitter = config.get('response_splitter', {})
        enable = input(f"启用回复分割器？[是/否]（当前：{'是' if splitter.get('enable_response_splitter', True) else '否'}）：").strip().lower()
        if enable in ['y', 'yes', '是']:
            splitter['enable_response_splitter'] = True
        elif enable in ['n', 'no', '否']:
            splitter['enable_response_splitter'] = False
        elif enable:
            logger.warning("输入非法，已使用默认值。")
        max_len = input(f"回复最大长度（当前：{splitter.get('response_max_length', 256)}）：").strip()
        if max_len.isdigit():
            splitter['response_max_length'] = int(max_len)
        elif max_len:
            logger.warning("输入非法，已使用默认值。")
        max_sent = input(f"回复最大句子数（当前：{splitter.get('response_max_sentence_num', 4)}）：").strip()
        if max_sent.isdigit():
            splitter['response_max_sentence_num'] = int(max_sent)
        elif max_sent:
            logger.warning("输入非法，已使用默认值。")
        kaomoji = input(f"启用颜文字保护？[是/否]（当前：{'是' if splitter.get('enable_kaomoji_protection', False) else '否'}）：").strip().lower()
        if kaomoji in ['y', 'yes', '是']:
            splitter['enable_kaomoji_protection'] = True
        elif kaomoji in ['n', 'no', '否']:
            splitter['enable_kaomoji_protection'] = False
        elif kaomoji:
            logger.warning("输入非法，已使用默认值。")
        config['response_splitter'] = splitter
    except Exception as e:
        logger.error(f"回复分割器配置异常: {str(e)}")
    return config

def step_experimental(config):
    print("\n=== 配置实验性功能 ===")
    print("实验性功能说明：")
    print("1. debug_show_chat_mode：是否在回复后显示当前聊天模式")
    print("2. enable_friend_chat：是否启用好友聊天，允许机器人在私聊中回复")
    try:
        experimental = config.get('experimental', {})
        
        # 配置调试显示聊天模式
        debug_show = input(f"是否在回复后显示当前聊天模式？[是/否]（当前：{'是' if experimental.get('debug_show_chat_mode', False) else '否'}）：").strip().lower()
        if debug_show in ['y', 'yes', '是']:
            experimental['debug_show_chat_mode'] = True
        elif debug_show in ['n', 'no', '否']:
            experimental['debug_show_chat_mode'] = False
        elif debug_show:
            logger.warning("输入非法，已使用默认值。")
            
        # 配置好友聊天
        friend_chat = input(f"是否启用好友聊天？[是/否]（当前：{'是' if experimental.get('enable_friend_chat', False) else '否'}）：").strip().lower()
        if friend_chat in ['y', 'yes', '是']:
            experimental['enable_friend_chat'] = True
        elif friend_chat in ['n', 'no', '否']:
            experimental['enable_friend_chat'] = False
        elif friend_chat:
            logger.warning("输入非法，已使用默认值。")
            
        config['experimental'] = experimental
    except Exception as e:
        logger.error(f"实验性功能配置异常: {str(e)}")
    return config

def step_info_extraction():
    print("\n=== 配置信息抽取线程数 ===")
    try:
        lpmm_data = {}
        if os.path.exists(LPMM_CONFIG_PATH):
            with open(LPMM_CONFIG_PATH, "r", encoding="utf-8") as f:
                lpmm_data = tomlkit.load(f)  # 使用 tomlkit.load
        info_extraction = lpmm_data.get("info_extraction", {})
        current_workers = info_extraction.get("workers", 10)
        value = input(f"实体提取最大线程数（当前：{current_workers}）：").strip()
        if value:
            if value.isdigit() and int(value) > 0:
                info_extraction["workers"] = int(value)
            else:
                logger.warning("输入非法，已使用默认值。")
                info_extraction["workers"] = 10
        else:
            info_extraction["workers"] = current_workers
        lpmm_data["info_extraction"] = info_extraction
        with open(LPMM_CONFIG_PATH, "w", encoding="utf-8") as f:
            tomlkit.dump(lpmm_data, f)  # 使用 tomlkit.dump
        logger.success(f"已保存info_extraction配置，workers={info_extraction['workers']}")
    except Exception as e:
        logger.error(f"写入lpmm_config.toml info_extraction配置失败: {str(e)}")

def step_api_keys(config):
    print("\n=== 配置API密钥 ===")
    try:
        env_path = os.path.join(BASE_DIR, ".env")
        current_env = dotenv_values(env_path)
        if os.path.exists(env_path):
            current_env = dotenv_values(env_path)
        current_key = current_env.get("SILICONFLOW_KEY", "")
        print("请前往 https://cloud.siliconflow.cn/account/ak 获取API秘钥")
        new_key = input(f"请输入SILICONFLOW_API密钥（当前：{current_key}，留空保持当前）：").strip()
        if new_key:
            os.environ["SILICONFLOW_KEY"] = new_key
            env_lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    env_lines = f.readlines()
            found = False
            for i, line in enumerate(env_lines):
                if line.startswith("SILICONFLOW_KEY="):
                    env_lines[i] = f"SILICONFLOW_KEY={new_key}\n"
                    found = True
                    break
            if not found:
                env_lines.append(f"\nSILICONFLOW_KEY={new_key}")
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(env_lines)
            # 同步写入lpmm_config.toml
            try:
                if os.path.exists(LPMM_CONFIG_PATH):
                    with open(LPMM_CONFIG_PATH, "r", encoding="utf-8") as f:
                        lpmm_data = tomlkit.load(f)  # 使用 tomlkit.load
                else:
                    lpmm_data = {}
                providers = lpmm_data.get("llm_providers", [])
                updated = False
                for provider in providers:
                    if provider.get("name") == "siliconflow":
                        provider["api_key"] = new_key
                        updated = True
                        break
                if not updated:
                    # 若没有则添加
                    providers.append({
                        "name": "siliconflow",
                        "base_url": "https://api.siliconflow.cn/v1/",
                        "api_key": new_key
                    })
                lpmm_data["llm_providers"] = providers
                with open(LPMM_CONFIG_PATH, "w", encoding="utf-8") as f:
                    tomlkit.dump(lpmm_data, f)  # 使用 tomlkit.dump
                logger.success("已同步API密钥到lpmm_config.toml")
            except Exception as e:
                logger.error(f"同步API密钥到lpmm_config.toml失败: {str(e)}")
    except Exception as e:
        logger.error(f"API密钥配置异常: {str(e)}")
    return config

def step_chat_details(config):
    print("\n=== 回复模式详细配置 ===")
    print("是否需要配置详细的回复模式参数？")
    print("（包括普通聊天和专注聊天的各项参数）")
    
    choice = input("是否进行详细配置？[是/否]（默认：否）：").strip().lower()
    if choice not in ['y', 'yes', '是']:
        print("跳过详细配置")
        return config
    
    try:
        # 配置普通聊天
        print("\n--- 配置普通聊天参数 ---")
        normal_chat = config.get('normal_chat', {})
        
        # 首要模型概率
        first_prob = input(f"首要模型选择概率（0-1，当前：{normal_chat.get('normal_chat_first_probability', 0.3)}）：").strip()
        if first_prob:
            try:
                normal_chat['normal_chat_first_probability'] = float(first_prob)
            except ValueError:
                logger.warning("输入非法，已使用默认值。")
        
        # 上下文长度
        context_size = input(f"上下文长度（当前：{normal_chat.get('max_context_size', 15)}）：").strip()
        if context_size.isdigit():
            normal_chat['max_context_size'] = int(context_size)
        elif context_size:
            logger.warning("输入非法，已使用默认值。")
        
        # 表情包概率
        emoji_chance = input(f"表情包使用概率（0-1，当前：{normal_chat.get('emoji_chance', 0.2)}）：").strip()
        if emoji_chance:
            try:
                normal_chat['emoji_chance'] = float(emoji_chance)
            except ValueError:
                logger.warning("输入非法，已使用默认值。")
        
        # 思考超时时间
        thinking_timeout = input(f"最长思考时间（秒，当前：{normal_chat.get('thinking_timeout', 120)}）：").strip()
        if thinking_timeout.isdigit():
            normal_chat['thinking_timeout'] = int(thinking_timeout)
        elif thinking_timeout:
            logger.warning("输入非法，已使用默认值。")
        
        # 回复意愿模式
        willing_mode = input(f"回复意愿模式（classical/mxp/custom，当前：{normal_chat.get('willing_mode', 'classical')}）：").strip()
        if willing_mode in ['classical', 'mxp', 'custom']:
            normal_chat['willing_mode'] = willing_mode
        elif willing_mode:
            logger.warning("输入非法，已使用默认值。")
        
        # 回复频率
        talk_freq = input(f"回复频率（当前：{normal_chat.get('talk_frequency', 1)}）：").strip()
        if talk_freq:
            try:
                normal_chat['talk_frequency'] = float(talk_freq)
            except ValueError:
                logger.warning("输入非法，已使用默认值。")
        
        # 回复意愿放大系数
        willing_amp = input(f"回复意愿放大系数（当前：{normal_chat.get('response_willing_amplifier', 1)}）：").strip()
        if willing_amp:
            try:
                normal_chat['response_willing_amplifier'] = float(willing_amp)
            except ValueError:
                logger.warning("输入非法，已使用默认值。")
        
        # 兴趣度放大系数
        interest_amp = input(f"回复兴趣度放大系数（当前：{normal_chat.get('response_interested_rate_amplifier', 1)}）：").strip()
        if interest_amp:
            try:
                normal_chat['response_interested_rate_amplifier'] = float(interest_amp)
            except ValueError:
                logger.warning("输入非法，已使用默认值。")
        
        # 表情包回复惩罚系数
        emoji_penalty = input(f"表情包回复惩罚系数（当前：{normal_chat.get('emoji_response_penalty', 0)}）：").strip()
        if emoji_penalty:
            try:
                normal_chat['emoji_response_penalty'] = float(emoji_penalty)
            except ValueError:
                logger.warning("输入非法，已使用默认值。")
        
        # 提及bot必然回复
        mentioned_reply = input(f"提及bot必然回复？[是/否]（当前：{'是' if normal_chat.get('mentioned_bot_inevitable_reply', True) else '否'}）：").strip().lower()
        if mentioned_reply in ['y', 'yes', '是']:
            normal_chat['mentioned_bot_inevitable_reply'] = True
        elif mentioned_reply in ['n', 'no', '否']:
            normal_chat['mentioned_bot_inevitable_reply'] = False
        elif mentioned_reply:
            logger.warning("输入非法，已使用默认值。")
        
        # @bot必然回复
        at_reply = input(f"@bot必然回复？[是/否]（当前：{'是' if normal_chat.get('at_bot_inevitable_reply', True) else '否'}）：").strip().lower()
        if at_reply in ['y', 'yes', '是']:
            normal_chat['at_bot_inevitable_reply'] = True
        elif at_reply in ['n', 'no', '否']:
            normal_chat['at_bot_inevitable_reply'] = False
        elif at_reply:
            logger.warning("输入非法，已使用默认值。")
        
        # 降低回复频率的群组系数
        down_freq_rate = input(f"降低回复频率群组系数（当前：{normal_chat.get('down_frequency_rate', 3)}）：").strip()
        if down_freq_rate:
            try:
                normal_chat['down_frequency_rate'] = float(down_freq_rate)
            except ValueError:
                logger.warning("输入非法，已使用默认值。")
        
        config['normal_chat'] = normal_chat
        
        # 配置专注聊天
        print("\n--- 配置专注聊天参数 ---")
        focus_chat = config.get('focus_chat', {})
        
        # 思考间隔
        think_interval = input(f"思考间隔（秒，当前：{focus_chat.get('think_interval', 3)}）：").strip()
        if think_interval.isdigit():
            focus_chat['think_interval'] = int(think_interval)
        elif think_interval:
            logger.warning("输入非法，已使用默认值。")
        
        # 连续回复能力
        consecutive = input(f"连续回复能力（当前：{focus_chat.get('consecutive_replies', 1)}）：").strip()
        if consecutive.isdigit():
            focus_chat['consecutive_replies'] = int(consecutive)
        elif consecutive:
            logger.warning("输入非法，已使用默认值。")
        
        # 并行处理
        parallel = input(f"是否并行处理回忆和处理器阶段？[是/否]（当前：{'是' if focus_chat.get('parallel_processing', True) else '否'}）：").strip().lower()
        if parallel in ['y', 'yes', '是']:
            focus_chat['parallel_processing'] = True
        elif parallel in ['n', 'no', '否']:
            focus_chat['parallel_processing'] = False
        elif parallel:
            logger.warning("输入非法，已使用默认值。")
        
        # 处理器最大时间
        proc_max_time = input(f"处理器最大时间（秒，当前：{focus_chat.get('processor_max_time', 25)}）：").strip()
        if proc_max_time.isdigit():
            focus_chat['processor_max_time'] = int(proc_max_time)
        elif proc_max_time:
            logger.warning("输入非法，已使用默认值。")
        
        # 观察上下文大小
        obs_context = input(f"观察上下文大小（当前：{focus_chat.get('observation_context_size', 16)}）：").strip()
        if obs_context.isdigit():
            focus_chat['observation_context_size'] = int(obs_context)
        elif obs_context:
            logger.warning("输入非法，已使用默认值。")
        
        # 压缩长度
        comp_length = input(f"上下文压缩长度（当前：{focus_chat.get('compressed_length', 8)}）：").strip()
        if comp_length.isdigit():
            focus_chat['compressed_length'] = int(comp_length)
        elif comp_length:
            logger.warning("输入非法，已使用默认值。")
        
        # 最多压缩份数
        comp_limit = input(f"最多压缩份数（当前：{focus_chat.get('compress_length_limit', 4)}）：").strip()
        if comp_limit.isdigit():
            focus_chat['compress_length_limit'] = int(comp_limit)
        elif comp_limit:
            logger.warning("输入非法，已使用默认值。")
        
        config['focus_chat'] = focus_chat
        
        # 配置专注聊天处理器
        print("\n--- 配置专注聊天处理器 ---")
        print("注意：打开处理器可以实现更多功能，但会增加token消耗")
        
        focus_processor = config.get('focus_chat_processor', {})
        
        # 自我识别处理器
        self_identify = input(f"启用自我识别处理器？[是/否]（当前：{'是' if focus_processor.get('self_identify_processor', True) else '否'}）：").strip().lower()
        if self_identify in ['y', 'yes', '是']:
            focus_processor['self_identify_processor'] = True
        elif self_identify in ['n', 'no', '否']:
            focus_processor['self_identify_processor'] = False
        elif self_identify:
            logger.warning("输入非法，已使用默认值。")
        
        # 工具使用处理器
        tool_use = input(f"启用工具使用处理器？[是/否]（当前：{'是' if focus_processor.get('tool_use_processor', False) else '否'}）：").strip().lower()
        if tool_use in ['y', 'yes', '是']:
            focus_processor['tool_use_processor'] = True
        elif tool_use in ['n', 'no', '否']:
            focus_processor['tool_use_processor'] = False
        elif tool_use:
            logger.warning("输入非法，已使用默认值。")
        
        # 工作记忆处理器
        working_memory = input(f"启用工作记忆处理器？（不稳定，消耗量大）[是/否]（当前：{'是' if focus_processor.get('working_memory_processor', False) else '否'}）：").strip().lower()
        if working_memory in ['y', 'yes', '是']:
            focus_processor['working_memory_processor'] = True
        elif working_memory in ['n', 'no', '否']:
            focus_processor['working_memory_processor'] = False
        elif working_memory:
            logger.warning("输入非法，已使用默认值。")
        
        config['focus_chat_processor'] = focus_processor
        
    except Exception as e:
        logger.error(f"回复模式详细配置异常: {str(e)}")
    return config

def main():
    print("---欢迎使用简易配置向导！---")
    print("请按照提示输入配置信息。")
    print("留空则使用默认值，直接回车跳过该步骤。")
    print("输入 Ctrl+C 取消配置。")
    print("------------------------")
    print("注意！！此配置向导只是简易版")
    print("想要配置更多请打开config/bot_config.toml进行手动编辑！")
    print("------------------------")
    print("制作 By MotricSeven")
    print("------------------------\n")

    try:
        backup_config()
        config = load_config()          
        steps = [
            step_bot,
            step_groups,
            step_personality,
            step_identity,
            step_expression,  # 新增
            step_response,
            step_chat_details,  # 新增：回复模式详细配置
            step_emoji,
            step_chinese_typo,
            step_response_splitter,
            step_experimental,
            step_info_extraction,  # 新增
            step_api_keys
        ]
        for step in steps:
            # step_info_extraction 不需要传参
            if step == step_info_extraction:
                step()
            else:
                config = step(config)
        save_config(config)
        print("\n配置已保存！")
    except Exception as e:
        logger.critical(f"配置流程异常终止: {str(e)}")

if __name__ == "__main__":
    main()