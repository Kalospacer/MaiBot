import asyncio
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from src.chat.utils.chat_message_builder import build_anonymous_messages, get_raw_msg_by_timestamp_random
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.llm_models.utils_model import LLMRequest

logger = get_logger("expressor_learner")
MAX_EXPRESSION_COUNT = 100

def init_prompt() -> None:
    learn_style_prompt = """
{chat_str}

请从上面这段群聊中概括除了人名为"SELF"之外的人的语言风格
1. 只考虑文字，不要考虑表情包和图片
2. 不要涉及具体的人名，只考虑语言风格
3. 语言风格包含特殊内容和情感
4. 思考有没有特殊的梗，一并总结成语言风格
5. 例子仅供参考，请严格根据群聊内容总结!!!
注意：总结成如下格式的规律，总结的内容要详细，但具有概括性：
当"xxx"时，可以"xxx", xxx不超过10个字

例如：
当"表示十分惊叹"时，使用"我嘞个xxxx"
当"表示讽刺的赞同，不想讲道理"时，使用"对对对"
当"想说明某个观点，但懒得明说"，使用"懂的都懂"

注意不要总结你自己（SELF）的发言
现在请你概括
"""
    Prompt(learn_style_prompt, "learn_style_prompt")

    learn_grammar_prompt = """
{chat_str}

请从上面这段群聊中概括除了人名为"SELF"之外的人的语法和句法特点，只考虑纯文字，不要考虑表情包和图片
1.不要总结【图片】，【动画表情】，[图片]，[动画表情]，不总结 表情符号 at @ 回复 和[回复]
2.不要涉及具体的人名，只考虑语法和句法特点,
3.语法和句法特点要包括，句子长短（具体字数），有何种语病，如何拆分句子。
4. 例子仅供参考，请严格根据群聊内容总结!!!
总结成如下格式的规律，总结的内容要简洁，不浮夸：
当"xxx"时，可以"xxx"

例如：
当"表达观点较复杂"时，使用"省略主语(3-6个字)"的句法
当"不用详细说明的一般表达"时，使用"非常简洁的句子"的句法
当"需要单纯简单的确认"时，使用"单字或几个字的肯定(1-2个字)"的句法

注意不要总结你自己（SELF）的发言
现在请你概括
"""
    Prompt(learn_grammar_prompt, "learn_grammar_prompt")


class ExpressionLearner:
    def __init__(self) -> None:
        # --- FINAL FIX: Use the new, correct LLMRequest initialization ---
        try:
            # Get the complete model configuration block and convert it to a dictionary
            expressor_model_config = global_config.model.focus_expressor
            # Initialize LLMRequest with the correct parameter name 'model_config'
            self.express_learn_model: LLMRequest = LLMRequest(
                model_config=expressor_model_config,
                temperature=0.1,
                max_tokens=256,
                request_type="expressor.learner",
            )
        except Exception as e:
            logger.error(f"加载 [model.focus_expressor] 配置失败，表达学习功能将不可用: {e}")
            self.express_learn_model = None
        # --- FIX END ---

    # ... (All other methods in ExpressionLearner remain unchanged) ...
    async def get_expression_by_chat_id(self, chat_id: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        learnt_style_file = os.path.join("data", "expression", "learnt_style", str(chat_id), "expressions.json")
        learnt_grammar_file = os.path.join("data", "expression", "learnt_grammar", str(chat_id), "expressions.json")
        personality_file = os.path.join("data", "expression", "personality", "expressions.json")
        
        def _load_json(file_path):
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except json.JSONDecodeError: pass
            return []

        return _load_json(learnt_style_file), _load_json(learnt_grammar_file), _load_json(personality_file)

    def is_similar(self, s1: str, s2: str) -> bool:
        if not s1 or not s2: return False
        min_len = min(len(s1), len(s2))
        if min_len < 5: return False
        return sum(1 for a, b in zip(s1, s2) if a == b) / min_len > 0.8

    async def learn_and_store_expression(self) -> Tuple[List, List]:
        learnt_style = await self.learn_and_store(type="style", num=15)
        learnt_grammar = await self.learn_and_store(type="grammar", num=15)
        return learnt_style or [], learnt_grammar or []

    async def learn_and_store(self, type: str, num: int = 10) -> Optional[List[Tuple[str, str, str]]]:
        if not self.express_learn_model: return None # Check if model was initialized
        
        type_str = "语言风格" if type == "style" else "句法特点"
        logger.info(f"开始学习{type_str}...")
        
        learnt_expressions = await self.learn_expression(type, num)
        if not learnt_expressions:
            logger.info(f"没有学习到新的{type_str}")
            return None

        logger.info(f"学习到{len(learnt_expressions)}条{type_str}")
        
        chat_dict: Dict[str, List[Dict[str, str]]] = {}
        for chat_id, situation, style in learnt_expressions:
            chat_dict.setdefault(chat_id, []).append({"situation": situation, "style": style})

        for chat_id, expr_list in chat_dict.items():
            dir_path = os.path.join("data", "expression", f"learnt_{type}", str(chat_id))
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, "expressions.json")
            
            old_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                except json.JSONDecodeError: pass

            if len(old_data) >= MAX_EXPRESSION_COUNT:
                old_data = [item for item in old_data if not (item.get("count", 1) == 1 and random.random() < 0.2)]

            for new_expr in expr_list:
                found = False
                for old_expr in old_data:
                    if self.is_similar(new_expr["situation"], old_expr.get("situation", "")) and self.is_similar(new_expr["style"], old_expr.get("style", "")):
                        if random.random() < 0.5:
                            old_expr.update(new_expr)
                        old_expr["count"] = old_expr.get("count", 1) + 1
                        found = True
                        break
                if not found:
                    new_expr["count"] = 1
                    old_data.append(new_expr)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(old_data, f, ensure_ascii=False, indent=2)
                
        return learnt_expressions

    async def learn_expression(self, type: str, num: int = 10) -> Optional[List[Tuple[str, str, str]]]:
        if type not in ["style", "grammar"]: raise ValueError(f"Invalid type: {type}")
        
        current_time = time.time()
        random_msg = get_raw_msg_by_timestamp_random(current_time - 3600 * 24, current_time, limit=num)
        if not random_msg: return None
        
        chat_id = random_msg[0]["chat_id"]
        random_msg_str = await build_anonymous_messages(random_msg)
        prompt_name = "learn_style_prompt" if type == "style" else "learn_grammar_prompt"
        prompt_str = await global_prompt_manager.format_prompt(prompt_name, chat_str=random_msg_str)
        
        try:
            response_tuple = await self.express_learn_model.generate_response_async(prompt_str)
            response = response_tuple[0] if response_tuple and len(response_tuple) > 0 else None
        except Exception as e:
            logger.error(f"学习{type}失败: {e}")
            return None
        
        if not response: return None
        return self.parse_expression_response(response, chat_id)

    def parse_expression_response(self, response: str, chat_id: str) -> List[Tuple[str, str, str]]:
        expressions: List[Tuple[str, str, str]] = []
        for line in response.splitlines():
            line = line.strip()
            if not line: continue
            match = re.search(r'当"(.*?)"时，.*?"(.*?)"', line)
            if match:
                situation, style = match.groups()
                expressions.append((chat_id, situation, style))
        return expressions


init_prompt()
expression_learner = ExpressionLearner()