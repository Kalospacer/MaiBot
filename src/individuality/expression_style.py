import asyncio
import json
import os
import random
import re
from typing import List, Tuple

from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.llm_models.utils_model import LLMRequest
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager

logger = get_logger("expressor_style")

def init_prompt() -> None:
    personality_expression_prompt = """
{personality}

请从以上人设中总结出这个角色可能的语言风格，你必须严格根据人设引申，不要输出例子
思考回复的特殊内容和情感
思考有没有特殊的梗，一并总结成语言风格
总结成如下格式的规律，总结的内容要详细，但具有概括性：
当"xxx"时，可以"xxx", xxx不超过10个字

例如（不要输出例子）：
当"表示十分惊叹"时，使用"我嘞个xxxx"
当"表示讽刺的赞同，不想讲道理"时，使用"对对对"
当"想说明某个观点，但懒得明说"，使用"懂的都懂"

现在请你概括
"""
    Prompt(personality_expression_prompt, "personality_expression_prompt")


class PersonalityExpression:
    def __init__(self):
        # --- FINAL FIX: Use the new, correct LLMRequest initialization ---
        try:
            # Get the complete model configuration block and convert it to a dictionary
            expressor_model_config = global_config.model.focus_expressor
            # Initialize LLMRequest with the correct parameter name 'model_config'
            self.express_learn_model: LLMRequest = LLMRequest(
                model_config=expressor_model_config,
                max_tokens=512,
                request_type="expressor.learner",
            )
        except Exception as e:
            logger.error(f"加载 [model.focus_expressor] 配置失败，表达学习功能将不可用: {e}")
            self.express_learn_model = None
        # --- FIX END ---

        self.meta_file_path = os.path.join("data", "expression", "personality", "expression_style_meta.json")
        self.expressions_file_path = os.path.join("data", "expression", "personality", "expressions.json")
        self.max_calculations = 20

    def _read_meta_data(self):
        if os.path.exists(self.meta_file_path):
            try:
                with open(self.meta_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"last_style_text": None, "count": 0}
        return {"last_style_text": None, "count": 0}

    def _write_meta_data(self, data):
        os.makedirs(os.path.dirname(self.meta_file_path), exist_ok=True)
        with open(self.meta_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def extract_and_store_personality_expressions(self):
        # Check if the model was initialized successfully
        if not self.express_learn_model:
            logger.warning("表达学习模型未初始化，跳过提取。")
            return

        os.makedirs(os.path.dirname(self.expressions_file_path), exist_ok=True)

        current_style_text = global_config.expression.expression_style
        meta_data = self._read_meta_data()

        last_style_text = meta_data.get("last_style_text")
        count = meta_data.get("count", 0)

        if current_style_text != last_style_text:
            logger.info("表达风格已更改，重置计数并删除旧文件。")
            count = 0
            if os.path.exists(self.expressions_file_path):
                try: os.remove(self.expressions_file_path)
                except OSError as e: logger.error(f"删除旧表达文件失败: {e}")

        if count >= self.max_calculations:
            # Update meta data even if we skip, to persist the current style text
            self._write_meta_data({"last_style_text": current_style_text, "count": count})
            return

        prompt = await global_prompt_manager.format_prompt("personality_expression_prompt", personality=current_style_text)
        
        try:
            # Unpack the response tuple correctly
            response_tuple = await self.express_learn_model.generate_response_async(prompt)
            if response_tuple and len(response_tuple) >= 2:
                response, _ = response_tuple
            else:
                logger.error("个性表达方式提取的响应格式不正确。")
                response = None
                
        except Exception as e:
            logger.error(f"个性表达方式提取失败: {e}")
            self._write_meta_data({"last_style_text": current_style_text, "count": count})
            return
        
        if not response:
             logger.warning("未能从LLM获取有效的表达方式响应。")
             return

        expressions = self.parse_expression_response(response, "personality")
        result = [{"situation": sit, "style": sty, "count": 100} for _, sit, sty in expressions]
        
        if len(result) > 50:
            result = random.sample(result, 50)
            
        with open(self.expressions_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"已写入{len(result)}条表达到{self.expressions_file_path}")

        count += 1
        self._write_meta_data({"last_style_text": current_style_text, "count": count})
        logger.info(f"成功处理。风格 '{current_style_text}' 的计数现在是 {count}。")

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