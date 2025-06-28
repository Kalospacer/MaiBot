import asyncio
import difflib
import json
from datetime import datetime
from typing import Dict, List, Optional

from json_repair import repair_json
from rich.traceback import install

from src.chat.heart_flow.observation.chatting_observation import ChattingObservation
from src.chat.heart_flow.observation.hfcloop_observation import HFCloopObservation
from src.chat.heart_flow.observation.structure_observation import StructureObservation
from src.chat.memory_system.Hippocampus import HippocampusManager
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.llm_models.utils_model import LLMRequest

install(extra_lines=3)
logger = get_logger("memory_activator")


def init_prompt():
    memory_activator_prompt = """
    你是一个记忆分析器，你需要根据以下信息来进行回忆
    以下是一场聊天中的信息，请根据这些信息，总结出几个关键词作为记忆回忆的触发词
    
    {obs_info_text}
    
    历史关键词（请避免重复提取这些关键词）：
    {cached_keywords}
    
    请输出一个json格式，包含以下字段：
    {{
        "keywords": ["关键词1", "关键词2", "关键词3",......]
    }}
    不要输出其他多余内容，只输出json格式就好
    """
    Prompt(memory_activator_prompt, "memory_activator_prompt")


class MemoryActivator:
    def __init__(self):
        try:
            summary_model_config = global_config.model.memory_summary
            model_config_for_llm = summary_model_config.copy()
            if 'temperature' not in model_config_for_llm:
                model_config_for_llm['temperature'] = 0.7
            if 'max_tokens' not in model_config_for_llm:
                model_config_for_llm['max_tokens'] = 50
            model_config_for_llm['request_type'] = "focus.memory_activator"
            
            self.summary_model = LLMRequest(model_config=model_config_for_llm)
        except Exception as e:
            logger.error(f"加载 [model.memory_summary] 配置失败，记忆激活功能将不可用: {e}")
            self.summary_model = None

        self.running_memory: List[Dict] = []
        self.cached_keywords: set = set()

    async def activate_memory(self, observations: List) -> List[Dict]:
        if not self.summary_model:
            logger.warning("记忆总结模型未初始化，跳过记忆激活。")
            return self.running_memory

        obs_info_text = ""
        for obs in observations:
            if hasattr(obs, 'get_observe_info'):
                info = obs.get_observe_info()
                if isinstance(info, list):
                    obs_info_text += "\n".join(f"{item.get('type', '')}: {item.get('content', '')}" for item in info) + "\n"
                elif isinstance(info, str):
                    obs_info_text += info + "\n"

        cached_keywords_str = ", ".join(self.cached_keywords) if self.cached_keywords else "暂无历史关键词"
        prompt = await global_prompt_manager.format_prompt(
            "memory_activator_prompt",
            obs_info_text=obs_info_text,
            cached_keywords=cached_keywords_str,
        )

        # generate_response returns (content, reasoning, tool_calls, model_name)
        # content could be str or Dict or None
        response_tuple = await self.summary_model.generate_response(prompt)
        
        keywords_data: Optional[Dict] = None
        # <<< 关键修正：安全处理 response_tuple[0] 可能为 None 的情况 >>>
        if response_tuple and response_tuple[0] is not None:
            raw_llm_content = response_tuple[0]
            
            if isinstance(raw_llm_content, dict):
                keywords_data = raw_llm_content
            elif isinstance(raw_llm_content, str):
                try:
                    fixed_json = repair_json(raw_llm_content)
                    keywords_data = json.loads(fixed_json) if isinstance(fixed_json, str) else fixed_json
                except Exception as e:
                    logger.error(f"memory_activator: 从LLM响应字符串解析JSON失败: {e}. 原始内容: {raw_llm_content[:200]}")
                    keywords_data = None
            else:
                logger.warning(f"memory_activator: LLM响应内容类型非预期: {type(raw_llm_content)}")
                keywords_data = None
        else:
            # 如果 response_tuple 为空 (LLM调用失败) 或 response_tuple[0] 为 None (结构化返回空)
            logger.warning("记忆激活时，LLM未能生成关键词或返回了空内容。")
            keywords_data = None

        keywords = keywords_data.get("keywords", []) if keywords_data else []


        if keywords:
            self.cached_keywords.update(keywords)
            if len(self.cached_keywords) > 20: # Limit cache size
                self.cached_keywords = set(list(self.cached_keywords)[-15:])
            logger.info(f"提取的关键词: {', '.join(keywords)}")

        related_memory = await HippocampusManager.get_instance().get_memory_from_topic(
            valid_keywords=keywords, max_memory_num=3, max_memory_length=2, max_depth=3
        )

        # Update running memory duration
        for m in self.running_memory:
            m["duration"] = m.get("duration", 1) + 1
        self.running_memory = [m for m in self.running_memory if m["duration"] < 3]

        if related_memory:
            for topic, memory in related_memory:
                if not any(m["topic"] == topic or difflib.SequenceMatcher(None, m["content"], memory).ratio() >= 0.7 for m in self.running_memory):
                    self.running_memory.append({"topic": topic, "content": memory, "timestamp": datetime.now().isoformat(), "duration": 1})

        self.running_memory = self.running_memory[-3:] # Keep only the last 3 memories
        return self.running_memory

init_prompt()