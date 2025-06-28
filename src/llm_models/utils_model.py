# 文件: src/llm_models/utils_model.py

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Tuple, Union, Dict, Any, Optional
import aiohttp
from aiohttp.client import ClientResponse
from src.common.logger import get_module_logger
from src.common.tcp_connector import get_tcp_connector
import base64
from PIL import Image
import io
import os
from src.common.database.database import db
from src.common.database.database_model import LLMUsage
from src.config.config import global_config
from rich.traceback import install

install(extra_lines=3)
logger = get_module_logger("model_utils")

# --- 全局LLM请求追踪字典和打印函数保持不变 ---
_ongoing_llm_requests: Dict[str, Dict[str, Any]] = {}

def log_ongoing_llm_requests():
    if not _ongoing_llm_requests:
        logger.info("当前没有正在等待的LLM请求。")
        return

    log_lines = ["--- 正在等待的LLM请求 (实时计时) ---"]
    current_time = time.monotonic()
    for req_id, req_info in _ongoing_llm_requests.items():
        elapsed_ms = (current_time - req_info['start_time']) * 1000
        log_lines.append(
            f"  请求ID: {req_id}, 模型: {req_info['model_name']}, 类型: {req_info['request_type']}, 已耗时: {elapsed_ms:.2f}ms"
        )
    log_lines.append("--- LLM请求等待状态结束 ---")
    logger.info("\n".join(log_lines))


# --- Custom Exception Classes (No Change) ---
class PayLoadTooLargeError(Exception):
    def __init__(self, message: str): super().__init__(message); self.message = message
class RequestAbortException(Exception):
    def __init__(self, message: str, response: ClientResponse): super().__init__(message); self.message, self.response = message, response
class PermissionDeniedException(Exception):
    def __init__(self, message: str): super().__init__(message); self.message = message

# --- Error Code Mapping (No Change) ---
error_code_mapping = {
    400: "参数不正确", 401: "API key 错误", 402: "账号余额不足", 403: "需要实名,或余额不足",
    404: "Not Found", 429: "请求过于频繁", 500: "服务器内部故障", 503: "服务器负载过高",
}

def extract_json_from_text(text: str) -> Optional[dict]:
    if not isinstance(text, str): return None
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if match: text = match.group(1)
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start: return json.loads(text[start:end+1])
    except (json.JSONDecodeError, TypeError): pass
    logger.warning("在文本中找不到有效的JSON对象。")
    return None

class LLMRequest:
    MODELS_NEEDING_TRANSFORMATION = ["o1", "o1-mini", "o1-preview", "o1-pro", "o3", "o3-mini", "o4-mini"]

    def __init__(self, model_config: dict, **kwargs):
        try:
            self.provider = model_config['provider']
            self.api_key = os.environ[f"{self.provider}_KEY"]
            self.base_url = os.environ[f"{self.provider}_BASE_URL"]
        except Exception as e:
            raise ValueError(f"为 provider '{model_config.get('provider')}' 加载配置时出错: {e}") from e
        self.model_name: str = model_config["name"]
        self.fallback_model_name: Optional[str] = model_config.get("fallback_model")
        # 确保 request_type 从 model_config 或 kwargs 中获取
        self.request_type = model_config.pop("request_type", kwargs.pop("request_type", "default")) # 优先从 model_config 取
        self.params = kwargs # 剩余的 kwargs
        
        self.stream = model_config.get("stream", False)
        self.pri_in, self.pri_out = model_config.get("pri_in", 0), model_config.get("pri_out", 0)
        self.enable_thinking = model_config.get("enable_thinking", False)
        self.temp = model_config.get("temp", 0.7)
        self.thinking_budget = model_config.get("thinking_budget", 4096)
        self._init_database()

    # --- _init_database, _record_usage, _calculate_cost are unchanged ---
    @staticmethod
    def _init_database(): db.create_tables([LLMUsage], safe=True)
    
    def _record_usage(self, p_tokens, c_tokens, t_tokens, user_id="system", request_type=None, endpoint="/chat/completions"):
        if request_type is None: request_type = self.request_type
        LLMUsage.create(model_name=self.model_name, user_id=user_id, request_type=request_type, endpoint=endpoint, prompt_tokens=p_tokens, completion_tokens=c_tokens, total_tokens=t_tokens, cost=self._calculate_cost(p_tokens, c_tokens), status="success", timestamp=datetime.now())
    
    def _calculate_cost(self, p_tokens, c_tokens): return round(((p_tokens / 1e6) * self.pri_in) + ((c_tokens / 1e6) * self.pri_out), 6)

    async def _prepare_request(self, requester: 'LLMRequest', endpoint: str, **kwargs) -> Dict[str, Any]:
        # policy={"max_retries": 2, ...} is the default for *any* request if not overridden by kwargs
        policy = {"max_retries": 2, "base_wait": 5, "retry_codes": [429, 500, 503], "abort_codes": [400, 401, 402, 403, 413]}
        if "retry_policy" in kwargs: policy.update(kwargs["retry_policy"]) # Apply kwargs policy
        api_url = f"{requester.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        payload = kwargs.get("payload") or await requester._build_payload(kwargs.get("prompt"), kwargs.get("image_base64"), kwargs.get("image_format"))
        if requester.stream: payload["stream"] = True
        # Ensure model name in payload matches the current requester
        payload["model"] = requester.model_name
        return {"policy": policy, "api_url": api_url, "payload": payload, **kwargs}

    async def _execute_request_internal(self, requester: 'LLMRequest', **kwargs) -> Optional[Dict]:
        request_content = await self._prepare_request(requester, **kwargs)
        # Generate a unique request ID for tracking
        request_id = f"{requester.model_name}-{requester.request_type}-{time.time_ns()}"
        
        # Add to ongoing requests at the very beginning
        _ongoing_llm_requests[request_id] = {
            "model_name": requester.model_name,
            "request_type": requester.request_type,
            "start_time": time.monotonic()
        }
        
        # try:
        #     final_payload_json = json.dumps(request_content["payload"], ensure_ascii=False, indent=2)
        #     logger.info(f"--- [PROMPT CAPTURE] ---\n发送给模型 '{requester.model_name}' (Provider: {requester.provider}, Type: {requester.request_type}, ReqID: {request_id}) 的最终Payload:\n{final_payload_json}\n--- [END PROMPT CAPTURE] ---")
        # except Exception as log_e:
        #     logger.error(f"无法序列化并记录请求Payload (ReqID: {request_id}): {log_e}")
            
        final_result = None # Track result to ensure it's removed from _ongoing_llm_requests
        for attempt in range(request_content["policy"]["max_retries"]):
            request_start_time = time.monotonic()
            try:
                headers = await requester._build_headers()
                if requester.stream: headers["Accept"] = "text/event-stream"
                async with aiohttp.ClientSession(connector=await get_tcp_connector()) as session:
                    async with session.post(request_content["api_url"], headers=headers, json=request_content["payload"], timeout=120) as response:
                        request_end_time = time.monotonic()
                        duration = (request_end_time - request_start_time) * 1000
                        logger.info(f"LLM请求 '{requester.request_type}' (模型: {requester.model_name}, Provider: {requester.provider}, ReqID: {request_id}) 完成! 耗时: {duration:.2f}ms (尝试 {attempt + 1})")
                        final_result = await requester._handle_response(response, request_content)
                        break # Exit loop on success
            except Exception as e:
                request_end_time = time.monotonic()
                duration = (request_end_time - request_start_time) * 1000
                if attempt >= request_content["policy"]["max_retries"] - 1:
                    logger.error(f"LLM请求 '{requester.request_type}' (模型: {requester.model_name}, ReqID: {request_id}) 已达到最大重试次数，最终失败! 耗时: {duration:.2f}ms. 错误: {e}")
                    final_result = None # Ensure final_result is None on ultimate failure
                else:
                    wait_time = request_content["policy"]["base_wait"] * (2 ** attempt)
                    logger.warning(f"LLM请求 '{requester.request_type}' (模型: {requester.model_name}, ReqID: {request_id}) 失败 (尝试 {attempt + 1}), 等待 {wait_time}s... 耗时: {duration:.2f}ms. 错误: {e}")
                    await asyncio.sleep(wait_time)
        
        # Remove request from ongoing tracking after it's completed or failed all retries
        if request_id in _ongoing_llm_requests: # Check existence before popping
            del _ongoing_llm_requests[request_id]
            logger.debug(f"LLM请求 {request_id} 已从追踪中移除。")

        return final_result

    async def _execute_request(self, **kwargs) -> Optional[Dict]:
        """
        Executes the request with the primary model. If it fails and a fallback is configured,
        it immediately retries with the fallback model. The fallback model then performs its own retries.
        This is the final, architecturally correct version for the new behavior.
        """
        # Step 1: Primary model: Attempt once (no retries here)
        logger.debug(f"🚀 正在尝试主引擎: {self.model_name} (Provider: {self.provider})")
        
        # Create a temporary policy for primary model's single attempt at this level
        primary_kwargs = kwargs.copy()
        temp_policy = primary_kwargs.get("retry_policy", {}).copy()
        temp_policy["max_retries"] = 1 # Force only one attempt for primary at this _execute_request level
        primary_kwargs["retry_policy"] = temp_policy

        primary_result = await self._execute_request_internal(self, **primary_kwargs)
        
        # Step 2: If primary fails AND a fallback is configured, immediately switch to fallback
        if primary_result is None and self.fallback_model_name:
            logger.warning(f"⚠️ 主引擎 {self.model_name} 调用失败，切换至备用引擎: {self.fallback_model_name}...")
            
            try:
                fallback_config_dict = getattr(global_config.model, self.fallback_model_name, None)
                
                if not fallback_config_dict:
                    logger.error(f"❌ 备用引擎配置 '{self.fallback_model_name}' 未在config/bot_config.toml中找到!")
                    return None
                
                # Create a NEW, temporary LLMRequest object with the fallback's configuration
                # Ensure request_type and other specific parameters are passed correctly
                fallback_model_config_for_llm = fallback_config_dict.copy()
                fallback_model_config_for_llm['request_type'] = self.request_type 
                
                # Preserve original parameters when creating fallback_requester
                # LLMRequest.__init__ expects model_config as first positional arg, and then **kwargs for self.params
                fallback_requester = LLMRequest(model_config=fallback_model_config_for_llm, **self.params)
                
                logger.info(f"⚙️ 正在使用备用引擎 {fallback_requester.model_name} (Provider: {fallback_requester.provider}) 重新发送请求...")
                
                # Execute the same request using the new, correctly configured fallback requester.
                # This internal call will handle its own retries based on its policy
                # (which defaults to max_retries=2 unless specified otherwise in the fallback's config)
                return await self._execute_request_internal(fallback_requester, **kwargs) # Pass original kwargs for fallback
            
            except Exception as e:
                logger.error(f"❌ 备用引擎 {self.fallback_model_name} 调用也失败了: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
    
        # Step 3: If primary succeeds (even on first try) or no fallback exists, return the primary result
        return primary_result

    async def _handle_response(self, response: ClientResponse, req_content: Dict[str, Any]) -> Optional[Dict]:
        response.raise_for_status()
        raw_text = await response.text()
        try:
            data = json.loads(raw_text)
            if "choices" in data and data["choices"]:
                message = data["choices"][0].get("message", {})
                content = message.get("content")
                if isinstance(content, str) and "{" in content and "}" in content:
                    nested_json = extract_json_from_text(content)
                    if nested_json: return nested_json
            return data
        except json.JSONDecodeError:
            json_result = extract_json_from_text(raw_text)
            if json_result is None: logger.error(f"无法从响应中解析出JSON: {raw_text[:1000]}")
            return json_result
    
    def _default_response_handler(self, result: Optional[Dict], **handler_kwargs) -> Optional[Tuple]:
        if result is None: return None

        # 优先处理标准聊天补全响应
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            content, tool_calls = message.get("content"), message.get("tool_calls")
            if tool_calls is None and (content is None or not str(content).strip()):
                logger.warning(f"模型 {self.model_name} 返回空回复（无内容或工具调用）。")
                return None
            content = content if content is not None else ""
            content_str, reasoning = self._extract_reasoning(str(content))
            if usage := result.get("usage"):
                self._record_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0), **handler_kwargs)
            return (content_str, reasoning, tool_calls) if tool_calls else (content_str, reasoning)
        
        # <<< 新增逻辑：处理不含'choices'字段的直接JSON响应 >>>
        # 如果没有 'choices' 字段，但结果本身是一个非空的字典，
        # 则将其视为模型直接返回的结构化数据（例如，来自嵌入或关键词提取模型）。
        # 在这种情况下，将整个字典作为内容的第一个元素返回。
        if isinstance(result, dict) and result:
            logger.debug(f"模型 {self.model_name} 返回直接JSON响应 (无'choices'字段)。")
            # 尝试记录使用量，如果result中包含usage信息
            if usage := result.get("usage"):
                self._record_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0), **handler_kwargs)
            # 返回一个包含该字典的元组作为 content，reasoning 和 tool_calls 为 None
            return (result, None, None) # (content, reasoning, tool_calls)
        
        logger.warning(f"响应中无'choices'字段且无法识别响应格式: {result}")
        return None

    @staticmethod
    def _extract_reasoning(content: str) -> Tuple[str, str]:
        if not isinstance(content, str): return "", ""
        match = re.search(r"(?:<think>)?(.*?)</think>", content, re.DOTALL)
        # 移除重复且已包含在其他地方的注释，恢复代码逻辑
        text = re.sub(r"(?:<think>)?.*?</think>", "", content, flags=re.DOTALL, count=1).strip()
        return text, match.group(1).strip() if match else ""
    async def _build_headers(self, no_key: bool = False) -> dict:
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        if not no_key: headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    async def _build_payload(self, prompt, image_base64=None, image_format=None):
        params = await self._transform_parameters(self.params.copy())
        content_parts = [{"type": "text", "text": prompt}]
        if image_base64: content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/{image_format.lower() if image_format else 'jpeg'};base64,{image_base64}"}})
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": content_parts}], **params}
        # Copying other relevant parameters
        if self.temp != 0.7: payload["temperature"] = self.temp
        if not self.enable_thinking: payload["enable_thinking"] = False
        if self.thinking_budget != 4096: payload["thinking_budget"] = self.thinking_budget
        if "max_tokens" not in payload and "max_completion_tokens" not in payload: payload["max_tokens"] = global_config.model.model_max_output_length
        return await self._transform_parameters(payload)
    async def _transform_parameters(self, params: dict) -> dict:
        if self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
        return params

    # --- FINAL FIX: Correctly call the default handler in the processing layer ---
    async def _process_response(self, raw_result: Optional[Dict], handler_kwargs: Dict, default_response: Tuple) -> Tuple:
        if raw_result is None:
            logger.warning(f"模型 {self.model_name} (包括备用模型) 调用失败，提供默认安全回复。")
            return default_response
        
        # Use a new `response_handler` kwarg for custom handlers, or fall back to the default
        handler = handler_kwargs.pop("response_handler", self._default_response_handler)
        processed_response = handler(raw_result, **handler_kwargs)

        if processed_response is None:
            logger.warning(f"模型 {self.model_name} 响应处理失败，提供默认安全回复。")
            return default_response
        
        # processed_response 此时可能是 (str, str, Optional[list]) 或者 (dict, None, None)
        # 我们需要确保它始终是 (content, reasoning, tool_calls)
        # 如果 processed_response 是 (dict, None, None)，那么它的长度是3，可以直接解包。
        # 如果是 (str, str)，它的长度是2，需要补齐 None。
        
        # Check if processed_response is a tuple and its length
        if isinstance(processed_response, tuple):
            if len(processed_response) == 3:
                return processed_response
            elif len(processed_response) == 2:
                # Assuming it's (content_str, reasoning) without tool_calls
                return (processed_response[0], processed_response[1], None)
        
        # Fallback for unexpected format (though _default_response_handler should prevent this now)
        logger.warning(f"模型 {self.model_name} 响应处理器返回了意外格式: {processed_response}，提供默认安全回复。")
        return default_response
        
    async def _call_and_process(self, default_response: Tuple, **kwargs) -> Tuple:
        raw_result = await self._execute_request(**kwargs)
        handler_kwargs = {"user_id": kwargs.get("user_id", "system"), "request_type": self.request_type}
        if "response_handler" in kwargs:
            handler_kwargs["response_handler"] = kwargs["response_handler"]
        return await self._process_response(raw_result, handler_kwargs, default_response)

    async def generate_response(self, prompt: str) -> Tuple:
        content, reasoning, tool_calls = await self._call_and_process(
            ("嗯... 麦麦好像开小差了，你能再说一遍吗？", "", None),
            prompt=prompt, endpoint="/chat/completions"
        )
        return content, reasoning, self.model_name, tool_calls

    async def generate_response_async(self, prompt: str, **kwargs) -> Union[str, Tuple]:
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], **self.params, **kwargs}
        user_id = kwargs.get("user_id", "system")
        content, reasoning, _ = await self._call_and_process(
            ("嗯... 我在听呢。", "", None),
            payload=payload, prompt=prompt, endpoint="/chat/completions", user_id=user_id
        )
        return content, (reasoning, self.model_name)
    
    async def generate_response_tool_async(self, prompt: str, tools: list, **kwargs) -> tuple[str, str, Optional[list]]:
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], **self.params, "tools": tools, **kwargs}
        user_id = kwargs.get("user_id", "system")
        content, reasoning, tool_calls = await self._call_and_process(
            ("我好像不知道该怎么做呢...", "", None),
            payload=payload, prompt=prompt, endpoint="/chat/completions", user_id=user_id
        )
        return content, reasoning, tool_calls
    
    async def generate_response_for_image(self, prompt: str, image_base64: str, image_format: str) -> Tuple:
        content, reasoning, tool_calls = await self._call_and_process(
            ("唔... 这张图片好有趣，但我好像有点没看懂呢。", "", None),
            prompt=prompt, image_base64=image_base64, image_format=image_format, endpoint="/chat/completions"
        )
        return content, reasoning, tool_calls

    async def get_embedding(self, text: str) -> Optional[list]:
        if not text: return None
        def handler(result, **kwargs): # embedding handler doesn't need other args
            if result and "data" in result and result["data"]:
                self._record_usage(result.get("usage", {}).get("prompt_tokens", 0), 0, result.get("usage", {}).get("total_tokens", 0), "system", "embedding", "/embeddings")
                return result["data"][0].get("embedding")
            return None # Explicitly return None if data or embedding is not found
        
        # Use _execute_request directly as embedding has a special handler
        raw_result = await self._execute_request(payload={"model": self.model_name, "input": text}, endpoint="/embeddings")
        return handler(raw_result)

def compress_base64_image_by_scale(base64_data: str, target_size: int = 0.8 * 1024 * 1024) -> str:
    try:
        image_data = base64.b64decode(base64_data)
        if len(image_data) <= target_size: return base64_data
        img = Image.open(io.BytesIO(image_data))
        scale = (target_size / len(image_data)) ** 0.5
        new_width, new_height = int(img.width * scale), int(img.height * scale)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        format_ = "PNG" if img.format == "PNG" and img.mode in ("RGBA", "LA") else "JPEG"
        resized_img.save(buffer, format=format_, optimize=True, quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"压缩图片失败: {e}")
        return base64_data