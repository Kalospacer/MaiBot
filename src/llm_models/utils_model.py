import asyncio
import json
import re
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
        self.params = kwargs
        self.stream = model_config.get("stream", False)
        self.pri_in, self.pri_out = model_config.get("pri_in", 0), model_config.get("pri_out", 0)
        self.request_type = kwargs.pop("request_type", "default")
        self.enable_thinking = model_config.get("enable_thinking", False)
        self.temp = model_config.get("temp", 0.7)
        self.thinking_budget = model_config.get("thinking_budget", 4096)
        self._init_database()

    # --- _init_database, _record_usage, _calculate_cost are unchanged ---
    @staticmethod
    def _init_database(): db.create_tables([LLMUsage], safe=True)
    def _record_usage(self, p_tokens, c_tokens, t_tokens, user_id="system", req_type=None, endpoint="/chat/completions"):
        if req_type is None: req_type = self.request_type
        LLMUsage.create(model_name=self.model_name, user_id=user_id, request_type=req_type, endpoint=endpoint, prompt_tokens=p_tokens, completion_tokens=c_tokens, total_tokens=t_tokens, cost=self._calculate_cost(p_tokens, c_tokens), status="success", timestamp=datetime.now())
    def _calculate_cost(self, p_tokens, c_tokens): return round(((p_tokens / 1e6) * self.pri_in) + ((c_tokens / 1e6) * self.pri_out), 6)

    async def _prepare_request(self, requester: 'LLMRequest', endpoint: str, **kwargs) -> Dict[str, Any]:
        policy = {"max_retries": 2, "base_wait": 5, "retry_codes": [429, 500, 503], "abort_codes": [400, 401, 402, 403, 413]}
        if "retry_policy" in kwargs: policy.update(kwargs["retry_policy"])
        api_url = f"{requester.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        payload = kwargs.get("payload") or await requester._build_payload(kwargs.get("prompt"), kwargs.get("image_base64"), kwargs.get("image_format"))
        if requester.stream: payload["stream"] = True
        # Ensure model name in payload matches the current requester
        payload["model"] = requester.model_name
        return {"policy": policy, "api_url": api_url, "payload": payload, **kwargs}

    async def _execute_request_internal(self, requester: 'LLMRequest', **kwargs) -> Optional[Dict]:
        request_content = await self._prepare_request(requester, **kwargs)
        try:
            final_payload_json = json.dumps(request_content["payload"], ensure_ascii=False, indent=2)
            logger.info(f"--- [PROMPT CAPTURE] ---\n发送给模型 '{requester.model_name}' (Provider: {requester.provider}) 的最终Payload:\n{final_payload_json}\n--- [END PROMPT CAPTURE] ---")
        except Exception as log_e:
            logger.error(f"无法序列化并记录请求Payload: {log_e}")
            
        for attempt in range(request_content["policy"]["max_retries"]):
            try:
                headers = await requester._build_headers()
                if requester.stream: headers["Accept"] = "text/event-stream"
                async with aiohttp.ClientSession(connector=await get_tcp_connector()) as session:
                    async with session.post(request_content["api_url"], headers=headers, json=request_content["payload"], timeout=120) as response:
                        return await requester._handle_response(response, request_content)
            except Exception as e:
                if attempt >= request_content["policy"]["max_retries"] - 1:
                    logger.error(f"模型 {requester.model_name} 已达到最大重试次数，最终错误: {e}")
                    return None
                wait_time = request_content["policy"]["base_wait"] * (2 ** attempt)
                logger.warning(f"模型 {requester.model_name} 请求失败 (尝试 {attempt + 1}), 等待 {wait_time}s... 错误: {e}")
                await asyncio.sleep(wait_time)
        return None

    async def _execute_request(self, **kwargs) -> Optional[Dict]:
        logger.debug(f"🚀 正在尝试主引擎: {self.model_name} (Provider: {self.provider})")
        primary_result = await self._execute_request_internal(self, **kwargs)
        if primary_result is None and self.fallback_model_name:
            logger.warning(f"⚠️ 主引擎 {self.model_name} 调用失败，正在启动备用引擎: {self.fallback_model_name}...")
            try:
                all_model_configs = global_config.model.dict()
                fallback_config_dict = all_model_configs.get(self.fallback_model_name)
                if not fallback_config_dict:
                    logger.error(f"❌ 备用引擎配置 '{self.fallback_model_name}' 未在config/bot_config.toml中找到!")
                    return None
                # Create a NEW requester with the correct fallback config
                fallback_requester = LLMRequest(model_config=fallback_config_dict, **self.params)
                logger.info(f"⚙️ 正在使用备用引擎 {fallback_requester.model_name} (Provider: {fallback_requester.provider}) 重新发送请求...")
                return await self._execute_request_internal(fallback_requester, **kwargs)
            except Exception as e:
                logger.error(f"❌ 备用引擎 {self.fallback_model_name} 调用也失败了: {e}")
                return None
        return primary_result

    # --- FINAL FIX: Signature Correction ---
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
    
    # --- FINAL FIX: Signature Correction ---
    def _default_response_handler(self, result: Optional[Dict], **handler_kwargs) -> Optional[Tuple]:
        if result is None: return None
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            content, tool_calls = message.get("content"), message.get("tool_calls")
            if tool_calls is None and (content is None or not str(content).strip()):
                logger.warning(f"模型 {self.model_name} 返回空回复。")
                return None
            content = content if content is not None else ""
            content_str, reasoning = self._extract_reasoning(str(content))
            if usage := result.get("usage"):
                self._record_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0), **handler_kwargs)
            return (content_str, reasoning, tool_calls) if tool_calls else (content_str, reasoning)
        logger.warning(f"响应中无'choices'字段: {result}")
        return None

    # ... (Rest of the helper methods are unchanged) ...
    @staticmethod
    def _extract_reasoning(content: str) -> Tuple[str, str]:
        if not isinstance(content, str): return "", ""
        match = re.search(r"(?:<think>)?(.*?)</think>", content, re.DOTALL)
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
        return (processed_response + (None, None))[:3]
        
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
                if usage := result.get("usage"): self._record_usage(usage.get("prompt_tokens", 0), 0, usage.get("total_tokens", 0), "system", "embedding", "/embeddings")
                return result["data"][0].get("embedding")
        
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