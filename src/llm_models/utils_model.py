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

# --- Custom Exception Classes ---
class PayLoadTooLargeError(Exception):
    def __init__(self, message: str): super().__init__(message); self.message = message
class RequestAbortException(Exception):
    def __init__(self, message: str, response: ClientResponse): super().__init__(message); self.message, self.response = message, response
class PermissionDeniedException(Exception):
    def __init__(self, message: str): super().__init__(message); self.message = message

# --- Error Code Mapping ---
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

    def __init__(self, model: dict, **kwargs):
        try:
            self.api_key = os.environ[f"{model['provider']}_KEY"]
            self.base_url = os.environ[f"{model['provider']}_BASE_URL"]
        except Exception as e:
            raise ValueError(f"配置错误: {e}") from e
        self.model_name: str = model["name"]
        self.params = kwargs
        self.stream = model.get("stream", False)
        self.pri_in, self.pri_out = model.get("pri_in", 0), model.get("pri_out", 0)
        self.request_type = kwargs.pop("request_type", "default")
        self.enable_thinking = model.get("enable_thinking", False)
        self.temp = model.get("temp", 0.7)
        self.thinking_budget = model.get("thinking_budget", 4096)
        self._init_database()

    @staticmethod
    def _init_database():
        db.create_tables([LLMUsage], safe=True)

    def _record_usage(self, p_tokens, c_tokens, t_tokens, user_id="system", req_type=None, endpoint="/chat/completions"):
        if req_type is None: req_type = self.request_type
        LLMUsage.create(model_name=self.model_name, user_id=user_id, request_type=req_type, endpoint=endpoint, prompt_tokens=p_tokens, completion_tokens=c_tokens, total_tokens=t_tokens, cost=self._calculate_cost(p_tokens, c_tokens), status="success", timestamp=datetime.now())

    def _calculate_cost(self, p_tokens, c_tokens):
        return round(((p_tokens / 1e6) * self.pri_in) + ((c_tokens / 1e6) * self.pri_out), 6)
    
    async def _prepare_request(self, endpoint, **kwargs) -> Dict[str, Any]:
        # THIS IS THE FIX FOR THE SYNTAX ERROR
        policy = {"max_retries": 3, "base_wait": 10, "retry_codes": [429, 500, 503], "abort_codes": [400, 401, 402, 403, 413]}
        if "retry_policy" in kwargs: policy.update(kwargs["retry_policy"])
        api_url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        payload = kwargs.get("payload") or await self._build_payload(kwargs.get("prompt"), kwargs.get("image_base64"), kwargs.get("image_format"))
        if self.stream: payload["stream"] = True
        return {"policy": policy, "api_url": api_url, "payload": payload, **kwargs}

    async def _execute_request(self, **kwargs) -> Optional[Dict]:
        request_content = await self._prepare_request(**kwargs)
        try:
            final_payload_json = json.dumps(request_content["payload"], ensure_ascii=False, indent=2)
            logger.info(f"--- [PROMPT CAPTURE] ---\n发送给模型 '{self.model_name}' 的最终Payload:\n{final_payload_json}\n--- [END PROMPT CAPTURE] ---")
        except Exception as log_e:
            logger.error(f"无法序列化并记录请求Payload: {log_e}")
            
        for attempt in range(request_content["policy"]["max_retries"] + 1):
            try:
                headers = await self._build_headers()
                if self.stream: headers["Accept"] = "text/event-stream"
                async with aiohttp.ClientSession(connector=await get_tcp_connector()) as session:
                    async with session.post(request_content["api_url"], headers=headers, json=request_content["payload"]) as response:
                        return await self._handle_response(response)
            except Exception as e:
                if attempt >= request_content["policy"]["max_retries"]:
                    logger.error(f"模型 {self.model_name} 已达到最大重试次数，最终错误: {e}")
                    return None
                wait_time = request_content["policy"]["base_wait"] * (2 ** attempt)
                logger.warning(f"模型 {self.model_name} 请求失败 (尝试 {attempt + 1}), 等待 {wait_time}s... 错误: {e}")
                await asyncio.sleep(wait_time)
        return None

    async def _handle_response(self, response: ClientResponse) -> Optional[Dict]:
        response.raise_for_status()
        raw_text = await response.text()
        json_result = extract_json_from_text(raw_text)
        if json_result is None:
            logger.warning(f"无法从模型 {self.model_name} 的响应中解析出JSON。原始响应: {raw_text[:500]}")
            # This is a fallback to try and get content even if JSON fails
            try:
                # A desperate attempt to see if the raw text is a simple content response
                # This is unlikely for chat models but better than nothing.
                maybe_json = json.loads(raw_text)
                if "choices" in maybe_json and maybe_json["choices"]:
                     return maybe_json
            except json.JSONDecodeError:
                pass
            return None
        return json_result
    
    def _default_response_handler(self, result: Optional[Dict], user_id: str = "system", request_type: str = None, endpoint: str = "/chat/completions") -> Optional[Tuple]:
        if result is None: return None
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            content, tool_calls = message.get("content"), message.get("tool_calls")
            if tool_calls is None and (content is None or not str(content).strip()):
                logger.warning(f"模型 {self.model_name} 返回了空的content且没有工具调用，视为回复失败。")
                return None
            content = content if content is not None else ""
            content_str, reasoning = self._extract_reasoning(str(content))
            if usage := result.get("usage"):
                self._record_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0), user_id, request_type, endpoint)
            return (content_str, reasoning, tool_calls) if tool_calls else (content_str, reasoning)
        logger.warning(f"模型 {self.model_name} 的响应中没有找到'choices'字段: {result}")
        return None

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
        if self.temp != 0.7: payload["temperature"] = self.temp
        if not self.enable_thinking: payload["enable_thinking"] = False
        if self.thinking_budget != 4096: payload["thinking_budget"] = self.thinking_budget
        if "max_tokens" not in payload and "max_completion_tokens" not in payload: payload["max_tokens"] = global_config.model.model_max_output_length
        return await self._transform_parameters(payload)

    async def _transform_parameters(self, params: dict) -> dict:
        if self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
        return params

    async def _process_response(self, raw_result: Optional[Dict], handler_kwargs: Dict, default_response: Tuple) -> Tuple:
        if raw_result is None:
            logger.warning(f"模型 {self.model_name} 调用失败或返回空，提供默认安全回复。")
            return default_response
        processed_response = self._default_response_handler(raw_result, **handler_kwargs)
        if processed_response is None:
            logger.warning(f"模型 {self.model_name} 响应处理失败，提供默认安全回复。")
            return default_response
        return (processed_response + (None, None))[:3]

    async def generate_response(self, prompt: str) -> Tuple:
        raw_result = await self._execute_request(prompt=prompt, endpoint="/chat/completions")
        content, reasoning, _ = await self._process_response(raw_result, {"user_id": "system", "request_type": self.request_type}, ("嗯... 爱丽丝好像开小差了，你能再说一遍吗？", "", None))
        return content, reasoning, self.model_name

    async def generate_response_async(self, prompt: str, **kwargs) -> Union[str, Tuple]:
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], **self.params, **kwargs}
        raw_result = await self._execute_request(payload=payload, prompt=prompt, endpoint="/chat/completions")
        content, reasoning, _ = await self._process_response(raw_result, {"user_id": kwargs.get("user_id", "system"), "request_type": self.request_type}, ("嗯... 我在听呢。", "", None))
        return content, (reasoning, self.model_name)
    
    async def generate_response_tool_async(self, prompt: str, tools: list, **kwargs) -> tuple[str, str, Optional[list]]:
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], **self.params, "tools": tools, **kwargs}
        raw_result = await self._execute_request(payload=payload, prompt=prompt, endpoint="/chat/completions")
        content, reasoning, tool_calls = await self._process_response(raw_result, {"user_id": kwargs.get("user_id", "system"), "request_type": self.request_type}, ("我好像不知道该怎么做呢...", "", None))
        return content, reasoning, tool_calls
    
    async def generate_response_for_image(self, prompt: str, image_base64: str, image_format: str) -> Tuple:
        raw_result = await self._execute_request(prompt=prompt, image_base64=image_base64, image_format=image_format, endpoint="/chat/completions")
        content, reasoning, tool_calls = await self._process_response(raw_result, {"user_id": "system", "request_type": self.request_type}, ("唔... 这张图片好有趣，但我好像有点没看懂呢。", "", None))
        return content, reasoning, tool_calls

    async def get_embedding(self, text: str) -> Optional[list]:
        if not text: return None
        def handler(result):
            if result and "data" in result and result["data"]:
                if usage := result.get("usage"): self._record_usage(usage.get("prompt_tokens", 0), 0, usage.get("total_tokens", 0), "system", "embedding", "/embeddings")
                return result["data"][0].get("embedding")
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