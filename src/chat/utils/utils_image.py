import base64
import os
import time
import hashlib
from typing import Optional, List, Tuple
from PIL import Image
import io
import numpy as np

from src.common.database.database import db
from src.common.database.database_model import Images, ImageDescriptions
from src.config.config import global_config
from src.llm_models.utils_model import LLMRequest # We are using the new LLMRequest

from src.common.logger_manager import get_logger
from rich.traceback import install

install(extra_lines=3)

logger = get_logger("chat_image")


class ImageManager:
    _instance = None
    IMAGE_DIR = "data"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initializes the ImageManager, ensuring the database connection and starting background tasks."""
        if not self._initialized:
            self._ensure_image_dir()
            
            # --- FINAL FIX: Use the new, correct LLMRequest initialization ---
            # We pass the entire model configuration block to the model_config parameter.
            try:
                # Get the entire VLM model configuration block and convert it to a dictionary
                vlm_model_config = global_config.model.vlm
                # Initialize LLMRequest with the correct parameter name 'model_config'
                self._llm = LLMRequest(model_config=vlm_model_config, temperature=0.4, max_tokens=300, request_type="image")
            except Exception as e:
                logger.error(f"加载VLM模型配置失败，请检查 [model.vlm] 部分: {e}")
                # Fallback to a dummy or raise an error if VLM is critical
                self._llm = None
            # --- FIX END ---

            try:
                db.connect(reuse_if_open=True)
                db.create_tables([Images, ImageDescriptions], safe=True)
            except Exception as e:
                logger.error(f"数据库连接或表创建失败: {e}")
            finally:
                if not db.is_closed():
                    db.close()

            self._initialized = True

    def _ensure_image_dir(self):
        os.makedirs(self.IMAGE_DIR, exist_ok=True)

    # ... (The rest of your file remains unchanged, but I'll include it for completeness) ...
    
    @staticmethod
    def _get_description_from_db(image_hash: str, description_type: str) -> Optional[str]:
        try:
            with db.atomic():
                record = ImageDescriptions.get_or_none(
                    (ImageDescriptions.image_description_hash == image_hash) & (ImageDescriptions.type == description_type)
                )
            return record.description if record else None
        except Exception as e:
            logger.error(f"从数据库获取描述失败: {str(e)}")
            return None

    @staticmethod
    def _save_description_to_db(image_hash: str, description: str, description_type: str) -> None:
        try:
            with db.atomic():
                current_timestamp = time.time()
                defaults = {"description": description, "timestamp": current_timestamp}
                desc_obj, created = ImageDescriptions.get_or_create(
                    image_description_hash=image_hash, type=description_type, defaults=defaults
                )
                if not created:
                    desc_obj.description = description
                    desc_obj.timestamp = current_timestamp
                    desc_obj.save()
        except Exception as e:
            logger.error(f"保存描述到数据库失败: {str(e)}")

    async def get_emoji_description(self, image_base64: str) -> str:
        if not self._llm: return "[表情包(VLM未配置)]"
        try:
            image_bytes = base64.b64decode(image_base64)
            image_hash = hashlib.md5(image_bytes).hexdigest()
            image_format = Image.open(io.BytesIO(image_bytes)).format.lower()

            cached_description = self._get_description_from_db(image_hash, "emoji")
            if cached_description:
                return f"[表情包，含义看起来是：{cached_description}]"

            if image_format == "gif":
                image_base64_processed = self.transform_gif(image_base64)
                if not image_base64_processed:
                    return "[表情包(GIF处理失败)]"
                prompt = "这是一个动态图表情包，每一张图代表了动态图的某一帧，黑色背景代表透明，使用1-2个词描述一下表情包表达的情感和内容，简短一些"
                response_tuple = await self._llm.generate_response_for_image(prompt, image_base64_processed, "jpg")
            else:
                prompt = "这是一个表情包，请用使用几个词描述一下表情包所表达的情感和内容，简短一些"
                response_tuple = await self._llm.generate_response_for_image(prompt, image_base64, image_format)
            
            description = response_tuple[0] if response_tuple else None

            if not description:
                return "[表情包(描述生成失败)]"

            self._save_description_to_db(image_hash, description, "emoji")
            return f"[表情包：{description}]"
        except Exception as e:
            logger.error(f"获取表情包描述失败: {str(e)}")
            return "[表情包]"

    async def get_image_description(self, image_base64: str) -> str:
        if not self._llm: return "[图片(VLM未配置)]"
        try:
            image_bytes = base64.b64decode(image_base64)
            image_hash = hashlib.md5(image_bytes).hexdigest()
            image_format = Image.open(io.BytesIO(image_bytes)).format.lower()

            cached_description = self._get_description_from_db(image_hash, "image")
            if cached_description:
                return f"[图片：{cached_description}]"

            prompt = "请用中文描述这张图片的内容。如果有文字，请把文字都描述出来。并尝试猜测这个图片的含义。最多100个字。"
            response_tuple = await self._llm.generate_response_for_image(prompt, image_base64, image_format)
            description = response_tuple[0] if response_tuple else None

            if not description:
                return "[图片(描述生成失败)]"

            self._save_description_to_db(image_hash, description, "image")
            return f"[图片：{description}]"
        except Exception as e:
            logger.error(f"获取图片描述失败: {str(e)}")
            return "[图片]"

    @staticmethod
    def transform_gif(gif_base64: str, similarity_threshold: float = 1000.0, max_frames: int = 15) -> Optional[str]:
        try:
            gif_data = base64.b64decode(gif_base64)
            gif = Image.open(io.BytesIO(gif_data))
            all_frames = []
            while True:
                try:
                    gif.seek(len(all_frames))
                    all_frames.append(gif.convert("RGB").copy())
                except EOFError:
                    break
            
            if not all_frames: return None

            selected_frames = [all_frames[0]]
            last_selected_frame_np = np.array(all_frames[0])

            for frame in all_frames[1:]:
                current_frame_np = np.array(frame)
                mse = np.mean((current_frame_np - last_selected_frame_np) ** 2)
                if mse > similarity_threshold:
                    selected_frames.append(frame)
                    last_selected_frame_np = current_frame_np
                    if len(selected_frames) >= max_frames: break
            
            if not selected_frames: return None

            frame_width, frame_height = selected_frames[0].size
            if frame_height == 0: return None
            target_height = 200
            target_width = int((target_height / frame_height) * frame_width)
            if target_width == 0: target_width = 1

            resized_frames = [f.resize((target_width, target_height), Image.Resampling.LANCZOS) for f in selected_frames]
            total_width = target_width * len(resized_frames)
            if total_width == 0: return None
            
            combined_image = Image.new("RGB", (total_width, target_height))
            for idx, frame in enumerate(resized_frames):
                combined_image.paste(frame, (idx * target_width, 0))

            buffer = io.BytesIO()
            combined_image.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"GIF转换失败: {str(e)}", exc_info=True)
            return None


image_manager = ImageManager()

def image_path_to_base64(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")