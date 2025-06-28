import asyncio
import base64
import hashlib
import os
import random
import time
import traceback
from typing import Optional, Tuple, List, Dict
from PIL import Image
import io
import re
import numpy as np

from src.common.database.database_model import Emoji
from src.common.database.database import db as peewee_db
from src.config.config import global_config
from src.chat.utils.utils_image import image_path_to_base64, image_manager
from src.llm_models.utils_model import LLMRequest
from src.common.logger_manager import get_logger
from rich.traceback import install

install(extra_lines=3)

logger = get_logger("emoji")

BASE_DIR = os.path.join("data")
EMOJI_DIR = os.path.join(BASE_DIR, "emoji")
EMOJI_REGISTED_DIR = os.path.join(BASE_DIR, "emoji_registed")
MAX_EMOJI_FOR_PROMPT = 20


class MaiEmoji:
    # ... (Your MaiEmoji class remains unchanged) ...
    def __init__(self, full_path: str):
        if not full_path: raise ValueError("full_path cannot be empty")
        self.full_path = full_path
        self.path = os.path.dirname(full_path)
        self.filename = os.path.basename(full_path)
        self.embedding, self.hash, self.description = [], "", ""
        self.emotion, self.usage_count = [], 0
        self.last_used_time, self.register_time = time.time(), time.time()
        self.is_deleted, self.format = False, ""
    async def initialize_hash_format(self) -> Optional[bool]:
        try:
            if not os.path.exists(self.full_path):
                self.is_deleted = True
                return None
            image_base64 = image_path_to_base64(self.full_path)
            if image_base64 is None:
                self.is_deleted = True
                return None
            image_bytes = base64.b64decode(image_base64)
            self.hash = hashlib.md5(image_bytes).hexdigest()
            with Image.open(io.BytesIO(image_bytes)) as img:
                self.format = img.format.lower()
            return True
        except Exception as e:
            logger.error(f"初始化表情包时发生错误 ({self.filename}): {e}")
            self.is_deleted = True
            return None
    async def register_to_db(self) -> bool:
        # ... (This method remains unchanged) ...
        try:
            destination_full_path = os.path.join(EMOJI_REGISTED_DIR, self.filename)
            if not os.path.exists(self.full_path): return False
            if os.path.exists(destination_full_path): os.remove(destination_full_path)
            os.rename(self.full_path, destination_full_path)
            self.full_path, self.path = destination_full_path, EMOJI_REGISTED_DIR
            emotion_str = ",".join(self.emotion) if self.emotion else ""
            Emoji.create(emoji_hash=self.hash, full_path=self.full_path, format=self.format, description=self.description, emotion=emotion_str, is_registered=True, register_time=self.register_time)
            return True
        except Exception as e:
            logger.error(f"注册表情包失败 ({self.filename}): {e}")
            return False
    async def delete(self) -> bool:
        # ... (This method remains unchanged) ...
        try:
            if os.path.exists(self.full_path): os.remove(self.full_path)
            query = Emoji.delete().where(Emoji.emoji_hash == self.hash)
            deleted_rows = query.execute()
            self.is_deleted = True
            return deleted_rows > 0
        except Exception as e:
            logger.error(f"删除表情包失败 ({self.filename}): {e}")
            return False

# ... (Your helper functions like _emoji_objects_to_readable_list remain unchanged) ...
def _emoji_objects_to_readable_list(emoji_objects: List["MaiEmoji"]) -> List[str]:
    return [f"编号: {i+1}\n描述: {e.description}\n使用次数: {e.usage_count}" for i, e in enumerate(emoji_objects)]
def _to_emoji_objects(data) -> Tuple[List["MaiEmoji"], int]:
    # ... (This function remains unchanged) ...
    emoji_objects, load_errors = [], 0
    for emoji_data in list(data):
        full_path = emoji_data.full_path
        if not full_path: load_errors += 1; continue
        try:
            emoji = MaiEmoji(full_path=full_path)
            emoji.hash = emoji_data.emoji_hash
            emoji.description = emoji_data.description
            emoji.emotion = emoji_data.emotion.split(",") if emoji_data.emotion else []
            emoji.usage_count = emoji_data.usage_count
            emoji.format = emoji_data.format
            emoji_objects.append(emoji)
        except Exception: load_errors += 1
    return emoji_objects, load_errors
def _ensure_emoji_dir() -> None: os.makedirs(EMOJI_DIR, exist_ok=True); os.makedirs(EMOJI_REGISTED_DIR, exist_ok=True)
async def clear_temp_emoji() -> None:
    # ... (This function remains unchanged) ...
    for need_clear in (EMOJI_DIR, os.path.join(BASE_DIR, "image")):
        if os.path.exists(need_clear) and len(os.listdir(need_clear)) > 100:
            for filename in os.listdir(need_clear): os.remove(os.path.join(need_clear, filename))
async def clean_unused_emojis(emoji_dir: str, emoji_objects: List["MaiEmoji"]) -> None:
    # ... (This function remains unchanged) ...
    if not os.path.exists(emoji_dir): return
    tracked_paths = {e.full_path for e in emoji_objects if not e.is_deleted}
    for file_name in os.listdir(emoji_dir):
        file_path = os.path.join(emoji_dir, file_name)
        if os.path.isfile(file_path) and file_path not in tracked_paths:
            try: os.remove(file_path)
            except Exception: pass


class EmojiManager:
    _instance = None

    def __new__(cls) -> "EmojiManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initializes the EmojiManager."""
        if self._initialized:
            return

        # --- FINAL FIX: Use the new, correct LLMRequest initialization for ALL models ---
        try:
            # 1. Initialize the VLM model
            vlm_model_config = global_config.model.vlm
            self.vlm = LLMRequest(model_config=vlm_model_config, temperature=0.3, max_tokens=1000, request_type="emoji")

            # 2. Initialize the utilities model for emotion judgment
            utils_model_config = global_config.model.utils
            self.llm_emotion_judge = LLMRequest(model_config=utils_model_config, max_tokens=600, request_type="emoji")
            
            logger.info("表情包管理器中的所有LLM客户端已成功初始化。")

        except Exception as e:
            logger.error(f"在EmojiManager中加载模型配置失败，部分功能将不可用: {e}")
            self.vlm = None
            self.llm_emotion_judge = None
        # --- FIX END ---

        self.emoji_num = 0
        self.emoji_num_max = global_config.emoji.max_reg_num
        self.emoji_num_max_reach_deletion = global_config.emoji.do_replace
        self.emoji_objects: List[MaiEmoji] = []
        self._scan_task = None
        
        # We call initialize separately now to control startup sequence
        self._initialized = True
        logger.info("表情包管理器已启动。")

    def initialize(self) -> None:
        """Initializes database connection and emoji directories."""
        try:
            peewee_db.connect(reuse_if_open=True)
            _ensure_emoji_dir()
            Emoji.create_table(safe=True)
        except Exception as e:
            logger.error(f"EmojiManager数据库初始化失败: {e}")
            raise RuntimeError("EmojiManager数据库初始化失败") from e

    # ... (All other methods of EmojiManager like record_usage, get_emoji_for_text, etc., remain unchanged) ...
    # ... I've omitted them here for brevity, please keep your existing methods. ...
    def _ensure_db(self) -> None:
        if peewee_db.is_closed(): self.initialize()
    def record_usage(self, emoji_hash: str) -> None:
        try:
            with peewee_db.atomic():
                query = Emoji.update(usage_count=Emoji.usage_count + 1, last_used_time=time.time()).where(Emoji.emoji_hash == emoji_hash)
                query.execute()
        except Exception as e: logger.error(f"记录表情使用失败: {e}")
    async def get_emoji_for_text(self, text_emotion: str) -> Optional[Tuple[str, str]]:
        if not self.emoji_objects: return None
        # ... (rest of the logic is unchanged)
        return None
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        # ... (unchanged)
        return 0
    async def check_emoji_file_integrity(self) -> None:
        # ... (unchanged)
        pass
    async def start_periodic_check_register(self) -> None:
        await self.get_all_emoji_from_db()
        while True:
            # ... (unchanged)
            await asyncio.sleep(global_config.emoji.check_interval * 60)
    async def get_all_emoji_from_db(self) -> None:
        # ... (unchanged)
        pass
    async def get_emoji_from_db(self, emoji_hash: Optional[str] = None) -> List["MaiEmoji"]:
        # ... (unchanged)
        return []
    async def get_emoji_from_manager(self, emoji_hash: str) -> Optional["MaiEmoji"]:
        # ... (unchanged)
        return None
    async def delete_emoji(self, emoji_hash: str) -> bool:
        # ... (unchanged)
        return False
    async def replace_a_emoji(self, new_emoji: "MaiEmoji") -> bool:
        if not self.llm_emotion_judge: return False
        # ... (unchanged)
        return False
    async def build_emoji_description(self, image_base64: str) -> Tuple[str, List[str]]:
        if not self.vlm or not self.llm_emotion_judge: return "", []
        # ... (unchanged)
        return "", []
    async def register_emoji_by_filename(self, filename: str) -> bool:
        # ... (unchanged)
        return False

# Create the global singleton
emoji_manager = EmojiManager()