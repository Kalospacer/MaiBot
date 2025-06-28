# 文件: src/person_info/person_info.py

import asyncio
import copy
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.traceback import install

from src.common.database.database import db
from src.common.database.database_model import PersonInfo
from src.config.config import global_config
from src.individuality.individuality import individuality
from src.llm_models.utils_model import LLMRequest
from src.common.logger_manager import get_logger

matplotlib.use("Agg")
install(extra_lines=3)

logger = get_logger("person_info")

person_info_default = {
    "person_name": None, "name_reason": None,
    "platform": "unknown", "user_id": "unknown", "nickname": "Unknown",
    "relationship_value": 0, "know_time": 0, "msg_interval": 2000,
    "msg_interval_list": [], "user_cardname": None, "user_avatar": None,
}


class PersonInfoManager:
    def __init__(self):
        self.person_name_list = {}
        
        try:
            utils_model_config = global_config.model.utils.copy()
            utils_model_config['request_type'] = "relation.qv_name"
            if 'max_tokens' not in utils_model_config:
                utils_model_config['max_tokens'] = 256
            self.qv_name_llm = LLMRequest(model_config=utils_model_config)
        except Exception as e:
            logger.error(f"加载 [model.utils] 配置失败，部分个人信息功能将不可用: {e}")
            self.qv_name_llm = None

        try:
            db.connect(reuse_if_open=True)
            db.create_tables([PersonInfo], safe=True)
            for record in PersonInfo.select(PersonInfo.person_id, PersonInfo.person_name).where(PersonInfo.person_name.is_null(False)):
                if record.person_name:
                    self.person_name_list[record.person_id] = record.person_name
            logger.debug(f"已加载 {len(self.person_name_list)} 个用户名称。")
        finally:
            if not db.is_closed():
                db.close()

    @staticmethod
    def get_person_id(platform: str, user_id: int) -> str:
        if "-" in platform: platform = platform.split("-")[1]
        key = f"{platform}_{user_id}"
        return hashlib.md5(key.encode()).hexdigest()

    async def is_person_known(self, platform: str, user_id: int):
        person_id = self.get_person_id(platform, user_id)
        return await asyncio.to_thread(lambda: PersonInfo.get_or_none(PersonInfo.person_id == person_id) is not None)

    def get_person_id_by_person_name(self, person_name: str):
        try:
            record = PersonInfo.get_or_none(PersonInfo.person_name == person_name)
            return record.person_id if record else ""
        except Exception as e:
            logger.error(f"根据用户名 {person_name} 获取用户ID时出错: {e}")
            return ""

    @staticmethod
    async def create_person_info(person_id: str, data: dict = None):
        if not person_id:
            logger.error(f"尝试创建个人信息失败：传入的person_id为空或无效。原始数据: {data}")
            return
        
        final_data = {
            **person_info_default,
            **(data or {}),
            "person_id": person_id
        }
        
        model_fields = PersonInfo._meta.fields.keys()
        db_data = {k: v for k, v in final_data.items() if k in model_fields}
        
        if "msg_interval_list" in db_data and isinstance(db_data["msg_interval_list"], list):
            db_data["msg_interval_list"] = json.dumps(db_data["msg_interval_list"])

        logger.debug(f"尝试创建PersonInfo记录，person_id: {person_id}, 最终传入数据: {db_data}")
        
        if 'person_id' not in db_data or not db_data['person_id']:
            logger.critical(f"致命错误：db_data中缺少person_id或其值为空！这不应发生。db_data: {db_data}")
            raise ValueError("Person ID is missing or invalid in database creation data.")
        
        try:
            await asyncio.to_thread(lambda: PersonInfo.create(**db_data))
            logger.info(f"成功为 person_id '{person_id}' 创建了PersonInfo记录。")
        except Exception as e:
            logger.error(f"为 person_id '{person_id}' 创建PersonInfo记录失败，数据: {db_data}。错误: {e}", exc_info=True)
            raise

    async def update_one_field(self, person_id: str, field_name: str, value, data: dict = None):
        if field_name not in PersonInfo._meta.fields: return
        def _db_update():
            record = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
            if record:
                val = json.dumps(value) if field_name == "msg_interval_list" else value
                setattr(record, field_name, val)
                record.save()
                return True
            return False
        if not await asyncio.to_thread(_db_update):
            creation_data = data or {}
            creation_data[field_name] = value
            await self.create_person_info(person_id, creation_data)

    @staticmethod
    async def has_one_field(person_id: str, field_name: str):
        """判断是否存在某一个字段"""
        if field_name not in PersonInfo._meta.fields:
            logger.debug(f"检查字段'{field_name}'失败，未在 PersonInfo Peewee 模型中定义。")
            return False
        def _db_has_field_sync(p_id: str, f_name: str):
            record = PersonInfo.select(getattr(PersonInfo, f_name)).where(PersonInfo.person_id == p_id).get_or_none()
            return record is not None and getattr(record, f_name) is not None
        try:
            return await asyncio.to_thread(_db_has_field_sync, person_id, field_name) # <<< 关键修正：确保这里传递的是 field_name >>>
        except Exception as e:
            # <<< 关键修正：使用外部作用域的变量 >>>
            logger.error(f"检查字段 {field_name} for {person_id} 时出错: {e}")
            return False

    async def get_value(self, person_id: str, field_name: str):
        if not person_id: return person_info_default.get(field_name)
        if field_name not in PersonInfo._meta.fields: return person_info_default.get(field_name)
        def _db_get():
            record = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
            if record:
                val = getattr(record, field_name)
                if field_name == "msg_interval_list" and isinstance(val, str):
                    try: return json.loads(val)
                    except json.JSONDecodeError: return []
                return val
            return None
        value = await asyncio.to_thread(_db_get)
        return value if value is not None else person_info_default.get(field_name)

    async def get_values(self, person_id: str, field_names: list) -> dict:
        result = {}
        record = await asyncio.to_thread(lambda: PersonInfo.get_or_none(PersonInfo.person_id == person_id))
        for field_name in field_names:
            if field_name not in PersonInfo._meta.fields:
                result[field_name] = person_info_default.get(field_name)
                continue
            if record:
                value = getattr(record, field_name)
                if field_name == "msg_interval_list" and isinstance(value, str):
                    try: result[field_name] = json.loads(value)
                    except json.JSONDecodeError: result[field_name] = []
                else:
                    result[field_name] = value if value is not None else person_info_default.get(field_name)
            else:
                result[field_name] = person_info_default.get(field_name)
        return result

    @staticmethod
    def _extract_json_from_text(text: str) -> dict:
        try: return json.loads(text)
        except json.JSONDecodeError: pass
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match: return json.loads(match.group())
        except json.JSONDecodeError: pass
        return {}
    
    async def qv_person_name(self, person_id: str, user_nickname: str, user_cardname: str, user_avatar: str, request: str = ""):
        if not self.qv_name_llm: return None
        return None
        
    async def get_or_create_person(self, platform: str, user_id: int, **kwargs) -> str:
        person_id = self.get_person_id(platform, user_id)
        if not await asyncio.to_thread(lambda: PersonInfo.get_or_none(PersonInfo.person_id == person_id)):
            initial_data = {"platform": platform, "user_id": str(user_id), **kwargs}
            await self.create_person_info(person_id, data=initial_data)
        return person_id

    async def personal_habit_deduction(self):
        while True:
            logger.info("个人信息推断任务正在运行...")
            await asyncio.sleep(86400)

    async def get_person_info_by_name(self, person_name: str) -> dict | None:
        return None

# Creating a singleton instance
person_info_manager = PersonInfoManager()