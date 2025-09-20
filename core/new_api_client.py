"""
新架构API客户端
按功能分类的API调用，支持硅基流动、魔搭原生格式和OpenAI兼容格式
独立的RPM管理和错误降级机制
"""

import asyncio
import time
import json
import hashlib
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import httpx
from loguru import logger
from collections import deque
from datetime import datetime, timedelta

from core.config_manager import get_new_config_manager, FunctionProviderConfig

class RateLimitError(Exception):
    """速率限制异常"""
    pass

class ProviderError(Exception):
    """提供商错误"""
    pass

class FunctionRateLimiter:
    """独立的功能速率限制器"""

    def __init__(self, rpm_limit: int, min_interval: float = 0.06):
        self.rpm_limit = rpm_limit
        self.min_interval = min_interval

        # 请求记录队列
        self.request_times = deque()
        self.last_request_time = 0

    async def wait_if_needed(self):
        """根据速率限制等待"""
        current_time = time.time()

        # 清理超过1分钟的记录
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()

        # 检查RPM限制
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"RPM limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

        # 检查最小请求间隔
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)

        # 记录请求
        self.request_times.append(current_time)
        self.last_request_time = time.time()

class BaseFunctionClient:
    """功能客户端基类"""

    def __init__(self, function_name: str):
        self.function_name = function_name
        self.config_manager = get_new_config_manager()

        # 获取功能配置
        if function_name == "text":
            self.function_config = self.config_manager.get_text_config()
        elif function_name == "vision":
            self.function_config = self.config_manager.get_vision_config()
        elif function_name == "generate":
            self.function_config = self.config_manager.get_generate_config()
        else:
            raise ValueError(f"不支持的功能: {function_name}")

        # 创建独立的速率限制器
        self.primary_limiter = None
        self.backup_limiter = None

        if self.function_config.has_primary() and self.function_config.primary_rpm:
            self.primary_limiter = FunctionRateLimiter(self.function_config.primary_rpm)

        if self.function_config.has_backup() and self.function_config.backup_rpm:
            self.backup_limiter = FunctionRateLimiter(self.function_config.backup_rpm)

        # HTTP客户端
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """关闭客户端"""
        await self.http_client.aclose()

    async def call_with_fallback(self, **kwargs) -> Optional[str]:
        """带降级的API调用"""

        # 尝试primary调用
        if self.function_config.has_primary():
            try:
                if self.primary_limiter:
                    await self.primary_limiter.wait_if_needed()

                result = await self._call_primary(**kwargs)
                if result:
                    return result
                else:
                    logger.warning(f"{self.function_name} primary调用返回空结果")

            except Exception as e:
                logger.error(f"{self.function_name} primary调用失败: {str(e)}")
                logger.warning(f"{self.function_name} primary失败，切换backup")

        # 尝试backup调用
        if self.function_config.has_backup():
            try:
                if self.backup_limiter:
                    await self.backup_limiter.wait_if_needed()

                result = await self._call_backup(**kwargs)
                if result:
                    logger.info(f"{self.function_name} backup调用成功")
                    return result
                else:
                    logger.warning(f"{self.function_name} backup调用返回空结果")

            except Exception as e:
                logger.error(f"{self.function_name} backup调用失败: {str(e)}")

        # 所有调用都失败
        logger.error(f"{self.function_name} 所有API调用失败")
        return None

    async def _call_primary(self, **kwargs) -> Optional[str]:
        """Primary API调用 - 子类实现"""
        raise NotImplementedError

    async def _call_backup(self, **kwargs) -> Optional[str]:
        """Backup API调用 - 使用OpenAI格式"""
        return await self._call_openai_format(
            url=self.function_config.backup_url,
            api_key=self.function_config.backup_api_key,
            model=self.function_config.backup_model,
            **kwargs
        )

    async def _call_openai_format(self, url: str, api_key: str, model: str, **kwargs) -> Optional[str]:
        """OpenAI格式的API调用"""

        # 构建OpenAI格式的消息
        messages = []

        if "system_prompt" in kwargs and kwargs["system_prompt"]:
            messages.append({
                "role": "system",
                "content": kwargs["system_prompt"]
            })

        if "prompt" in kwargs:
            if "image_base64" in kwargs:
                # 多模态消息格式
                content = [
                    {"type": "text", "text": kwargs["prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{kwargs['image_base64']}"
                        }
                    }
                ]
                messages.append({"role": "user", "content": content})
            else:
                # 纯文本消息
                messages.append({"role": "user", "content": kwargs["prompt"]})

        # 构建请求数据
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }

        # 发送请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = await self.http_client.post(
            url,
            json=data,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenAI格式API调用失败: {response.status_code} - {response.text}")

        return None

class TextUnderstandingClient(BaseFunctionClient):
    """文本理解客户端"""

    def __init__(self):
        super().__init__("text")

    async def _call_primary(self, **kwargs) -> Optional[str]:
        """硅基流动格式的文本理解调用"""

        # 构建硅基流动格式的请求
        data = {
            "model": self.function_config.primary_model,
            "messages": [
                {
                    "role": "system",
                    "content": kwargs.get("system_prompt", "You are a financial text analysis expert.")
                },
                {
                    "role": "user",
                    "content": kwargs.get("prompt", "")
                }
            ],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }

        headers = {
            "Authorization": f"Bearer {self.function_config.primary_api_key}",
            "Content-Type": "application/json"
        }

        response = await self.http_client.post(
            f"{self.function_config.primary_url}/chat/completions",
            json=data,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"硅基流动API调用失败: {response.status_code} - {response.text}")

        return None

class VisionUnderstandingClient(BaseFunctionClient):
    """多模态理解客户端"""

    def __init__(self):
        super().__init__("vision")

    async def _call_primary(self, **kwargs) -> Optional[str]:
        """魔搭格式的多模态理解调用"""

        # 构建魔搭格式的请求
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": kwargs.get("prompt", "")},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{kwargs.get('image_base64', '')}"
                    }
                }
            ]
        }]

        data = {
            "model": self.function_config.primary_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }

        headers = {
            "Authorization": f"Bearer {self.function_config.primary_api_key}",
            "Content-Type": "application/json"
        }

        response = await self.http_client.post(
            f"{self.function_config.primary_url}/chat/completions",
            json=data,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"魔搭API调用失败: {response.status_code} - {response.text}")

        return None

class LLMGenerationClient(BaseFunctionClient):
    """LLM生成客户端"""

    def __init__(self):
        super().__init__("generate")

    async def _call_primary(self, **kwargs) -> Optional[str]:
        """AnyRouter格式的LLM生成调用"""

        # AnyRouter使用OpenAI兼容格式
        return await self._call_openai_format(
            url=self.function_config.primary_url,
            api_key=self.function_config.primary_api_key,
            model=self.function_config.primary_model,
            **kwargs
        )

# 全局客户端实例
_text_client = None
_vision_client = None
_generate_client = None

async def get_text_client() -> TextUnderstandingClient:
    """获取文本理解客户端"""
    global _text_client
    if _text_client is None:
        _text_client = TextUnderstandingClient()
    return _text_client

async def get_vision_client() -> VisionUnderstandingClient:
    """获取多模态理解客户端"""
    global _vision_client
    if _vision_client is None:
        _vision_client = VisionUnderstandingClient()
    return _vision_client

async def get_generate_client() -> LLMGenerationClient:
    """获取LLM生成客户端"""
    global _generate_client
    if _generate_client is None:
        _generate_client = LLMGenerationClient()
    return _generate_client

async def close_all_clients():
    """关闭所有客户端"""
    global _text_client, _vision_client, _generate_client

    if _text_client:
        await _text_client.close()
        _text_client = None

    if _vision_client:
        await _vision_client.close()
        _vision_client = None

    if _generate_client:
        await _generate_client.close()
        _generate_client = None