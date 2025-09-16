"""
API客户端基类，支持批量调用、RPM/TPM限制和重试机制
适用于硅基流动和魔搭的API调用
"""

import asyncio
import time
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import httpx
from loguru import logger
from collections import deque
from datetime import datetime, timedelta

@dataclass
class APIConfig:
    """API配置"""
    base_url: str
    api_key: str
    model: str
    rpm_limit: int = 1000  # 每分钟请求数限制
    tpm_limit: int = 5000  # 每分钟token数限制
    min_request_interval: float = 0.06  # 最小请求间隔(秒)
    max_retries: int = 3  # 最大重试次数
    timeout: int = 30  # 请求超时时间(秒)
    batch_size: int = 10  # 批量处理大小

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int, min_interval: float):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.min_interval = min_interval
        
        # 请求记录队列
        self.request_times = deque()
        self.token_counts = deque()
        self.last_request_time = 0
    
    async def wait_if_needed(self, estimated_tokens: int = 100):
        """根据速率限制等待"""
        current_time = time.time()
        
        # 清理超过1分钟的记录
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        while self.token_counts and current_time - self.token_counts[0][0] > 60:
            self.token_counts.popleft()
        
        # 检查RPM限制
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"RPM limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # 检查TPM限制
        total_tokens = sum(count for _, count in self.token_counts)
        if total_tokens + estimated_tokens > self.tpm_limit:
            wait_time = 60 - (current_time - self.token_counts[0][0])
            if wait_time > 0:
                logger.info(f"TPM limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # 检查最小请求间隔
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        # 记录请求
        self.request_times.append(current_time)
        self.token_counts.append((current_time, estimated_tokens))
        self.last_request_time = time.time()

class CheckpointManager:
    """断点管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, task_id: str, data: Dict[str, Any]):
        """保存断点"""
        checkpoint_file = self.checkpoint_dir / f"{task_id}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f)
        logger.info(f"Checkpoint saved: {task_id}")
    
    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """加载断点"""
        checkpoint_file = self.checkpoint_dir / f"{task_id}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                logger.info(f"Checkpoint loaded: {task_id} from {checkpoint['timestamp']}")
                return checkpoint['data']
        return None
    
    def remove_checkpoint(self, task_id: str):
        """删除断点"""
        checkpoint_file = self.checkpoint_dir / f"{task_id}.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Checkpoint removed: {task_id}")

class BaseAPIClient:
    """通用API客户端基类"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            config.rpm_limit, 
            config.tpm_limit,
            config.min_request_interval
        )
        self.checkpoint_manager = CheckpointManager()
        self.client = httpx.AsyncClient(timeout=config.timeout)
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
    
    def estimate_tokens(self, text: str) -> int:
        """估算token数量"""
        # 简单估算：中文约1.5个字符一个token，英文约4个字符一个token
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fa5'])
        english_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + english_chars / 4)
    
    async def make_request(
        self, 
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """发送单个请求"""
        
        estimated_tokens = self.estimate_tokens(prompt)
        
        for attempt in range(self.config.max_retries):
            try:
                # 等待速率限制
                await self.rate_limiter.wait_if_needed(estimated_tokens)
                
                # 构建请求
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                request_data = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # 发送请求
                response = await self.client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.warning(f"Request failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                
            # 重试前等待
            if attempt < self.config.max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying after {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        return None
    
    async def batch_process(
        self,
        items: List[Dict[str, Any]],
        process_func: Callable,
        task_id: str = None,
        resume_from_checkpoint: bool = True
    ) -> List[Any]:
        """批量处理"""
        
        # 生成任务ID
        if task_id is None:
            task_id = hashlib.md5(json.dumps(items).encode()).hexdigest()[:8]
        
        # 尝试从断点恢复
        results = []
        start_index = 0
        
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint(task_id)
            if checkpoint:
                results = checkpoint.get('results', [])
                start_index = checkpoint.get('processed_count', 0)
                logger.info(f"Resuming from checkpoint: {start_index}/{len(items)} processed")
        
        # 批量处理
        batch_size = self.config.batch_size
        
        for i in range(start_index, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = []
            
            # 处理批次
            tasks = [process_func(item) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            
            results.extend(batch_results)
            
            # 保存断点
            self.checkpoint_manager.save_checkpoint(task_id, {
                'results': results,
                'processed_count': i + len(batch),
                'total_count': len(items)
            })
            
            logger.info(f"Processed batch: {i + len(batch)}/{len(items)}")
            
            # 批次间休息
            if i + batch_size < len(items):
                await asyncio.sleep(1)
        
        # 删除断点
        self.checkpoint_manager.remove_checkpoint(task_id)
        
        return results

class SiliconFlowClient(BaseAPIClient):
    """硅基流动API客户端(Qwen3-8B)"""
    
    def __init__(self, api_key: str, batch_mode: bool = False):
        config = APIConfig(
            base_url="https://api.siliconflow.cn/v1",
            api_key=api_key,
            model="Qwen/Qwen3-8B",
            rpm_limit=100 if batch_mode else 1000,
            tpm_limit=50000 if batch_mode else 5000,
            min_request_interval=0.6 if batch_mode else 0.06,
            max_retries=3,
            batch_size=5 if batch_mode else 1
        )
        super().__init__(config)
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> Optional[str]:
        """生成文本"""
        return await self.make_request(prompt, system_prompt, temperature, max_tokens)

class ModelScopeVLClient(BaseAPIClient):
    """魔搭视觉语言模型客户端(Qwen2.5-VL-72B)"""
    
    def __init__(self, api_key: str, batch_mode: bool = False):
        config = APIConfig(
            base_url="https://api-inference.modelscope.cn/v1",
            api_key=api_key,
            model="qwen-vl-72b-instruct",
            rpm_limit=10 if batch_mode else 30,
            tpm_limit=10000 if batch_mode else 3000,
            min_request_interval=6 if batch_mode else 2,
            max_retries=3,
            batch_size=2 if batch_mode else 1
        )
        super().__init__(config)
    
    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """分析图像"""
        
        # 构建包含图像的prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        
        estimated_tokens = self.estimate_tokens(prompt) + 500  # 图像额外token
        
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.wait_if_needed(estimated_tokens)
                
                request_data = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = await self.client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
            except Exception as e:
                logger.error(f"Image analysis error (attempt {attempt + 1}): {str(e)}")
                
            if attempt < self.config.max_retries - 1:
                wait_time = 2 ** attempt * 2
                await asyncio.sleep(wait_time)
        
        return None