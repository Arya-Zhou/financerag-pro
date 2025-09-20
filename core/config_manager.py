"""
新架构配置管理器
按功能分类的API提供商配置管理，支持primary/backup双重配置
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger

@dataclass
class FunctionProviderConfig:
    """功能提供商配置结构"""
    primary_url: Optional[str]
    primary_api_key: Optional[str]
    primary_model: Optional[str]
    primary_rpm: Optional[int]

    backup_url: Optional[str]
    backup_api_key: Optional[str]
    backup_model: Optional[str]
    backup_rpm: Optional[int]

    def has_primary(self) -> bool:
        """检查是否有primary配置"""
        return all([self.primary_url, self.primary_api_key, self.primary_model])

    def has_backup(self) -> bool:
        """检查是否有backup配置"""
        return all([self.backup_url, self.backup_api_key, self.backup_model])

class NewConfigManager:
    """新架构配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器"""

        # 加载环境变量
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()  # 加载默认的.env文件

        logger.info("正在验证新架构API配置...")

        # 加载功能配置
        self.text_config = self._load_function_config("TEXT")
        self.vision_config = self._load_function_config("VISION")
        self.generate_config = self._load_function_config("GENERATE")

        self.validate_configurations()
        logger.info("新架构API配置验证通过")

    def _load_function_config(self, function_name: str) -> FunctionProviderConfig:
        """加载指定功能的配置"""
        return FunctionProviderConfig(
            primary_url=os.getenv(f"{function_name}_PRIMARY_URL"),
            primary_api_key=os.getenv(f"{function_name}_PRIMARY_API_KEY"),
            primary_model=os.getenv(f"{function_name}_PRIMARY_MODEL"),
            primary_rpm=self._get_int_env(f"{function_name}_PRIMARY_RPM"),

            backup_url=os.getenv(f"{function_name}_BACKUP_URL"),
            backup_api_key=os.getenv(f"{function_name}_BACKUP_API_KEY"),
            backup_model=os.getenv(f"{function_name}_BACKUP_MODEL"),
            backup_rpm=self._get_int_env(f"{function_name}_BACKUP_RPM"),
        )

    def _get_int_env(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """安全地获取整数环境变量"""
        value = os.getenv(key)
        if value:
            try:
                return int(value)
            except ValueError:
                logger.warning(f"无效的整数配置 {key}: {value}")
                return default
        return default

    def validate_configurations(self):
        """验证必需的配置"""

        available_functions = []
        missing_functions = []

        # 检查文本理解配置
        if self.text_config.has_primary():
            primary_status = "硅基流动(primary)"
            backup_status = " + 文本理解(backup)" if self.text_config.has_backup() else ""
            available_functions.append(f"文本理解: {primary_status}{backup_status}")
            logger.info(f"✓ 文本理解: {primary_status}{backup_status}")
        else:
            missing_functions.append("文本理解")
            logger.warning("⚠ 文本理解: 未配置primary")

        # 检查多模态理解配置
        if self.vision_config.has_primary():
            primary_status = "魔搭(primary)"
            backup_status = " + 多模态理解(backup)" if self.vision_config.has_backup() else ""
            available_functions.append(f"多模态理解: {primary_status}{backup_status}")
            logger.info(f"✓ 多模态理解: {primary_status}{backup_status}")
        else:
            missing_functions.append("多模态理解")
            logger.warning("⚠ 多模态理解: 未配置primary")

        # 检查LLM生成配置
        if self.generate_config.has_primary():
            primary_status = "AnyRouter(primary)"
            backup_status = " + LLM生成(backup)" if self.generate_config.has_backup() else ""
            available_functions.append(f"LLM生成: {primary_status}{backup_status}")
            logger.info(f"✓ LLM生成: {primary_status}{backup_status}")
        else:
            missing_functions.append("LLM生成")
            logger.warning("⚠ LLM生成: 未配置primary")

        # 至少需要一个功能可用
        if not available_functions:
            error_msg = "至少需要配置一个功能的API提供商"
            raise ValueError(error_msg)

        # 记录配置状态
        logger.info(f"可用的API功能: {', '.join(available_functions)}")

        if missing_functions:
            logger.warning(f"未配置的功能: {', '.join(missing_functions)}")

    def get_text_config(self) -> FunctionProviderConfig:
        """获取文本理解配置"""
        return self.text_config

    def get_vision_config(self) -> FunctionProviderConfig:
        """获取多模态理解配置"""
        return self.vision_config

    def get_generate_config(self) -> FunctionProviderConfig:
        """获取LLM生成配置"""
        return self.generate_config

    def is_function_available(self, function_name: str) -> bool:
        """检查指定功能是否可用"""
        if function_name == "text":
            return self.text_config.has_primary()
        elif function_name == "vision":
            return self.vision_config.has_primary()
        elif function_name == "generate":
            return self.generate_config.has_primary()
        else:
            return False

    def get_function_fallback_status(self, function_name: str) -> Tuple[bool, bool]:
        """获取功能的primary和backup可用状态"""
        if function_name == "text":
            return self.text_config.has_primary(), self.text_config.has_backup()
        elif function_name == "vision":
            return self.vision_config.has_primary(), self.vision_config.has_backup()
        elif function_name == "generate":
            return self.generate_config.has_primary(), self.generate_config.has_backup()
        else:
            return False, False

# 全局配置管理器实例
_config_manager = None

def get_new_config_manager(config_path: Optional[str] = None) -> NewConfigManager:
    """获取新架构配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = NewConfigManager(config_path)
    return _config_manager

# 向后兼容的接口
def get_config_manager(config_path: Optional[str] = None):
    """向后兼容的配置管理器获取函数"""
    return get_new_config_manager(config_path)