"""
统一配置管理器
学习spark_multi_rag项目的配置管理方式，提供统一的环境变量验证和管理
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger

@dataclass
class ProviderConfig:
    """提供商配置结构"""
    api_key_env: str
    base_url: str
    models: Dict[str, str]
    rate_limits: Dict[str, Any]

class ConfigManager:
    """统一配置管理器"""
    
    # 预定义的提供商配置
    PROVIDER_CONFIGS = {
        "siliconflow": ProviderConfig(
            api_key_env="SILICONFLOW_API_KEY",
            base_url="https://api.siliconflow.cn/v1",
            models={
                "text": "Qwen/Qwen3-8B",
                "embedding": "BAAI/bge-m3"
            },
            rate_limits={
                "rpm": 1000,
                "tpm": 5000,
                "min_interval": 0.06,
                "batch_rpm": 100,
                "batch_tpm": 50000,
                "batch_min_interval": 0.6
            }
        ),
        "modelscope": ProviderConfig(
            api_key_env="MODELSCOPE_API_KEY",
            base_url="https://api-inference.modelscope.cn/v1",
            models={
                "vision": "qwen-vl-72b-instruct"
            },
            rate_limits={
                "rpm": 30,
                "tpm": 3000,
                "min_interval": 2,
                "batch_rpm": 10,
                "batch_tpm": 10000,
                "batch_min_interval": 6
            }
        ),
        "zhipu": ProviderConfig(
            api_key_env="ZHIPU_API_KEY",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            models={
                "text": "glm-4",
                "vision": "glm-4v-flash"
            },
            rate_limits={
                "rpm": 200,
                "tpm": 20000,
                "min_interval": 0.3,
                "batch_rpm": 50,
                "batch_tpm": 100000,
                "batch_min_interval": 1.2
            }
        ),
        "openai": ProviderConfig(
            api_key_env="OPENAI_API_KEY", 
            base_url="https://api.openai.com/v1",
            models={
                "text": "gpt-3.5-turbo",
                "vision": "gpt-4-vision-preview"
            },
            rate_limits={
                "rpm": 3500,
                "tpm": 90000,
                "min_interval": 0.02,
                "batch_rpm": 500,
                "batch_tpm": 450000,
                "batch_min_interval": 0.12
            }
        )
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器"""
        
        # 加载环境变量
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()  # 加载默认的.env文件
        
        logger.info("正在验证配置...")
        self.validate_configurations()
        logger.info("配置验证通过")
    
    def validate_configurations(self):
        """验证必需的配置"""
        
        missing_configs = []
        available_providers = []
        
        # 检查每个提供商的配置
        for provider_name, config in self.PROVIDER_CONFIGS.items():
            api_key = os.getenv(config.api_key_env)
            
            if api_key:
                available_providers.append(provider_name)
                logger.info(f"✓ {provider_name}: API密钥已配置")
            else:
                missing_configs.append(f"{config.api_key_env} ({provider_name})")
        
        # 至少需要一个提供商配置
        if not available_providers:
            error_msg = f"至少需要配置一个API提供商的密钥:\n" + \
                       "\n".join([f"  - {config}" for config in missing_configs])
            raise ValueError(error_msg)
        
        # 记录可用的提供商
        logger.info(f"可用的API提供商: {', '.join(available_providers)}")
        
        # 验证其他必需配置
        optional_configs = {
            'BATCH_MODE': '批处理模式（可选）',
            'BATCH_SIZE': '批处理大小（可选）',
            'MAX_CONCURRENT': '最大并发数（可选）'
        }
        
        for key, desc in optional_configs.items():
            value = os.getenv(key)
            if value:
                logger.info(f"✓ {key}: {value}")
        
        # 警告缺失的可选配置
        if missing_configs:
            logger.warning(f"未配置的API提供商: {', '.join(missing_configs)}")
    
    def get_provider_config(self, provider: str) -> ProviderConfig:
        """获取提供商配置"""
        
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"不支持的提供商: {provider}. 支持的提供商: {list(self.PROVIDER_CONFIGS.keys())}")
        
        config = self.PROVIDER_CONFIGS[provider]
        api_key = os.getenv(config.api_key_env)
        
        if not api_key:
            raise ValueError(f"提供商 {provider} 的API密钥未配置，请设置环境变量 {config.api_key_env}")
        
        return config
    
    def get_api_key(self, provider: str) -> str:
        """获取API密钥"""
        config = self.get_provider_config(provider)
        return os.getenv(config.api_key_env)
    
    def get_available_providers(self) -> List[str]:
        """获取可用的提供商列表"""
        
        available = []
        for provider_name, config in self.PROVIDER_CONFIGS.items():
            if os.getenv(config.api_key_env):
                available.append(provider_name)
        
        return available
    
    def is_batch_mode(self) -> bool:
        """检查是否启用批处理模式"""
        return os.getenv('BATCH_MODE', 'false').lower() == 'true'
    
    def get_batch_size(self) -> int:
        """获取批处理大小"""
        return int(os.getenv('BATCH_SIZE', '10'))
    
    def get_max_concurrent(self) -> int:
        """获取最大并发数"""
        return int(os.getenv('MAX_CONCURRENT', '5'))
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        
        return {
            'batch_mode': self.is_batch_mode(),
            'batch_size': self.get_batch_size(),
            'max_concurrent': self.get_max_concurrent(),
            'available_providers': self.get_available_providers(),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'checkpoint_enabled': os.getenv('ENABLE_CHECKPOINT', 'true').lower() == 'true',
            'checkpoint_dir': os.getenv('CHECKPOINT_DIR', './checkpoints')
        }
    
    def print_config_summary(self):
        """打印配置摘要"""
        
        print("\n" + "="*50)
        print("FinanceRAG-Pro 配置摘要")
        print("="*50)
        
        system_config = self.get_system_config()
        
        print(f"批处理模式: {'启用' if system_config['batch_mode'] else '禁用'}")
        print(f"批处理大小: {system_config['batch_size']}")
        print(f"最大并发数: {system_config['max_concurrent']}")
        print(f"断点续传: {'启用' if system_config['checkpoint_enabled'] else '禁用'}")
        print(f"日志级别: {system_config['log_level']}")
        
        print(f"\n可用API提供商:")
        for provider in system_config['available_providers']:
            config = self.PROVIDER_CONFIGS[provider]
            print(f"  - {provider}: {config.base_url}")
            print(f"    模型: {config.models}")
            
            # 显示速率限制
            limits = config.rate_limits
            mode = "批量" if system_config['batch_mode'] else "标准"
            if system_config['batch_mode']:
                print(f"    速率限制({mode}): {limits.get('batch_rpm', 'N/A')} RPM, {limits.get('batch_tpm', 'N/A')} TPM")
            else:
                print(f"    速率限制({mode}): {limits.get('rpm', 'N/A')} RPM, {limits.get('tpm', 'N/A')} TPM")
        
        print("="*50 + "\n")

# 创建全局配置管理器实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def validate_environment():
    """验证环境配置（兼容性函数）"""
    try:
        config_manager = get_config_manager()
        config_manager.print_config_summary()
        return True
    except Exception as e:
        logger.error(f"环境配置验证失败: {str(e)}")
        print(f"\n❌ 配置错误: {str(e)}")
        print("\n请检查以下配置:")
        print("1. 复制 .env.api 到 .env")
        print("2. 编辑 .env 文件，填入正确的API密钥")
        print("3. 确保至少配置一个API提供商\n")
        return False