import os
from typing import Dict, Any
from pathlib import Path
import yaml

class Config:
    """System configuration management"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return os.path.join(Path(__file__).parent.parent, "config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "system": {
                "name": "FinanceRAG-Pro",
                "version": "1.0.0",
                "debug": False,
                "log_level": "INFO"
            },
            
            "models": {
                "query_understanding": {
                    "model": "gpt-3.5-turbo",
                    "fallback": "Qwen2-7B-GPTQ",
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                "embedding": {
                    "model": "BAAI/bge-m3",
                    "dimension": 1024,
                    "batch_size": 32,
                    "precision": "fp16"
                },
                "reranking": {
                    "model": "BAAI/bge-reranker-base",
                    "top_k": 10,
                    "precision": "int8"
                },
                "generation": {
                    "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ",
                    "precision": "4bit",
                    "max_length": 2048,
                    "temperature": 0.7
                }
            },
            
            "database": {
                "postgresql": {
                    "host": os.getenv("PG_HOST", "localhost"),
                    "port": int(os.getenv("PG_PORT", 5432)),
                    "database": os.getenv("PG_DATABASE", "financerag"),
                    "user": os.getenv("PG_USER", "postgres"),
                    "password": os.getenv("PG_PASSWORD", "password")
                },
                "redis": {
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": int(os.getenv("REDIS_PORT", 6379)),
                    "db": int(os.getenv("REDIS_DB", 0)),
                    "password": os.getenv("REDIS_PASSWORD", None)
                },
                "chromadb": {
                    "persist_directory": "./data/chromadb",
                    "collection_name": "finance_documents"
                },
                "minio": {
                    "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
                    "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                    "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                    "bucket": os.getenv("MINIO_BUCKET", "finance-docs"),
                    "secure": False
                }
            },
            
            "retrieval": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 20,
                "similarity_threshold": 0.7,
                "enable_rerank": True
            },
            
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000,
                "eviction_policy": "LRU"
            },
            
            "performance": {
                "max_workers": 10,
                "batch_processing": True,
                "async_enabled": True,
                "timeout": 30
            },
            
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_enabled": True,
                "docs_enabled": True,
                "max_request_size": 10485760
            }
        }
    
    def get(self, key_path: str, default=None):
        """Get configuration value by key path (e.g., 'models.embedding.model')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value by key path"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)