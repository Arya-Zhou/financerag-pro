"""
Patches and fixes for main_lite.py potential issues
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException

# Issue 1: Configuration file path checking
def get_config_path() -> str:
    """Get the appropriate config file path"""
    config_files = [
        "config_lite.yaml",
        "config.yaml", 
        "config_api.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            return config_file
    
    # Create default config if none exists
    create_default_config()
    return "config_lite.yaml"


def create_default_config():
    """Create default configuration file"""
    default_config = """system:
  name: FinanceRAG-Pro-Lite
  version: 1.0.0-lite
  debug: false
  log_level: INFO

models:
  embedding:
    model: sentence-transformers/all-MiniLM-L6-v2
    dimension: 384
    batch_size: 16
  
  generation:
    use_api: true
    model: gpt-3.5-turbo

database:
  sqlite:
    path: ./data/financerag.db
  chromadb:
    persist_directory: ./data/chromadb
    collection_name: finance_documents

retrieval:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 10
  similarity_threshold: 0.7

api:
  host: 0.0.0.0
  port: 8000
  cors_enabled: true
  docs_enabled: true
"""
    
    with open("config_lite.yaml", "w", encoding="utf-8") as f:
        f.write(default_config)
    print("Created default config file: config_lite.yaml")


# Issue 2: Improved file validation
async def validate_upload_file_improved(file: UploadFile) -> None:
    """Enhanced file validation with better error messages"""
    
    # Constants
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
    MIN_FILE_SIZE = 100  # 100 bytes
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "message": f"File type {file_ext} not allowed",
                "allowed_types": list(ALLOWED_EXTENSIONS)
            }
        )
    
    # Read file content to check actual size
    try:
        contents = await file.read()
        file_size = len(contents)
        
        # Reset file pointer for later use
        await file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "File too large",
                    "message": f"File size {file_size / 1024 / 1024:.1f}MB exceeds maximum {MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
                    "max_size_mb": MAX_FILE_SIZE / 1024 / 1024
                }
            )
        
        if file_size < MIN_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "File too small",
                    "message": f"File size {file_size} bytes is below minimum {MIN_FILE_SIZE} bytes",
                    "min_size_bytes": MIN_FILE_SIZE
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "File validation failed",
                "message": str(e)
            }
        )


# Issue 3: Specific exception handlers
class SpecificExceptionHandlers:
    """Collection of specific exception handlers"""
    
    @staticmethod
    async def handle_database_error(e: Exception):
        """Handle database-related errors"""
        if "sqlite" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Database connection failed. Please try again later."
            )
        elif "chromadb" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Vector database unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    
    @staticmethod
    async def handle_model_error(e: Exception):
        """Handle model-related errors"""
        if "api" in str(e).lower() or "openai" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="API service unavailable. Please check your API key and try again."
            )
        elif "embedding" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail="Embedding model error. Please check model configuration."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Model error: {str(e)}"
            )
    
    @staticmethod
    async def handle_file_error(e: Exception):
        """Handle file-related errors"""
        if isinstance(e, FileNotFoundError):
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        elif isinstance(e, PermissionError):
            raise HTTPException(
                status_code=403,
                detail="Permission denied to access file"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"File processing error: {str(e)}"
            )


# Issue 4: Memory management improvements
class ImprovedMemoryManager:
    """Enhanced memory management with dynamic thresholds"""
    
    def __init__(self, base_threshold_mb: int = 1000):
        """Initialize with dynamic threshold based on available memory"""
        import psutil
        
        # Get total system memory
        total_memory_mb = psutil.virtual_memory().total / 1024 / 1024
        
        # Set threshold to 60% of available memory or base threshold, whichever is higher
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        self.threshold_mb = max(
            base_threshold_mb,
            int(available_memory_mb * 0.6)
        )
        
        self.cleanup_count = 0
        self.last_cleanup = None
        
        print(f"Memory manager initialized with threshold: {self.threshold_mb}MB")
    
    async def cleanup_if_needed(self) -> bool:
        """Perform intelligent cleanup based on memory pressure"""
        import gc
        import psutil
        from datetime import datetime
        
        memory_info = psutil.Process().memory_info()
        current_usage_mb = memory_info.rss / 1024 / 1024
        
        # Calculate memory pressure (0.0 to 1.0)
        memory_pressure = current_usage_mb / self.threshold_mb
        
        if memory_pressure > 0.8:  # High pressure
            print(f"High memory pressure detected: {memory_pressure:.2%}")
            
            # Aggressive cleanup
            gc.collect(2)  # Full collection
            
            # Clear caches if implemented
            self._clear_caches()
            
            self.cleanup_count += 1
            self.last_cleanup = datetime.now()
            
            # Check if cleanup was effective
            new_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            freed_mb = current_usage_mb - new_usage_mb
            
            print(f"Memory cleanup completed. Freed: {freed_mb:.2f}MB")
            return True
            
        elif memory_pressure > 0.6:  # Medium pressure
            # Light cleanup
            gc.collect(0)  # Quick collection
            return False
            
        return False
    
    def _clear_caches(self):
        """Clear application caches"""
        # This would clear various caches in the application
        # For example: embedding cache, query cache, etc.
        pass


# Additional fixes and improvements
class ConfigurationValidator:
    """Validate configuration on startup"""
    
    @staticmethod
    def validate_config(config_path: str) -> bool:
        """Validate configuration file"""
        import yaml
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = [
                'system',
                'models',
                'database',
                'retrieval',
                'api'
            ]
            
            for field in required_fields:
                if field not in config:
                    print(f"Warning: Missing required field '{field}' in config")
                    return False
            
            # Validate API keys if using API models
            if config.get('models', {}).get('generation', {}).get('use_api'):
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    print("Warning: OPENAI_API_KEY not set but API mode is enabled")
            
            return True
            
        except Exception as e:
            print(f"Config validation failed: {e}")
            return False


# Startup checks
def perform_startup_checks():
    """Perform all necessary startup checks"""
    checks = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        checks.append("Python 3.8+ required")
    
    # Check required directories
    required_dirs = ['data', 'logs', 'data/pdfs', 'data/chromadb']
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            checks.append(f"Created directory: {dir_path}")
    
    # Check available disk space
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024 ** 3)
    if free_gb < 1:
        checks.append(f"Warning: Low disk space ({free_gb:.1f}GB free)")
    
    return checks


if __name__ == "__main__":
    # Test the fixes
    print("Testing configuration fixes...")
    
    # Test config path
    config_path = get_config_path()
    print(f"Config path: {config_path}")
    
    # Test configuration validation
    validator = ConfigurationValidator()
    is_valid = validator.validate_config(config_path)
    print(f"Config valid: {is_valid}")
    
    # Test startup checks
    checks = perform_startup_checks()
    for check in checks:
        print(f"  - {check}")
    
    # Test memory manager
    mem_manager = ImprovedMemoryManager()
    print(f"Memory threshold: {mem_manager.threshold_mb}MB")