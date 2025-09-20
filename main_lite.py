"""
Lightweight main application for Cloud Studio environment
No Docker, PostgreSQL, or Redis required
Integrated with ConfigManager for unified configuration management
Enhanced with production-ready optimizations
"""

# SQLite兼容性修复 - 必须在导入其他模块之前
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
    print("✓ SQLite兼容性修复已应用")
except ImportError:
    print("⚠ pysqlite3未安装，使用系统SQLite（可能遇到版本问题）")

import asyncio
import os
import sys
import time
import gc
import psutil
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uvicorn
from loguru import logger
import tempfile
import shutil
import hashlib

# Import lightweight components
from configs.config import Config
from core.config_manager import get_new_config_manager  # 使用新的配置管理器
from core.new_api_client import get_text_client, get_vision_client, get_generate_client, close_all_clients  # 新的API客户端
from core.query_engine import QueryRoutingEngine
from core.document_processor import MultiModalPreprocessor
from core.lightweight_storage import LightweightMetadataManager, InMemoryMetadataManager
from core.retrieval_engine import MultiPathRetrievalEngine, OptimizedVectorRetriever
from core.validation_engine import ValidationEngine
from core.inference_engine import InferenceEngine, InferenceContext

# Request/Response Models (same as original)
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: Optional[int] = Field(10, description="Number of results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Answer confidence score")
    sources: List[str] = Field(..., description="Source citations")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")

# =========================
# Production Optimizations
# =========================

# File upload validation constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
MIN_FILE_SIZE = 100  # 100 bytes minimum

# Setup enhanced logging
# Configuration path utilities (from main_lite_fixes.py)
def get_config_path() -> str:
    """Get the appropriate config file path"""
    config_files = [
        "configs/lite.yaml",
        "configs/production.yaml",
        "config_lite.yaml",  # legacy support
        "config.yaml", 
        "config_api.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            return config_file
    
    # Create default config if none exists
    create_default_config()
    return "configs/lite.yaml"

def create_default_config():
    """Create default configuration file"""
    os.makedirs("configs", exist_ok=True)
    
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

performance:
  max_workers: 4
  batch_processing: true
  async_enabled: true
  timeout: 30
"""
    
    with open("configs/lite.yaml", "w", encoding="utf-8") as f:
        f.write(default_config)
    logger.info("Created default configuration: configs/lite.yaml")

def setup_logging():
    """Setup production-ready logging configuration"""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Console logger with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # File logger for all logs
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )
    
    # Separate error logger
    logger.add(
        "logs/error.log",
        rotation="10 MB",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}\n{exception}"
    )
    
    # Performance logger
    logger.add(
        "logs/performance.log",
        rotation="10 MB",
        retention="3 days",
        level="INFO",
        filter=lambda record: "performance" in record["extra"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}"
    )

# Initialize logging
setup_logging()

# Performance tracking context manager
@asynccontextmanager
async def track_performance(operation_name: str):
    """Track performance of operations"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.bind(performance=True).info(
            f"{operation_name} | Duration: {duration:.3f}s | Memory Δ: {memory_delta:.2f}MB | Final: {end_memory:.2f}MB"
        )

# Performance Monitoring Middleware
class PerformanceMiddleware:
    """Middleware to track request performance"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.request_count = 0
        self.total_duration = 0
        self.slow_requests = []
    
    async def __call__(self, request: Request, call_next):
        """Process request and track performance"""
        self.request_count += 1
        start_time = time.time()
        
        # Add request ID
        request_id = hashlib.md5(f"{time.time()}{self.request_count}".encode()).hexdigest()[:8]
        
        logger.info(f"Request [{request_id}] started: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            self.total_duration += duration
            
            # Track slow requests (> 2 seconds)
            if duration > 2.0:
                self.slow_requests.append({
                    "path": str(request.url.path),
                    "method": request.method,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                })
                logger.warning(f"Slow request [{request_id}]: {duration:.3f}s")
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(duration)
            
            logger.info(f"Request [{request_id}] completed: {response.status_code} in {duration:.3f}s")
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request [{request_id}] failed after {duration:.3f}s: {str(e)}")
            raise
    
    def get_stats(self):
        """Get performance statistics"""
        avg_duration = self.total_duration / max(self.request_count, 1)
        return {
            "total_requests": self.request_count,
            "average_duration": avg_duration,
            "total_duration": self.total_duration,
            "slow_requests": len(self.slow_requests),
            "recent_slow_requests": self.slow_requests[-10:]  # Last 10 slow requests
        }

# File validation function
def validate_upload_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (estimate from content-length header)
    if hasattr(file, 'size') and file.size:
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        if file.size < MIN_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too small. Minimum size: {MIN_FILE_SIZE} bytes"
            )

# Memory management utilities
class MemoryManager:
    """Memory management and optimization"""
    
    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self.cleanup_count = 0
        self.last_cleanup = datetime.now()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    async def cleanup_if_needed(self) -> bool:
        """Perform cleanup if memory usage is high"""
        memory_stats = self.get_memory_usage()
        
        if memory_stats["rss_mb"] > self.threshold_mb:
            logger.warning(f"High memory usage detected: {memory_stats['rss_mb']:.2f}MB")
            
            # Force garbage collection
            gc.collect()
            
            self.cleanup_count += 1
            self.last_cleanup = datetime.now()
            
            # Get new stats
            new_stats = self.get_memory_usage()
            freed = memory_stats["rss_mb"] - new_stats["rss_mb"]
            
            logger.info(f"Memory cleanup completed. Freed: {freed:.2f}MB")
            return True
        
        return False
    
    async def periodic_cleanup(self, interval_minutes: int = 30):
        """Periodic memory cleanup task"""
        while True:
            await asyncio.sleep(interval_minutes * 60)
            
            try:
                await self.cleanup_if_needed()
                
                # Also clear caches if implemented
                logger.debug("Periodic cleanup completed")
                
            except Exception as e:
                logger.error(f"Error during periodic cleanup: {str(e)}")

# Global memory manager instance
memory_manager = MemoryManager(threshold_mb=800)

# Lightweight Entity Linking Engine
class LightweightEntityLinkingEngine:
    """Lightweight entity linking engine using SQLite/in-memory storage"""
    
    def __init__(self, config: Dict[str, Any], use_sqlite: bool = True):
        self.config = config
        
        # Import the original entity linking components
        from core.entity_linker import FinancialEntityKnowledgeBase, EntityLinker, RelationExtractor
        
        self.kb = FinancialEntityKnowledgeBase()
        self.linker = EntityLinker(self.kb)
        self.extractor = RelationExtractor(self.kb)
        
        # Use lightweight storage
        if use_sqlite:
            self.metadata_manager = LightweightMetadataManager(config)
        else:
            self.metadata_manager = InMemoryMetadataManager(config)
    
    async def initialize(self):
        """Initialize the engine"""
        await self.metadata_manager.connect()
    
    async def process_document_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process document chunks with entity linking"""
        
        processed_chunks = []
        
        for chunk in chunks:
            # Extract entities
            entities = await self.linker.link_entities(chunk["content"])
            
            # Extract relations
            relations = await self.extractor.extract_relations(
                chunk["content"],
                entities
            )
            
            # Store metadata
            await self.metadata_manager.store_chunk_metadata(
                chunk_id=chunk["chunk_id"],
                document_id=document_id,
                content=chunk["content"],
                entities=entities,
                relations=relations,
                metadata=chunk.get("metadata", {}),
                embedding=chunk.get("embedding")
            )
            
            processed_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "entities": entities,
                "relations": relations
            })
        
        logger.info(f"Processed {len(processed_chunks)} chunks with entity linking")
        
        return processed_chunks
    
    async def search_by_entities(self, query_entities: List[str]) -> List[Dict[str, Any]]:
        """Search chunks by entities"""
        
        # Map query entities to entity IDs
        entity_ids = []
        for query_entity in query_entities:
            entity = self.kb.search_entity(query_entity)
            if entity:
                entity_ids.append(entity.entity_id)
        
        if not entity_ids:
            return []
        
        # Query database
        results = await self.metadata_manager.query_by_entities(entity_ids)
        
        return results
    
    async def close(self):
        """Close the engine"""
        await self.metadata_manager.close()

# Lightweight Cache
class LightweightCache:
    """Simple in-memory cache using cachetools"""
    
    def __init__(self, max_size: int = 1000):
        from cachetools import TTLCache
        self.cache = TTLCache(maxsize=max_size, ttl=3600)  # 1 hour TTL
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = value

class LightweightFinanceRAG:
    """Lightweight FinanceRAG application for Cloud Studio"""
    
    def __init__(self, config_path: str = "config_lite.yaml"):
        # 初始化配置管理器
        try:
            self.config_manager = get_config_manager()
            self.config_manager.print_config_summary()
        except Exception as e:
            logger.warning(f"配置管理器初始化失败: {e}")
            self.config_manager = None
        
        # Load configuration
        self.config = Config(config_path)
        
        # 从配置管理器获取系统配置
        if self.config_manager:
            system_config = self.config_manager.get_system_config()
            # 更新配置
            self.config.set("batch_mode", system_config.get("batch_mode", False))
            self.config.set("batch_size", system_config.get("batch_size", 10))
            self.config.set("max_concurrent", system_config.get("max_concurrent", 5))
        
        # Initialize components
        self.query_engine = None
        self.document_processor = None
        self.entity_linker = None
        self.retrieval_engine = None
        self.validation_engine = None
        self.inference_engine = None
        self.cache = LightweightCache()
        self.optimized_retriever = None  # 优化的向量检索器
        
        # Performance tracking
        self.performance_middleware = None
        self.startup_time = None
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with enhanced features"""
        
        app = FastAPI(
            title="FinanceRAG-Pro Lite",
            description="Lightweight Financial RAG System for Cloud Studio",
            version="1.0.0-lite-enhanced",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add performance monitoring middleware
        self.performance_middleware = PerformanceMiddleware(app)
        app.middleware("http")(self.performance_middleware)
        
        # Setup routes
        self._setup_routes(app)
        
        return app
    
    def _setup_routes(self, app: FastAPI):
        """Setup API routes"""
        
        @app.on_event("startup")
        async def startup_event():
            """Initialize components on startup"""
            await self.initialize_components()
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            await self.cleanup_components()
        
        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "status": "healthy",
                "version": "1.0.0-lite",
                "environment": "Cloud Studio",
                "timestamp": datetime.now().isoformat()
            }
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy"}
        
        @app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            """Process user query"""
            
            # Check cache first
            cache_key = f"query_{hash(request.query)}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
            
            # Process query
            result = await self.handle_query(request)
            
            # Cache result
            await self.cache.set(cache_key, result)
            
            return result
        
        @app.post("/upload", response_model=DocumentUploadResponse)
        async def upload_document(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...)
        ):
            """Upload and process document"""
            return await self.handle_upload(file, background_tasks)
        
        @app.get("/documents")
        async def list_documents():
            """List indexed documents"""
            # Simple implementation for lightweight version
            return {"documents": [], "message": "Document listing not implemented in lite version"}
    
    async def initialize_components(self):
        """初始化所有组件"""
        
        logger.info("Initializing FinanceRAG-Pro Lite components...")
        
        try:
            # 初始化查询引擎
            self.query_engine = QueryRoutingEngine(self.config.config)
            
            # 初始化文档处理器
            self.document_processor = MultiModalPreprocessor(self.config.config)
            
            # 初始化轻量级实体链接器
            self.entity_linker = LightweightEntityLinkingEngine(
                self.config.config,
                use_sqlite=True  # 使用SQLite持久化
            )
            await self.entity_linker.initialize()
            
            # 初始化优化的向量检索器
            self.optimized_retriever = OptimizedVectorRetriever(self.config.config)
            logger.info("优化向量检索器初始化成功")
            
            # 初始化检索引擎（简化版）
            self.retrieval_engine = MultiPathRetrievalEngine(
                self.config.config,
                self.entity_linker
            )
            
            # 如果配置管理器可用，显示可用的API提供商
            if self.config_manager:
                providers = self.config_manager.get_available_providers()
                logger.info(f"可用的API提供商: {', '.join(providers)}")
            
            # 初始化验证引擎
            self.validation_engine = ValidationEngine(self.config.config)
            
            # 初始化推理引擎
            self.inference_engine = InferenceEngine(self.config.config)
            await self.inference_engine.initialize()
            
            logger.info("所有组件初始化成功")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {str(e)}")
            logger.warning("运行在降级模式 - 某些功能可能无法工作")
    
    async def cleanup_components(self):
        """Cleanup components"""
        
        logger.info("Cleaning up components...")
        
        if self.entity_linker:
            await self.entity_linker.close()
        
        logger.info("Cleanup completed")
    
    async def handle_query(self, request: QueryRequest) -> QueryResponse:
        """Handle query request"""
        
        try:
            logger.info(f"Processing query: {request.query}")
            
            # Simplified query processing for lightweight version
            
            # Step 1: Basic query understanding (without full LLM if no API key)
            try:
                routing_result = await self.query_engine.process_query(request.query)
            except Exception as e:
                logger.warning(f"Query routing failed: {e}, using default")
                # Fallback to simple routing
                routing_result = {
                    "original_query": request.query,
                    "analysis": {
                        "query_type": "factual",
                        "key_entities": [],
                        "sub_queries": [request.query],
                        "required_modalities": ["text"],
                        "expected_output": "summary"
                    },
                    "routing_decisions": [{
                        "query": request.query,
                        "routing": {
                            "primary_strategy": "semantic_search",
                            "metadata_filters": {}
                        }
                    }]
                }
            
            # Step 2: 简化的检索
            retrieval_results = []
            
            # 尝试使用优化的向量检索器
            if self.optimized_retriever:
                try:
                    results = await self.optimized_retriever.search(
                        query=request.query,
                        top_k=request.top_k,
                        threshold=0.5
                    )
                    retrieval_results = results
                except Exception as e:
                    logger.warning(f"优化检索失败，回退到标准检索: {e}")
            
            # 如果优化检索器不可用或失败，使用标准检索
            if not retrieval_results:
                from core.retrieval_engine import RetrievalContext
                
                context = RetrievalContext(
                    query=request.query,
                    strategy="semantic_search",
                    metadata_filters={},
                    top_k=request.top_k,
                    similarity_threshold=0.5
                )
                
                try:
                    results = await self.retrieval_engine.vector_retriever.search(context)
                    retrieval_results = [
                        {
                            "chunk_id": r.chunk_id,
                            "content": r.content,
                            "score": r.score,
                            "metadata": r.metadata,
                            "source": r.source_type
                        }
                        for r in results
                    ]
                except Exception as e:
                    logger.warning(f"检索失败: {e}")
                retrieval_results = []
            
            # Step 3: Generate answer (simplified)
            if retrieval_results:
                answer_content = f"Based on the available information:\n\n"
                answer_content += retrieval_results[0]["content"][:500]
                confidence = retrieval_results[0]["score"]
            else:
                answer_content = "I couldn't find specific information to answer your question."
                confidence = 0.0
            
            return QueryResponse(
                answer=answer_content,
                confidence=confidence,
                sources=[],
                metadata={
                    "query_type": routing_result["analysis"].get("query_type", "unknown"),
                    "processing_time": datetime.now().isoformat(),
                    "version": "lite"
                }
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_upload(
        self,
        file: UploadFile,
        background_tasks: BackgroundTasks
    ) -> DocumentUploadResponse:
        """Handle document upload"""
        
        try:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported"
                )
            
            # Check file size (limit to 10MB for lite version)
            if file.size > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail="File size exceeds 10MB limit"
                )
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            # Generate document ID
            doc_id = hashlib.md5(file.filename.encode()).hexdigest()
            
            # Process document in background
            background_tasks.add_task(
                self.process_document_background,
                doc_id,
                tmp_path,
                file.filename
            )
            
            return DocumentUploadResponse(
                document_id=doc_id,
                status="processing",
                message="Document uploaded and processing started",
                metadata={
                    "filename": file.filename,
                    "size": file.size,
                    "upload_time": datetime.now().isoformat()
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_document_background(
        self,
        document_id: str,
        file_path: str,
        filename: str
    ):
        """Process document in background"""
        
        try:
            logger.info(f"Processing document: {filename}")
            
            # Simplified document processing for lite version
            processed_data = await self.document_processor.preprocess_document(file_path)
            
            # Process chunks
            chunks = processed_data.get("text_chunks", [])
            chunk_dicts = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
                for chunk in chunks[:100]  # Limit to first 100 chunks for lite version
            ]
            
            # Entity linking
            await self.entity_linker.process_document_chunks(document_id, chunk_dicts)
            
            # Basic indexing
            await self.retrieval_engine.vector_retriever.index_documents(chunk_dicts)
            
            # Clean up temp file
            os.remove(file_path)
            
            logger.info(f"Document processing completed: {filename}")
            
        except Exception as e:
            logger.error(f"Background processing failed: {str(e)}")
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)

# Create application instance
app_instance = LightweightFinanceRAG()
app = app_instance.app

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )