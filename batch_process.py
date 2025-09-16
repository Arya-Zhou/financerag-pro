"""
批量处理文档脚本
支持断点续传和批量API调用
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from dotenv import load_dotenv

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from core.document_processor import MultiModalPreprocessor
from core.entity_linker import EntityLinkingEngine
from core.retrieval_engine import MultiPathRetrievalEngine
from core.api_client import CheckpointManager

class BatchDocumentProcessor:
    """批量文档处理器"""
    
    def __init__(self, config_path: str = "config_api.yaml"):
        # 加载环境变量
        load_dotenv(".env.api")
        
        # 加载配置
        self.config = Config(config_path)
        
        # 启用批量模式
        self.config.set("batch_mode", True)
        self.config.set("performance.batch_mode", True)
        
        # 初始化组件
        self.document_processor = None
        self.entity_linker = None
        self.retrieval_engine = None
        self.checkpoint_manager = CheckpointManager("./checkpoints/batch_processing")
        
        # 批处理配置
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.max_concurrent = 3  # 最大并发处理数
    
    async def initialize(self):
        """初始化组件"""
        logger.info("Initializing batch processing components...")
        
        # 初始化文档处理器
        self.document_processor = MultiModalPreprocessor(self.config.config)
        
        # 初始化实体链接器
        self.entity_linker = EntityLinkingEngine(self.config.config)
        await self.entity_linker.initialize()
        
        # 初始化检索引擎
        self.retrieval_engine = MultiPathRetrievalEngine(
            self.config.config,
            self.entity_linker
        )
        
        logger.info("Components initialized successfully")
    
    async def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """处理单个PDF文件"""
        try:
            logger.info(f"Processing: {pdf_path}")
            
            # 文档预处理（支持断点续传）
            result = await self.document_processor.preprocess_document(pdf_path)
            
            # 实体链接
            if "text_chunks" in result:
                chunks = result["text_chunks"][:100]  # 限制处理数量
                chunk_dicts = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks
                ]
                
                # 生成文档ID
                import hashlib
                doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
                
                # 处理chunks
                await self.entity_linker.process_document_chunks(doc_id, chunk_dicts)
                
                # 索引到检索引擎
                await self.retrieval_engine.vector_retriever.index_documents(chunk_dicts)
            
            return {
                "status": "success",
                "file": pdf_path,
                "chunks": len(result.get("text_chunks", [])),
                "tables": len(result.get("tables", [])),
                "charts": len(result.get("charts", []))
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "status": "error",
                "file": pdf_path,
                "error": str(e)
            }
    
    async def process_batch(
        self,
        pdf_files: List[str],
        task_id: str = None
    ) -> List[Dict[str, Any]]:
        """批量处理PDF文件"""
        
        if not task_id:
            import hashlib
            task_id = hashlib.md5(str(pdf_files).encode()).hexdigest()[:8]
        
        # 尝试从断点恢复
        checkpoint = self.checkpoint_manager.load_checkpoint(task_id)
        if checkpoint:
            processed = checkpoint.get("processed", [])
            results = checkpoint.get("results", [])
            start_index = len(processed)
            logger.info(f"Resuming from checkpoint: {start_index}/{len(pdf_files)} processed")
        else:
            processed = []
            results = []
            start_index = 0
        
        # 处理剩余文件
        remaining_files = pdf_files[start_index:]
        
        for i in range(0, len(remaining_files), self.batch_size):
            batch = remaining_files[i:i + self.batch_size]
            
            # 限制并发数
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_with_limit(pdf_path):
                async with semaphore:
                    return await self.process_pdf(pdf_path)
            
            # 并发处理批次
            batch_tasks = [process_with_limit(pdf) for pdf in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # 更新结果
            results.extend(batch_results)
            processed.extend(batch)
            
            # 保存断点
            self.checkpoint_manager.save_checkpoint(task_id, {
                "processed": processed,
                "results": results,
                "total": len(pdf_files)
            })
            
            logger.info(f"Batch progress: {len(processed)}/{len(pdf_files)}")
            
            # 批次间休息
            if i + self.batch_size < len(remaining_files):
                await asyncio.sleep(2)
        
        # 删除断点
        self.checkpoint_manager.remove_checkpoint(task_id)
        
        return results
    
    async def cleanup(self):
        """清理资源"""
        if self.entity_linker:
            await self.entity_linker.close()
        
        if self.document_processor:
            await self.document_processor.pdf_processor.close()
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """生成处理报告"""
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        failed = total - success
        
        total_chunks = sum(r.get("chunks", 0) for r in results if r["status"] == "success")
        total_tables = sum(r.get("tables", 0) for r in results if r["status"] == "success")
        total_charts = sum(r.get("charts", 0) for r in results if r["status"] == "success")
        
        report = f"""
========================================
批量处理报告
========================================
总文件数: {total}
成功: {success}
失败: {failed}

提取统计:
- 文本块: {total_chunks}
- 表格: {total_tables}
- 图表: {total_charts}

失败文件:
"""
        
        for result in results:
            if result["status"] == "error":
                report += f"- {result['file']}: {result.get('error', 'Unknown error')}\n"
        
        return report

async def main():
    """主函数"""
    
    # 设置日志
    logger.add("batch_processing.log", rotation="100 MB")
    
    # 获取PDF文件列表
    pdf_dir = Path("./data/pdfs")  # PDF文件目录
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found in ./data/pdfs")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # 初始化处理器
    processor = BatchDocumentProcessor()
    
    try:
        # 初始化
        await processor.initialize()
        
        # 批量处理
        pdf_paths = [str(pdf) for pdf in pdf_files]
        results = await processor.process_batch(pdf_paths)
        
        # 生成报告
        report = processor.generate_report(results)
        print(report)
        
        # 保存报告
        with open("batch_processing_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info("Batch processing completed")
        
    finally:
        # 清理
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())