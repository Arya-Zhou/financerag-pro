"""
批量处理文档脚本
支持断点续传和批量API调用
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from core.document_processor import MultiModalPreprocessor
from core.entity_linker import EntityLinkingEngine
from core.retrieval_engine import MultiPathRetrievalEngine
from core.api_client import CheckpointManager
from core.config_manager import get_config_manager

class BatchDocumentProcessor:
    """批量文档处理器"""
    
    def __init__(self, config_path: str = "configs/production.yaml"):
        # 使用统一配置管理器
        self.config_manager = get_config_manager()
        
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
        
        # 批处理配置（从配置管理器获取）
        self.batch_size = self.config_manager.get_batch_size()
        self.max_concurrent = self.config_manager.get_max_concurrent()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'error': 0,
            'start_time': None,
            'total_chunks': 0,
            'total_tables': 0,
            'total_charts': 0
        }
    
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
    
    async def process_pdf(self, pdf_path: str, pbar: tqdm = None) -> Dict[str, Any]:
        """处理单个PDF文件"""
        async with self.semaphore:
            try:
                if pbar:
                    pbar.set_description(f"处理中: {os.path.basename(pdf_path)[:30]}...")
            
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
            
                chunks_count = len(result.get("text_chunks", []))
                tables_count = len(result.get("tables", []))
                charts_count = len(result.get("charts", []))
                
                # 更新统计
                self.stats['success'] += 1
                self.stats['total_chunks'] += chunks_count
                self.stats['total_tables'] += tables_count
                self.stats['total_charts'] += charts_count
                
                if pbar:
                    pbar.set_postfix({
                        '成功': self.stats['success'],
                        '失败': self.stats['error'],
                        '块': self.stats['total_chunks']
                    })
                
                return {
                    "status": "success",
                    "file": pdf_path,
                    "chunks": chunks_count,
                    "tables": tables_count,
                    "charts": charts_count
                }
                
            except Exception as e:
                self.stats['error'] += 1
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                
                if pbar:
                    pbar.set_postfix({
                        '成功': self.stats['success'],
                        '失败': self.stats['error'],
                        '块': self.stats['total_chunks']
                    })
                
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
        
        # 初始化统计
        self.stats['total'] = len(pdf_files)
        self.stats['start_time'] = time.time()
        
        # 尝试从断点恢复
        checkpoint = self.checkpoint_manager.load_checkpoint(task_id)
        if checkpoint:
            processed = checkpoint.get("processed", [])
            results = checkpoint.get("results", [])
            start_index = len(processed)
            
            # 恢复统计信息
            for r in results:
                if r["status"] == "success":
                    self.stats['success'] += 1
                    self.stats['total_chunks'] += r.get("chunks", 0)
                    self.stats['total_tables'] += r.get("tables", 0)
                    self.stats['total_charts'] += r.get("charts", 0)
                else:
                    self.stats['error'] += 1
            
            logger.info(f"从断点恢复: {start_index}/{len(pdf_files)} 已处理")
        else:
            processed = []
            results = []
            start_index = 0
        
        # 处理剩余文件
        remaining_files = pdf_files[start_index:]
        
        # 创建总进度条
        with tqdm(total=len(pdf_files), initial=start_index, desc="批量处理PDF") as pbar:
            # 显示初始统计
            pbar.set_postfix({
                '成功': self.stats['success'],
                '失败': self.stats['error'],
                '块': self.stats['total_chunks']
            })
            
            for i in range(0, len(remaining_files), self.batch_size):
                batch = remaining_files[i:i + self.batch_size]
                
                # 创建批次任务
                batch_tasks = []
                for pdf_path in batch:
                    batch_tasks.append(self.process_pdf(pdf_path, pbar))
                
                # 并发处理批次
                batch_results = await asyncio.gather(*batch_tasks)
                
                # 更新结果
                results.extend(batch_results)
                processed.extend(batch)
                
                # 更新进度条
                pbar.update(len(batch))
                
                # 保存断点
                self.checkpoint_manager.save_checkpoint(task_id, {
                    "processed": processed,
                    "results": results,
                    "total": len(pdf_files)
                })
                
                # 批次间休息
                if i + self.batch_size < len(remaining_files):
                    await asyncio.sleep(2)
            
            # 显示最终统计
            elapsed = time.time() - self.stats['start_time']
            pbar.set_postfix({
                '成功': self.stats['success'],
                '失败': self.stats['error'],
                '块': self.stats['total_chunks'],
                '耗时': f"{elapsed:.1f}s",
                '速度': f"{self.stats['success']/max(elapsed, 1):.2f}个/秒"
            })
        
        # 删除断点
        self.checkpoint_manager.remove_checkpoint(task_id)
        
        return results
    
    async def cleanup(self):
        """清理资源"""
        if self.entity_linker:
            await self.entity_linker.close()
        
        if self.document_processor:
            await self.document_processor.pdf_processor.close()
    
    def get_progress_info(self) -> Dict[str, Any]:
        """获取进度信息"""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        return {
            'total': self.stats['total'],
            'success': self.stats['success'],
            'error': self.stats['error'],
            'success_rate': self.stats['success'] / max(self.stats['total'], 1),
            'elapsed_time': elapsed,
            'avg_time_per_doc': elapsed / max(self.stats['success'], 1),
            'total_chunks': self.stats['total_chunks'],
            'total_tables': self.stats['total_tables'],
            'total_charts': self.stats['total_charts']
        }
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """生成处理报告"""
        progress_info = self.get_progress_info()
        total = len(results)
        success = progress_info['success']
        failed = progress_info['error']
        
        total_chunks = progress_info['total_chunks']
        total_tables = progress_info['total_tables']
        total_charts = progress_info['total_charts']
        
        report = f"""
========================================
批量处理报告
========================================
总文件数: {total}
成功: {success} ({progress_info['success_rate']:.1%})
失败: {failed}

处理性能:
- 总耗时: {progress_info['elapsed_time']:.1f} 秒
- 平均速度: {progress_info['avg_time_per_doc']:.2f} 秒/文档
- 处理速率: {success/max(progress_info['elapsed_time'], 1):.2f} 文档/秒

提取统计:
- 文本块: {total_chunks}
- 表格: {total_tables}
- 图表: {total_charts}

配置信息:
- 批处理大小: {self.batch_size}
- 最大并发数: {self.max_concurrent}
- API提供商: {', '.join(self.config_manager.get_available_providers())}

失败文件:
"""
        
        for result in results:
            if result["status"] == "error":
                report += f"- {result['file']}: {result.get('error', 'Unknown error')}\n"
        
        return report

async def main():
    """主函数"""
    
    # 验证配置
    try:
        config_manager = get_config_manager()
        config_manager.print_config_summary()
    except Exception as e:
        logger.error(f"配置验证失败: {str(e)}")
        return
    
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