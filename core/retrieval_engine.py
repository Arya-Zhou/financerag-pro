import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import redis
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import json
from numpy.linalg import norm

@dataclass
class RetrievalResult:
    """Retrieval result structure"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_type: str  # entity/semantic/keyword/table/chart

@dataclass
class RetrievalContext:
    """Context for retrieval execution"""
    query: str
    strategy: str
    metadata_filters: Dict[str, Any]
    top_k: int
    similarity_threshold: float

class OptimizedVectorRetriever:
    """优化的向量检索器，使用NumPy加速计算"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = []  # 存储向量
        self.documents = []  # 存储文档
        self.embedding_matrix = None  # NumPy矩阵用于快速计算
        self.needs_rebuild = True  # 是否需要重建矩阵
        
        # 初始化嵌入模型
        model_name = config.get("models", {}).get("embedding", {}).get("model", "BAAI/bge-m3")
        self.embedding_model = SentenceTransformer(model_name)
        
        logger.info("优化向量检索器初始化成功")
    
    def add_documents(self, docs: List[Dict], embeddings: List[List[float]] = None):
        """添加文档到索引"""
        if embeddings is None:
            # 如果没有提供嵌入，生成它们
            contents = [doc.get("content", "") for doc in docs]
            embeddings = self.embedding_model.encode(contents)
        
        self.documents.extend(docs)
        self.embeddings.extend(embeddings)
        self.needs_rebuild = True
        
        logger.info(f"添加了 {len(docs)} 个文档到索引")
    
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """索引文档（兼容接口）"""
        contents = [doc.get("content", "") for doc in documents]
        embeddings = self.embedding_model.encode(contents)
        self.add_documents(documents, embeddings.tolist())
    
    def _rebuild_matrix(self):
        """重建 NumPy 矩阵以加速计算"""
        if self.embeddings and self.needs_rebuild:
            self.embedding_matrix = np.array(self.embeddings)
            # 预计算范数以加速余弦相似度计算
            self.embedding_norms = norm(self.embedding_matrix, axis=1)
            self.needs_rebuild = False
            logger.debug(f"重建嵌入矩阵，形状: {self.embedding_matrix.shape}")
    
    async def search(
        self, 
        query_embedding: List[float] = None,
        query: str = None,
        top_k: int = 10,
        threshold: float = 0.0,
        context: RetrievalContext = None
    ) -> List[Dict]:
        """快速向量搜索"""
        
        # 兼容 RetrievalContext 参数
        if context:
            query = context.query
            top_k = context.top_k
            threshold = context.similarity_threshold
        
        if not self.embeddings:
            return []
        
        # 确保矩阵是最新的
        if self.needs_rebuild:
            self._rebuild_matrix()
        
        # 生成查询向量
        if query_embedding is None:
            if query is None:
                return []
            query_embedding = self.embedding_model.encode(query)
        
        query_emb = np.array(query_embedding)
        
        # 批量计算余弦相似度
        # 优化：使用矩阵乘法代替循环
        similarities = self.embedding_matrix @ query_emb / (
            self.embedding_norms * norm(query_emb) + 1e-8
        )
        
        # 获取top_k索引
        # 优化：使用argpartition代替argsort以提高性能
        if top_k < len(similarities):
            # 使用argpartition找到top_k个最大值
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # 对这top_k个值进行排序
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            # 如果top_k大于文档数，直接排序
            top_indices = similarities.argsort()[::-1][:top_k]
        
        # 构建结果
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                # 返回 RetrievalResult 形式的结果（如果有 context）
                if context:
                    result = RetrievalResult(
                        chunk_id=self.documents[idx].get("chunk_id", str(idx)),
                        content=self.documents[idx].get("content", ""),
                        score=score,
                        metadata=self.documents[idx].get("metadata", {}),
                        source_type="semantic"
                    )
                else:
                    result = {
                        **self.documents[idx],
                        "score": score,
                        "rank": len(results) + 1
                    }
                results.append(result)
        
        logger.debug(f"向量搜索返回 {len(results)} 个结果")
        return results
    
    async def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[List[Dict]]:
        """批量向量搜索，进一步优化性能"""
        
        if not self.embeddings:
            return [[] for _ in queries]
        
        # 确保矩阵是最新的
        if self.needs_rebuild:
            self._rebuild_matrix()
        
        # 批量生成查询向量
        query_embeddings = self.embedding_model.encode(queries)
        
        # 一次性计算所有查询的相似度
        # 形状：(n_queries, n_documents)
        similarities = query_embeddings @ self.embedding_matrix.T / (
            np.outer(norm(query_embeddings, axis=1), self.embedding_norms) + 1e-8
        )
        
        # 为每个查询构建结果
        all_results = []
        for query_idx, query_sims in enumerate(similarities):
            # 获取当前查询的top_k结果
            if top_k < len(query_sims):
                top_indices = np.argpartition(query_sims, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(query_sims[top_indices])[::-1]]
            else:
                top_indices = query_sims.argsort()[::-1][:top_k]
            
            # 构建结果
            results = []
            for idx in top_indices:
                score = float(query_sims[idx])
                if score >= threshold:
                    result = {
                        **self.documents[idx],
                        "score": score,
                        "rank": len(results) + 1
                    }
                    results.append(result)
            
            all_results.append(results)
        
        logger.debug(f"批量搜索完成，处理了 {len(queries)} 个查询")
        return all_results
    
    def clear(self):
        """清空索引"""
        self.embeddings = []
        self.documents = []
        self.embedding_matrix = None
        self.needs_rebuild = True
        logger.info("向量索引已清空")

class VectorRetriever:
    """Semantic vector retrieval using ChromaDB"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config.get("database", {}).get("chromadb", {}).get("collection_name", "finance_documents")
        self.persist_directory = config.get("database", {}).get("chromadb", {}).get("persist_directory", "./data/chromadb")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        model_name = config.get("models", {}).get("embedding", {}).get("model", "BAAI/bge-m3")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        
        try:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Financial documents collection"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents into vector database"""
        
        for doc in documents:
            # Generate embedding
            embedding = self.embedding_model.encode(doc["content"])
            
            # Add to collection
            self.collection.add(
                ids=[doc["chunk_id"]],
                embeddings=[embedding.tolist()],
                metadatas=[doc.get("metadata", {})],
                documents=[doc["content"]]
            )
        
        logger.info(f"Indexed {len(documents)} documents")
    
    async def search(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Perform semantic vector search"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(context.query)
        
        # Build where clause from metadata filters
        where_clause = self._build_where_clause(context.metadata_filters)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=context.top_k,
            where=where_clause if where_clause else None
        )
        
        # Convert to RetrievalResult
        retrieval_results = []
        
        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                score = 1.0 - results["distances"][0][i] if results["distances"] else 0.0
                
                if score >= context.similarity_threshold:
                    result = RetrievalResult(
                        chunk_id=results["ids"][0][i],
                        content=results["documents"][0][i] if results["documents"] else "",
                        score=score,
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        source_type="semantic"
                    )
                    retrieval_results.append(result)
        
        logger.info(f"Vector search returned {len(retrieval_results)} results")
        
        return retrieval_results
    
    def _build_where_clause(self, metadata_filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from metadata filters"""
        
        if not metadata_filters:
            return None
        
        where_clause = {}
        
        # Add entity filters
        if "entities" in metadata_filters:
            where_clause["entities"] = {"$in": metadata_filters["entities"]}
        
        # Add temporal filters
        if "temporal_range" in metadata_filters:
            temporal = metadata_filters["temporal_range"]
            if "start" in temporal:
                where_clause["year"] = {"$gte": int(temporal["start"])}
            if "end" in temporal:
                where_clause["year"] = {"$lte": int(temporal["end"])}
        
        return where_clause if where_clause else None

class KeywordRetriever:
    """BM25-based keyword retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents = {}  # In-memory document store
        self.inverted_index = {}  # Inverted index for BM25
    
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for keyword search"""
        
        for doc in documents:
            # Store document
            self.documents[doc["chunk_id"]] = doc
            
            # Build inverted index
            tokens = self._tokenize(doc["content"])
            
            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                
                self.inverted_index[token].append({
                    "doc_id": doc["chunk_id"],
                    "freq": tokens.count(token)
                })
        
        logger.info(f"Indexed {len(documents)} documents for keyword search")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        
        import re
        
        # Convert to lowercase and split
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        
        return tokens
    
    async def search(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Perform BM25 keyword search"""
        
        query_tokens = self._tokenize(context.query)
        
        # Calculate BM25 scores
        scores = {}
        
        for token in query_tokens:
            if token in self.inverted_index:
                docs = self.inverted_index[token]
                
                # IDF calculation
                N = len(self.documents)
                df = len(docs)
                idf = np.log((N - df + 0.5) / (df + 0.5))
                
                for doc_info in docs:
                    doc_id = doc_info["doc_id"]
                    tf = doc_info["freq"]
                    
                    # BM25 parameters
                    k1 = 1.2
                    b = 0.75
                    avgdl = np.mean([len(self._tokenize(d["content"])) for d in self.documents.values()])
                    dl = len(self._tokenize(self.documents[doc_id]["content"]))
                    
                    # BM25 score
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                    
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += score
        
        # Sort by score and get top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:context.top_k]
        
        # Convert to RetrievalResult
        results = []
        for doc_id, score in sorted_docs:
            if score >= context.similarity_threshold:
                doc = self.documents[doc_id]
                result = RetrievalResult(
                    chunk_id=doc_id,
                    content=doc["content"],
                    score=score,
                    metadata=doc.get("metadata", {}),
                    source_type="keyword"
                )
                results.append(result)
        
        logger.info(f"Keyword search returned {len(results)} results")
        
        return results

class EntityRetriever:
    """Entity-based metadata retrieval"""
    
    def __init__(self, config: Dict[str, Any], entity_linking_engine):
        self.config = config
        self.entity_engine = entity_linking_engine
    
    async def search(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Perform entity-based retrieval"""
        
        # Extract entities from query
        query_entities = context.metadata_filters.get("entities", [])
        
        if not query_entities:
            # Try to extract entities from query text
            from core.entity_linker import EntityLinker, FinancialEntityKnowledgeBase
            kb = FinancialEntityKnowledgeBase()
            linker = EntityLinker(kb)
            
            entity_mentions = await linker.link_entities(context.query)
            query_entities = [e.entity_id for e in entity_mentions]
        
        if not query_entities:
            return []
        
        # Search by entities
        results = await self.entity_engine.search_by_entities(query_entities)
        
        # Convert to RetrievalResult
        retrieval_results = []
        for result in results[:context.top_k]:
            retrieval_result = RetrievalResult(
                chunk_id=result["chunk_id"],
                content=result["content"],
                score=1.0,  # Entity match is binary
                metadata=result.get("metadata", {}),
                source_type="entity"
            )
            retrieval_results.append(retrieval_result)
        
        logger.info(f"Entity search returned {len(retrieval_results)} results")
        
        return retrieval_results

class TableRetriever:
    """Table-specific retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tables = {}  # In-memory table store
    
    async def index_tables(self, tables: List[Dict[str, Any]]):
        """Index tables for retrieval"""
        
        for table in tables:
            self.tables[table["table_id"]] = table
        
        logger.info(f"Indexed {len(tables)} tables")
    
    async def search(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Search in tables"""
        
        results = []
        
        # Search for relevant tables
        for table_id, table in self.tables.items():
            relevance_score = self._calculate_table_relevance(table, context)
            
            if relevance_score >= context.similarity_threshold:
                # Convert table to text representation
                table_text = self._table_to_text(table)
                
                result = RetrievalResult(
                    chunk_id=table_id,
                    content=table_text,
                    score=relevance_score,
                    metadata=table.get("metadata", {}),
                    source_type="table"
                )
                results.append(result)
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Table search returned {len(results[:context.top_k])} results")
        
        return results[:context.top_k]
    
    def _calculate_table_relevance(self, table: Dict[str, Any], context: RetrievalContext) -> float:
        """Calculate relevance score for table"""
        
        score = 0.0
        query_lower = context.query.lower()
        
        # Check table type
        if table.get("table_type") in query_lower:
            score += 0.3
        
        # Check metrics
        if "metrics" in table:
            for metric in table["metrics"]:
                if metric.lower() in query_lower:
                    score += 0.2
        
        # Check periods
        if "periods" in table:
            for period in table["periods"]:
                if str(period) in context.query:
                    score += 0.2
        
        # Check metadata filters
        if context.metadata_filters:
            if "entities" in context.metadata_filters:
                for entity in context.metadata_filters["entities"]:
                    if entity in str(table):
                        score += 0.3
        
        return min(score, 1.0)
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table to text representation"""
        
        text_parts = []
        
        # Add table type
        if "table_type" in table:
            text_parts.append(f"Table Type: {table['table_type']}")
        
        # Add metrics
        if "metrics" in table:
            text_parts.append(f"Metrics: {', '.join(table['metrics'].keys())}")
            
            # Add metric values
            for metric, values in table["metrics"].items():
                text_parts.append(f"{metric}: {values}")
        
        # Add original data if available
        if "original_data" in table:
            text_parts.append(f"Data: {json.dumps(table['original_data'], ensure_ascii=False)}")
        
        return "\n".join(text_parts)

class ChartRetriever:
    """Chart-specific retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.charts = {}  # In-memory chart store
    
    async def index_charts(self, charts: List[Dict[str, Any]]):
        """Index charts for retrieval"""
        
        for chart in charts:
            self.charts[chart["chart_id"]] = chart
        
        logger.info(f"Indexed {len(charts)} charts")
    
    async def search(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Search in charts"""
        
        results = []
        
        # Search for relevant charts
        for chart_id, chart in self.charts.items():
            relevance_score = self._calculate_chart_relevance(chart, context)
            
            if relevance_score >= context.similarity_threshold:
                # Convert chart to text representation
                chart_text = self._chart_to_text(chart)
                
                result = RetrievalResult(
                    chunk_id=chart_id,
                    content=chart_text,
                    score=relevance_score,
                    metadata=chart.get("metadata", {}),
                    source_type="chart"
                )
                results.append(result)
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Chart search returned {len(results[:context.top_k])} results")
        
        return results[:context.top_k]
    
    def _calculate_chart_relevance(self, chart: Dict[str, Any], context: RetrievalContext) -> float:
        """Calculate relevance score for chart"""
        
        score = 0.0
        query_lower = context.query.lower()
        
        # Check chart title
        if "title" in chart and chart["title"].lower() in query_lower:
            score += 0.4
        
        # Check chart type
        if "chart_type" in chart and chart["chart_type"] in query_lower:
            score += 0.2
        
        # Check axis labels
        if "x_axis" in chart and chart["x_axis"].get("name", "").lower() in query_lower:
            score += 0.2
        if "y_axis" in chart and chart["y_axis"].get("name", "").lower() in query_lower:
            score += 0.2
        
        # Check insights
        if "insights" in chart and any(word in chart["insights"].lower() for word in query_lower.split()):
            score += 0.3
        
        return min(score, 1.0)
    
    def _chart_to_text(self, chart: Dict[str, Any]) -> str:
        """Convert chart to text representation"""
        
        text_parts = []
        
        # Add chart info
        if "chart_type" in chart:
            text_parts.append(f"Chart Type: {chart['chart_type']}")
        
        if "title" in chart:
            text_parts.append(f"Title: {chart['title']}")
        
        # Add axis info
        if "x_axis" in chart:
            x_axis = chart["x_axis"]
            text_parts.append(f"X-Axis: {x_axis.get('name', '')} ({x_axis.get('unit', '')})")
            if "values" in x_axis:
                text_parts.append(f"X Values: {x_axis['values']}")
        
        if "y_axis" in chart:
            y_axis = chart["y_axis"]
            text_parts.append(f"Y-Axis: {y_axis.get('name', '')} ({y_axis.get('unit', '')})")
            if "values" in y_axis:
                text_parts.append(f"Y Values: {y_axis['values']}")
        
        # Add data series
        if "data_series" in chart:
            for series in chart["data_series"]:
                text_parts.append(f"Series: {series.get('name', '')}")
                if "data_points" in series:
                    text_parts.append(f"Data Points: {series['data_points']}")
        
        # Add insights
        if "insights" in chart:
            text_parts.append(f"Insights: {chart['insights']}")
        
        return "\n".join(text_parts)

class HybridRetriever:
    """Hybrid retrieval combining multiple strategies"""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        entity_retriever: EntityRetriever,
        table_retriever: TableRetriever,
        chart_retriever: ChartRetriever
    ):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.entity_retriever = entity_retriever
        self.table_retriever = table_retriever
        self.chart_retriever = chart_retriever
    
    async def search(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Perform hybrid search combining multiple strategies"""
        
        # Execute searches in parallel
        tasks = [
            self.vector_retriever.search(context),
            self.keyword_retriever.search(context),
            self.entity_retriever.search(context)
        ]
        
        # Add table and chart search if needed
        if "table" in context.metadata_filters.get("modalities", []):
            tasks.append(self.table_retriever.search(context))
        
        if "chart" in context.metadata_filters.get("modalities", []):
            tasks.append(self.chart_retriever.search(context))
        
        results_lists = await asyncio.gather(*tasks)
        
        # Merge and re-rank results
        merged_results = self._merge_results(results_lists)
        
        logger.info(f"Hybrid search returned {len(merged_results)} results")
        
        return merged_results
    
    def _merge_results(self, results_lists: List[List[RetrievalResult]]) -> List[RetrievalResult]:
        """Merge and re-rank results from multiple sources"""
        
        # Combine all results
        all_results = {}
        
        for results in results_lists:
            for result in results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                else:
                    # Combine scores with weighted average
                    existing = all_results[result.chunk_id]
                    
                    # Weight by source type
                    weights = {
                        "entity": 1.2,
                        "semantic": 1.0,
                        "keyword": 0.8,
                        "table": 1.1,
                        "chart": 0.9
                    }
                    
                    existing_weight = weights.get(existing.source_type, 1.0)
                    new_weight = weights.get(result.source_type, 1.0)
                    
                    combined_score = (
                        existing.score * existing_weight + result.score * new_weight
                    ) / (existing_weight + new_weight)
                    
                    existing.score = combined_score
        
        # Sort by score
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        return sorted_results

class RetrievalCache:
    """Cache for retrieval results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_config = config.get("database", {}).get("redis", {})
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0),
            password=self.redis_config.get("password", None),
            decode_responses=True
        )
        
        self.ttl = config.get("cache", {}).get("ttl", 3600)
    
    async def get(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """Get cached results"""
        
        try:
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                results_data = json.loads(cached_data)
                
                # Convert back to RetrievalResult objects
                results = []
                for data in results_data:
                    result = RetrievalResult(
                        chunk_id=data["chunk_id"],
                        content=data["content"],
                        score=data["score"],
                        metadata=data["metadata"],
                        source_type=data["source_type"]
                    )
                    results.append(result)
                
                logger.info(f"Cache hit for key: {cache_key}")
                return results
        
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
        
        return None
    
    async def set(self, cache_key: str, results: List[RetrievalResult]):
        """Set cache results"""
        
        try:
            # Convert to JSON-serializable format
            results_data = [asdict(r) for r in results]
            
            self.redis_client.setex(
                cache_key,
                self.ttl,
                json.dumps(results_data, ensure_ascii=False)
            )
            
            logger.info(f"Cached results for key: {cache_key}")
        
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

class MultiPathRetrievalEngine:
    """Main multi-path retrieval execution engine"""
    
    def __init__(self, config: Dict[str, Any], entity_linking_engine):
        self.config = config
        
        # Initialize retrievers
        self.vector_retriever = VectorRetriever(config)
        self.keyword_retriever = KeywordRetriever(config)
        self.entity_retriever = EntityRetriever(config, entity_linking_engine)
        self.table_retriever = TableRetriever(config)
        self.chart_retriever = ChartRetriever(config)
        
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever,
            self.keyword_retriever,
            self.entity_retriever,
            self.table_retriever,
            self.chart_retriever
        )
        
        # Initialize cache
        self.cache = RetrievalCache(config) if config.get("cache", {}).get("enabled", True) else None
    
    async def execute_retrieval(
        self,
        query: str,
        routing_decision: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Execute retrieval based on routing decision"""
        
        # Create retrieval context
        context = RetrievalContext(
            query=query,
            strategy=routing_decision["primary_strategy"],
            metadata_filters=routing_decision.get("metadata_filters", {}),
            top_k=self.config.get("retrieval", {}).get("top_k", 20),
            similarity_threshold=self.config.get("retrieval", {}).get("similarity_threshold", 0.7)
        )
        
        # Check cache
        cache_key = self._generate_cache_key(context)
        if self.cache:
            cached_results = await self.cache.get(cache_key)
            if cached_results:
                return cached_results
        
        # Execute retrieval based on strategy
        strategy = routing_decision["primary_strategy"]
        
        if strategy == "entity_search":
            results = await self.entity_retriever.search(context)
        elif strategy == "semantic_search":
            results = await self.vector_retriever.search(context)
        elif strategy == "keyword_match":
            results = await self.keyword_retriever.search(context)
        elif strategy == "table_search":
            results = await self.table_retriever.search(context)
        elif strategy == "chart_search":
            results = await self.chart_retriever.search(context)
        elif strategy == "hybrid_search":
            results = await self.hybrid_retriever.search(context)
        else:
            # Default to hybrid
            results = await self.hybrid_retriever.search(context)
        
        # Cache results
        if self.cache and results:
            await self.cache.set(cache_key, results)
        
        return results
    
    async def parallel_execute(
        self,
        routing_decisions: List[Dict[str, Any]]
    ) -> List[List[RetrievalResult]]:
        """Execute multiple retrievals in parallel"""
        
        tasks = []
        
        for decision in routing_decisions:
            task = self.execute_retrieval(
                decision["query"],
                decision["routing"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Parallel execution completed: {len(results)} queries")
        
        return results
    
    def _generate_cache_key(self, context: RetrievalContext) -> str:
        """Generate cache key from context"""
        
        import hashlib
        
        key_parts = [
            context.query,
            context.strategy,
            json.dumps(context.metadata_filters, sort_keys=True),
            str(context.top_k),
            str(context.similarity_threshold)
        ]
        
        key_string = "|".join(key_parts)
        
        return hashlib.md5(key_string.encode()).hexdigest()