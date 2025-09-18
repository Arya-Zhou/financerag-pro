"""
Unit tests for retrieval engine module
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.retrieval_engine import (
    MultiPathRetrievalEngine, 
    OptimizedVectorRetriever,
    RetrievalContext,
    RetrievalResult
)


@pytest.mark.unit
class TestMultiPathRetrievalEngine:
    """Test suite for MultiPath Retrieval Engine"""
    
    @pytest.fixture
    def retrieval_engine(self, test_config):
        """Create retrieval engine instance for testing"""
        mock_entity_linker = Mock()
        return MultiPathRetrievalEngine(test_config.config, mock_entity_linker)
    
    @pytest.fixture
    def retrieval_context(self):
        """Create a sample retrieval context"""
        return RetrievalContext(
            query="What was Apple's revenue in 2023?",
            strategy="semantic_search",
            metadata_filters={"year": 2023},
            top_k=10,
            similarity_threshold=0.7
        )
    
    @pytest.fixture
    def mock_retrieval_results(self):
        """Create mock retrieval results"""
        return [
            RetrievalResult(
                chunk_id="chunk_001",
                content="Apple Inc. reported revenue of $394.3 billion for fiscal year 2023.",
                score=0.95,
                metadata={"source": "apple_2023.pdf", "page": 1},
                source_type="semantic"
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                content="The company's revenue grew by 2.8% year-over-year.",
                score=0.88,
                metadata={"source": "apple_2023.pdf", "page": 2},
                source_type="semantic"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, retrieval_engine, retrieval_context, mock_retrieval_results):
        """Test semantic search retrieval"""
        with patch.object(retrieval_engine.vector_retriever, 'search', 
                         new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_retrieval_results
            
            results = await retrieval_engine.retrieve(retrieval_context)
            
            mock_search.assert_called_once_with(retrieval_context)
            assert len(results) == 2
            assert results[0].score > results[1].score
            assert results[0].content == mock_retrieval_results[0].content
    
    @pytest.mark.asyncio
    async def test_entity_based_retrieval(self, retrieval_engine, retrieval_context):
        """Test entity-based retrieval"""
        retrieval_context.strategy = "entity_search"
        
        mock_entities = ["Apple Inc.", "2023", "$394.3 billion"]
        mock_entity_results = [
            RetrievalResult(
                chunk_id="chunk_003",
                content="Apple Inc. financial results",
                score=0.85,
                metadata={"entities": mock_entities},
                source_type="entity"
            )
        ]
        
        with patch.object(retrieval_engine.entity_linker, 'extract_entities', 
                         return_value=mock_entities):
            with patch.object(retrieval_engine, '_entity_search', 
                            new_callable=AsyncMock) as mock_entity_search:
                mock_entity_search.return_value = mock_entity_results
                
                results = await retrieval_engine.retrieve(retrieval_context)
                
                assert len(results) == 1
                assert results[0].source_type == "entity"
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, retrieval_engine, retrieval_context):
        """Test hybrid retrieval combining multiple strategies"""
        retrieval_context.strategy = "hybrid"
        
        semantic_results = [
            RetrievalResult("chunk_001", "Semantic result", 0.9, {}, "semantic")
        ]
        entity_results = [
            RetrievalResult("chunk_002", "Entity result", 0.85, {}, "entity")
        ]
        keyword_results = [
            RetrievalResult("chunk_003", "Keyword result", 0.8, {}, "keyword")
        ]
        
        with patch.object(retrieval_engine.vector_retriever, 'search', 
                         new_callable=AsyncMock, return_value=semantic_results):
            with patch.object(retrieval_engine, '_entity_search', 
                            new_callable=AsyncMock, return_value=entity_results):
                with patch.object(retrieval_engine, '_keyword_search', 
                                new_callable=AsyncMock, return_value=keyword_results):
                    
                    results = await retrieval_engine.retrieve(retrieval_context)
                    
                    # Should combine all results
                    assert len(results) >= 3
                    source_types = [r.source_type for r in results]
                    assert "semantic" in source_types
                    assert "entity" in source_types
                    assert "keyword" in source_types
    
    @pytest.mark.asyncio
    async def test_result_reranking(self, retrieval_engine, mock_retrieval_results):
        """Test result reranking functionality"""
        # Add duplicate with lower score
        duplicate_result = RetrievalResult(
            chunk_id="chunk_001",  # Same as first result
            content="Duplicate content",
            score=0.70,
            metadata={},
            source_type="keyword"
        )
        all_results = mock_retrieval_results + [duplicate_result]
        
        reranked = retrieval_engine._rerank_results(all_results)
        
        # Should remove duplicates and keep highest score
        chunk_ids = [r.chunk_id for r in reranked]
        assert chunk_ids.count("chunk_001") == 1
        
        # Should be sorted by score
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_apply_filters(self, retrieval_engine, mock_retrieval_results):
        """Test applying metadata filters"""
        filters = {"page": 1}
        filtered = retrieval_engine._apply_filters(mock_retrieval_results, filters)
        
        assert len(filtered) == 1
        assert filtered[0].metadata["page"] == 1
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, retrieval_engine):
        """Test handling of empty query"""
        context = RetrievalContext(
            query="",
            strategy="semantic_search",
            metadata_filters={},
            top_k=10
        )
        
        with pytest.raises(ValueError) as exc_info:
            await retrieval_engine.retrieve(context)
        
        assert "empty query" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_retrieval(self, retrieval_engine):
        """Test concurrent retrieval operations"""
        contexts = [
            RetrievalContext(f"Query {i}", "semantic_search", {}, 5)
            for i in range(3)
        ]
        
        with patch.object(retrieval_engine.vector_retriever, 'search', 
                         new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []
            
            tasks = [retrieval_engine.retrieve(ctx) for ctx in contexts]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert mock_search.call_count == 3


@pytest.mark.unit
class TestOptimizedVectorRetriever:
    """Test suite for Optimized Vector Retriever"""
    
    @pytest.fixture
    def vector_retriever(self, test_config):
        """Create vector retriever instance for testing"""
        return OptimizedVectorRetriever(test_config.config)
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings"""
        # Create normalized random vectors
        vectors = np.random.randn(5, 384).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    
    @pytest.fixture
    def mock_documents(self):
        """Create mock documents for indexing"""
        return [
            {
                "chunk_id": f"chunk_{i:03d}",
                "content": f"Document content {i}",
                "embedding": None,  # Will be replaced with mock embeddings
                "metadata": {"doc_id": i, "page": i}
            }
            for i in range(5)
        ]
    
    @pytest.mark.asyncio
    async def test_index_documents(self, vector_retriever, mock_documents, mock_embeddings):
        """Test document indexing"""
        # Add embeddings to documents
        for i, doc in enumerate(mock_documents):
            doc["embedding"] = mock_embeddings[i].tolist()
        
        with patch.object(vector_retriever, '_generate_embeddings', 
                         return_value=mock_embeddings):
            await vector_retriever.index_documents(mock_documents)
            
            # Verify documents were indexed
            assert vector_retriever.index_size() == len(mock_documents)
    
    @pytest.mark.asyncio
    async def test_vector_search(self, vector_retriever, mock_embeddings):
        """Test vector similarity search"""
        # Setup mock index
        vector_retriever.index = Mock()
        vector_retriever.index.search = Mock(return_value=(
            np.array([[0.95, 0.88, 0.75]]),  # distances
            np.array([[0, 2, 4]])  # indices
        ))
        
        query_embedding = mock_embeddings[0]
        
        results = await vector_retriever._search_vectors(
            query_embedding, 
            top_k=3
        )
        
        assert len(results) == 3
        assert results[0]["score"] == 0.95
        assert results[0]["index"] == 0
    
    def test_cosine_similarity(self, vector_retriever, mock_embeddings):
        """Test cosine similarity calculation"""
        vec1 = mock_embeddings[0]
        vec2 = mock_embeddings[1]
        
        similarity = vector_retriever._cosine_similarity(vec1, vec2)
        
        assert -1 <= similarity <= 1
        
        # Same vector should have similarity ~1
        self_similarity = vector_retriever._cosine_similarity(vec1, vec1)
        assert abs(self_similarity - 1.0) < 0.0001
    
    @pytest.mark.asyncio
    async def test_search_with_threshold(self, vector_retriever, retrieval_context):
        """Test search with similarity threshold filtering"""
        mock_results = [
            {"chunk_id": "1", "score": 0.95, "content": "High match"},
            {"chunk_id": "2", "score": 0.60, "content": "Low match"},
            {"chunk_id": "3", "score": 0.85, "content": "Good match"}
        ]
        
        retrieval_context.similarity_threshold = 0.8
        
        with patch.object(vector_retriever, '_search_vectors', 
                         new_callable=AsyncMock, return_value=mock_results):
            results = await vector_retriever.search(retrieval_context)
            
            # Should filter out results below threshold
            assert len(results) == 2
            assert all(r.score >= 0.8 for r in results)
    
    def test_batch_embedding_generation(self, vector_retriever):
        """Test batch embedding generation"""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        
        with patch.object(vector_retriever, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.random.randn(5, 384)
            
            embeddings = vector_retriever._generate_embeddings(texts, batch_size=2)
            
            # Should process in batches
            assert mock_model.encode.call_count >= 2
            assert embeddings.shape == (5, 384)
    
    @pytest.mark.asyncio
    async def test_empty_index_search(self, vector_retriever, retrieval_context):
        """Test searching empty index"""
        # Don't index any documents
        results = await vector_retriever.search(retrieval_context)
        
        assert results == []
    
    def test_embedding_cache(self, vector_retriever):
        """Test embedding caching mechanism"""
        text = "Test query for caching"
        
        with patch.object(vector_retriever, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.array([[0.1] * 384])
            
            # First call should generate embedding
            emb1 = vector_retriever._get_cached_embedding(text)
            assert mock_model.encode.call_count == 1
            
            # Second call should use cache
            emb2 = vector_retriever._get_cached_embedding(text)
            assert mock_model.encode.call_count == 1  # Still 1
            
            # Embeddings should be identical
            np.testing.assert_array_equal(emb1, emb2)


@pytest.mark.unit
class TestRetrievalContext:
    """Test suite for RetrievalContext class"""
    
    def test_context_creation(self):
        """Test creating retrieval context"""
        context = RetrievalContext(
            query="Test query",
            strategy="semantic_search",
            metadata_filters={"year": 2023},
            top_k=10,
            similarity_threshold=0.7
        )
        
        assert context.query == "Test query"
        assert context.strategy == "semantic_search"
        assert context.metadata_filters["year"] == 2023
        assert context.top_k == 10
        assert context.similarity_threshold == 0.7
    
    def test_context_validation(self):
        """Test context validation"""
        # Invalid top_k
        with pytest.raises(ValueError):
            RetrievalContext("query", "semantic", {}, top_k=-1)
        
        # Invalid threshold
        with pytest.raises(ValueError):
            RetrievalContext("query", "semantic", {}, top_k=10, similarity_threshold=1.5)
    
    def test_context_serialization(self):
        """Test context serialization"""
        context = RetrievalContext(
            query="Serialize me",
            strategy="hybrid",
            metadata_filters={"doc_type": "pdf"},
            top_k=5
        )
        
        serialized = context.to_dict()
        assert serialized["query"] == "Serialize me"
        assert serialized["strategy"] == "hybrid"
        assert serialized["metadata_filters"]["doc_type"] == "pdf"


@pytest.mark.unit
class TestRetrievalResult:
    """Test suite for RetrievalResult class"""
    
    def test_result_creation(self):
        """Test creating retrieval result"""
        result = RetrievalResult(
            chunk_id="chunk_001",
            content="Test content",
            score=0.95,
            metadata={"page": 1},
            source_type="semantic"
        )
        
        assert result.chunk_id == "chunk_001"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata["page"] == 1
        assert result.source_type == "semantic"
    
    def test_result_comparison(self):
        """Test result comparison by score"""
        result1 = RetrievalResult("1", "Content 1", 0.9, {}, "semantic")
        result2 = RetrievalResult("2", "Content 2", 0.8, {}, "semantic")
        
        assert result1 > result2  # Higher score
        assert result2 < result1
        
        # Sort should work
        results = [result2, result1]
        sorted_results = sorted(results, reverse=True)
        assert sorted_results[0] == result1
    
    def test_result_serialization(self):
        """Test result serialization"""
        result = RetrievalResult(
            chunk_id="chunk_002",
            content="Financial data",
            score=0.88,
            metadata={"source": "report.pdf"},
            source_type="entity"
        )
        
        serialized = result.to_dict()
        assert serialized["chunk_id"] == "chunk_002"
        assert serialized["content"] == "Financial data"
        assert serialized["score"] == 0.88
        assert serialized["metadata"]["source"] == "report.pdf"
        assert serialized["source_type"] == "entity"