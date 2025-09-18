"""
Pytest configuration and shared fixtures for FinanceRAG-Pro tests
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from core.lightweight_storage import InMemoryMetadataManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Create test configuration."""
    config_data = {
        "system": {
            "name": "FinanceRAG-Test",
            "debug": True,
            "log_level": "ERROR"
        },
        "models": {
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "batch_size": 4
            }
        },
        "database": {
            "sqlite": {
                "path": ":memory:"
            },
            "chromadb": {
                "persist_directory": ":memory:",
                "collection_name": "test_documents"
            }
        },
        "retrieval": {
            "chunk_size": 256,
            "chunk_overlap": 25,
            "top_k": 5,
            "similarity_threshold": 0.5
        },
        "api": {
            "host": "localhost",
            "port": 8000
        }
    }
    return Config.from_dict(config_data)


@pytest.fixture
async def memory_storage():
    """Create in-memory storage for testing."""
    storage = InMemoryMetadataManager({})
    await storage.connect()
    yield storage
    await storage.close()


@pytest.fixture
def mock_api_client():
    """Create mock API client."""
    mock_client = Mock()
    mock_client.query_async = AsyncMock()
    mock_client.query_async.return_value = {
        "choices": [{
            "message": {
                "content": "Mock response from API"
            }
        }],
        "usage": {
            "total_tokens": 100
        }
    }
    return mock_client


@pytest.fixture
def mock_embedding_model():
    """Create mock embedding model."""
    mock_model = Mock()
    mock_model.encode = Mock(return_value=[[0.1] * 384])  # 384-dimensional embeddings
    return mock_model


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Apple Inc. reported revenue of $394.3 billion for fiscal year 2023, 
    representing a 2.8% increase from the previous year. The company's 
    research and development expenses were $29.9 billion, accounting for 
    approximately 7.6% of total revenue.
    """


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "content": "Apple Inc. reported revenue of $394.3 billion for fiscal year 2023.",
            "metadata": {"source": "apple_2023.pdf", "page": 1},
            "embedding": [0.1] * 384
        },
        {
            "chunk_id": "chunk_2", 
            "content": "Research and development expenses were $29.9 billion.",
            "metadata": {"source": "apple_2023.pdf", "page": 2},
            "embedding": [0.2] * 384
        }
    ]


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a sample PDF file for testing."""
    # This would be a real PDF file in practice
    pdf_path = temp_dir / "sample_financial_report.pdf"
    pdf_path.write_text("Mock PDF content for testing")
    return pdf_path


@pytest.fixture
def mock_vector_db():
    """Create mock vector database."""
    mock_db = Mock()
    mock_db.add = AsyncMock()
    mock_db.query = AsyncMock()
    mock_db.query.return_value = [
        {
            "id": "chunk_1",
            "score": 0.95,
            "metadata": {"source": "test.pdf"},
            "document": "Sample document content"
        }
    ]
    return mock_db


# Test markers
pytestmark = pytest.mark.asyncio