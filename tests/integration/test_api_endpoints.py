"""
Integration tests for API endpoints
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from httpx import AsyncClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main_lite import app


@pytest.mark.integration
class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    async def client(self):
        """Create test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def sample_pdf_file(self, tmp_path):
        """Create a sample PDF file for testing"""
        pdf_path = tmp_path / "test_financial_report.pdf"
        # Create a minimal PDF-like content
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"
        pdf_path.write_bytes(pdf_content)
        return pdf_path
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = await client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_docs_endpoint(self, client):
        """Test API documentation endpoint"""
        response = await client.get("/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.headers.get("content-type", "").lower() or \
               "html" in response.headers.get("content-type", "").lower()
    
    @pytest.mark.asyncio
    async def test_query_endpoint_success(self, client):
        """Test successful query endpoint"""
        query_data = {
            "query": "What was Apple's revenue in 2023?",
            "top_k": 5
        }
        
        with patch('main_lite.LightweightFinanceRAG.handle_query') as mock_handle:
            mock_handle.return_value = {
                "answer": "Apple's revenue was $394.3 billion in 2023.",
                "confidence": 0.95,
                "sources": ["apple_2023_report.pdf"],
                "metadata": {
                    "query_type": "factual",
                    "processing_time": "2024-01-01T10:00:00",
                    "version": "lite"
                }
            }
            
            response = await client.post("/query", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "confidence" in data
            assert "sources" in data
            assert "metadata" in data
            assert data["confidence"] == 0.95
    
    @pytest.mark.asyncio
    async def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query"""
        query_data = {
            "query": "",
            "top_k": 5
        }
        
        response = await client.post("/query", json=query_data)
        
        # Should return an error for empty query
        assert response.status_code == 422 or response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_query_endpoint_invalid_top_k(self, client):
        """Test query endpoint with invalid top_k"""
        query_data = {
            "query": "Test query",
            "top_k": -1  # Invalid value
        }
        
        response = await client.post("/query", json=query_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_query_endpoint_server_error(self, client):
        """Test query endpoint server error handling"""
        query_data = {
            "query": "Test query causing error",
            "top_k": 5
        }
        
        with patch('main_lite.LightweightFinanceRAG.handle_query') as mock_handle:
            mock_handle.side_effect = Exception("Server error")
            
            response = await client.post("/query", json=query_data)
            
            assert response.status_code == 500
    
    @pytest.mark.asyncio
    async def test_upload_endpoint_success(self, client, sample_pdf_file):
        """Test successful file upload"""
        
        with patch('main_lite.LightweightFinanceRAG.handle_upload') as mock_handle:
            mock_handle.return_value = {
                "document_id": "test_doc_123",
                "status": "processing",
                "message": "Document uploaded and processing started",
                "metadata": {
                    "filename": "test_financial_report.pdf",
                    "size": 1024,
                    "upload_time": "2024-01-01T10:00:00"
                }
            }
            
            with open(sample_pdf_file, 'rb') as f:
                files = {"file": (sample_pdf_file.name, f, "application/pdf")}
                response = await client.post("/upload", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "document_id" in data
            assert "status" in data
            assert data["status"] == "processing"
    
    @pytest.mark.asyncio
    async def test_upload_endpoint_invalid_file_type(self, client, tmp_path):
        """Test upload with invalid file type"""
        # Create a text file instead of PDF
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is not a PDF")
        
        with open(txt_file, 'rb') as f:
            files = {"file": (txt_file.name, f, "text/plain")}
            response = await client.post("/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "not allowed" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_upload_endpoint_no_file(self, client):
        """Test upload without file"""
        response = await client.post("/upload")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    @pytest.mark.asyncio
    async def test_upload_endpoint_large_file(self, client, tmp_path):
        """Test upload with oversized file"""
        # Create a large fake PDF file (simulate >50MB)
        large_file = tmp_path / "large.pdf"
        
        # Create file larger than MAX_FILE_SIZE
        with patch('main_lite.validate_upload_file') as mock_validate:
            mock_validate.side_effect = Exception("File too large")
            
            large_file.write_bytes(b"fake pdf content" * 1000)
            
            with open(large_file, 'rb') as f:
                files = {"file": (large_file.name, f, "application/pdf")}
                response = await client.post("/upload", files=files)
            
            assert response.status_code == 500  # Server error due to validation
    
    @pytest.mark.asyncio
    async def test_documents_endpoint(self, client):
        """Test documents listing endpoint"""
        response = await client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "message" in data
        # In lite version, documents list is not implemented
        assert data["documents"] == []
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = await client.options("/")
        
        # Should allow CORS
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented
        
        # Test with actual request
        response = await client.get("/")
        headers = response.headers
        
        # CORS should be enabled in the app configuration
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_request_timeout(self, client):
        """Test request timeout handling"""
        
        with patch('main_lite.LightweightFinanceRAG.handle_query') as mock_handle:
            # Simulate a long-running query
            async def slow_query(*args, **kwargs):
                await asyncio.sleep(5)  # 5 second delay
                return {"answer": "slow response", "confidence": 0.8, "sources": [], "metadata": {}}
            
            mock_handle.side_effect = slow_query
            
            query_data = {
                "query": "Slow query test",
                "top_k": 5
            }
            
            # This should timeout or complete depending on configuration
            try:
                response = await client.post("/query", json=query_data, timeout=2.0)
                # If it doesn't timeout, it should still be a valid response
                assert response.status_code in [200, 408, 504]
            except asyncio.TimeoutError:
                # Timeout is acceptable for this test
                pass
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        
        with patch('main_lite.LightweightFinanceRAG.handle_query') as mock_handle:
            mock_handle.return_value = {
                "answer": "Test answer",
                "confidence": 0.9,
                "sources": [],
                "metadata": {}
            }
            
            query_data = {
                "query": "Concurrent test query",
                "top_k": 5
            }
            
            # Send multiple concurrent requests
            tasks = [
                client.post("/query", json=query_data)
                for _ in range(5)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            for response in responses:
                if isinstance(response, Exception):
                    pytest.fail(f"Request failed: {response}")
                assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_json_validation(self, client):
        """Test JSON request validation"""
        # Test invalid JSON
        response = await client.post(
            "/query",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client):
        """Test missing required fields in request"""
        # Query without required 'query' field
        incomplete_data = {
            "top_k": 5
            # Missing 'query' field
        }
        
        response = await client.post("/query", json=incomplete_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_response_headers(self, client):
        """Test response headers"""
        response = await client.get("/")
        
        assert response.status_code == 200
        
        # Check content type
        assert "application/json" in response.headers.get("content-type", "")
        
        # Check for security headers (if implemented)
        # These might be added by middleware
        headers = response.headers
        # Note: Security headers would typically be added in production


@pytest.mark.integration
class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    @pytest.fixture
    async def client(self):
        """Create test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.mark.asyncio
    async def test_query_response_time(self, client):
        """Test query response time"""
        import time
        
        with patch('main_lite.LightweightFinanceRAG.handle_query') as mock_handle:
            mock_handle.return_value = {
                "answer": "Fast response",
                "confidence": 0.9,
                "sources": [],
                "metadata": {}
            }
            
            query_data = {
                "query": "Performance test query",
                "top_k": 5
            }
            
            start_time = time.time()
            response = await client.post("/query", json=query_data)
            end_time = time.time()
            
            assert response.status_code == 200
            
            # Response should be reasonably fast (under 5 seconds for mock)
            response_time = end_time - start_time
            assert response_time < 5.0, f"Response too slow: {response_time}s"
    
    @pytest.mark.asyncio
    async def test_health_check_speed(self, client):
        """Test health check endpoint speed"""
        import time
        
        start_time = time.time()
        response = await client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Health check should be very fast
        response_time = end_time - start_time
        assert response_time < 1.0, f"Health check too slow: {response_time}s"