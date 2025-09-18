"""
Unit tests for document processor module
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.document_processor import MultiModalPreprocessor, DocumentChunk


@pytest.mark.unit
class TestMultiModalPreprocessor:
    """Test suite for MultiModal Document Preprocessor"""
    
    @pytest.fixture
    def preprocessor(self, test_config):
        """Create preprocessor instance for testing"""
        return MultiModalPreprocessor(test_config.config)
    
    @pytest.fixture
    def mock_pdf_file(self, tmp_path):
        """Create a mock PDF file for testing"""
        pdf_path = tmp_path / "test_financial_report.pdf"
        pdf_path.write_bytes(b"Mock PDF content for testing")
        return str(pdf_path)
    
    @pytest.fixture
    def mock_pdf_content(self):
        """Mock PDF content structure"""
        return {
            "text_chunks": [
                DocumentChunk(
                    chunk_id="chunk_001",
                    content="Apple Inc. reported revenue of $394.3 billion for fiscal year 2023.",
                    metadata={
                        "page": 1,
                        "source": "test_financial_report.pdf",
                        "chunk_type": "text"
                    }
                ),
                DocumentChunk(
                    chunk_id="chunk_002",
                    content="Research and development expenses were $29.9 billion.",
                    metadata={
                        "page": 2,
                        "source": "test_financial_report.pdf",
                        "chunk_type": "text"
                    }
                )
            ],
            "tables": [
                {
                    "table_id": "table_001",
                    "page": 3,
                    "data": [
                        ["Revenue", "2023", "2022"],
                        ["Product", "$298.1B", "$316.2B"],
                        ["Services", "$96.2B", "$78.1B"]
                    ]
                }
            ],
            "images": [],
            "metadata": {
                "total_pages": 10,
                "file_size": 2048576,
                "extraction_time": "2024-01-01T10:00:00"
            }
        }
    
    @pytest.mark.asyncio
    async def test_preprocess_document_success(self, preprocessor, mock_pdf_file, mock_pdf_content):
        """Test successful document preprocessing"""
        with patch.object(preprocessor, '_extract_pdf_content') as mock_extract:
            mock_extract.return_value = mock_pdf_content
            
            result = await preprocessor.preprocess_document(mock_pdf_file)
            
            # Verify extraction was called
            mock_extract.assert_called_once_with(mock_pdf_file)
            
            # Verify result structure
            assert "text_chunks" in result
            assert "tables" in result
            assert "metadata" in result
            assert len(result["text_chunks"]) == 2
            assert len(result["tables"]) == 1
    
    @pytest.mark.asyncio
    async def test_preprocess_document_file_not_found(self, preprocessor):
        """Test preprocessing with non-existent file"""
        with pytest.raises(FileNotFoundError):
            await preprocessor.preprocess_document("/non/existent/file.pdf")
    
    def test_chunk_text_basic(self, preprocessor):
        """Test basic text chunking"""
        text = "This is a test document. " * 100  # Create long text
        chunks = preprocessor._chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
        
        # Test overlap
        for i in range(len(chunks) - 1):
            # There should be some overlap between consecutive chunks
            assert chunks[i][-20:] in chunks[i+1][:40] or len(chunks[i]) < 20
    
    def test_extract_tables_from_text(self, preprocessor):
        """Test table extraction from text content"""
        text_with_table = """
        Financial Results:
        | Revenue | 2023 | 2022 |
        |---------|------|------|
        | Product | $298B| $316B|
        | Service | $96B | $78B |
        
        Additional text after table.
        """
        
        tables = preprocessor._extract_tables(text_with_table)
        
        assert len(tables) > 0
        assert "Revenue" in str(tables[0])
        assert "$298B" in str(tables[0])
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_images(self, preprocessor, mock_pdf_file):
        """Test PDF processing with image extraction"""
        mock_content = {
            "text_chunks": [],
            "tables": [],
            "images": [
                {
                    "image_id": "img_001",
                    "page": 5,
                    "caption": "Revenue Growth Chart",
                    "type": "chart"
                }
            ],
            "metadata": {"total_pages": 10}
        }
        
        with patch.object(preprocessor, '_extract_pdf_content', return_value=mock_content):
            result = await preprocessor.preprocess_document(mock_pdf_file)
            
            assert len(result["images"]) == 1
            assert result["images"][0]["caption"] == "Revenue Growth Chart"
    
    def test_validate_pdf_structure(self, preprocessor, mock_pdf_content):
        """Test PDF structure validation"""
        # Valid structure should pass
        assert preprocessor._validate_structure(mock_pdf_content) == True
        
        # Invalid structure should fail
        invalid_content = {"invalid": "structure"}
        assert preprocessor._validate_structure(invalid_content) == False
    
    def test_clean_text(self, preprocessor):
        """Test text cleaning functionality"""
        dirty_text = "  This   has\n\nextra    whitespace\t\tand\nnewlines.  "
        clean = preprocessor._clean_text(dirty_text)
        
        assert clean == "This has extra whitespace and newlines."
        assert "  " not in clean
        assert "\n\n" not in clean
        assert "\t" not in clean
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, preprocessor, tmp_path):
        """Test concurrent processing of multiple documents"""
        # Create multiple mock PDFs
        pdf_files = []
        for i in range(3):
            pdf_path = tmp_path / f"test_{i}.pdf"
            pdf_path.write_bytes(f"Mock PDF {i}".encode())
            pdf_files.append(str(pdf_path))
        
        with patch.object(preprocessor, '_extract_pdf_content') as mock_extract:
            mock_extract.return_value = {"text_chunks": [], "tables": [], "metadata": {}}
            
            # Process concurrently
            tasks = [preprocessor.preprocess_document(pdf) for pdf in pdf_files]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert mock_extract.call_count == 3
    
    def test_metadata_extraction(self, preprocessor, mock_pdf_content):
        """Test metadata extraction from document"""
        metadata = preprocessor._extract_metadata(mock_pdf_content)
        
        assert "total_pages" in metadata
        assert "file_size" in metadata
        assert metadata["total_pages"] == 10
    
    def test_handle_special_characters(self, preprocessor):
        """Test handling of special financial characters"""
        text = "Revenue: $394.3B (↑2.8%) | P/E: 31.46 | β: 1.25"
        chunks = preprocessor._chunk_text(text, chunk_size=50)
        
        # Should preserve special characters
        full_text = " ".join(chunks)
        assert "$" in full_text
        assert "%" in full_text
        assert "β" in full_text
    
    @pytest.mark.asyncio 
    async def test_error_handling(self, preprocessor):
        """Test error handling in document processing"""
        with patch.object(preprocessor, '_extract_pdf_content') as mock_extract:
            mock_extract.side_effect = Exception("PDF extraction failed")
            
            with pytest.raises(Exception) as exc_info:
                await preprocessor.preprocess_document("test.pdf")
            
            assert "PDF extraction failed" in str(exc_info.value)
    
    def test_chunk_overlap_consistency(self, preprocessor):
        """Test that chunk overlaps are consistent"""
        text = "word " * 100  # Simple repeated text
        chunks = preprocessor._chunk_text(text, chunk_size=25, overlap=10)
        
        for i in range(len(chunks) - 1):
            current_end = chunks[i][-10:] if len(chunks[i]) >= 10 else chunks[i]
            next_start = chunks[i+1][:10] if len(chunks[i+1]) >= 10 else chunks[i+1]
            
            # Check if there's actual overlap for chunks that are long enough
            if len(chunks[i]) >= 10 and len(chunks[i+1]) >= 10:
                # At least some characters should match
                assert any(c in next_start for c in current_end[-5:])


@pytest.mark.unit  
class TestDocumentChunk:
    """Test suite for DocumentChunk class"""
    
    def test_document_chunk_creation(self):
        """Test creating a document chunk"""
        chunk = DocumentChunk(
            chunk_id="test_001",
            content="Test content",
            metadata={"page": 1, "source": "test.pdf"}
        )
        
        assert chunk.chunk_id == "test_001"
        assert chunk.content == "Test content"
        assert chunk.metadata["page"] == 1
    
    def test_document_chunk_serialization(self):
        """Test chunk serialization"""
        chunk = DocumentChunk(
            chunk_id="test_002",
            content="Financial data",
            metadata={"page": 2}
        )
        
        serialized = chunk.to_dict()
        assert serialized["chunk_id"] == "test_002"
        assert serialized["content"] == "Financial data"
        assert serialized["metadata"]["page"] == 2
    
    def test_document_chunk_with_embedding(self):
        """Test chunk with embedding vector"""
        embedding = [0.1] * 384  # Mock embedding
        chunk = DocumentChunk(
            chunk_id="test_003",
            content="Test with embedding",
            metadata={"page": 3},
            embedding=embedding
        )
        
        assert chunk.embedding == embedding
        assert len(chunk.embedding) == 384