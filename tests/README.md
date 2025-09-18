# Tests Directory

This directory contains all test files for the FinanceRAG-Pro project.

## Directory Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_document_processor.py
│   ├── test_retrieval_engine.py
│   ├── test_query_engine.py
│   ├── test_entity_linker.py
│   ├── test_validation_engine.py
│   └── test_inference_engine.py
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── test_api_endpoints.py
│   ├── test_pdf_processing.py
│   └── test_full_pipeline.py
├── fixtures/                   # Test data and fixtures
│   ├── __init__.py
│   ├── sample_pdfs/
│   ├── mock_responses/
│   └── test_configs/
└── mocks/                      # Mock objects and utilities
    ├── __init__.py
    ├── mock_api_client.py
    ├── mock_vector_db.py
    └── mock_llm_responses.py
```

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov pytest-asyncio httpx
```

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -m unit

# Integration tests only
pytest tests/integration/ -m integration

# API tests only
pytest tests/ -m api

# PDF processing tests only
pytest tests/ -m pdf
```

### Run Tests with Coverage
```bash
pytest --cov=core --cov-report=html
```

### Run Tests in Cloud Studio
```bash
# Set environment variables
export TESTING=true
export LOG_LEVEL=ERROR

# Run lightweight tests (no external dependencies)
pytest tests/unit/ -v

# Run full integration tests (requires services)
pytest tests/integration/ -v
```

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and API endpoints
- **PDF Tests**: Test real PDF document processing
- **Vector Tests**: Test vector database operations
- **API Tests**: Test REST API endpoints

## Adding New Tests

1. Follow naming convention: `test_*.py`
2. Use appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`
3. Place test data in `fixtures/` directory
4. Use mocks for external dependencies in unit tests