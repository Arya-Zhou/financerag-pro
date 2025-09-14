#!/bin/bash

# Lightweight setup script for Cloud Studio environment
# No sudo required, no Docker needed

echo "========================================="
echo "FinanceRAG-Pro Lightweight Setup"
echo "For Cloud Studio Environment"
echo "========================================="

# Create necessary directories
echo "Creating directories..."
mkdir -p data/chromadb
mkdir -p data/cache
mkdir -p logs
mkdir -p models

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install lightweight requirements
echo "Installing Python dependencies..."
cat > requirements_lite.txt << EOF
# Core dependencies (lightweight version)
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# Document processing
PyPDF2==3.0.1
pdfplumber==0.10.3
pillow==10.1.0

# ML and NLP (CPU only)
torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
transformers==4.35.0
sentence-transformers==2.2.2

# Vector database (lightweight)
chromadb==0.4.22

# Storage (SQLite instead of PostgreSQL)
# No psycopg2 needed

# Cache (in-memory instead of Redis)
cachetools==5.3.2

# Utilities
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
tqdm==4.66.1
python-dotenv==1.0.0
pyyaml==6.0.1

# Async support
aiofiles==23.2.1
httpx==0.25.2

# Logging
loguru==0.7.2
EOF

pip install -r requirements_lite.txt

# Create lightweight config
echo "Creating lightweight configuration..."
cat > config_lite.yaml << EOF
system:
  name: FinanceRAG-Pro-Lite
  version: 1.0.0
  debug: false
  log_level: INFO

models:
  # Use smaller models for Cloud Studio
  embedding:
    model: sentence-transformers/all-MiniLM-L6-v2
    dimension: 384
    batch_size: 16
  
  # Use API for generation (no local LLM)
  generation:
    use_api: true
    model: gpt-3.5-turbo

database:
  # Use SQLite instead of PostgreSQL
  sqlite:
    path: ./data/financerag.db
  
  # Use in-memory cache instead of Redis
  cache:
    type: memory
    max_size: 1000
  
  # ChromaDB with persistent storage
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
EOF

# Create .env file
echo "Creating environment file..."
cat > .env << EOF
# OpenAI API Key (required for LLM features)
OPENAI_API_KEY=your_api_key_here

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Storage paths
DATA_PATH=./data
LOG_PATH=./logs
MODEL_CACHE_DIR=./models
EOF

echo "========================================="
echo "Setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the application: python main_lite.py"
echo ""
echo "Note: This is a lightweight version optimized for Cloud Studio"
echo "- Uses SQLite instead of PostgreSQL"
echo "- Uses in-memory cache instead of Redis"
echo "- No Docker required"
echo "- Runs with user permissions only"