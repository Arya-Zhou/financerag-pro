#!/bin/bash

# FinanceRAG-Pro Cloud Studio Setup Script
# 专为Cloud Studio环境设计的一键部署脚本

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Banner
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  FinanceRAG-Pro Cloud Studio Setup Script    ${NC}"
echo -e "${CYAN}     金融多模态智能检索系统部署工具            ${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# Check if we're in Cloud Studio environment
check_environment() {
    print_header "检查运行环境"
    
    if [ -n "$CLOUDSTUDIO" ] || [ -n "$CS_PROJECT_PATH" ]; then
        print_status "检测到Cloud Studio环境"
    else
        print_warning "未检测到Cloud Studio环境，继续安装..."
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python版本: $PYTHON_VERSION"
        
        # Check if Python version is >= 3.8
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python版本满足要求 (>=3.8)"
        else
            print_error "Python版本过低，需要3.8或更高版本"
            exit 1
        fi
    else
        print_error "未找到Python3，请先安装Python"
        exit 1
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 2 ]; then
        print_warning "可用磁盘空间不足2GB ($AVAILABLE_SPACE GB)，可能导致安装失败"
    else
        print_status "磁盘空间充足: ${AVAILABLE_SPACE}GB"
    fi
}

# Create necessary directories
create_directories() {
    print_header "创建必要的目录结构"
    
    directories=(
        "data"
        "data/pdfs"
        "data/chromadb"
        "data/sqlite"
        "data/cache"
        "logs"
        "models"
        "checkpoints"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "创建目录: $dir"
        else
            print_status "目录已存在: $dir"
        fi
    done
}

# Install system dependencies (if needed)
install_system_deps() {
    print_header "安装系统依赖"
    
    # Check if we have sudo privileges
    if sudo -n true 2>/dev/null; then
        print_status "检测到sudo权限，安装系统依赖..."
        
        # Update package list
        sudo apt-get update -qq
        
        # Install essential packages
        sudo apt-get install -y -qq \
            build-essential \
            python3-dev \
            python3-pip \
            python3-venv \
            curl \
            wget \
            unzip \
            git \
            sqlite3 \
            libsqlite3-dev \
            pkg-config \
            libhdf5-dev
        
        print_status "系统依赖安装完成"
    else
        print_warning "无sudo权限，跳过系统依赖安装"
        print_warning "如果安装失败，请手动安装: build-essential python3-dev"
    fi
}

# Setup Python virtual environment
setup_virtual_env() {
    print_header "设置Python虚拟环境"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "创建Python虚拟环境..."
        python3 -m venv venv
        print_success "虚拟环境创建完成"
    else
        print_status "虚拟环境已存在"
    fi
    
    # Activate virtual environment
    print_status "激活虚拟环境..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "升级pip..."
    pip install --upgrade pip -q
    
    # Install wheel for faster package installation
    pip install wheel -q
}

# Install Python dependencies
install_python_deps() {
    print_header "安装Python依赖包"
    
    # Make sure we're in virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "虚拟环境未激活，正在激活..."
        source venv/bin/activate
    fi
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        print_status "从requirements.txt安装依赖..."
        
        # Use pip with timeout and retries for better reliability
        pip install -r requirements.txt \
            --timeout 300 \
            --retries 3 \
            --disable-pip-version-check \
            -q
        
        print_success "Python依赖安装完成"
    else
        print_error "未找到requirements.txt文件"
        exit 1
    fi
    
    # Install additional Cloud Studio optimized packages
    print_status "安装Cloud Studio优化包..."
    pip install -q \
        httpx \
        rich \
        psutil \
        memory-profiler
}

# Create configuration files
setup_configuration() {
    print_header "设置配置文件"
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "创建.env环境配置文件..."
        cat > .env << 'EOF'
# FinanceRAG-Pro Environment Configuration for Cloud Studio

# API Keys (请替换为您的实际API密钥)
OPENAI_API_KEY=your_openai_api_key_here
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
MODELSCOPE_API_KEY=your_modelscope_api_key_here

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=cloud_studio

# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*

# Database Configuration
DATABASE_TYPE=sqlite
DATABASE_PATH=./data/financerag.db
CHROMADB_PATH=./data/chromadb

# Performance Settings (Cloud Studio Optimized)
MAX_WORKERS=2
BATCH_SIZE=8
MEMORY_THRESHOLD_MB=600
CLEANUP_INTERVAL_MINUTES=15

# Model Configuration
USE_LOCAL_MODELS=false
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Cache Settings
ENABLE_CACHE=true
CACHE_TTL=1800
CACHE_MAX_SIZE=500

# Logging
LOG_TO_FILE=true
LOG_ROTATION=true
LOG_MAX_SIZE=10MB
LOG_RETENTION_DAYS=3
EOF
        print_success "环境配置文件创建完成"
        print_warning "请编辑.env文件并添加您的API密钥"
    else
        print_status "环境配置文件已存在"
    fi
    
    # Create Cloud Studio optimized config if it doesn't exist
    if [ ! -f "config_lite.yaml" ]; then
        print_status "创建轻量级配置文件..."
        cat > config_lite.yaml << 'EOF'
system:
  name: FinanceRAG-Pro-Lite
  version: 1.0.0-cloud-studio
  debug: false
  log_level: INFO
  environment: cloud_studio

models:
  # 使用轻量级嵌入模型
  embedding:
    model: sentence-transformers/all-MiniLM-L6-v2
    dimension: 384
    batch_size: 8
    precision: fp16
  
  # 使用API调用而非本地模型
  generation:
    use_api: true
    model: gpt-3.5-turbo
    temperature: 0.7
    max_tokens: 1500

database:
  # SQLite数据库
  sqlite:
    path: ./data/financerag.db
    timeout: 30
    journal_mode: WAL
  
  # 内存缓存
  cache:
    type: memory
    max_size: 500
    ttl: 1800
  
  # ChromaDB向量数据库
  chromadb:
    persist_directory: ./data/chromadb
    collection_name: finance_documents
    anonymized_telemetry: false

retrieval:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 8
  similarity_threshold: 0.7
  enable_rerank: true
  max_chunks: 50

api:
  host: 0.0.0.0
  port: 8000
  cors_enabled: true
  docs_enabled: true
  max_request_size: 26214400  # 25MB
  request_timeout: 120

performance:
  max_workers: 2
  batch_processing: true
  async_enabled: true
  timeout: 60
  memory_threshold_mb: 600
  periodic_cleanup_minutes: 15

logging:
  level: INFO
  format: detailed
  file_rotation: "5 MB"
  file_retention: "3 days"
  performance_log: true

security:
  enable_cors: true
  allowed_origins: ["*"]
  max_file_size_mb: 25
  allowed_extensions: [".pdf", ".docx", ".txt", ".md"]
  rate_limiting: false
EOF
        print_success "轻量级配置文件创建完成"
    else
        print_status "轻量级配置文件已存在"
    fi
}

# Test installation
test_installation() {
    print_header "测试安装结果"
    
    # Make sure we're in virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi
    
    print_status "测试Python模块导入..."
    
    # Test core module imports
    python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from fastapi import FastAPI
    print('✓ FastAPI导入成功')
    
    from core.config_manager import get_config_manager
    print('✓ 配置管理器导入成功')
    
    from core.document_processor import MultiModalPreprocessor
    print('✓ 文档处理器导入成功')
    
    from core.retrieval_engine import MultiPathRetrievalEngine
    print('✓ 检索引擎导入成功')
    
    import sqlite3
    print('✓ SQLite支持正常')
    
    print('\\n🎉 所有核心模块导入测试通过！')
    
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ 测试失败: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "模块导入测试通过"
    else
        print_error "模块导入测试失败"
        return 1
    fi
    
    # Test configuration loading
    print_status "测试配置文件加载..."
    if python3 -c "
import yaml
with open('config_lite.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    print('✓ 配置文件加载成功')
    print(f'  - 系统名称: {config[\"system\"][\"name\"]}')
    print(f'  - 环境类型: {config[\"system\"][\"environment\"]}')
"; then
        print_success "配置文件测试通过"
    else
        print_error "配置文件测试失败"
        return 1
    fi
}

# Create startup script
create_startup_script() {
    print_header "创建启动脚本"
    
    cat > start_service.sh << 'EOF'
#!/bin/bash

# FinanceRAG-Pro Service Startup Script for Cloud Studio

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}启动 FinanceRAG-Pro 服务...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}错误: 虚拟环境不存在，请先运行 bash setup_cloud_studio.sh${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate

# Check if .env file exists and has API key
if [ -f ".env" ]; then
    if grep -q "your_openai_api_key_here" .env; then
        echo -e "${YELLOW}警告: 请在.env文件中设置您的API密钥${NC}"
    fi
else
    echo -e "${YELLOW}警告: .env文件不存在${NC}"
fi

# Start the service
echo -e "${GREEN}启动服务 (端口: 8000)...${NC}"
echo -e "${YELLOW}Cloud Studio会自动生成访问链接${NC}"
echo -e "${YELLOW}通常格式为: https://xxx-8000.preview.mycs.com${NC}"
echo ""

# Run the application
python3 main_lite.py
EOF

    chmod +x start_service.sh
    print_success "启动脚本创建完成: start_service.sh"
}

# Create quick test script caller
create_test_runner() {
    print_header "创建测试运行脚本"
    
    cat > run_tests.sh << 'EOF'
#!/bin/bash

# FinanceRAG-Pro Test Runner for Cloud Studio

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}FinanceRAG-Pro 测试运行器${NC}"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}虚拟环境已激活${NC}"
else
    echo -e "${YELLOW}警告: 虚拟环境不存在${NC}"
fi

echo ""
echo "选择测试类型:"
echo "1. 快速系统检查 (离线)"
echo "2. 单元测试"
echo "3. 集成测试 (需要服务运行)"
echo "4. 性能测试"
echo ""

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo -e "${YELLOW}运行快速系统检查...${NC}"
        python3 quick_test.py
        ;;
    2)
        echo -e "${YELLOW}运行单元测试...${NC}"
        if command -v pytest &> /dev/null; then
            pytest tests/unit/ -v
        else
            python3 -m pytest tests/unit/ -v
        fi
        ;;
    3)
        echo -e "${YELLOW}运行集成测试...${NC}"
        echo -e "${YELLOW}请确保服务正在运行 (python3 main_lite.py)${NC}"
        if command -v pytest &> /dev/null; then
            pytest tests/integration/ -v
        else
            python3 -m pytest tests/integration/ -v
        fi
        ;;
    4)
        echo -e "${YELLOW}运行性能测试...${NC}"
        python3 quick_test.py
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
EOF

    chmod +x run_tests.sh
    print_success "测试运行脚本创建完成: run_tests.sh"
}

# Display final instructions
show_completion_message() {
    print_header "安装完成"
    
    echo ""
    echo -e "${GREEN}🎉 FinanceRAG-Pro 已成功部署到Cloud Studio！${NC}"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}                        使用指南                          ${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}📋 下一步操作:${NC}"
    echo ""
    echo -e "1️⃣  ${BLUE}配置API密钥${NC}"
    echo -e "    nano .env"
    echo -e "    # 编辑OPENAI_API_KEY=your_actual_key_here"
    echo ""
    echo -e "2️⃣  ${BLUE}上传PDF文件${NC}"
    echo -e "    # 将财报PDF放入 data/pdfs/ 目录"
    echo ""
    echo -e "3️⃣  ${BLUE}启动服务${NC}"
    echo -e "    bash start_service.sh"
    echo -e "    # 或者: source venv/bin/activate && python3 main_lite.py"
    echo ""
    echo -e "4️⃣  ${BLUE}测试系统${NC}"
    echo -e "    bash run_tests.sh"
    echo -e "    # 或者: python3 quick_test.py"
    echo ""
    echo -e "5️⃣  ${BLUE}运行演示${NC}"
    echo -e "    python3 demo_script.py"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}                       重要提示                          ${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  API密钥配置${NC}"
    echo -e "    请在.env文件中设置您的实际API密钥"
    echo ""
    echo -e "${YELLOW}🌐  Cloud Studio访问${NC}"
    echo -e "    启动后系统会自动生成访问链接"
    echo -e "    通常格式: https://xxx-8000.preview.mycs.com"
    echo ""
    echo -e "${YELLOW}📁  文件管理${NC}"
    echo -e "    PDF文件: data/pdfs/"
    echo -e "    日志文件: logs/"
    echo -e "    数据库: data/"
    echo ""
    echo -e "${GREEN}✅ 系统已准备就绪，可以开始使用！${NC}"
    echo ""
}

# Main execution
main() {
    check_environment
    create_directories
    install_system_deps
    setup_virtual_env
    install_python_deps
    setup_configuration
    create_startup_script
    create_test_runner
    
    print_header "验证安装"
    if test_installation; then
        show_completion_message
    else
        print_error "安装验证失败，请检查错误信息"
        exit 1
    fi
}

# Run main function
main "$@"