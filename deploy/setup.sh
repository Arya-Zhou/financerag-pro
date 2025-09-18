#!/bin/bash

# ===================================================================
# FinanceRAG-Pro 统一部署脚本
# 支持多种部署模式：lite, cloud-studio, production
# ===================================================================

set -e  # Exit on any error

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 输出函数
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

# 显示横幅
show_banner() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${CYAN}     FinanceRAG-Pro 统一部署脚本              ${NC}"
    echo -e "${CYAN}     金融多模态智能检索系统部署工具            ${NC}"
    echo -e "${CYAN}================================================${NC}"
    echo ""
}

# 显示使用说明
show_usage() {
    cat << EOF
使用方法:
    bash deploy/setup.sh [模式] [选项]

部署模式:
    lite            轻量级部署（适合开发和测试）
    cloud-studio    Cloud Studio专用部署
    production      生产环境部署（需要Docker）

选项:
    -h, --help      显示帮助信息
    -v, --verbose   详细输出
    --skip-deps     跳过依赖检查
    --force         强制重新部署

示例:
    bash deploy/setup.sh lite
    bash deploy/setup.sh cloud-studio --verbose
    bash deploy/setup.sh production --force

EOF
}

# 检查Python环境
check_python() {
    print_header "检查Python环境"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 未安装，请先安装Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        print_error "需要Python 3.8+，当前版本: $PYTHON_VERSION"
        exit 1
    fi
    
    print_status "Python版本检查通过: $PYTHON_VERSION"
}

# 检查环境
check_environment() {
    print_header "检查运行环境"
    
    # 检测Cloud Studio
    if [ -n "$CLOUDSTUDIO" ] || [ -n "$CS_PROJECT_PATH" ]; then
        ENV_TYPE="cloud-studio"
        print_status "检测到Cloud Studio环境"
    elif command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        ENV_TYPE="production"
        print_status "检测到Docker环境，支持生产部署"
    else
        ENV_TYPE="lite"
        print_status "使用轻量级环境"
    fi
}

# 创建目录结构
create_directories() {
    print_header "创建目录结构"
    
    mkdir -p data/{pdfs,chromadb,sqlite,cache}
    mkdir -p logs
    mkdir -p models
    mkdir -p checkpoints
    
    print_status "目录结构创建完成"
}

# 创建虚拟环境
setup_virtualenv() {
    print_header "设置Python虚拟环境"
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "虚拟环境创建完成"
    else
        print_warning "虚拟环境已存在，跳过创建"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    print_status "虚拟环境配置完成"
}

# 轻量级部署
deploy_lite() {
    print_header "执行轻量级部署"
    
    # 安装轻量级依赖
    cat > requirements_lite.txt << EOF
# 核心框架
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# 文档处理
PyPDF2==3.0.1
pdfplumber==0.10.3
pillow==10.1.0

# ML和NLP (CPU版本)
torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
transformers==4.35.0
sentence-transformers==2.2.2

# 向量数据库
chromadb==0.4.22

# 工具库
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
tqdm==4.66.1
python-dotenv==1.0.0
pyyaml==6.0.1
aiofiles==23.2.1
httpx==0.25.2
loguru==0.7.2
cachetools==5.3.2
EOF
    
    pip install -r requirements_lite.txt
    
    # 创建环境文件
    if [ ! -f ".env" ]; then
        cp configs/.env.template .env
        print_status "环境配置文件已创建，请编辑 .env 添加API密钥"
    fi
    
    # 使用新的配置文件路径
    if [ ! -f "configs/lite.yaml" ]; then
        print_warning "配置文件 configs/lite.yaml 不存在，将创建默认配置"
    fi
    
    print_success "轻量级部署完成"
}

# Cloud Studio部署
deploy_cloud_studio() {
    print_header "执行Cloud Studio专用部署"
    
    # Cloud Studio特定优化
    export TOKENIZERS_PARALLELISM=false
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 安装依赖（Cloud Studio优化版本）
    pip install -r requirements.txt
    
    # 创建Cloud Studio配置
    if [ ! -f ".env" ]; then
        cp configs/.env.template .env
        # Cloud Studio特定配置
        cat >> .env << EOF

# Cloud Studio特定配置
LITE_MODE=true
MAX_WORKERS=2
ASYNC_TIMEOUT=60
USE_LOCAL_MODEL=false
EOF
        print_status "Cloud Studio环境配置已创建"
    fi
    
    print_success "Cloud Studio部署完成"
}

# 生产环境部署
deploy_production() {
    print_header "执行生产环境部署"
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，生产部署需要Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose未安装"
        exit 1
    fi
    
    # 创建环境文件
    if [ ! -f ".env" ]; then
        cp configs/.env.template .env
        print_warning "请编辑 .env 文件配置生产环境参数"
    fi
    
    # 构建和启动服务
    print_status "构建Docker镜像..."
    docker-compose -f deploy/docker-compose.production.yml build
    
    print_status "启动生产服务..."
    docker-compose -f deploy/docker-compose.production.yml up -d
    
    # 检查服务状态
    sleep 5
    if docker-compose -f deploy/docker-compose.production.yml ps | grep -q "Up"; then
        print_success "生产环境部署完成"
        print_status "服务状态："
        docker-compose -f deploy/docker-compose.production.yml ps
    else
        print_error "服务启动失败，请检查日志"
        docker-compose -f deploy/docker-compose.production.yml logs
        exit 1
    fi
}

# 验证部署
verify_deployment() {
    print_header "验证部署结果"
    
    # 检查主要文件
    if [ -f "main_lite.py" ]; then
        print_status "✓ 主程序文件存在"
    else
        print_error "✗ 主程序文件缺失"
        return 1
    fi
    
    # 检查配置
    if [ -f ".env" ]; then
        print_status "✓ 环境配置文件存在"
    else
        print_error "✗ 环境配置文件缺失"
        return 1
    fi
    
    # 检查数据目录
    if [ -d "data" ]; then
        print_status "✓ 数据目录存在"
    else
        print_error "✗ 数据目录缺失"
        return 1
    fi
    
    print_success "部署验证通过"
}

# 显示完成信息
show_completion() {
    print_success "========================================="
    print_success "部署完成！"
    print_success "========================================="
    echo ""
    
    case $DEPLOY_MODE in
        lite|cloud-studio)
            echo "启动命令:"
            echo "  source venv/bin/activate"
            echo "  python main_lite.py"
            echo ""
            echo "访问地址:"
            echo "  http://localhost:8000"
            echo "  API文档: http://localhost:8000/docs"
            ;;
        production)
            echo "管理命令:"
            echo "  docker-compose -f deploy/docker-compose.production.yml ps"
            echo "  docker-compose -f deploy/docker-compose.production.yml logs"
            echo "  docker-compose -f deploy/docker-compose.production.yml down"
            ;;
    esac
    
    echo ""
    echo "注意事项:"
    echo "1. 请编辑 .env 文件添加必要的API密钥"
    echo "2. 将PDF文档放入 data/pdfs/ 目录"
    echo "3. 运行测试: python quick_test.py"
}

# 主函数
main() {
    # 解析参数
    DEPLOY_MODE=""
    VERBOSE=false
    SKIP_DEPS=false
    FORCE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            lite|cloud-studio|production)
                DEPLOY_MODE="$1"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定模式，自动检测
    if [ -z "$DEPLOY_MODE" ]; then
        check_environment
        DEPLOY_MODE="$ENV_TYPE"
        print_warning "自动选择部署模式: $DEPLOY_MODE"
    fi
    
    show_banner
    
    # 基础检查
    if [ "$SKIP_DEPS" = false ]; then
        check_python
    fi
    
    # 创建目录
    create_directories
    
    # 根据模式执行部署
    case $DEPLOY_MODE in
        lite)
            setup_virtualenv
            deploy_lite
            ;;
        cloud-studio)
            setup_virtualenv
            deploy_cloud_studio
            ;;
        production)
            deploy_production
            ;;
        *)
            print_error "不支持的部署模式: $DEPLOY_MODE"
            show_usage
            exit 1
            ;;
    esac
    
    # 验证部署
    verify_deployment
    
    # 显示完成信息
    show_completion
}

# 执行主函数
main "$@"