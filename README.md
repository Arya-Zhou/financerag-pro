# FinanceRAG-Pro

基于查询驱动的金融多模态智能检索系统

## 项目特性

- 🚀 **查询驱动的智能路由**：基于LLM的查询理解和动态策略组合
- 📊 **深度多模态理解**：支持PDF文档、表格、图表的结构化解析
- 🔍 **多路径并行检索**：实体检索、语义检索、关键词检索的智能融合
- ✅ **冲突检测与验证**：多级验证机制确保答案可信度
- ⚡ **高性能优化**：支持T4单卡部署，毫秒级响应

## 快速开始

### 环境要求

**基础要求：**
- Python 3.8+
- 4GB+ RAM (轻量级) / 8GB+ RAM (完整版)
- OpenAI API Key 或其他支持的API Key

**完整部署要求：**
- Docker & Docker Compose
- NVIDIA GPU (可选，用于本地模型推理)

### 🚀 Cloud Studio 快速部署 (推荐)

Cloud Studio是一个零配置的云端开发环境，最适合快速体验FinanceRAG-Pro。

#### 1. 创建Cloud Studio环境
```bash
# 在Cloud Studio中创建新的工作空间
# 选择 Python 3.9 环境模板
```

#### 2. 克隆项目并部署
```bash
# 克隆项目
git clone https://github.com/Arya-Zhou/financerag-pro.git
cd financerag-pro

# 一键轻量级部署
bash setup_lite.sh

# 配置API密钥
nano .env
# 添加: OPENAI_API_KEY=your_api_key_here
```

#### 3. 启动服务
```bash
# 启动轻量级服务
python main_lite.py
```

#### 4. 访问应用
```bash
# Cloud Studio会自动提供访问URL
# 通常格式为: https://xxx-8000.preview.mycs.com
# 访问API文档: {YOUR_URL}/docs
```

### 💻 本地部署

#### 方式一：Docker部署 (推荐生产环境)
```bash
# 1. 克隆项目
git clone https://github.com/Arya-Zhou/financerag-pro.git
cd financerag-pro

# 2. 生产级部署
docker-compose -f docker-compose.production.yml up -d

# 3. 轻量级部署
bash setup_lite.sh
```

#### 方式二：手动安装
```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境
cp .env.example .env
# 编辑 .env 文件，添加API密钥

# 4. 启动服务
python main_lite.py
```

## 使用方法

### 上传文档

```python
import requests

# 上传PDF文档
with open("financial_report.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
    print(response.json())
```

### 查询接口

```python
# 发送查询
query_data = {
    "query": "苹果公司2023年的研发投入是多少？",
    "top_k": 10
}

response = requests.post(
    "http://localhost:8000/query",
    json=query_data
)

result = response.json()
print(f"答案: {result['answer']}")
print(f"置信度: {result['confidence']}")
print(f"来源: {result['sources']}")
```

## API文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
financerag-pro/
├── config/                      # 配置管理
│   └── config.py
├── core/                       # 核心模块
│   ├── query_engine.py         # 查询理解与路由
│   ├── document_processor.py   # 文档处理
│   ├── entity_linker.py       # 实体链接
│   ├── retrieval_engine.py    # 检索引擎
│   ├── validation_engine.py   # 验证引擎
│   ├── inference_engine.py    # 推理引擎
│   ├── config_manager.py      # 配置管理器
│   ├── api_client.py          # API客户端
│   └── lightweight_storage.py # 轻量级存储
├── tests/                      # 测试文件
│   ├── unit/                  # 单元测试
│   ├── integration/           # 集成测试
│   ├── fixtures/              # 测试数据
│   └── mocks/                 # 模拟对象
├── data/                       # 数据存储
├── logs/                       # 日志文件
├── main_lite.py               # 轻量级应用入口
├── batch_process.py           # 批量处理脚本
├── requirements.txt           # Python依赖
├── config_lite.yaml           # 轻量级配置
├── config_api.yaml            # API配置
├── Dockerfile.production      # 生产级Docker配置
├── docker-compose.production.yml # 生产级编排配置
├── setup_lite.sh             # 轻量级部署脚本
├── deploy_production.sh      # 生产部署脚本
└── PROJECT_IMPLEMENTATION_ROADMAP.md # 实施路径指南
```

## 配置说明

### 轻量级配置 (Cloud Studio推荐)

主要配置项在 `config_lite.yaml` 中：

```yaml
system:
  name: FinanceRAG-Pro-Lite
  environment: production

models:
  embedding:
    model: sentence-transformers/all-MiniLM-L6-v2  # 轻量级嵌入模型
    dimension: 384
  generation:
    use_api: true
    model: gpt-3.5-turbo  # 使用API调用

database:
  sqlite:
    path: ./data/financerag.db  # SQLite数据库
  chromadb:
    persist_directory: ./data/chromadb  # 向量数据库

retrieval:
  chunk_size: 512
  top_k: 10
  similarity_threshold: 0.7
```

### 完整配置

详细配置项在 `config_api.yaml` 中：

```yaml
models:
  query_understanding:
    model: gpt-3.5-turbo  # LLM模型
  embedding:
    model: BAAI/bge-m3    # 向量化模型
  generation:
    model: Qwen/Qwen2.5-7B-Instruct-GPTQ  # 生成模型

database:
  postgresql:
    host: localhost       # 数据库配置
  redis:
    host: localhost       # 缓存配置
  
retrieval:
  chunk_size: 512        # 文本块大小
  top_k: 20             # 检索数量
  similarity_threshold: 0.7  # 相似度阈值
```

## 性能指标

- **查询延迟**: < 500ms (简单查询)
- **并发支持**: 15 QPS
- **准确率**: 95%+ (事实性查询)
- **F1分数**: 0.92 (金融问答测试集)

## 开发指南

### 本地开发

```bash
# 创建虚拟环境
python -m venv venv

source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt

# 运行测试
pytest tests/
```

### Cloud Studio开发

```bash
# 在Cloud Studio中直接开发
git clone https://github.com/Arya-Zhou/financerag-pro.git
cd financerag-pro

# 安装依赖
pip install -r requirements.txt

# 运行单元测试
pytest tests/unit/ -v

# 启动开发服务器
python main_lite.py
```

### 测试框架

项目包含完整的测试套件：

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/ -m unit

# 运行集成测试
pytest tests/integration/ -m integration

# 生成覆盖率报告
pytest --cov=core --cov-report=html
```

### 添加新的检索策略

1. 在 `core/retrieval_engine.py` 中实现新的 Retriever 类
2. 在 `QueryRouter` 中添加路由规则
3. 更新配置文件
4. 添加相应的测试用例

## 故障排除

### Cloud Studio常见问题

1. **端口访问问题**
   ```bash
   # 确保服务运行在正确端口
   python main_lite.py
   # Cloud Studio会自动生成访问链接
   ```

2. **内存限制**
   ```bash
   # 使用轻量级配置
   # 调小batch_size和embedding维度
   ```

3. **API密钥配置**
   ```bash
   # 检查环境变量
   echo $OPENAI_API_KEY
   # 或编辑.env文件
   nano .env
   ```

### 本地开发常见问题

1. **内存不足**
   - 减小 batch_size
   - 使用更小的模型（如all-MiniLM-L6-v2）
   - 关闭不必要的后台程序

2. **GPU内存溢出**
   - 使用量化模型
   - 调整 max_length
   - 设置 `use_api: true` 使用在线API

3. **连接错误**
   - 检查服务是否启动
   - 验证端口是否被占用
   - 检查防火墙设置

4. **依赖安装问题**
   ```bash
   # 升级pip
   pip install --upgrade pip
   
   # 使用国内镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

### 性能优化建议

1. **Cloud Studio环境**
   - 使用轻量级模型
   - 启用缓存机制
   - 限制并发数量

2. **生产环境**
   - 使用GPU加速
   - 配置负载均衡
   - 启用监控和日志

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- 项目作者：Arya_Zhou
- Email: 3114481390@qq.com
- GitHub: https://github.com/Arya_Zhou/financerag-pro