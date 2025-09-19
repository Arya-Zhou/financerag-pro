# FinanceRAG-Pro

> 🚀 **项目已重构优化** - 文件结构更加清晰，部署更加简便！

## 📁 新的项目结构

```
FinanceRAG-Pro/
├── 📁 configs/                    # 配置文件目录
│   ├── .env.example             # 环境变量示例
│   ├── lite.yaml                 # 轻量级配置
│   ├── production.yaml           # 生产环境配置
│   └── config.py                 # 配置管理类
├── 📁 deploy/                     # 部署文件目录
│   ├── setup.sh                  # 统一部署脚本
│   ├── docker-compose.production.yml
│   ├── Dockerfile.production
│   └── nginx.conf
├── 📁 docs/                       # 文档目录
│   ├── INDEX.md                  # 文档索引
│   ├── README.md                 # 详细说明
│   ├── API_USAGE.md              # API使用指南
│   └── DEMO_QUERY_CASES.md       # 演示用例
├── 📁 core/                       # 核心模块
├── 📁 tests/                      # 测试文件
├── 📁 data/                       # 数据存储
├── main_lite.py                   # 主程序
└── requirements.txt              # 依赖文件
```

## 📋 环境要求

### 系统要求
- **Python**: 3.8+ (推荐 3.9 或 3.10)
- **操作系统**: Linux, macOS, Windows
- **内存**: 最少 2GB (推荐 4GB+)
- **磁盘空间**: 最少 1GB

### API 密钥要求
- **LLM服务密钥** (必需) - 默认使用 OpenAI API Key，可替代为其他兼容的LLM服务密钥
- **多模态模型密钥** (图文融合检索) - 默认使用 ModelScope API Key，可替代为其他多模态服务密钥
- **文本理解密钥** (文本检索优化) - 默认使用 SiliconFlow API Key，可替代为其他文本理解服务密钥

> **注意**: 至少需要配置一个LLM服务密钥才能正常运行系统。多模态和文本理解密钥为可选配置，用于增强相应功能。

## 🚀 快速开始

### 方法一：使用统一部署脚本（推荐）

```bash
# 轻量级部署（适合开发和Cloud Studio）
bash deploy/setup.sh lite

# Cloud Studio专用部署
bash deploy/setup.sh cloud-studio

# 生产环境部署
bash deploy/setup.sh production
```

### 方法二：手动部署

```bash
# 1. 创建环境配置
cp configs/.env.example .env
# 编辑 .env 添加API密钥

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python main_lite.py
```

## 📚 文档

详细文档请查看 **[docs/](./docs/)** 目录：

- **[完整文档](./docs/README.md)** - 项目详细说明
- **[API使用指南](./docs/API_USAGE.md)** - 接口文档
- **[实施路径](./docs/PROJECT_IMPLEMENTATION_ROADMAP.md)** - 部署指南
- **[演示用例](./docs/DEMO_QUERY_CASES.md)** - 测试案例

## 🔧 配置说明

### 环境配置
- 复制 `configs/.env.example` 为 `.env`
- 根据需要修改API密钥和其他配置

### 系统配置
- **轻量级**: `configs/lite.yaml`
- **生产环境**: `configs/production.yaml`

## ⚡ 快速测试

```bash
# 系统健康检查
python quick_test.py

# 运行演示
python demo_script.py
```

## 🎯 核心特性

- 🚀 **查询驱动的智能路由**
- 📊 **深度多模态理解** 
- 🔍 **多路径并行检索**
- ✅ **冲突检测与验证**
- ⚡ **高性能优化**

---

**注意**: 项目结构已优化，如遇到路径问题请参考新的目录结构。