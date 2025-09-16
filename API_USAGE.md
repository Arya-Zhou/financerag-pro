# FinanceRAG-Pro API配置使用说明

## 概述

本项目已替换原有的OpenAI API调用，现在使用：
- **硅基流动 Qwen3-8B**：用于查询理解和文本生成
- **魔搭 Qwen2.5-VL-72B**：用于图表视觉理解

## 主要特性

1. **速率限制管理**：
   - 自动RPM/TPM限制
   - 最小请求间隔控制
   - 自动重试机制

2. **批量处理支持**：
   - 批量模式自动调整限制参数
   - 支持断点续传
   - 并发控制

3. **断点续传**：
   - 自动保存处理进度
   - 支持从中断点恢复
   - 避免重复处理

## 配置步骤

### 1. 获取API密钥

#### 硅基流动（Qwen3-8B）
1. 访问 [https://siliconflow.cn](https://siliconflow.cn)
2. 注册并获取API密钥
3. 查看API限制：
   - 标准限制：1000 RPM, 5000 TPM
   - 批量限制：100 RPM, 50000 TPM

#### 魔搭（Qwen2.5-VL-72B）
1. 访问 [https://modelscope.cn](https://modelscope.cn)
2. 注册并申请API访问权限
3. 查看API限制：
   - 标准限制：30 RPM, 3000 TPM
   - 批量限制：10 RPM, 10000 TPM

### 2. 设置环境变量

复制环境变量模板：
```bash
cp .env.api .env
```

编辑 `.env` 文件：
```bash
# 硅基流动API密钥
SILICONFLOW_API_KEY=your_actual_siliconflow_key_here

# 魔搭API密钥
MODELSCOPE_API_KEY=your_actual_modelscope_key_here

# 批处理模式（处理大量文档时设置为true）
BATCH_MODE=false
BATCH_SIZE=10
```

### 3. 使用配置

#### 标准模式运行
```python
# 使用默认配置
python main.py
```

#### 批量处理模式
```python
# 启用批量模式处理大量PDF
BATCH_MODE=true python batch_process.py
```

## API调用示例

### 单个查询
```python
from core.api_client import SiliconFlowClient

# 初始化客户端
client = SiliconFlowClient(api_key="your_key", batch_mode=False)

# 生成文本
response = await client.generate_text(
    prompt="分析苹果公司的财务状况",
    temperature=0.7,
    max_tokens=1000
)
```

### 批量处理
```python
from core.api_client import SiliconFlowClient

# 批量模式客户端
client = SiliconFlowClient(api_key="your_key", batch_mode=True)

# 批量处理项目
items = ["query1", "query2", "query3", ...]

async def process_item(item):
    return await client.generate_text(item)

# 批量处理（自动控制速率）
results = await client.batch_process(
    items=items,
    process_func=process_item,
    task_id="batch_001"
)
```

### 图像分析
```python
from core.api_client import ModelScopeVLClient

# 初始化视觉语言模型客户端
vl_client = ModelScopeVLClient(api_key="your_key", batch_mode=False)

# 分析图表
result = await vl_client.analyze_image(
    image_base64=image_data,
    prompt="提取这个财务图表的数据",
    temperature=0.1
)
```

## 批量处理PDF文档

### 准备工作
1. 将PDF文件放入 `./data/pdfs/` 目录
2. 确保已设置API密钥

### 运行批量处理
```bash
# 标准批量处理
python batch_process.py

# 启用更大批次（注意API限制）
BATCH_SIZE=20 python batch_process.py
```

### 断点续传
如果处理中断，脚本会自动从断点恢复：
```bash
# 重新运行相同命令即可继续
python batch_process.py
```

断点文件保存在 `./checkpoints/` 目录。

## 速率限制参数

### 硅基流动（Qwen3-8B）

| 模式 | RPM | TPM | 最小间隔 |
|-----|-----|-----|---------|
| 标准 | 1000 | 5000 | 0.06秒 |
| 批量 | 100 | 50000 | 0.6秒 |

### 魔搭（Qwen2.5-VL-72B）

| 模式 | RPM | TPM | 最小间隔 |
|-----|-----|-----|---------|
| 标准 | 30 | 3000 | 2秒 |
| 批量 | 10 | 10000 | 6秒 |

## 监控与日志

### 查看处理日志
```bash
tail -f batch_processing.log
```

### 查看断点状态
```bash
ls -la ./checkpoints/
```

### 查看处理报告
```bash
cat batch_processing_report.txt
```

## 故障排除

### 1. API调用失败
- 检查API密钥是否正确
- 确认账户余额充足
- 查看是否达到速率限制

### 2. 断点恢复失败
- 删除损坏的断点文件：`rm ./checkpoints/*.pkl`
- 重新开始处理

### 3. 内存不足
- 减小批处理大小：`BATCH_SIZE=5`
- 限制并发数（在batch_process.py中调整max_concurrent）

### 4. 处理速度慢
- 适当增加批处理大小
- 检查网络连接
- 考虑使用批量模式

## 性能优化建议

1. **批量处理**：处理多个文档时始终使用批量模式
2. **合理设置批次大小**：根据API限制和系统资源调整
3. **使用断点续传**：长时间任务启用checkpoint
4. **监控API使用**：定期检查API调用量和费用

## 费用估算

### 硅基流动（Qwen3-8B）
- 输入：约 ¥0.008/1K tokens
- 输出：约 ¥0.008/1K tokens

### 魔搭（Qwen2.5-VL-72B）  
- 输入：约 ¥0.02/1K tokens
- 输出：约 ¥0.02/1K tokens
- 图像：约 ¥0.01/张

### 处理132个PDF（400MB）估算
- 文本处理：约 ¥50-100
- 图表分析：约 ¥20-50
- 总计：约 ¥70-150

## 注意事项

1. **API密钥安全**：
   - 不要将密钥提交到代码仓库
   - 使用环境变量管理密钥
   - 定期更换密钥

2. **成本控制**：
   - 监控API使用量
   - 合理使用批量模式
   - 设置预算警报

3. **合规要求**：
   - 确保数据处理符合隐私法规
   - 不要上传敏感财务数据到公共API

## 联系支持

- 硅基流动支持：[https://siliconflow.cn/support](https://siliconflow.cn/support)
- 魔搭支持：[https://modelscope.cn/help](https://modelscope.cn/help)
- 项目问题：提交GitHub Issue