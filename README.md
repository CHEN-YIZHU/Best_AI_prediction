# AI公司分析API 使用指南


### 文件说明

| 文件 | 功能描述 |
|------|----------|
| `api_server.py` | FastAPI主服务器，提供RESTful API接口 |
| `requirements_api.txt` | 依赖package |
| `start_api.sh` | 一键启动脚本 |
| `api_example.py` | API使用示例和测试代码 |


## 快速启动API服务

1. 安装依赖
```bash
pip install -r requirements_api.txt
```

2. 启动服务
```bash
# 从项目根目录启动
python -m api.api_server --reload
```

## 结果获取

### 1. 使用一键测试脚本 (推荐)
```bash
python call_api.py
```

### 2. Python手动调用示例
```python
import requests

# 提交分析任务
response = requests.post("http://localhost:8000/analyze", json={
    "companies": ["OpenAI (GPT系列)", "Google (Gemini)", "字节跳动 (Doubao)"],
    "max_workers": 1
})

# 获取任务ID并查询结果
task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# ... (稍后查询)
results = requests.get(f"http://localhost:8000/results/{task_id}").json()
print(results)
```

### 3. 使用原有脚本
```bash
# 需要修改api key
bash scripts/fetch_result.sh
```

### 3. cURL命令行
```bash
# 提交任务
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"companies": ["OpenAI", "Google"], "max_workers": 1}'

# 查询结果
curl "http://localhost:8000/results/{task_id}"
```

## 环境配置

### API密钥设置
```bash
export AI_PROVIDER=qwen
export AI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=sk..
export OPENAI_MODEL=qwen-max
```

### 配置参数说明
- `companies`: 要分析的AI公司列表
- `max_workers`: 并发线程数(1-10)
- `api_keys`: 资料收集API Key
- `inference_key`: 推理API密钥
