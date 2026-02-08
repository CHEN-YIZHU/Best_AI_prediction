# 🔮 AI Ranking Prediction System

Predict future AI company rankings using **LMArena data** and **LLM-based web search**.

Focus: **Text/Chat Models** (LLM)

## 🚀 Quick Start

### Option 1: CLI

```bash
# Install dependencies
pip install -r requirements_api.txt
pip install playwright && playwright install chromium

# Run prediction (Default: OpenAI, Google, Anthropic, xAI, DeepSeek)
python run_prediction.py
```

### Option 2: API Server

```bash
# Start server
python -m api.api_server --reload

# Send request (Synchronous, wait for result)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{}' --max-time 300
```

## 📊 Features

| Feature | Description |
|---|---|
| **Arena Precise Data** | Directly scrape text model Elo scores from lmarena.ai |
| **Timeline Analysis** | Web search for new model leaks, rumors, and release cycles |
| **Forecast** | Predict rankings for Now / 1 Month / 3 Months / 6 Months |

## 📁 Structure

```
Best_AI_prediction/
├── run_prediction.py      # Entry point
├── prediction_engine.py   # Core logic
├── arena_scraper.py       # Playwright scraper
├── api_wrapper.py         # API wrapper
├── .env                   # Config
└── api/                   # API Server
```

## ⚙️ Configuration

Set up `.env`:

```bash
AI_PROVIDER=qwen
AI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=qwen-max
```

## 📝 License

MIT
