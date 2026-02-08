import argparse
import logging
import os
import sys
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 添加项目根目录到路径，以便导入 prediction_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_engine import PredictionEngine


# 定义请求模型
class PredictionRequest(BaseModel):
    companies: List[str] = Field(
        default=["OpenAI", "Google", "Anthropic", "xAI", "DeepSeek"],
        description="要分析的AI公司列表"
    )

class PredictionResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# 全局任务存储
# {task_id: {status: 'pending/running/completed/failed', result: ..., error: ...}}
tasks: Dict[str, Dict[str, Any]] = {}


def run_prediction_task(task_id: str, companies: List[str]):
    """后台运行预测任务"""
    try:
        tasks[task_id]["status"] = "running"
        engine = PredictionEngine(companies=companies)
        result = engine.run()
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        logging.error(f"Prediction task {task_id} failed: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="AI Prediction API",
    description="AI Company Ranking Prediction API",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "AI Prediction API",
        "endpoints": {
            "predict": "POST /predict",
            "status": "GET /status/{task_id}"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest):
    """提交并等待预测结果 (同步模式)"""
    task_id = str(uuid.uuid4())
    
    try:
        # 记录任务开始
        tasks[task_id] = {
            "id": task_id,
            "status": "running",
            "companies": request.companies,
            "created_at": datetime.now().isoformat()
        }
        
        # 同步运行预测 (这会阻塞连接直到完成)
        engine = PredictionEngine(companies=request.companies)
        result = engine.run()
        
        # 更新并返回结果
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        return PredictionResponse(
            task_id=task_id,
            status="completed",
            message="Prediction completed",
            result=result
        )
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        logging.error(f"Prediction task {task_id} failed: {e}")
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/status/{task_id}", response_model=PredictionResponse)
async def get_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return PredictionResponse(
        task_id=task_id,
        status=task["status"],
        message=f"Task is {task['status']}",
        result=task.get("result"),
        error=task.get("error")
    )


def main():
    """启动API服务器"""
    parser = argparse.ArgumentParser(description="AI Prediction API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Auto reload")
    
    args = parser.parse_args()
    
    print(f"Starting API Server at http://{args.host}:{args.port}")
    
    uvicorn.run(
        "api.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()