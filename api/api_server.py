import argparse
import json
import logging
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import uuid
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from engine_analysis import AICompanyAnalyzer


class AnalysisRequest(BaseModel):
    """API分析请求模型"""
    companies: List[str] = Field(
        default=["OpenAI (GPT系列)", "Google (Gemini)", "Anthropic (Claude)"],
        description="要分析的公司列表"
    )
    max_workers: int = Field(default=1, ge=1, le=10, description="并发工作线程数")
    api_keys: Optional[List[str]] = Field(default=None, description="API密钥列表")
    inference_key: Optional[str] = Field(default=None, description="推理API密钥")


class AnalysisResponse(BaseModel):
    """API分析响应模型"""
    task_id: str
    status: str
    companies: List[str]
    timestamp: str
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AnalysisTask:
    """分析任务管理器"""
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    def create_task(self, request: AnalysisRequest) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        
        # 模拟argparse参数
        args = argparse.Namespace()
        args.companies = request.companies
        args.max_workers = request.max_workers
        args.api_keys = request.api_keys or []
        args.inference_key = request.inference_key
        args.log_level = "INFO"
        
        analyzer = AICompanyAnalyzer(args)
        
        # 更新公司列表
        analyzer.target_companies = request.companies
        
        self.tasks[task_id] = {
            "status": "running",
            "request": request.dict(),
            "analyzer": analyzer,
            "start_time": datetime.now(),
            "results": None,
            "error": None
        }
        
        return task_id
    
    def update_task(self, task_id: str, results: List[Dict[str, Any]] = None, error: str = None):
        """更新任务状态"""
        if task_id in self.tasks:
            if results:
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["results"] = results
                self.tasks[task_id]["end_time"] = datetime.now()
            elif error:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = error
                self.tasks[task_id]["end_time"] = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self.tasks.get(task_id)


# 创建FastAPI应用
app = FastAPI(
    title="AI公司分析API",
    description="基于FastAPI的AI公司综合分析服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局任务管理器
task_manager = AnalysisTask()


def _ensure_api_keys_if_needed(request: AnalysisRequest) -> None:
    """确保API密钥已设置"""
    if not request.api_keys:
        request.api_keys = [os.environ.get("OPENAI_API_KEY", "")]
    if not request.inference_key:
        request.inference_key = os.environ.get("OPENAI_API_KEY", "").strip()


def run_analysis(task_id: str, request: AnalysisRequest):
    """后台运行分析任务"""
    try:
        _ensure_api_keys_if_needed(request)
        
        # 模拟argparse参数
        args = argparse.Namespace()
        args.companies = request.companies
        args.max_workers = request.max_workers
        args.api_keys = request.api_keys or []
        args.inference_key = request.inference_key
        args.log_level = "INFO"
        
        analyzer = AICompanyAnalyzer(args)
        analyzer.target_companies = request.companies
        
        # 执行分析
        results = analyzer.analyze_all_companies(max_workers=request.max_workers)
        
        # 生成总结报告
        try:
            summary = analyzer.generate_summary_report(results)
        except Exception as e:
            summary = {"error": f"生成总结报告失败: {str(e)}"}
        
        # 更新任务状态
        task_manager.update_task(task_id, results, None)
        
    except Exception as e:
        logging.error(f"分析任务 {task_id} 失败: {str(e)}")
        task_manager.update_task(task_id, None, str(e))


@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "AI公司分析API服务",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - 提交分析任务",
            "status": "GET /status/{task_id} - 查询任务状态",
            "results": "GET /results/{task_id} - 获取分析结果"
        }
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_companies(
    request: AnalysisRequest, 
    background_tasks: BackgroundTasks
):
    """提交AI公司分析任务"""
    try:
        # 验证请求
        if not request.companies:
            raise HTTPException(status_code=400, detail="公司列表不能为空")
        
        # 创建任务
        task_id = task_manager.create_task(request)
        
        # 在后台运行分析任务
        background_tasks.add_task(run_analysis, task_id, request)
        
        return AnalysisResponse(
            task_id=task_id,
            status="running",
            companies=request.companies,
            timestamp=datetime.now().isoformat(),
            message="分析任务已提交，正在后台运行"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建分析任务失败: {str(e)}")


@app.get("/status/{task_id}", response_model=AnalysisResponse)
async def get_task_status(task_id: str):
    """查询分析任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    status_info = {
        "task_id": task_id,
        "status": task["status"],
        "companies": task["request"]["companies"],
        "timestamp": datetime.now().isoformat(),
        "message": f"任务状态: {task['status']}"
    }
    
    if task["status"] == "completed":
        status_info["message"] = "分析已完成"
    elif task["status"] == "failed":
        status_info["message"] = f"分析失败: {task.get('error', '未知错误')}"
        status_info["error"] = task.get("error")
    
    return AnalysisResponse(**status_info)


@app.get("/results/{task_id}", response_model=AnalysisResponse)
async def get_analysis_results(task_id: str):
    """获取分析结果"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task["status"] == "running":
        raise HTTPException(status_code=425, detail="分析仍在进行中")
    
    if task["status"] == "failed":
        raise HTTPException(
            status_code=500, 
            detail=f"分析失败: {task.get('error', '未知错误')}"
        )
    
    # 生成总结报告
    summary = {}
    if task["results"]:
        try:
            analyzer = task["analyzer"]
            summary = analyzer.generate_summary_report(task["results"])
        except Exception as e:
            summary = {"error": f"生成总结报告失败: {str(e)}"}
    
    return AnalysisResponse(
        task_id=task_id,
        status="completed",
        companies=task["request"]["companies"],
        timestamp=datetime.now().isoformat(),
        message="分析已完成",
        results=task["results"],
        summary=summary
    )


@app.get("/tasks")
async def list_tasks():
    """列出所有任务"""
    tasks_info = []
    for task_id, task in task_manager.tasks.items():
        tasks_info.append({
            "task_id": task_id,
            "status": task["status"],
            "companies": task["request"]["companies"],
            "start_time": task["start_time"].isoformat(),
            "end_time": task.get("end_time", ""),
            "error": task.get("error")
        })
    
    return {
        "total_tasks": len(tasks_info),
        "tasks": tasks_info
    }


def main():
    """启动API服务器"""
    parser = argparse.ArgumentParser(description="AI公司分析API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    print(f"AI公司分析API服务启动中...")
    print(f"访问地址: http://{args.host}:{args.port}")
    print(f"文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )



if __name__ == "__main__":
    main()