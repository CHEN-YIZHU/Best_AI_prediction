"""
AI公司分析API使用示例
"""

import requests
import json
import time

# API基础URL
BASE_URL = "http://localhost:8000"

def test_api():
    
    print("🧪 测试AI公司分析API...")
    
    # 1. 检查API状态
    try:
        response = requests.get(f"{BASE_URL}/")
        print("API服务状态正常")
        print(f"服务信息: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("无法连接到API服务，请先启动服务器")
        return
    
    # 2. 提交分析任务
    analysis_request = {
        "companies": ["Google (Gemini)", "Anthropic (Claude)", "OpenAI (GPT系列)"],
        "max_workers": 10,
        # "api_keys": ["your-api-key-here"],  # 可选
        # "inference_key": "your-inference-key"  # 可选
    }
    
    print("\n📨 提交分析任务...")
    response = requests.post(f"{BASE_URL}/analyze", json=analysis_request)
    
    if response.status_code == 200:
        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"任务提交成功，任务ID: {task_id}")
        print(f"分析公司: {task_data['companies']}")
        print(f"状态: {task_data['status']}")
    else:
        print(f"任务提交失败: {response.text}")
        return
    
    # 3. 轮询任务状态
    print("\n 等待分析完成...")
    max_attempts = 60  # 最多等待5分钟
    attempt = 0
    
    while attempt < max_attempts:
        status_response = requests.get(f"{BASE_URL}/status/{task_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                print("✅ 分析已完成！")
                break
            elif status_data["status"] == "failed":
                print(f"❌ 分析失败: {status_data.get('error', '未知错误')}")
                return
            else:
                print(f"⏳ 分析进行中... ({attempt + 1}/{max_attempts})")
        else:
            print(f"❌ 查询状态失败: {status_response.text}")
            return
        
        time.sleep(5)  # 每5秒查询一次
        attempt += 1
    
    if attempt >= max_attempts:
        print("⏰ 分析超时，请稍后手动查询结果")
        return
    
    # 4. 获取分析结果
    print("\n📊 获取分析结果...")
    results_response = requests.get(f"{BASE_URL}/results/{task_id}")
    
    if results_response.status_code == 200:
        results_data = results_response.json()
        
        print("🎯 分析结果摘要:")
        if results_data.get("summary"):
            summary = results_data["summary"]
            print(f"📈 分析公司数: {summary.get('summary', {}).get('total_companies_analyzed', 0)}")
            print(f"🏆 最佳表现: {summary.get('summary', {}).get('top_performer', 'N/A')}")
            print(f"⭐ 最高得分: {summary.get('summary', {}).get('top_score', 0):.3f}")
        
        if results_data.get("results"):
            print("\n📋 详细排名:")
            for i, company in enumerate(results_data["results"][:5], 1):
                print(f"{i}. {company['company']}: {company.get('final_score', 0):.3f}")
        
        # 保存结果到文件
        with open(f"analysis_result_{task_id}.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: analysis_result_{task_id}.json")
        
    else:
        print(f"❌ 获取结果失败: {results_response.text}")


def quick_analysis(companies=None):
    """快速分析函数"""
    if companies is None:
        companies = ["OpenAI (GPT系列)", "Google (Gemini)"]
    
    analysis_request = {
        "companies": companies,
        "max_workers": 1
    }
    
    # 提交任务
    response = requests.post(f"{BASE_URL}/analyze", json=analysis_request)
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        print(f"✅ 任务提交成功，任务ID: {task_id}")
        return task_id
    else:
        print(f"❌ 任务提交失败: {response.text}")
        return None


if __name__ == "__main__":
    # 测试API
    test_api()
    
    # 快速使用示例
    # task_id = quick_analysis(["字节跳动 (Doubao)", "百度 (文心一言)"])
    # if task_id:
    #     # 稍后查询结果
    #     time.sleep(30)
    #     response = requests.get(f"{BASE_URL}/results/{task_id}")
    #     if response.status_code == 200:
    #         print("分析结果:", json.dumps(response.json(), ensure_ascii=False, indent=2))