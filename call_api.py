import requests
import json
import time
import sys

# API 地址
BASE_URL = "http://localhost:8000"

def submit_analysis_task(companies=None):
    if companies is None:
        companies = ["OpenAI (GPT-5.2)", "Google (Gemini 3 Pro)", "Anthropic (Claude Opus 4.6 Thinking)"]
    
    url = f"{BASE_URL}/analyze"
    payload = {
        "companies": companies,
        "max_workers": 2
    }
    
    print(f"📡 提交分析任务: {companies}")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"✅ 任务提交成功! Task ID: {data['task_id']}")
        return data['task_id']
    except Exception as e:
        print(f"❌ 提交失败: {e}")
        if 'response' in locals():
            print(f"   响应: {response.text}")
        sys.exit(1)

def wait_for_result(task_id):
    url = f"{BASE_URL}/results/{task_id}"
    print(f"⏳ 等待分析结果 (Task ID: {task_id})...")
    
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'completed':
                    print("\n🎉 分析完成!")
                    return data
                elif data['status'] == 'failed':
                    print(f"\n❌ 分析失败: {data.get('error')}")
                    return None
                else:
                    # 仍在运行中
                    print(".", end="", flush=True)
                    time.sleep(2)
            elif response.status_code == 425: # Too Early (Running)
                 print(".", end="", flush=True)
                 time.sleep(2)
            else:
                print(f"\n❌ 查询出错: {response.status_code} - {response.text}")
                time.sleep(2)
        except Exception as e:
            print(f"\n❌ 请求异常: {e}")
            time.sleep(2)

def main():
    # 1. 提交任务
    task_id = submit_analysis_task()
    
    # 2. 轮询结果
    result = wait_for_result(task_id)
    
    if result:
        print("\n" + "="*50)
        print("📊 分析报告摘要")
        print("="*50)
        
        # 打印总结
        summary = result.get('summary', {})
        if summary:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        # 保存完整结果
        filename = f"analysis_result_{task_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 完整结果已保存至: {filename}")

if __name__ == "__main__":
    main()
