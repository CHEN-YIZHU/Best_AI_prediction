#!/usr/bin/env python3
"""
简单的API测试脚本 - 验证阿里云DashScope API连通性
"""

import os
import json
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
import json
import requests

def test_dashscope_api():
    """测试DashScope API连通性"""
    
    # 设置环境变量
    os.environ.update({
        'OPENAI_API_KEY': 'sk-a5003d95f24b49ebb40c1927f126fba1',
        'AI_BASE_URL': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'OPENAI_MODEL': 'qwen-max'
    })
    
    url = os.environ['AI_BASE_URL'] + '/chat/completions'
    api_key = os.environ['OPENAI_API_KEY']
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'qwen-max',
        'messages': [
            {
                'role': 'user',
                'content': '请简单介绍一下你自己，并告诉我现在的日期和时间。'
            }
        ],
        'max_tokens': 500
    }
    
    print("=== 开始测试DashScope API ===")
    print(f"API URL: {url}")
    print(f"模型: qwen-max")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API调用成功!")
            print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 提取模型回复
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']
                print(f"\n模型回复: {message['content']}")
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {str(e)}")
        print(f"原始响应: {response.text}")
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")

if __name__ == "__main__":
    test_dashscope_api()