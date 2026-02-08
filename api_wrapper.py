import json
import logging
import os
import random
import requests
from time import sleep
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ModelAPIWrapper:
    
    def __init__(
        self,
        timeout: int = 120,
        retry_count: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.retry_count = retry_count
        self.configs: Dict[str, Dict[str, Any]] = {
            "qwen": {
                # 使用原生 DashScope 接口以支持 enable_search
                "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                "model_id": os.environ.get("OPENAI_MODEL", "qwen-max"),
                "api_keys": [os.environ.get("OPENAI_API_KEY")],
                "customized_args": {
                    "max_tokens": 8192,
                    "enable_search": True,  # 开启联网搜索
                    "result_format": "message"
                },
                "post_request_kwargs": {"timeout": timeout}
            }
        }

    def _get_config(self, api_type: str) -> Dict[str, Any]:
        if api_type not in self.configs:
            raise ValueError(f"不支持的API类型: {api_type}. 可选: {list(self.configs.keys())}")
        cfg = self.configs[api_type]
        keys = cfg.get("api_keys") or []
        if not keys or not keys[0]:
            raise ValueError(f"{api_type} 的 api_keys 为空或无效")
        return cfg

    def _request_with_retry(self, url: str, headers: dict, payload: dict) -> dict:
        """手动实现请求并进行重试机制"""
        last_exception = None
        for attempt in range(self.retry_count):
            try:
                # 注意：requests timeout
                timeout = self.configs["qwen"]["post_request_kwargs"]["timeout"]
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(f"API调用失败，重试 {attempt + 1}/{self.retry_count}: {str(e)}")
                sleep(random.uniform(1, 3))

        raise last_exception

    def call_text(
        self,
        prompt: str,
        api_type: str = "qwen",
    ) -> str:
        cfg = self._get_config(api_type)
        keys: List[str] = list(cfg["api_keys"])
        random.shuffle(keys)
        last_err: Optional[Exception] = None
        
        for sk in keys:
            try:
                url = cfg["base_url"]
                headers = {
                    "Authorization": f"Bearer {sk}",
                    "Content-Type": "application/json"
                }
                # 构造原生 DashScope API 请求体
                payload = {
                    "model": cfg["model_id"],
                    "input": {
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    "parameters": cfg["customized_args"]
                }
                
                response_data = self._request_with_retry(url, headers, payload)
                
                # 解析原生 API 响应
                if 'output' in response_data and 'choices' in response_data['output']:
                    return response_data['output']['choices'][0]['message']['content']
                elif 'choices' in response_data:
                    return response_data['choices'][0]['message']['content']
                
                last_err = RuntimeError(f"Empty response. Response: {str(response_data)}")
                
            except Exception as e:
                last_err = e
                self.logger.warning(f"API调用失败，准备切换key重试: api_type=%s err=%s", api_type, repr(e))
                continue
                
        raise RuntimeError(f"API调用失败({api_type})。最后错误: {repr(last_err)}")

    def call_json(self, prompt: str, api_type: str = "qwen") -> Dict[str, Any]:
        """调用API并返回JSON格式响应"""
        text = self.call_text(prompt, api_type=api_type)
        # 尝试提取 JSON
        try:
            # 简单的 markdown json 块提取
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                 text = text.split("```")[1].split("```")[0].strip()
            
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "error": "Response is not valid JSON",
                "raw_text": text
            }
