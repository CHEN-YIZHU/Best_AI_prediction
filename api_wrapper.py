import json
import logging
import os
import random
import re
import requests
from typing import Any, Dict, List, Optional
from time import sleep

import os
import random
import json
import logging
import requests
from typing import Any, Dict, List, Optional
from openai import OpenAI  # 引入 OpenAI SDK

class ModelAPIWrapper:
    """AI模型API调用包装器：使用 requests 库发请求 + 自动轮询 key + JSON 解析增强"""

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
                "base_url": os.environ.get("AI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completion") + '/chat/completions',
                "model_id": os.environ.get("OPENAI_MODEL", "qwen-max"),
                "api_keys": [os.environ.get("OPENAI_API_KEY")],
                "customized_args": {"max_tokens": 4096},
                "post_request_kwargs": {"timeout": timeout}
            }
        }
        # print(self.configs)

    def _get_config(self, api_type: str) -> Dict[str, Any]:
        if api_type not in self.configs:
            raise ValueError(f"不支持的API类型: {api_type}. 可选: {list(self.configs.keys())}")
        cfg = self.configs[api_type]
        keys = cfg.get("api_keys") or []
        if not keys:
            raise ValueError(f"{api_type} 的 api_keys 为空")
        return cfg

    def aliChat(self, question: str) -> str:
        """使用OpenAI接口进行联网搜索"""
        print("此次使用的模型是{}".format(self.configs["qwen"]["model_id"]))
        
        # 使用OpenAI客户端进行联网调用
        client = OpenAI(
            api_key=self.configs["qwen"]["api_keys"][0],  # 使用API密钥
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 基础URL
        )
        
        # 发起聊天请求
        try:
            completion = client.chat.completions.create(
                model=self.configs["qwen"]["model_id"],  # 使用的模型ID
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': question}
                ],
            )
            return completion.model_dump_json()  # 返回json格式的响应
        except Exception as e:
            self.logger.error(f"联网请求失败: {str(e)}")
            return str(e)

    def _request_with_retry(self, url: str, headers: dict, payload: dict) -> dict:
        """手动实现请求并进行重试机制"""
        last_exception = None
        for attempt in range(self.retry_count):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.configs["qwen"]["post_request_kwargs"]["timeout"])
                response.raise_for_status()  # 检查请求是否成功
                return response.json()
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(f"API调用失败，重试 {attempt + 1}/{self.retry_count}: {str(e)}")
                sleep(random.uniform(1, 3))  # 随机延时，避免频繁请求

        # 如果重试次数耗尽，抛出最后一次异常
        raise last_exception

    def call_text(self, prompt: str, api_type: str = "qwen") -> str:
        """调用API并返回文本响应"""
        if api_type == "qwen":
            return self.aliChat(prompt)  # 直接使用 aliChat 进行联网搜索
        
        # 其他API类型处理
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
                payload = {
                    "model": cfg["model_id"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": cfg["customized_args"]["max_tokens"],
                }
                response_data = self._request_with_retry(url, headers, payload)
                if 'choices' in response_data:
                    return response_data['choices'][0]['message']['content']
                last_err = RuntimeError(f"Empty response. api_type={api_type}")
            except Exception as e:
                last_err = e
                self.logger.warning(f"API调用失败，准备切换key重试: api_type=%s err=%s", api_type, repr(e))
                continue
        raise RuntimeError(f"API调用失败({api_type})，已轮询所有key。最后错误: {repr(last_err)}")

    def call_json(self, prompt: str, api_type: str = "qwen") -> Dict[str, Any]:
        """调用API并返回JSON格式响应"""
        text = self.call_text(prompt, api_type=api_type)
        return json.loads(text)

    def collect_arena_benchmark_data(self, company: str) -> Dict[str, Any]:
        
        prompt = f"""
        你是“事实核验优先”的数据采集员。目标：收集公开最新数据，关于 {company} 的模型在 Chatbot Arena（LMSYS / lmarena）及常见基准上的表现。

        【硬性规则（必须遵守）】
        1) 只输出一个严格合法的 JSON（双引号、无尾逗号、无注释、无 markdown、无 ``` 代码块），不要输出任何额外文本。
        2) 任何“数值/排名/日期/发布信息/结论”都必须给出可核验来源 source_url（真实可打开的网页/论文/官方公告链接）与 source_date（该来源页面或公告日期；不确定则为 null）。
        3) 如果你无法提供真实且可核验的 source_url，则对应数值必须设为 null，并在 missing_fields 中列出该字段路径；严禁编造链接、编造分数、编造排名。
        4) 优先使用官方网站/榜单页/论文原文/官方博客；其次才用媒体报道。来源不一致时，保留冲突并标注 conflict=true。

        【你需要覆盖的信息】
        - {company} 相关的“公开可用聊天模型/旗舰模型”列表（尽量包含版本/发布日期）
        - Chatbot Arena：总体/主要维度（如 overall 或等价主榜）的 Elo/排名（如果来源支持就给；否则 null）
        - 其他基准（MMLU/GSM8K/HumanEval）：仅在你能给出可信来源时填入（论文/技术报告/官方评测页等）；否则 null
        - 相对优势/劣势：必须用 evidence 列表逐条给出处

        【输出 JSON 结构（必须严格遵循字段名；可填 null，但不要新增顶层字段）】
        {{
        "company": "{company}",
        "sources": [
            {{
            "name": "LMSYS / Chatbot Arena (or lmarena)",
            "url": null,
            "source_date": null,
            "accessed_at": null
            }}
        ],
        "models": [
            {{
            "model_name": null,
            "variant_or_version": null,
            "provider": "{company}",
            "release_date": null,
            "release_source_url": null,
            "arena": {{
                "leaderboard_name": "chatbot_arena_overall",
                "elo": null,
                "rank": null,
                "source_url": null,
                "source_date": null,
                "conflict": false,
                "conflict_notes": null
            }},
            "benchmarks": {{
                "MMLU": {{ "score": null, "source_url": null, "source_date": null }},
                "GSM8K": {{ "score": null, "source_url": null, "source_date": null }},
                "HumanEval": {{ "score": null, "source_url": null, "source_date": null }}
            }}
            }}
        ],
        "comparative_summary": {{
            "strengths": [],
            "weaknesses": [],
            "evidence": [
            {{
                "claim": null,
                "supports": ["model_name_or_scope"],
                "source_url": null,
                "source_date": null
            }}
            ]
        }},
        "quality_checks": {{
            "needs_verification": false,
            "missing_fields": [],
            "notes": null
        }}
        }}
        """
        return self.call_json(prompt, api_type="qwen")

    def analyze_leadership_persistence(self, company: str) -> Dict[str, Any]:
        prompt = f"""分析{company}在AI领域的领导地位持久性。
【输出要求】只输出严格 JSON，不要额外文本/markdown。
{{
  "current_technological_advantage": "...",
  "competition_speed": "...",
  "structural_moat": "...",
  "long_term_competitiveness": "...",
  "future_predictions": "..."
}}
"""
        return self.call_json(prompt, api_type="qwen")

#     def assess_risks(self, company: str) -> Dict[str, Any]:
#         prompt = f"""评估{company}面临的法律、伦理、监管风险。
# 【输出要求】只输出严格 JSON，不要额外文本/markdown。
# {{
#   "legal_challenges": "...",
#   "ethical_issues": "...",
#   "contract_risks": "...",
#   "impact_on_commercialization": "...",
#   "risk_management": "..."
# }}
# """
#         return self.call_json(prompt, api_type="qwen")

    def analyze_business_value(self, company: str) -> Dict[str, Any]:
        prompt = f"""分析{company}AI模型质量与商业价值的关系。
【输出要求】只输出严格 JSON，不要额外文本/markdown。
{{
  "quality_vs_revenue": "...",
  "market_share_growth": "...",
  "business_strategies": "...",
  "customer_adoption": "...",
  "return_on_investment": "..."
}}
"""
        return self.call_json(prompt, api_type="qwen")
