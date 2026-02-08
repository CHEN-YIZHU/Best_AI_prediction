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
from datetime import datetime
import dashscope

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
                "base_url": os.environ.get("AI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completion") + '/chat/completions',
                "model_id": os.environ.get("OPENAI_MODEL", "qwen-max"),
                "api_keys": [os.environ.get("OPENAI_API_KEY")],
                "customized_args": {"max_tokens": 18964},
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
        
        
        client = OpenAI(
            api_key=self.configs["qwen"]["api_keys"][0],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 发起聊天请求
        try:
            # completion = client.chat.completions.create(
            #     model=self.configs["qwen"]["model_id"],
            #     messages=[
            #         {'role': 'system', 'content': 'You are a helpful assistant.'},
            #         {'role': 'user', 'content': question}
            #     ],
            #     extra_body={
            #         "enable_search": True   # 为true是支持联网
            #     }
            # )
            # return completion.model_dump_json()
            

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
            response = dashscope.Generation.call(
                api_key=self.configs["qwen"]["api_keys"][0],
                model="qwen-max",
                messages=messages,
                enable_search=True,
                result_format="message"
            )
            # print(response)
            if response.status_code == 200:
                return response["output"]["choices"][0]["message"]["content"]
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
        # if api_type == "qwen":
            
        return self.aliChat(prompt)  # 直接使用 aliChat 进行联网搜索
        
        # 其他API类型处理
        # cfg = self._get_config(api_type)
        # keys: List[str] = list(cfg["api_keys"])
        # random.shuffle(keys)
        # last_err: Optional[Exception] = None
        # for sk in keys:
        #     try:
        #         url = cfg["base_url"]
        #         headers = {
        #             "Authorization": f"Bearer {sk}",
        #             "Content-Type": "application/json"
        #         }
        #         payload = {
        #             "model": cfg["model_id"],
        #             "messages": [{"role": "user", "content": prompt}],
        #             "max_tokens": cfg["customized_args"]["max_tokens"],
        #         }
        #         response_data = self._request_with_retry(url, headers, payload)
        #         if 'choices' in response_data:
        #             return response_data['choices'][0]['message']['content']
        #         last_err = RuntimeError(f"Empty response. api_type={api_type}")
        #     except Exception as e:
        #         last_err = e
        #         self.logger.warning(f"API调用失败，准备切换key重试: api_type=%s err=%s", api_type, repr(e))
        #         continue
        # raise RuntimeError(f"API调用失败({api_type})，已轮询所有key。最后错误: {repr(last_err)}")

    def call_json(self, prompt: str, api_type: str = "qwen") -> Dict[str, Any]:
        """调用API并返回JSON格式响应"""
        text = self.call_text(prompt, api_type=api_type)
        return json.loads(text)
    
    def get_newest_model_info(self, company: str) -> str:
        """获取公司最新发布的LLM模型和版本信息
        
        Args:
            company: 公司名称 (如: OpenAI, Google, Anthropic等)
            
        Returns:
            字符串格式的最新模型信息描述
        """
        as_of = datetime.utcnow().strftime("%Y-%m-%d")
        prompt = f"""请搜索{company}公司截止{as_of}最新发布的大型语言模型(LLM)是什么？
        
        只返回一个简洁的字符串描述，格式示例：
        "[模型名称]，版本[版本号]，发布于[发布日期]。"

        要求：
        1. 确保信息准确，基于可靠的网络搜索
        2. 只返回最终结果，不要返回JSON或其他格式!!!!
        """
        
        return self.call_text(prompt, api_type="qwen")


    def collect_arena_benchmark_data(self, company: str) -> Dict[str, Any]:
        
        # prompt = f"""
        # 你是“事实核验优先”的数据采集员。目标：收集公开最新数据，关于 {company} 在 Chatbot Arena（LMSYS / lmarena）及常见基准上的表现。

        # 【硬性规则（必须遵守）】
        # 1) 只输出一个严格合法的 JSON（双引号、无尾逗号、无注释、无 markdown、无 ``` 代码块），不要输出任何额外文本。
        # 2) 任何“数值/排名/日期/发布信息/结论”都必须给出可核验来源 source_url（真实可打开的网页/论文/官方公告链接）与 source_date（该来源页面或公告日期；不确定则为 null）。
        # 3) 如果你无法提供真实且可核验的 source_url，则对应数值必须设为 null，并在 missing_fields 中列出该字段路径；严禁编造链接、编造分数、编造排名。
        # 4) 优先使用官方网站/榜单页/论文原文/官方博客；其次才用媒体报道。来源不一致时，保留冲突并标注 conflict=true。

        # 【你需要覆盖的信息】
        # - {company} 相关的“公开可用聊天模型/旗舰模型”列表（尽量包含版本/发布日期）
        # - Chatbot Arena：总体/主要维度（如 overall 或等价主榜）的 Elo/排名（如果来源支持就给；否则 null）
        # - 其他基准（MMLU/GSM8K/HumanEval）：仅在你能给出可信来源时填入（论文/技术报告/官方评测页等）；否则 null
        # - 相对优势/劣势：必须用 evidence 列表逐条给出处

        # 【输出 JSON 结构（必须严格遵循字段名；可填 null，但不要新增顶层字段）】
        # {{
        # "company": "{company}",
        # "sources": [
        #     {{
        #     "name": "LMSYS / Chatbot Arena (or lmarena)",
        #     "url": null,
        #     "source_date": null,
        #     "accessed_at": null
        #     }}
        # ],
        # "models": [
        #     {{
        #     "model_name": null,
        #     "variant_or_version": null,
        #     "provider": "{company}",
        #     "release_date": null,
        #     "release_source_url": null,
        #     "arena": {{
        #         "leaderboard_name": "chatbot_arena_overall",
        #         "elo": null,
        #         "rank": null,
        #         "source_url": null,
        #         "source_date": null,
        #         "conflict": false,
        #         "conflict_notes": null
        #     }},
        #     "benchmarks": {{
        #         "MMLU": {{ "score": null, "source_url": null, "source_date": null }},
        #         "GSM8K": {{ "score": null, "source_url": null, "source_date": null }},
        #         "HumanEval": {{ "score": null, "source_url": null, "source_date": null }}
        #     }}
        #     }}
        # ],
        # "comparative_summary": {{
        #     "strengths": [],
        #     "weaknesses": [],
        #     "evidence": [
        #     {{
        #         "claim": null,
        #         "supports": ["model_name_or_scope"],
        #         "source_url": null,
        #         "source_date": null
        #     }}
        #     ]
        # }},
        # "quality_checks": {{
        #     "needs_verification": false,
        #     "missing_fields": [],
        #     "notes": null
        # }}
        # }}
        # """
        as_of = datetime.utcnow().strftime("%Y-%m-%d")
        accessed_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        prompt = f"""
        你是“事实核验优先”的数据采集员，并且你具备联网检索能力。你的目标：收集截至 {as_of}（UTC）的公开最新数据，关于 {company} 在 Chatbot Arena（LMSYS / LMArena）及常见基准上的表现。

        【不可违背的输出规则】
        1) 只输出一个严格合法的 JSON（双引号、无尾逗号、无注释、无 markdown、无 ``` 代码块），不要输出任何额外文本。
        2) 任何“数值/排名/日期/发布信息/结论”都必须给出可核验来源：source_url（真实可打开链接）与 source_date（页面/公告日期；不确定则为 null）。
        3) 如果你无法提供真实且可核验的 source_url，则对应字段必须设为 null，并把字段路径写入 quality_checks.missing_fields；严禁编造链接/分数/排名/发布日期。
        4) “最新”以 {as_of} 为截止：优先官方榜单页/官方博客/论文原文/官方技术报告；其次才用媒体。来源不一致时保留冲突：conflict=true，并写 conflict_notes。

        【执行要求（内部执行，不要在输出里复述）】
        A. 先找 Arena 主榜（Overall/默认主榜）可用来源（至少命中一个）：
        - https://lmarena.ai/leaderboard/text （优先）
        - https://lmarena.ai/leaderboard
        - https://chat.lmsys.org/?leaderboard （旧入口，若仍可用）
        - 兜底结构化来源（可用于取 Elo/Rank）：
            * https://huggingface.co/datasets/mathewhe/chatbot-arena-elo
            * https://huggingface.co/spaces/lmarena-ai/lmarena-leaderboard

        B. 再找 {company} 的“公开可用聊天模型/旗舰模型”：
        - 必须从官方发布页/官方文档/官方技术报告/论文中确认 model_name、variant_or_version（若有）与 release_date；
        - 如果找不到“发布日期”，release_date=null 且 missing_fields 记录路径。

        C. 最后再补常见基准（只在能找到论文/技术报告/官方评测页时填写；否则 null）：
        - MMLU / GSM8K / HumanEval

        【字段一致性硬约束】
        - 输出 JSON 的 company 必须严格等于 "{company}"（不得改写、不得替换成模型名）。
        - 每个 models[i].provider 必须严格等于 "{company}"。
        - Arena 数值字段：若来源页显示为 “Arena Score / Elo / Rating”，统一填入 arena.elo；若只有排名则只填 rank。

        【重要：不要输出“全是 null 的占位模型”】【models 数组规则】
        - models 可以是空数组 []。
        - 只有当你至少拿到以下任意一项的可核验来源时，才把该模型写入 models：
        (1) release_source_url（发布/版本/日期来源） 或
        (2) arena.source_url（Arena 主榜中该模型行可核验来源）
        - 否则不要把该模型写入 models（避免产出大量 null）。

        【输出 JSON 结构（必须严格遵循字段名；可填 null，但不要新增顶层字段）】
        {{
        "company": "{company}",
        "sources": [],
        "models": [],
        "comparative_summary": {{
            "strengths": [],
            "weaknesses": [],
            "evidence": []
        }},
        "quality_checks": {{
            "needs_verification": false,
            "missing_fields": [],
            "notes": null
        }}
        }}

        【sources 填写要求】
        - 把你实际使用过的每个来源都加入 sources（至少 1 条，建议 3-10 条），每条包括：
        name, url, source_date, accessed_at="{accessed_at}"

        【models 每个元素结构（写入时必须严格按此结构）】
        - 对于每个写入 models 的模型，必须包含如下字段（可为 null，但若为 null 需有 missing_fields）：
        model_name, variant_or_version, provider, release_date, release_source_url,
        arena{{leaderboard_name="chatbot_arena_overall", elo, rank, source_url, source_date, conflict, conflict_notes}},
        benchmarks{{MMLU{{score,source_url,source_date}}, GSM8K{{...}}, HumanEval{{...}}}}

        【quality_checks 自动规则】
        - 只要 missing_fields 非空，needs_verification 必须为 true。
        - missing_fields 用 JSONPath 风格，例如：
        "models[0].release_date", "models[0].arena.elo", "models[1].benchmarks.MMLU.score"
        """
        return self.call_json(prompt, api_type="qwen")

    def analyze_leadership_persistence(self, company: str) -> Dict[str, Any]:
#         prompt = f"""分析{company}在AI领域的领导地位持久性。
# 【输出要求】只输出严格 JSON，不要额外文本/markdown。
# {{
#   "current_technological_advantage": "...",
#   "competition_speed": "...",
#   "structural_moat": "...",
#   "long_term_competitiveness": "...",
#   "future_predictions": "..."
# }}
# """   
        as_of = datetime.utcnow().strftime("%Y-%m-%d")
        prompt = f"""
你是“可证据化的战略分析师”。目标：基于截至 {as_of if 'as_of' in locals() else '今天'} 的公开信息，评估 {company} 在 AI/大模型领域领导地位的“可持续性（2-3年）”。

【硬性规则】
1) 只输出一个严格合法的 JSON（无额外文本/markdown/```）。
2) 任何事实性断言（如“领先”“投入”“份额”“论文数量”“算力”“产品用户量”“收入”）都必须有 source_url + source_date；没有就写 null，并在 missing_fields 标注字段路径。
3) 严禁编造来源/编造数据；无法核验时必须显式 uncertainty。

【分析框架（必须覆盖）】
- 技术领先：模型能力/迭代节奏/研发效率（用可引用证据支撑）
- 竞争追赶：主要竞争对手列表、追赶速度迹象（发布频率/开源/评测）
- 护城河：分发渠道、生态、开发者/企业客户、数据/产品闭环等（有证据则写）
- 风险因素：监管/诉讼/供应链/算力/人才流动等（有证据则写）
- 2-3年情景：base/bull/bear 三情景 + 触发条件（尽量证据化）

【输出 JSON 结构（不要新增顶层字段）】
{{
  "company": "{company}",
  "as_of": "{as_of if 'as_of' in locals() else ''}",
  "time_horizon_years": 3,
  "current_technological_advantage": {{
    "summary": null,
    "evidence": [
      {{ "claim": null, "source_url": null, "source_date": null }}
    ]
  }},
  "competition_speed": {{
    "main_competitors": [],
    "summary": null,
    "evidence": [
      {{ "claim": null, "source_url": null, "source_date": null }}
    ]
  }},
  "structural_moat": {{
    "distribution": null,
    "ecosystem": null,
    "customers_or_users": null,
    "evidence": [
      {{ "claim": null, "source_url": null, "source_date": null }}
    ]
  }},
  "long_term_competitiveness": {{
    "thesis": null,
    "key_risks": [],
    "key_upside_drivers": [],
    "uncertainties": []
  }},
  "future_predictions": {{
    "base_case": {{ "probability": 0.5, "outlook": null, "triggers": [] }},
    "bull_case": {{ "probability": 0.25, "outlook": null, "triggers": [] }},
    "bear_case": {{ "probability": 0.25, "outlook": null, "triggers": [] }}
  }},
  "quality_checks": {{
    "needs_verification": false,
    "missing_fields": [],
    "notes": null
  }}
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
#         prompt = f"""分析{company}AI模型质量与商业价值的关系。
# 【输出要求】只输出严格 JSON，不要额外文本/markdown。
# {{
#   "quality_vs_revenue": "...",
#   "market_share_growth": "...",
#   "business_strategies": "...",
#   "customer_adoption": "...",
#   "return_on_investment": "..."
# }}
# """
        as_of = datetime.utcnow().strftime("%Y-%m-%d")
        prompt = f"""
你是“模型质量 -> 商业价值”分析师。目标：基于截至 {as_of if 'as_of' in locals() else '今天'} 的公开信息，分析 {company} 的 AI 模型质量与商业价值之间的关系，并给出可核验证据。

【硬性规则】
1) 只输出一个严格合法的 JSON（无额外文本/markdown/```）。
2) 任何财务/用户/客户/价格/合同/市场份额等“事实性数字”必须有 source_url + source_date；否则数值必须为 null，并写入 missing_fields。
3) 严禁编造财务/用户数据与来源；无法核验必须写 uncertainty 或 null。

【必须覆盖】
- 质量指标与收入/采用的关联：用 evidence 解释“为何相关”，不要凭空断言
- 商业化策略：产品形态（API/订阅/企业方案/云合作/硬件等）
- 采用与满意度：仅引用可核验数据（公告/财报/采访/可信报告）
- ROI：如果无公开数据，可给“分析框架”但数值必须为 null

【输出 JSON 结构（不要新增顶层字段）】
{{
  "company": "{company}",
  "as_of": "{as_of if 'as_of' in locals() else ''}",
  "quality_vs_revenue": {{
    "summary": null,
    "evidence": [
      {{ "claim": null, "source_url": null, "source_date": null }}
    ]
  }},
  "market_share_growth": {{
    "summary": null,
    "reported_numbers": [
      {{ "metric": null, "value": null, "unit": null, "source_url": null, "source_date": null }}
    ],
    "uncertainties": []
  }},
  "business_strategies": {{
    "summary": null,
    "channels": [
      {{ "channel": null, "details": null, "source_url": null, "source_date": null }}
    ]
  }},
  "customer_adoption": {{
    "summary": null,
    "reported_numbers": [
      {{ "metric": null, "value": null, "unit": null, "source_url": null, "source_date": null }}
    ],
    "case_studies": [
      {{ "customer_or_partner": null, "what_deployed": null, "source_url": null, "source_date": null }}
    ]
  }},
  "return_on_investment": {{
    "summary": null,
    "reported_numbers": [
      {{ "metric": null, "value": null, "unit": null, "source_url": null, "source_date": null }}
    ],
    "framework_if_no_public_data": [
      "costs: training/inference/serving",
      "benefits: revenue/savings/retention",
      "payback_period, gross_margin, CAC/LTV"
    ],
    "uncertainties": []
  }},
  "quality_checks": {{
    "needs_verification": false,
    "missing_fields": [],
    "notes": null
  }}
}}
"""

        return self.call_json(prompt, api_type="qwen")
