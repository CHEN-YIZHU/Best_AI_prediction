import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Union

from utils import RequestWrapperOne


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    """如果模型把 JSON 放在 ```json ...``` 里，提取出来。"""
    if not isinstance(text, str):
        return str(text)
    m = _JSON_FENCE_RE.search(text)
    return (m.group(1).strip() if m else text.strip())


def _extract_json_candidate(text: str) -> str:
    """
    尝试从文本中截取 JSON 片段：
    - 优先截取最外层 {...}
    - 否则截取最外层 [...]
    - 都找不到就返回原文本
    """
    t = _strip_code_fence(text)

    # object
    s = t.find("{")
    e = t.rfind("}")
    if 0 <= s < e:
        return t[s : e + 1].strip()

    # array
    s = t.find("[")
    e = t.rfind("]")
    if 0 <= s < e:
        return t[s : e + 1].strip()

    return t.strip()


def _remove_trailing_commas(s: str) -> str:
    """去掉 JSON 常见的尾逗号： ,} 或 ,] """
    return re.sub(r",\s*([}\]])", r"\1", s)


class ModelAPIWrapper:
    """AI模型API调用包装器：统一用 RequestWrapperOne 发请求 + 自动轮询 key + JSON 解析增强"""

    def __init__(
        self,
        doubao_api_keys: Optional[List[str]] = None,
        doubao_base_url: str = "http://yy.dbh.baidu-int.com/v1/chat/completions",
        doubao_model_id: str = "doubao-seed-1-8-251228",
        doubao_customized_args: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        self.configs: Dict[str, Dict[str, Any]] = {
            "doubao": {
                "base_url": doubao_base_url,
                "model_id": doubao_model_id,
                # 建议外部注入（环境变量/配置文件），不要硬编码
                "api_keys": doubao_api_keys or ["sk-REPLACE_WITH_YOUR_KEY"],
                # RequestWrapperOne._api_request 会把 customized_args 传给 _request(messages, **customized_args)
                # 其中 max_tokens 会进入 payload
                "customized_args": doubao_customized_args or {"max_tokens": 18192},
                "post_request_kwargs": {"timeout": timeout},
            }
        }

    def _get_config(self, api_type: str) -> Dict[str, Any]:
        if api_type not in self.configs:
            raise ValueError(f"不支持的API类型: {api_type}. 可选: {list(self.configs.keys())}")
        cfg = self.configs[api_type]
        keys = cfg.get("api_keys") or []
        if not keys:
            raise ValueError(f"{api_type} 的 api_keys 为空")
        return cfg

    def _new_wrapper(self, api_type: str, sk: str) -> RequestWrapperOne:
        """每次请求新建 wrapper：更适合并发，避免共享状态。"""
        cfg = self._get_config(api_type)
        return RequestWrapperOne(
            cfg["base_url"],
            cfg["model_id"],
            sk=sk,
            # 不强制指定，让 RequestWrapperOne 自己 parse_method_type 也行；
            # 但这里明确 api 更稳（sk 合法时会走 Authorization）
            request_method_type="api",
            customized_args=cfg.get("customized_args", {}),
            **cfg.get("post_request_kwargs", {}),
        )

    def call_text(
        self,
        prompt: str,
        api_type: str = "doubao",
        history: Optional[List[dict]] = None,
        system: Optional[str] = None,
        media: Optional[List] = None,
    ) -> str:
        """
        统一走 RequestWrapperOne.make_a_valid_request
        - 自动轮询/随机 key
        - 任意一个 key 成功就返回
        """
        cfg = self._get_config(api_type)
        keys: List[str] = list(cfg["api_keys"])
        # 打乱顺序做轮询，避免所有并发都打到同一把 key
        random.shuffle(keys)

        last_err: Optional[Exception] = None
        for sk in keys:
            try:
                wrapper = self._new_wrapper(api_type, sk)
                responses, _messages = wrapper.make_a_valid_request(
                    prompt, history=history, media=media, system=system
                )
                if responses and isinstance(responses[0], str) and responses[0].strip():
                    return responses[0]
                last_err = RuntimeError(f"Empty response. api_type={api_type}")
            except Exception as e:
                last_err = e
                self.logger.warning(
                    "API调用失败，准备切换key重试: api_type=%s err=%s", api_type, repr(e)
                )
                continue

        raise RuntimeError(f"API调用失败({api_type})，已轮询所有key。最后错误: {repr(last_err)}")

    def call_json(
        self,
        prompt: str,
        api_type: str = "doubao",
        history: Optional[List[dict]] = None,
        system: Optional[str] = None,
        media: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        调用并解析 JSON：
        - 自动去 code fence
        - 自动截取 JSON 主体
        - 自动去尾逗号
        """
        text = self.call_text(prompt, api_type=api_type, history=history, system=system, media=media)
        cand = _extract_json_candidate(text)
        cand = _remove_trailing_commas(cand)

        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
            # 你这里的下游都期望 dict，array 就包一层
            return {"data": obj}
        except json.JSONDecodeError:
            # 兜底：返回原始文本，方便你日志排查
            return {
                "error": "Response is not valid JSON",
                "raw_text": text,
                "json_candidate": cand,
                "api_type": api_type,
            }

    # -------- 下面是你原来的四个业务方法：改成用 call_json --------

    def collect_arena_benchmark_data(self, company: str) -> Dict[str, Any]:
        prompt = f"""请提供以下关于{company}在chatbot-arena榜单上的最新表现数据。
【输出要求】
- 只输出一个严格合法的 JSON（双引号、无尾逗号、无注释）
- 不要输出任何额外文本，不要用```包裹
【字段】
{{
  "company": "{company}",
  "models": [{{"model_name": "...", "version": "...", "release_date": "YYYY-MM-DD or null"}}],
  "benchmark_scores": {{
    "chatbot_arena": {{}},
    "MMLU": {{}},
    "GSM8K": {{}},
    "HumanEval": {{}}
  }},
  "user_ratings": {{"rating": null, "rank": null}},
  "strengths": "...",
  "weaknesses": "...",
  "tech_details": "..."
}}
"""
        return self.call_json(prompt, api_type="doubao")

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
        return self.call_json(prompt, api_type="doubao")

    def assess_risks(self, company: str) -> Dict[str, Any]:
        prompt = f"""评估{company}面临的法律、伦理、监管风险。
【输出要求】只输出严格 JSON，不要额外文本/markdown。
{{
  "legal_challenges": "...",
  "ethical_issues": "...",
  "contract_risks": "...",
  "impact_on_commercialization": "...",
  "risk_management": "..."
}}
"""
        return self.call_json(prompt, api_type="doubao")

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
        return self.call_json(prompt, api_type="doubao")
