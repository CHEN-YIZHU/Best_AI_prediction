import random
import requests
import json
from typing import Dict, Any, Optional
from utils import RequestWrapperOne

import json
import re

from datetime import datetime

from vllm import LLM, SamplingParams


def prepare_model(model_path, max_model_len=2048, tp_size=8):
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        max_model_len=max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,   # 关键
    )
    return llm


def build_sampling_params(max_tokens=1024, temperature=1.0, top_p=1.0):
    # Qwen 常见的 stop token ids（你原来给的保持不变）
    stop_token_ids = [151329, 151336, 151338]
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )

def normalize_score(score_val: Any) -> float:
    """
    统一把 score 归一化到 0~1
    """
    if score_val is None:
        return 0.0

    try:
        s = float(score_val)
    except Exception:
        m = re.search(r"-?\d+(\.\d+)?", str(score_val))
        s = float(m.group(0)) if m else 0.0

    # 如果是 0-100，转成 0-1
    if s > 1.0:
        s = s / 100.0

    return max(0.0, min(s, 1.0))


def safe_json_dump(data):
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except:
        # 如果是字符串，直接返回
        if isinstance(data, str):
            return str(data)
        try:
            # 尝试用repr来避免格式化问题
            return repr(data)
        except:
            return "无法解析的数据"

def safe_format_prompt(
    company: str,
    analysis_data: Dict[str, Any],
    template: str,
    **kwargs
) -> str:
    """
    安全格式化 prompt：
    - 不做全局大括号替换
    - 自动补齐 company / time
    - format 失败时仍保持 JSON 约束
    """
    try:
        fmt_kwargs = dict(kwargs)
        fmt_kwargs.setdefault("company", company)
        fmt_kwargs.setdefault(
            "time",
            analysis_data.get("timestamp", datetime.now().isoformat())
        )
        return template.format(**fmt_kwargs)
    except Exception:
        # 最小但安全的 fallback（仍强制 JSON）
        return (
            f"你是一个专业的AI分析师，请基于以下数据对 {company} 打分（0-100），"
            f"并且【只输出一个JSON对象，不要输出任何解释文字】。\n\n"
            f"基准数据:\n{safe_json_dump(analysis_data.get('benchmark_data', {}))}\n\n"
            f"领导力数据:\n{safe_json_dump(analysis_data.get('leadership_analysis', {}))}\n\n"
            f"风险数据:\n{safe_json_dump(analysis_data.get('risk_assessment', {}))}\n\n"
            f"商业数据:\n{safe_json_dump(analysis_data.get('business_analysis', {}))}\n\n"
            "请输出："
            '{ "score": 0-100, "reasoning": "...", "strengths": "...", "weaknesses": "..." }'
        )


def extract_first_json_obj(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取第一个 JSON 对象：
    - 允许前后有解释
    - 允许 ```json ``` 包裹
    """
    if not text:
        raise ValueError("Empty model output")

    t = text.strip()

    # 去掉 ```json ``` 包裹
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)

    start = t.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found: {t[:200]}")

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(t[start:])

    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")

    return obj