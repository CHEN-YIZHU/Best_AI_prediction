import json
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd

from api_wrapper import ModelAPIWrapper
from utils import extract_first_json_obj, safe_json_dump, safe_format_prompt, normalize_score


class AICompanyAnalyzer:
    """AI公司综合分析引擎"""
    
    def __init__(self, args):
        # Initialize other components
        self.api_wrapper = ModelAPIWrapper()
        
        # 外部API配置（使用fuwuqi.txt中的配置）
        self.external_api_config = {
            "provider": "qwen",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": args.inference_key,
            "model": "qwen-max"
        }

        self.logger = logging.getLogger(__name__)
        
        # 主要AI公司列表
        self.target_companies = [
            "OpenAI (Chatgpt)", "Google (Gemini)",
            "Anthropic (Claude)", "DeepSeek", 
            "字节跳动 (Doubao)", "Alibaba (Qwen)", "Meituan",
            "xAI", "Mistral", "百度 (文心一言)"
        ]

    
    def analyze_single_company(self, company: str) -> Dict[str, Any]:
        """
        分析单个公司，支持API和本地模型双模式
        """
        self.logger.info(f"开始分析公司: {company}")
        
        analysis_data = {
            "company": company,
            "timestamp": datetime.now().isoformat(),
            "analysis_method": "api",
            "benchmark_data": {},
            "leadership_analysis": {},
            # "risk_assessment": {},
            "business_analysis": {},
            "final_score": 0.0
        }
        
        try:
            
                # 使用API收集数据
            analysis_data["benchmark_data"] = self.api_wrapper.collect_arena_benchmark_data(company)
            analysis_data["leadership_analysis"] = self.api_wrapper.analyze_leadership_persistence(company)
            # analysis_data["risk_assessment"] = self.api_wrapper.assess_risks(company)
            analysis_data["business_analysis"] = self.api_wrapper.analyze_business_value(company)
            
            # 将API收集的数据传递给本地LLM进行综合评分
            analysis_data["final_score"] = self.analyze_with_local_llm(analysis_data)

            print("analysis_data", analysis_data)
        except Exception as e:
            self.logger.error(f"分析公司 {company} 时出错: {str(e)}")
            analysis_data["error"] = str(e)
        
        return analysis_data
    
    def analyze_with_local_llm(self, analysis_data: Dict[str, Any]) -> float:
        """
        使用外部API（通义千问）对API数据做综合分析并返回0~1的得分
        """
        company = analysis_data["company"]
        
        # 构建LLM提示词
        template = """你是一个专业的AI投资分析师，你需要预测截至在{time}前
Chatbot Arena榜单上第一的AI模型更可能来自哪个公司。

请基于以下四个维度的数据，对 {company} 进行综合评分（0-100分）：

基准表现数据:
{benchmark_data}

领导地位分析:
{leadership_data}

商业价值分析:
{business_data}

评分标准:
1. 技术实力和基准测试表现 (40%)
2. 市场竞争力和领导者地位 (30%)
3. 商业价值和盈利能力 (20%)

注意：
- 请只输出一个JSON对象
- 不要输出解释性文字
- 不要使用```json```代码块

JSON格式：
{{
"score": 0-100,
"reasoning": "评分理由",
"strengths": "主要优势",
"weaknesses": "潜在风险"
}}
"""

        prompt = f"""你是一个专业的AI投资分析师，需要基于以下数据对{company}进行综合评分（0-100分）：

基准表现数据:
{safe_json_dump(analysis_data.get("benchmark_data", {}))}

领导地位分析:
{safe_json_dump(analysis_data.get("leadership_analysis", {}))}

商业价值分析:
{safe_json_dump(analysis_data.get("business_analysis", {}))}

请严格按照以下JSON格式输出，只输出JSON对象，不要有其他文字：
{{
"score": 分数值（0-100）,
"reasoning": "评分理由",
"strengths": "主要优势",
"weaknesses": "潜在风险"
}}"""

        try:
            # 使用外部API调用
            from openai import OpenAI
            
            client = OpenAI(
                base_url=self.external_api_config["base_url"],
                api_key=self.external_api_config["api_key"]
            )
            
            response = client.chat.completions.create(
                model=self.external_api_config["model"],
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的AI投资分析师，需要基于提供的商业数据对AI公司进行评分。请严格按要求格式输出JSON。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    "enable_search": True   # 为true是支持联网
                },
                temperature=0.3,
                max_tokens=2000
            )
            
            text = response.choices[0].message.content
            analysis_data["external_api_raw_output"] = text

            # 提取JSON对象
            obj = extract_first_json_obj(text)
            analysis_data["llm_analysis"] = obj
            analysis_data["llm_reasoning"] = obj.get("reasoning", "")

            return normalize_score(obj.get("score", 0))

        except Exception as e:
            self.logger.error(f"外部API分析失败: {str(e)}，回退到规则评分")
            analysis_data["external_api_error"] = str(e)
            return float(self.calculate_comprehensive_score(analysis_data) or 0.0)


    
    def calculate_comprehensive_score(self, analysis_data: Dict[str, Any]) -> float:
        """
        规则兜底综合评分（0~1）
        """
        weights = {
            "benchmark": 0.4,
            "leadership": 0.3,
            "business": 0.2,
            "risk": 0.1,   # 修复：风险应占 10%
        }

        score = 0.0
        score += self._score_benchmark_data(analysis_data["benchmark_data"]) * weights["benchmark"]
        score += self._score_leadership(analysis_data["leadership_analysis"]) * weights["leadership"]
        score += self._score_business_value(analysis_data["business_analysis"]) * weights["business"]
        score += self._score_risk(analysis_data["risk_assessment"]) * weights["risk"]

        return max(0.0, min(score, 1.0))

    
    def _score_benchmark_data(self, benchmark_data: Dict[str, Any]) -> float:
        """评估基准测试数据"""
        if "error" in benchmark_data:
            return 0.0
        
        score = 50.0  # 基础分
        
        # 根据基准测试分数加分
        if "mmlu_score" in str(benchmark_data).lower():
            score += 20
        if "gsm8k" in str(benchmark_data).lower():
            score += 15
        if "humaneval" in str(benchmark_data).lower():
            score += 15
        
        # 根据排名信息加分
        if "rank" in str(benchmark_data).lower():
            score += 20
        
        return min(score, 100.0) / 100.0
    
    def _score_leadership(self, leadership_data: Dict[str, Any]) -> float:
        """评估领导地位"""
        if "error" in leadership_data:
            return 0.0
        
        score = 50.0  # 基础分
        
        # 根据领导力分析内容加分
        content = str(leadership_data).lower()
        
        if "领先" in content or "leader" in content:
            score += 20
        if "护城河" in content or "moat" in content:
            score += 15
        if "竞争优势" in content or "competitive" in content:
            score += 15
        if "用户基数" in content or "user base" in content:
            score += 10
        
        return min(score, 100.0) / 100.0
    
    def _score_business_value(self, business_data: Dict[str, Any]) -> float:
        """评估商业价值"""
        if "error" in business_data:
            return 0.0
        
        score = 50.0  # 基础分
        
        content = str(business_data).lower()
        
        if "收入" in content or "revenue" in content:
            score += 20
        if "市场份额" in content or "market share" in content:
            score += 15
        if "商业化" in content or "commercial" in content:
            score += 15
        if "投资回报" in content or "roi" in content:
            score += 10
        
        return min(score, 100.0) / 100.0
    
    def _score_risk(self, risk_data: Dict[str, Any]) -> float:
        """评估风险（负向指标，分越高风险越低）"""
        if "error" in risk_data:
            return 0.5  # 中等风险
        
        score = 70.0  # 基础分（假设大多数公司风险可控）
        
        content = str(risk_data).lower()
        
        # 发现风险因素时减分
        if "法律" in content or "legal" in content:
            score -= 10
        if "监管" in content or "regulation" in content:
            score -= 10
        if "伦理" in content or "ethical" in content:
            score -= 5
        if "合同" in content or "contract" in content:
            score -= 5
        
        return max(score, 0.0) / 100.0
    
    def analyze_all_companies(self, max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        并行分析所有目标公司
        """
        self.logger.info(f"开始分析所有{len(self.target_companies)}家公司")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_company = {
                executor.submit(self.analyze_single_company, company): company
                for company in self.target_companies
            }
            
            # 收集结果
            for future in as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"完成分析: {company}")
                except Exception as e:
                    self.logger.error(f"分析公司 {company} 失败: {str(e)}")
                    results.append({
                        "company": company,
                        "error": str(e),
                        "final_score": 0.0
                    })
        
        # 按得分排序
        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        return results
    
    def generate_summary_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成总结报告
        """
        # 过滤有效结果
        valid_results = [r for r in analysis_results if "error" not in r]
        
        if not valid_results:
            return {"error": "没有有效的分析结果"}
        
        # 统计信息
        top_company = valid_results[0]
        avg_score = sum(r["final_score"] for r in valid_results) / len(valid_results)
        
        report = {
            "summary": {
                "total_companies_analyzed": len(valid_results),
                "top_performer": top_company["company"],
                "top_score": top_company["final_score"],
                "average_score": avg_score,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "ranking": [
                {
                    "rank": i + 1,
                    "company": result["company"],
                    "score": result["final_score"],
                    "analysis_method": result.get("analysis_method", "unknown")
                }
                for i, result in enumerate(valid_results[:10])  # 前10名
            ],
            "detailed_analysis": valid_results
        } 

        return report