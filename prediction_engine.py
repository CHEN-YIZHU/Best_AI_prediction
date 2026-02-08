"""
AI 公司排名预测引擎 - 精简版

数据来源:
1. Arena Scraper → 精确 Elo 分数
2. LLM Search → 新模型预期 + 发布时间线

输出:
- 现在: 当前最佳公司
- 1个月后: 预测最佳
- 3个月后: 预测最佳  
- 6个月后: 预测最佳
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from arena_scraper import ArenaLeaderboardScraper
from api_wrapper import ModelAPIWrapper


class PredictionEngine:
    """AI 公司排名预测引擎"""
    
    # 关注的公司列表
    DEFAULT_COMPANIES = [
        "OpenAI", "Google", "Anthropic", "xAI", 
        "DeepSeek", "Baidu", "Zhipu AI", "Moonshot",
        "Meta", "Mistral", "MiniMax", "Alibaba"
    ]
    
    def __init__(self, companies: Optional[List[str]] = None):
        self.companies = companies or self.DEFAULT_COMPANIES
        self.api = ModelAPIWrapper()
        self.arena_scraper = ArenaLeaderboardScraper(headless=True)
    
    def run(self) -> Dict[str, Any]:
        """执行完整预测流程"""
        print("🚀 启动 AI 排名预测引擎")
        print(f"📊 分析公司: {', '.join(self.companies)}")
        print("=" * 60)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "companies": self.companies,
            "arena_data": {},
            "search_data": {},
            "predictions": {}
        }
        
        # Step 1: 获取 Arena 精确排名
        print("\n📊 Step 1: 抓取 Arena 排行榜...")
        arena_rankings = self._fetch_arena_rankings()
        result["arena_data"] = arena_rankings
        
        # Step 2: LLM 搜索新模型预期 + 发布时间线
        print("\n🔍 Step 2: 搜索新模型预期与发布时间线...")
        search_data = self._search_model_predictions(arena_rankings)
        result["search_data"] = search_data
        
        # Step 3: 综合预测
        print("\n🔮 Step 3: 生成预测...")
        predictions = self._generate_predictions(arena_rankings, search_data)
        result["predictions"] = predictions
        
        return result
    
    def _fetch_arena_rankings(self) -> Dict[str, Any]:
        """获取 Arena 排名数据"""
        try:
            data = self.arena_scraper.fetch_leaderboard()
            
            # 按公司筛选
            company_rankings = {}
            for company in self.companies:
                company_lower = company.lower()
                best_model = None
                
                for model in data.get("models", []):
                    name = model.get("model_name", "").lower()
                    org = model.get("organization", "").lower()
                    
                    # 匹配公司
                    if self._match_company(company_lower, name, org):
                        if best_model is None or model.get("rank", 999) < best_model.get("rank", 999):
                            best_model = model
                
                if best_model:
                    company_rankings[company] = {
                        "rank": best_model["rank"],
                        "elo": best_model["elo_score"],
                        "model": best_model["model_name"],
                        "organization": best_model.get("organization", "")
                    }
                else:
                    company_rankings[company] = {"rank": None, "elo": None, "model": None}
            
            # 按排名排序找出当前领先者
            ranked = sorted(
                [(c, d) for c, d in company_rankings.items() if d.get("rank")],
                key=lambda x: x[1]["rank"]
            )
            
            return {
                "last_updated": data.get("last_updated"),
                "total_models": data.get("total_count"),
                "company_rankings": company_rankings,
                "current_leader": ranked[0][0] if ranked else None,
                "current_top5": [c for c, _ in ranked[:5]]
            }
            
        except Exception as e:
            print(f"❌ Arena 抓取失败: {e}")
            return {"error": str(e)}
    
    def _match_company(self, company: str, model_name: str, org: str) -> bool:
        """匹配公司名"""
        keywords_map = {
            "openai": ["openai", "gpt", "chatgpt", "o1", "o3"],
            "google": ["google", "gemini"],
            "anthropic": ["anthropic", "claude"],
            "xai": ["xai", "grok"],
            "deepseek": ["deepseek"],
            "meta": ["meta", "llama"],
            "mistral": ["mistral"],
            "alibaba": ["alibaba", "qwen"],
            "baidu": ["baidu", "ernie"],
            "zhipu": ["zhipu", "glm", "智谱"],
            "minimax": ["minimax"],
            "moonshot": ["moonshot", "kimi"],
        }
        
        company_clean = company.lower().replace(" ", "").replace("ai", "")
        keywords = keywords_map.get(company_clean, [company.lower()])
        
        return any(kw in model_name or kw in org for kw in keywords)
    
    def _search_model_predictions(self, arena_data: Dict) -> Dict[str, Any]:
        """LLM 搜索：新模型预期 + 发布时间线"""
        
        # 构建当前状态摘要
        rankings_summary = []
        for company, data in arena_data.get("company_rankings", {}).items():
            if data.get("rank"):
                rankings_summary.append(f"{company}: #{data['rank']} ({data['model']}, Elo {data['elo']})")
        
        prompt = f"""
请使用联网搜索，分析以下 AI 公司的模型发布计划和预期。

【当前 Arena 排名】(来自 lmarena.ai, {arena_data.get('last_updated', '最新')})
{chr(10).join(rankings_summary)}

【需要搜索的信息】
对于每家公司，请搜索（**重点关注文本/代码生成大模型**，忽略纯图像/视频模型）：
1. **即将发布的新一代文本模型**：是否有官方预告或泄露信息？（如 GPT-5, Gemini 2, Claude 4 等）
2. **预期文本能力**：基于技术博客、论文、社区反馈，新模型的推理/编程/写作能力预期如何？
3. **发布周期**：该公司的文本模型历史发布间隔是多少？距上次发布多久了？

【输出 JSON】
```json
{{
  "companies": [
    {{
      "name": "公司名",
      "upcoming_model": {{
        "name": "模型名或null",
        "expected_release": "2026-Q1 / 2026-03 / 未知",
        "expected_arena_rank": "预计排名1-5/5-10/10+/未知",
        "confidence": "高/中/低",
        "evidence": "依据说明"
      }},
      "release_interval_months": 6,
      "months_since_last_release": 3,
      "momentum": "上升/稳定/下降"
    }}
  ],
  "key_signals": ["重要信号1", "重要信号2"],
  "search_date": "2026-02-09"
}}
```
"""
        try:
            return self.api.call_json(prompt, api_type="qwen")
        except Exception as e:
            print(f"❌ LLM 搜索失败: {e}")
            return {"error": str(e)}
    
    def _generate_predictions(self, arena_data: Dict, search_data: Dict) -> Dict[str, Any]:
        """生成最终预测"""
        
        prompt = f"""
基于以下数据，预测 AI 公司排名的未来变化。

【当前 Arena 排名】
当前领先: {arena_data.get('current_leader')}
Top 5: {', '.join(arena_data.get('current_top5', []))}

公司详情:
{json.dumps(arena_data.get('company_rankings', {}), ensure_ascii=False, indent=2)}

【新模型预期与发布时间线】
{json.dumps(search_data, ensure_ascii=False, indent=2)}

【预测规则】
1. **聚焦文本/对话能力**：仅考虑 LLM (Text/Chat) 模型的竞争力。
2. 如果某公司有即将发布的强力文本模型(预期排名1-5)，应在预测中权重较高。
3. 发布周期分析：距上次发布已超过平均间隔的公司，可能即将发新模型。
4. 当前领先者若无新模型计划，长期可能被超越。

【输出 JSON】
```json
{{
  "now": {{
    "leader": "当前领先公司",
    "model": "当前最强模型",
    "elo": 1500,
    "reason": "原因"
  }},
  "1_month": {{
    "leader": "预测领先公司",
    "likely_model": "可能的模型",
    "confidence": 0.8,
    "key_change": "关键变化说明"
  }},
  "3_months": {{
    "leader": "预测领先公司",
    "likely_model": "可能的模型",
    "confidence": 0.6,
    "key_change": "关键变化说明"
  }},
  "6_months": {{
    "leader": "预测领先公司",
    "likely_model": "可能的模型", 
    "confidence": 0.5,
    "key_change": "关键变化说明"
  }},
  "ranking_trend": [
    {{"company": "公司名", "direction": "上升/下降/稳定", "reason": "原因"}}
  ],
  "prediction_summary": "总体预测摘要"
}}
```
"""
        try:
            return self.api.call_json(prompt, api_type="qwen")
        except Exception as e:
            print(f"❌ 预测生成失败: {e}")
            return {"error": str(e)}
    
    def print_results(self, result: Dict[str, Any]):
        """打印预测结果"""
        print("\n" + "=" * 60)
        print("📊 预测结果")
        print("=" * 60)
        
        pred = result.get("predictions", {})
        
        # 当前
        now = pred.get("now", {})
        print(f"\n🏆 现在: {now.get('leader', 'N/A')}")
        print(f"   模型: {now.get('model', 'N/A')} (Elo: {now.get('elo', 'N/A')})")
        
        # 未来预测
        for period in ["1_month", "3_months", "6_months"]:
            p = pred.get(period, {})
            label = period.replace("_", " ")
            print(f"\n🔮 {label}: {p.get('leader', 'N/A')} (置信度: {p.get('confidence', 'N/A')})")
            if p.get("key_change"):
                print(f"   关键变化: {p.get('key_change')}")
        
        # 趋势
        if pred.get("ranking_trend"):
            print("\n📈 趋势预测:")
            for t in pred.get("ranking_trend", [])[:5]:
                print(f"   {t.get('company')}: {t.get('direction')} - {t.get('reason', '')[:50]}")
