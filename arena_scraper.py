"""
Arena Leaderboard Scraper using Playwright
使用浏览器自动化抓取 lmarena.ai 数据 (和 Antigravity 的 browser_subagent 原理一样)

表格列结构（2026年2月确认）:
- td:nth-child(1) = Rank
- td:nth-child(2) = Rank Spread
- td:nth-child(3) = Model (含链接)
- td:nth-child(4) = Score (Elo)
- td:nth-child(5) = 95% CI
- td:nth-child(6) = Votes
- td:nth-child(7) = Organization
"""
import json
from datetime import datetime
from typing import Dict, List, Any
from playwright.sync_api import sync_playwright


class ArenaLeaderboardScraper:
    """使用 Playwright 浏览器自动化抓取 Arena 排行榜"""
    
    LEADERBOARD_URL = "https://lmarena.ai/leaderboard/text"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    def fetch_leaderboard(self) -> Dict[str, Any]:
        """抓取排行榜数据"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "source": "lmarena.ai",
            "category": "text",
            "last_updated": None,
            "total_votes": None,
            "total_models": None,
            "models": [],
            "error": None
        }
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            
            try:
                print("🌐 打开 lmarena.ai/leaderboard/text ...")
                page.goto(self.LEADERBOARD_URL, timeout=30000)
                
                print("⏳ 等待页面加载...")
                page.wait_for_load_state("networkidle", timeout=20000)
                page.wait_for_selector("table tbody tr", timeout=15000)
                
                print("📊 提取表格数据...")
                models = self._extract_table_data(page)
                result["models"] = models
                result["total_count"] = len(models)
                
                # 尝试提取页面元数据（Last Updated, Total Votes）
                try:
                    meta = page.evaluate('''
                        () => {
                            const text = document.body.innerText;
                            const lastUpdated = text.match(/Last Updated\\s*([A-Za-z]+\\s+\\d+,?\\s*\\d*)/)?.[1];
                            const totalVotes = text.match(/Total Votes\\s*([\\d,]+)/)?.[1];
                            const totalModels = text.match(/Total Models\\s*(\\d+)/)?.[1];
                            return { lastUpdated, totalVotes, totalModels };
                        }
                    ''')
                    result["last_updated"] = meta.get("lastUpdated")
                    result["total_votes"] = meta.get("totalVotes")
                    result["total_models"] = meta.get("totalModels")
                except:
                    pass
                
            except Exception as e:
                result["error"] = str(e)
                print(f"❌ 错误: {e}")
            finally:
                browser.close()
        
        return result
    
    def _extract_table_data(self, page) -> List[Dict[str, Any]]:
        """从页面表格提取数据 - 使用正确的选择器"""
        
        table_data = page.evaluate('''
            () => {
                const rows = Array.from(document.querySelectorAll('table tbody tr'));
                return rows.map(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length < 7) return null;
                    
                    // 正确的列映射
                    const rankText = cells[0]?.innerText?.trim() || '';
                    const modelCell = cells[2];  // Model 列
                    const scoreText = cells[3]?.innerText?.trim() || '';
                    const votesText = cells[5]?.innerText?.trim() || '';
                    const orgText = cells[6]?.innerText?.trim() || '';
                    
                    // 提取模型名（从链接或文本）
                    const modelLink = modelCell?.querySelector('a');
                    const modelName = modelLink?.innerText?.trim() || modelCell?.innerText?.trim() || '';
                    
                    // 解析数值
                    const rank = parseInt(rankText) || null;
                    const score = parseInt(scoreText.replace(/,/g, '')) || null;
                    const votes = votesText.replace(/,/g, '');
                    
                    return {
                        rank: rank,
                        model_name: modelName,
                        elo_score: score,
                        votes: votes,
                        organization: orgText
                    };
                }).filter(x => x && x.model_name);
            }
        ''')
        
        return table_data or []
    
    def get_top_models(self, n: int = 20) -> List[Dict[str, Any]]:
        """获取前 N 名模型"""
        data = self.fetch_leaderboard()
        return data.get("models", [])[:n]
    
    def get_company_rankings(self) -> Dict[str, Dict[str, Any]]:
        """获取各公司最佳模型排名"""
        data = self.fetch_leaderboard()
        company_best = {}
        
        company_keywords = {
            "OpenAI": ["openai", "gpt", "chatgpt", "o1", "o3"],
            "Anthropic": ["anthropic", "claude"],
            "Google": ["google", "gemini"],
            "xAI": ["xai", "grok"],
            "DeepSeek": ["deepseek"],
            "Meta": ["meta", "llama"],
            "Mistral": ["mistral"],
            "Alibaba": ["alibaba", "qwen", "通义"],
            "ByteDance": ["bytedance", "doubao", "豆包", "seed"],
            "Baidu": ["baidu", "ernie", "文心"],
            "Zhipu AI": ["zhipu", "glm", "智谱"],
            "MiniMax": ["minimax"],
            "Moonshot": ["moonshot", "kimi"],
            "Cohere": ["cohere", "command"],
            "AI21 Labs": ["ai21", "jamba"],
        }
        
        for model in data.get("models", []):
            name_lower = model.get("model_name", "").lower()
            org_lower = model.get("organization", "").lower()
            
            for company, keywords in company_keywords.items():
                if any(kw in name_lower or kw in org_lower for kw in keywords):
                    if company not in company_best:
                        company_best[company] = {
                            "best_model": model.get("model_name"),
                            "best_rank": model.get("rank"),
                            "best_elo": model.get("elo_score"),
                            "all_models": []
                        }
                    company_best[company]["all_models"].append(model)
                    break
        
        return company_best


def main():
    """测试抓取"""
    scraper = ArenaLeaderboardScraper(headless=True)
    
    print("=" * 75)
    print("🔍 LMArena 排行榜抓取器 (Playwright 浏览器自动化)")
    print("=" * 75)
    
    data = scraper.fetch_leaderboard()
    
    if data.get("error"):
        print(f"❌ 错误: {data['error']}")
        return
    
    print(f"\n✅ 成功抓取 {data.get('total_count', 0)} 个模型")
    print(f"⏰ 抓取时间: {data['timestamp']}")
    if data.get("last_updated"):
        print(f"📅 榜单更新: {data['last_updated']}")
    if data.get("total_votes"):
        print(f"🗳️  总投票数: {data['total_votes']}")
    
    print("\n📊 Top 15:")
    print("-" * 75)
    print(f"{'Rank':>4} | {'Model':<40} | {'Elo':>5} | {'Organization':<12}")
    print("-" * 75)
    
    for model in data.get("models", [])[:15]:
        name = model.get('model_name', '')[:40]
        elo = model.get('elo_score') or 'N/A'
        rank = model.get('rank') or 'N/A'
        org = model.get('organization', '')[:12]
        print(f"{rank:>4} | {name:<40} | {elo:>5} | {org:<12}")
    
    # 保存完整数据
    with open("arena_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 完整数据已保存至 arena_data.json")
    
    # 公司排名
    print("\n📈 各公司最佳排名:")
    print("-" * 75)
    
    # 重新获取数据（使用已有数据，不重复请求）
    company_keywords = {
        "OpenAI": ["openai", "gpt", "chatgpt", "o1", "o3"],
        "Anthropic": ["anthropic", "claude"],
        "Google": ["google", "gemini"],
        "xAI": ["xai", "grok"],
        "DeepSeek": ["deepseek"],
        "Meta": ["meta", "llama"],
        "Mistral": ["mistral"],
        "Alibaba / Qwen": ["alibaba", "qwen"],
        "Baidu": ["baidu", "ernie"],
        "Zhipu AI": ["zhipu", "glm"],
        "MiniMax": ["minimax"],
        "Moonshot": ["moonshot", "kimi"],
    }
    
    company_best = {}
    for model in data.get("models", []):
        name_lower = model.get("model_name", "").lower()
        org_lower = model.get("organization", "").lower()
        
        for company, keywords in company_keywords.items():
            if any(kw in name_lower or kw in org_lower for kw in keywords):
                if company not in company_best:
                    company_best[company] = model
                break
    
    for company, model in sorted(company_best.items(), key=lambda x: x[1].get("rank", 999)):
        print(f"{company:18s} | #{model['rank']:3d} | {model['model_name'][:35]:<35} | Elo: {model['elo_score']}")


if __name__ == "__main__":
    main()
