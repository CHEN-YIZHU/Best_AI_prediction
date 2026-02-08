import argparse
import json
import logging
import os
from typing import List

from engine_analysis import AICompanyAnalyzer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for AICompanyAnalyzer")

    # ===== 测试运行相关 =====
    p.add_argument(
        "--companies",
        nargs="+",
        default=["OpenAI", "Google", "Anthropic"],
        help="要分析的公司列表（空格分隔）",
    )
    p.add_argument("--max-workers", type=int, default=1, help="并发线程数（建议先用1跑通）")

    # 是否使用 API（默认 True；可用 --no-api 
    # 输出
    p.add_argument("--save-json", type=str, default="", help="保存结果到 JSON 文件（可选）")
    p.add_argument("--log-level", type=str, default="INFO", help="日志等级：DEBUG/INFO/WARNING/ERROR")
    p.add_argument("--api-keys", nargs="+", help="Doubao API Key（可选）")
    p.add_argument("--inference-key", type=str, help="API Key（可选）")
    return p.parse_args()


def _ensure_api_keys_if_needed(args) -> None:
    """
    如果你已将 ModelAPIWrapper 改造成从环境变量读取 key（推荐），
    这里提前检查一下，避免跑到一半才报错。
    """


    if not args.api_keys:
        # 使用新的环境变量配置
        args.api_keys = [os.environ.get("OPENAI_API_KEY", "")]
    if not args.api_keys or not args.api_keys[0]:
        logging.warning(
            "API 调用需要 API Key，这会导致 API 调用失败。\n"
            "示例：export OPENAI_API_KEY='sk-xxx'"
        )
    if not args.inference_key:
        args.inference_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not args.inference_key:
        logging.warning(
            "API 调用需要 API Key，这会导致 API 调用失败。\n"
            "示例：export OPENAI_API_KEY='sk-xxx'"
        )




def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    _ensure_api_keys_if_needed(args)

    # ✅ 关键：必须传 args
    analyser = AICompanyAnalyzer(args)

    # 覆盖公司列表
    # analyser.target_companies = args.companies

    results = analyser.analyze_all_companies(
        max_workers=args.max_workers,
    )

    # print("\n===== RAW RESULTS =====")
    # print(json.dumps(results, ensure_ascii=False, indent=2))

    # 可选：输出简单总结（如果你的 generate_summary_report 可用）
    try:
        report = analyser.generate_summary_report(results)
        print("\n===== SUMMARY =====")
        print(json.dumps(report.get("summary", report), ensure_ascii=False, indent=2))
    except Exception as e:
        logging.warning(f"generate_summary_report 调用失败（可忽略，仅影响摘要打印）：{e}")



if __name__ == "__main__":
    main()


