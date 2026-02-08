#!/usr/bin/env python3
"""
AI 公司排名预测 - 运行入口

使用方法:
    python run_prediction.py
    python run_prediction.py --companies "OpenAI,Google,Anthropic"
"""
import os
import sys
import json
import argparse
from datetime import datetime

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(override=True)  # 强制覆盖环境变量

from prediction_engine import PredictionEngine


def main():
    parser = argparse.ArgumentParser(description="AI 公司排名预测")
    parser.add_argument(
        "--companies", 
        type=str, 
        default=None,
        help="逗号分隔的公司列表，例如: OpenAI,Google,Anthropic"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="输出文件名（不指定则自动生成）"
    )
    args = parser.parse_args()
    
    # 解析公司列表
    companies = None
    if args.companies:
        companies = [c.strip() for c in args.companies.split(",")]
    
    # 运行预测
    engine = PredictionEngine(companies=companies)
    result = engine.run()
    
    # 打印结果
    engine.print_results(result)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = args.output or f"prediction_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 完整结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
