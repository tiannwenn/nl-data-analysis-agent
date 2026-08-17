"""Run end-to-end agent checks for the five homework examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import DataAnalysisAgent

EXAMPLES = [
    "查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。",
    "查询 2025 年上半年华东战区即时零售渠道动销最好的 3 个 SKU，按销额排序，并给出每个 SKU 的销量、销额和所属品类。",
    "查询各门店 2025 年上半年的退损情况，画出全部门店退损率对比图；再找出退损率超过 5% 的门店，并分析这些高退损门店的主要退款原因。",
    "比较每家门店 2025 年第一季度和第二季度的销售额，找出第二季度销售额比第一季度增长超过 10% 的门店，并生成两个季度销售额对比图。",
    "找出 2025 年第二季度销额比第一季度增长超过 10%，但毛利率下降的门店。进一步分析这些门店是否存在低毛利 SKU 放量导致整体毛利率下降，并生成合适的图表。",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="*", help="1-based example indexes")
    args = parser.parse_args()
    indexes = args.only or [1]
    agent = DataAnalysisAgent(max_steps=8)

    for idx in indexes:
        if idx < 1 or idx > len(EXAMPLES):
            print(f"skip invalid index {idx}")
            continue
        query = EXAMPLES[idx - 1]
        print("=" * 60)
        print(f"EXAMPLE {idx}: {query}")
        result = agent.run(query, on_progress=lambda m: print(" ", m))
        print("--- answer preview ---")
        print(result.answer[:800])
        print(
            f"steps={result.steps} figures={len(result.figures)} "
            f"tables={len(result.tables)} error={result.error}"
        )
        if not result.tables:
            print("WARNING: no tables produced")
        if idx in (1, 3, 4, 5) and not result.figures:
            print("WARNING: expected a chart but figures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
