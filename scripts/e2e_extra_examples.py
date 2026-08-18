"""Extra examples beyond homework 1–5, plus independent SQL ground truth."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import DataAnalysisAgent
from src.config import DB_PATH

EXAMPLES = [
    "查询 2025 年上半年每家门店在到店、即时零售、团购三个渠道的销售额，生成分渠道对比柱状图。",
    "查询 2025 年上半年各品类的退损率，画出品类退损率对比图，并指出退损率最高的品类。",
    "查询 2025 年上半年每家门店的净销额（成交销额减去退款金额），从高到低排序并画柱状图。",
    "找出 2025 年第二季度销售额比第一季度下降的门店，比较这些门店两个季度的销售额，并生成对比图。",
]


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


def ground_truth() -> None:
    conn = _conn()
    print("=" * 60)
    print("GROUND TRUTH (independent SQL, not via Agent)")

    print("\n--- Extra 1: store x channel sales H1 ---")
    print(
        pd.read_sql(
            """
SELECT st.store_name,
       CASE fs.channel_code
         WHEN 'POS' THEN '到店'
         WHEN 'O2O' THEN '即时零售'
         WHEN 'B2B' THEN '团购'
         ELSE fs.channel_code
       END AS channel,
       ROUND(SUM(fs.quantity * fs.sale_price - fs.discount_amount), 2) AS sales
FROM fact_sales fs
JOIN dim_store st ON fs.store_id = st.store_id
WHERE fs.order_date BETWEEN '2025-01-01' AND '2025-06-30'
GROUP BY 1, 2
ORDER BY st.store_name, sales DESC
""",
            conn,
        ).to_string(index=False)
    )

    print("\n--- Extra 2: category refund rate H1 ---")
    print(
        pd.read_sql(
            """
WITH sales AS (
  SELECT p.category,
         SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS sales_amt
  FROM fact_sales fs
  JOIN dim_product p ON fs.product_id = p.product_id
  WHERE fs.order_date BETWEEN '2025-01-01' AND '2025-06-30'
  GROUP BY 1
),
refund AS (
  SELECT p.category, SUM(fr.refund_amount) AS refund_amt
  FROM fact_refund fr
  JOIN fact_sales fs ON fr.order_id = fs.order_id
  JOIN dim_product p ON fs.product_id = p.product_id
  WHERE fr.refund_date BETWEEN '2025-01-01' AND '2025-06-30'
  GROUP BY 1
)
SELECT s.category,
       ROUND(s.sales_amt, 2) AS sales,
       ROUND(COALESCE(r.refund_amt, 0), 2) AS refund,
       ROUND(100.0 * COALESCE(r.refund_amt, 0) / s.sales_amt, 2) AS refund_rate_pct
FROM sales s
LEFT JOIN refund r ON s.category = r.category
ORDER BY refund_rate_pct DESC
""",
            conn,
        ).to_string(index=False)
    )

    print("\n--- Extra 3: net sales H1 ---")
    print(
        pd.read_sql(
            """
WITH sales AS (
  SELECT st.store_name,
         SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS sales_amt
  FROM fact_sales fs
  JOIN dim_store st ON fs.store_id = st.store_id
  WHERE fs.order_date BETWEEN '2025-01-01' AND '2025-06-30'
  GROUP BY 1
),
refund AS (
  SELECT st.store_name, SUM(fr.refund_amount) AS refund_amt
  FROM fact_refund fr
  JOIN fact_sales fs ON fr.order_id = fs.order_id
  JOIN dim_store st ON fs.store_id = st.store_id
  WHERE fr.refund_date BETWEEN '2025-01-01' AND '2025-06-30'
  GROUP BY 1
)
SELECT s.store_name,
       ROUND(s.sales_amt, 2) AS sales,
       ROUND(COALESCE(r.refund_amt, 0), 2) AS refund,
       ROUND(s.sales_amt - COALESCE(r.refund_amt, 0), 2) AS net_sales
FROM sales s
LEFT JOIN refund r ON s.store_name = r.store_name
ORDER BY net_sales DESC
""",
            conn,
        ).to_string(index=False)
    )

    print("\n--- Extra 4: Q2 sales down vs Q1 ---")
    print(
        pd.read_sql(
            """
WITH q1 AS (
  SELECT st.store_name,
         SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS q1_sales
  FROM fact_sales fs
  JOIN dim_store st ON fs.store_id = st.store_id
  WHERE fs.order_date BETWEEN '2025-01-01' AND '2025-03-31'
  GROUP BY 1
),
q2 AS (
  SELECT st.store_name,
         SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS q2_sales
  FROM fact_sales fs
  JOIN dim_store st ON fs.store_id = st.store_id
  WHERE fs.order_date BETWEEN '2025-04-01' AND '2025-06-30'
  GROUP BY 1
)
SELECT q1.store_name,
       ROUND(q1.q1_sales, 2) AS q1_sales,
       ROUND(q2.q2_sales, 2) AS q2_sales,
       ROUND(100.0 * (q2.q2_sales - q1.q1_sales) / q1.q1_sales, 2) AS growth_pct
FROM q1
JOIN q2 ON q1.store_name = q2.store_name
WHERE q2.q2_sales < q1.q1_sales
ORDER BY growth_pct
""",
            conn,
        ).to_string(index=False)
    )
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="*", help="1-based extra example indexes")
    parser.add_argument("--truth-only", action="store_true")
    args = parser.parse_args()

    ground_truth()
    if args.truth_only:
        return 0

    indexes = args.only or list(range(1, len(EXAMPLES) + 1))
    agent = DataAnalysisAgent(max_steps=8, pace_mode="auto")
    out_dir = ROOT / "data" / "extra_example_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indexes:
        if idx < 1 or idx > len(EXAMPLES):
            print(f"skip invalid index {idx}")
            continue
        query = EXAMPLES[idx - 1]
        print("=" * 60)
        print(f"EXTRA {idx}: {query}")
        result = agent.run(query, on_progress=lambda m: print(" ", m))
        print("--- answer ---")
        print(result.answer)
        print(
            f"steps={result.steps} figures={len(result.figures)} "
            f"tables={len(result.tables)} error={result.error}"
        )
        if not result.tables:
            print("WARNING: no tables produced")
        if not result.figures:
            print("WARNING: expected a chart but figures=0")
        (out_dir / f"extra_{idx}.md").write_text(result.answer or "", encoding="utf-8")
        for i, (name, df) in enumerate(result.tables[-4:], 1):
            print(f"\n[table {i}] {name} ({len(df)} rows)")
            print(df.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
