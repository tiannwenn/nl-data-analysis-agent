"""Smoke tests for SQL layer and one agent example (optional --live)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db import run_sql, get_schema_text
from src.schema_pack import build_schema_pack


def test_sql_layer() -> None:
    schema = get_schema_text()
    assert "dim_store" in schema and "fact_sales" in schema
    pack = build_schema_pack()
    assert "成交销额" in pack and "即时零售" in pack

    blocked = run_sql("DELETE FROM fact_sales")
    assert not blocked.ok and blocked.error

    r = run_sql(
        """
        SELECT s.store_name,
               SUM(f.quantity * f.sale_price - f.discount_amount) AS sales_amount
        FROM fact_sales f
        JOIN dim_store s ON f.store_id = s.store_id
        WHERE f.order_date BETWEEN '2025-01-01' AND '2025-06-30'
        GROUP BY s.store_name
        ORDER BY sales_amount DESC
        """
    )
    assert r.ok and r.row_count > 0, r.error
    print("SQL layer OK:", r.row_count, "stores", r.preview[:2])

    r2 = run_sql(
        """
        SELECT p.product_id, p.product_name, p.category,
               SUM(f.quantity) AS qty,
               SUM(f.quantity * f.sale_price - f.discount_amount) AS sales_amount
        FROM fact_sales f
        JOIN dim_store s ON f.store_id = s.store_id
        JOIN dim_product p ON f.product_id = p.product_id
        WHERE f.order_date BETWEEN '2025-01-01' AND '2025-06-30'
          AND s.region = '华东' AND f.channel_code = 'O2O'
        GROUP BY p.product_id, p.product_name, p.category
        ORDER BY sales_amount DESC
        LIMIT 3
        """
    )
    assert r2.ok and r2.row_count == 3, r2.error
    print("Example2 SQL OK:", r2.preview)

    r3 = run_sql(
        """
        WITH sales AS (
          SELECT store_id,
                 SUM(quantity * sale_price - discount_amount) AS sales_amount
          FROM fact_sales
          WHERE order_date BETWEEN '2025-01-01' AND '2025-06-30'
          GROUP BY store_id
        ),
        refunds AS (
          SELECT s.store_id, SUM(r.refund_amount) AS refund_amount
          FROM fact_refund r
          JOIN fact_sales s ON r.order_id = s.order_id
          WHERE r.refund_date BETWEEN '2025-01-01' AND '2025-06-30'
          GROUP BY s.store_id
        )
        SELECT st.store_name,
               COALESCE(rf.refund_amount, 0) AS refund_amount,
               sa.sales_amount,
               COALESCE(rf.refund_amount, 0) * 1.0 / NULLIF(sa.sales_amount, 0) AS refund_rate,
               COALESCE(rf.refund_amount, 0) * 100.0 / NULLIF(sa.sales_amount, 0) AS refund_rate_pct
        FROM sales sa
        JOIN dim_store st ON sa.store_id = st.store_id
        LEFT JOIN refunds rf ON sa.store_id = rf.store_id
        ORDER BY refund_rate DESC
        """
    )
    assert r3.ok and r3.row_count == 4, r3.error
    assert float(r3.dataframe["refund_rate_pct"].max()) < 15, r3.preview
    print("Example3 correct refund SQL OK:", r3.preview[:2])

    bad = run_sql(
        """
        SELECT st.store_name,
          SUM(fr.refund_amount) AS refund_amount,
          SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS sales_amount,
          SUM(fr.refund_amount)
            / NULLIF(SUM(fs.quantity * fs.sale_price - fs.discount_amount), 0) * 100
            AS refund_rate_pct
        FROM fact_refund fr
        JOIN fact_sales fs ON fr.order_id = fs.order_id
        JOIN dim_store st ON fs.store_id = st.store_id
        WHERE fr.refund_date BETWEEN '2025-01-01' AND '2025-06-30'
        GROUP BY st.store_name
        """
    )
    assert not bad.ok, "refund denominator trap should be rejected"
    assert "denominator" in (bad.error or "").lower()
    print("Example3 bad refund SQL blocked:", (bad.error or "")[:90])

    r3_high = run_sql(
        """
        WITH sales AS (
          SELECT store_id,
                 SUM(quantity * sale_price - discount_amount) AS sales_amount
          FROM fact_sales
          WHERE order_date BETWEEN '2025-01-01' AND '2025-06-30'
          GROUP BY store_id
        ),
        refunds AS (
          SELECT s.store_id, SUM(r.refund_amount) AS refund_amount
          FROM fact_refund r
          JOIN fact_sales s ON r.order_id = s.order_id
          WHERE r.refund_date BETWEEN '2025-01-01' AND '2025-06-30'
          GROUP BY s.store_id
        )
        SELECT st.store_name,
               COALESCE(rf.refund_amount, 0) AS refund_amount,
               sa.sales_amount,
               COALESCE(rf.refund_amount, 0) * 1.0 / NULLIF(sa.sales_amount, 0) AS refund_rate
        FROM sales sa
        JOIN dim_store st ON sa.store_id = st.store_id
        LEFT JOIN refunds rf ON sa.store_id = rf.store_id
        WHERE COALESCE(rf.refund_amount, 0) * 1.0 / NULLIF(sa.sales_amount, 0) > 0.05
        ORDER BY refund_rate DESC
        """
    )
    assert r3_high.ok, r3_high.error
    print("Example3 SQL OK:", r3_high.row_count, "stores over 5%", r3_high.preview)

    r4 = run_sql(
        """
        WITH q AS (
          SELECT store_id,
            SUM(CASE WHEN order_date BETWEEN '2025-01-01' AND '2025-03-31'
                     THEN quantity * sale_price - discount_amount ELSE 0 END) AS q1,
            SUM(CASE WHEN order_date BETWEEN '2025-04-01' AND '2025-06-30'
                     THEN quantity * sale_price - discount_amount ELSE 0 END) AS q2
          FROM fact_sales
          GROUP BY store_id
        )
        SELECT st.store_name, q.q1, q.q2,
               (q.q2 - q.q1) * 1.0 / NULLIF(q.q1, 0) AS growth
        FROM q JOIN dim_store st ON q.store_id = st.store_id
        WHERE (q.q2 - q.q1) * 1.0 / NULLIF(q.q1, 0) > 0.10
        ORDER BY growth DESC
        """
    )
    assert r4.ok, r4.error
    print("Example4 SQL OK:", r4.row_count, r4.preview)

    r5 = run_sql(
        """
        WITH q AS (
          SELECT f.store_id,
            SUM(CASE WHEN order_date BETWEEN '2025-01-01' AND '2025-03-31'
                     THEN quantity * sale_price - discount_amount ELSE 0 END) AS sales_q1,
            SUM(CASE WHEN order_date BETWEEN '2025-04-01' AND '2025-06-30'
                     THEN quantity * sale_price - discount_amount ELSE 0 END) AS sales_q2,
            SUM(CASE WHEN order_date BETWEEN '2025-01-01' AND '2025-03-31'
                     THEN quantity * sale_price - discount_amount - quantity * p.unit_cost ELSE 0 END) AS gp_q1,
            SUM(CASE WHEN order_date BETWEEN '2025-04-01' AND '2025-06-30'
                     THEN quantity * sale_price - discount_amount - quantity * p.unit_cost ELSE 0 END) AS gp_q2
          FROM fact_sales f
          JOIN dim_product p ON f.product_id = p.product_id
          GROUP BY f.store_id
        )
        SELECT st.store_name, sales_q1, sales_q2,
               gp_q1 * 1.0 / NULLIF(sales_q1, 0) AS margin_q1,
               gp_q2 * 1.0 / NULLIF(sales_q2, 0) AS margin_q2
        FROM q JOIN dim_store st ON q.store_id = st.store_id
        WHERE (sales_q2 - sales_q1) * 1.0 / NULLIF(sales_q1, 0) > 0.10
          AND gp_q2 * 1.0 / NULLIF(sales_q2, 0) < gp_q1 * 1.0 / NULLIF(sales_q1, 0)
        """
    )
    assert r5.ok, r5.error
    print("Example5 SQL OK:", r5.row_count, r5.preview)
    print("All SQL smoke tests passed.")


def test_live_agent(query: str) -> None:
    from src.agent import DataAnalysisAgent

    agent = DataAnalysisAgent(max_steps=8)
    result = agent.run(query, on_progress=print)
    print("--- ANSWER ---")
    print(result.answer)
    print("--- figures ---", len(result.figures))
    print("--- tables ---", len(result.tables))
    if result.error:
        print("error:", result.error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call LLM for example 1")
    parser.add_argument(
        "--query",
        type=str,
        default="查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。",
    )
    args = parser.parse_args()
    test_sql_layer()
    if args.live:
        test_live_agent(args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
