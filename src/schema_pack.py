"""Assemble Markdown docs + DDL into a schema pack for the LLM."""

from __future__ import annotations

from pathlib import Path

from .config import DOCS, ROOT
from .db import get_schema_text


METRIC_RULES = """
## 核心指标与业务约定（必须遵守）

- 成交销额 = quantity * sale_price - discount_amount（口语「销额」默认指成交销额）
- 毛利额 = 成交销额 - quantity * unit_cost（unit_cost 来自 dim_product）
- 毛利率 = 毛利额 / 成交销额；分母用 NULLIF(成交销额, 0) 防止除零
- 退损金额 = refund_amount；退损率 = 退款金额 / 成交销额
- 净销额 = 成交销额 - 退款金额
- 统计某时段退损时，以 fact_refund.refund_date 判断；销额以 fact_sales.order_date 判断
- **退损率分母必须是该范围全部成交销额**，不是「发生过退款的订单销额」。
  禁止：`FROM fact_refund JOIN fact_sales` 后同时 `SUM(refund_amount)` 与 `SUM(quantity*sale_price-…)` 再相除
  （这会把分母缩成退款订单销额，退损率会被抬到 70%+，完全错误）。
  正确：用两个 CTE **分别**汇总退款与销额，再按 store_id 相除（见下方模板）。
  本数据集上门店退损率通常在个位数百分比，若算出 >30% 必为 SQL 写错，必须重写。
- 战区 = dim_store.region（如「华东战区」→ region = '华东'）
- SKU = dim_product.product_id，展示时同时给出 product_name
- 即时零售 = channel_code = 'O2O'；到店 = 'POS'；团购/企业团购 = 'B2B'
- 动销最好并按销额排序 → 按成交销额降序

## 时间解析约定

- 2025 年上半年 / H1 2025 → order_date BETWEEN '2025-01-01' AND '2025-06-30'
- 2025 年第一季度 / Q1 → BETWEEN '2025-01-01' AND '2025-03-31'
- 2025 年第二季度 / Q2 → BETWEEN '2025-04-01' AND '2025-06-30'

## SQL 硬约束

- 只允许只读 SELECT / WITH ... SELECT
- 禁止 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/PRAGMA 等写操作
- 表名：dim_store, dim_product, fact_sales, fact_refund
- 跨表用 store_id / product_id / order_id 关联
- 增长率 = (本期 - 上期) / NULLIF(上期, 0)
- **禁止空 SQL**：每次 run_sql 必须带完整非空 sql 字符串
- 禁止在 WHERE 中写 SUM()/AVG() 等聚合（会报 misuse of aggregate / 工具会直接拒收）。应：
  1) 先 GROUP BY 算出销额/毛利率，再用 **HAVING** 过滤；或
  2) 在外层 SELECT 对已算好的列做 WHERE（推荐）
- 低毛利 SKU 的 Q1/Q2 对比**正确模板**（可直接仿写；须覆盖阈值内全部 SKU，禁止点名抽样）。
  **重要：成交销额公式里的乘号 `*` 必须保留**，写成 `quantity * sale_price`，禁止写成 `quantitysale_price` / `quantityfs.sale_price`。
  ```
  WITH q1 AS (
    SELECT st.store_name, p.product_name,
           SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS q1_sales,
           SUM(fs.quantity * fs.sale_price - fs.discount_amount - fs.quantity * p.unit_cost)
             / NULLIF(SUM(fs.quantity * fs.sale_price - fs.discount_amount), 0) AS q1_margin
    FROM fact_sales fs
    JOIN dim_store st ON fs.store_id = st.store_id
    JOIN dim_product p ON fs.product_id = p.product_id
    WHERE fs.order_date BETWEEN '2025-01-01' AND '2025-03-31'
      AND st.store_name IN ('星河路店', '科技园店')
    GROUP BY st.store_name, p.product_name
  ),
  q2 AS (
    SELECT st.store_name, p.product_name,
           SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS q2_sales,
           SUM(fs.quantity * fs.sale_price - fs.discount_amount - fs.quantity * p.unit_cost)
             / NULLIF(SUM(fs.quantity * fs.sale_price - fs.discount_amount), 0) AS q2_margin
    FROM fact_sales fs
    JOIN dim_store st ON fs.store_id = st.store_id
    JOIN dim_product p ON fs.product_id = p.product_id
    WHERE fs.order_date BETWEEN '2025-04-01' AND '2025-06-30'
      AND st.store_name IN ('星河路店', '科技园店')
    GROUP BY st.store_name, p.product_name
  ),
  keys AS (
    SELECT store_name, product_name FROM q1
    UNION
    SELECT store_name, product_name FROM q2
  )
  SELECT k.store_name, k.product_name,
         COALESCE(q1.q1_sales, 0) AS q1_sales,
         COALESCE(q2.q2_sales, 0) AS q2_sales,
         q1.q1_margin, q2.q2_margin
  FROM keys k
  LEFT JOIN q1 USING (store_name, product_name)
  LEFT JOIN q2 USING (store_name, product_name)
  WHERE COALESCE(q1.q1_margin, q2.q2_margin) < 0.3
  ```
  说明：不要用 INNER JOIN 只保留两季都有的 SKU；不要 LIMIT；阈值 0.3（30%）可按题调整。
  在本数据集上，毛利率&lt;30% 的 SKU 通常只有激光打印机、24寸显示器等少数办公设备，两家店合计约 4 行是正常全量，不是抽样。
- 门店退损率**正确模板**（可直接仿写；日期按题调整；必须输出 refund_rate_pct）：
  ```
  WITH sales AS (
    SELECT fs.store_id,
           SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS sales_amt
    FROM fact_sales fs
    WHERE fs.order_date BETWEEN '2025-01-01' AND '2025-06-30'
    GROUP BY fs.store_id
  ),
  refunds AS (
    SELECT fs.store_id, SUM(fr.refund_amount) AS refund_amt
    FROM fact_refund fr
    JOIN fact_sales fs ON fr.order_id = fs.order_id
    WHERE fr.refund_date BETWEEN '2025-01-01' AND '2025-06-30'
    GROUP BY fs.store_id
  )
  SELECT st.store_name,
         COALESCE(r.refund_amt, 0) AS refund_amount,
         s.sales_amt AS sales_amount,
         COALESCE(r.refund_amt, 0) / NULLIF(s.sales_amt, 0) AS refund_rate,
         COALESCE(r.refund_amt, 0) / NULLIF(s.sales_amt, 0) * 100 AS refund_rate_pct
  FROM dim_store st
  JOIN sales s ON st.store_id = s.store_id
  LEFT JOIN refunds r ON st.store_id = r.store_id
  ORDER BY refund_rate_pct DESC
  ```
  高退损门店退款原因下钻（先筛出门店名，再按原因汇总；不要空 sql）：
  ```
  SELECT st.store_name, fr.refund_reason,
         COUNT(*) AS reason_count,
         SUM(fr.refund_amount) AS refund_amount
  FROM fact_refund fr
  JOIN fact_sales fs ON fr.order_id = fs.order_id
  JOIN dim_store st ON fs.store_id = st.store_id
  WHERE fr.refund_date BETWEEN '2025-01-01' AND '2025-06-30'
    AND st.store_name IN ('星河路店', '科技园店')
  GROUP BY st.store_name, fr.refund_reason
  ORDER BY st.store_name, refund_amount DESC
  ```

## 画图约定

- 「各门店 / 每家门店对比图」= 全部有数据的门店都必须出现在图上
- 阈值筛选（如退损率>5%、销额增长>10%）只用于文字名单与原因下钻，禁止作为对比图的唯一数据源
- 退损率作图：输出 `refund_rate_pct = 退损率 * 100`，纵轴为「退损率（%）」；可用 plot_chart 的 threshold_y=5 高亮超阈值门店，但仍须传入全部门店结果
- 门店 Q1/Q2 销额对比**正确模板**（「比较每家门店两季度销售额并找出增长>10%」时仿写；先全量再文字筛选）：
  ```
  WITH q1 AS (
    SELECT st.store_name,
           SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS q1_sales
    FROM fact_sales fs
    JOIN dim_store st ON fs.store_id = st.store_id
    WHERE fs.order_date BETWEEN '2025-01-01' AND '2025-03-31'
    GROUP BY st.store_name
  ),
  q2 AS (
    SELECT st.store_name,
           SUM(fs.quantity * fs.sale_price - fs.discount_amount) AS q2_sales
    FROM fact_sales fs
    JOIN dim_store st ON fs.store_id = st.store_id
    WHERE fs.order_date BETWEEN '2025-04-01' AND '2025-06-30'
    GROUP BY st.store_name
  )
  SELECT q1.store_name, q1.q1_sales, q2.q2_sales,
         (q2.q2_sales - q1.q1_sales) * 1.0 / NULLIF(q1.q1_sales, 0) AS growth
  FROM q1
  JOIN q2 USING (store_name)
  ORDER BY q1.store_name
  ```
  说明：plot_chart 用全量结果（x=store_name, y=['q1_sales','q2_sales']）；增长>10% 只在文字里点名，不要先 WHERE growth>0.1 再画图。
""".strip()


def _read_md(path: Path) -> str:
    if not path.exists():
        return f"(missing: {path.name})"
    return path.read_text(encoding="utf-8").strip()


def build_schema_pack(db_path: Path | None = None) -> str:
    ddl = get_schema_text(db_path)
    parts = [
        "# 零售数据分析 Schema Pack",
        "",
        "## 当前数据库 DDL",
        ddl,
        "",
        METRIC_RULES,
        "",
        "## 门店说明 (store_info.md)",
        _read_md(DOCS["store"]),
        "",
        "## 商品说明 (product_info.md)",
        _read_md(DOCS["product"]),
        "",
        "## 销售说明 (sales_order.md)",
        _read_md(DOCS["sales"]),
        "",
        "## 退款说明 (refund_record.md)",
        _read_md(DOCS["refund"]),
        "",
        "## 业务术语 (business_terms.md)",
        _read_md(DOCS["terms"]),
    ]
    return "\n".join(parts)


def project_root() -> Path:
    return ROOT
