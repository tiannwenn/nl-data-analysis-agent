"""Agent tools: schema, SQL, charting, finish."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from .config import CHARTS_DIR
from .db import get_schema_text, run_sql


_AXIS_LABELS = {
    "store_name": "门店名称",
    "store_id": "门店编号",
    "product_name": "商品名称",
    "product_id": "SKU",
    "category": "品类",
    "region": "战区",
    "sales_amount": "销售额（元）",
    "sales": "销售额（元）",
    "qty": "销量",
    "quantity": "销量",
    "refund_rate": "退损率",
    "refund_rate_pct": "退损率（%）",
    "refund_amount": "退损金额（元）",
    "growth": "增长率",
    "margin_q1": "Q1毛利率",
    "margin_q2": "Q2毛利率",
    "q1": "Q1销售额（元）",
    "q2": "Q2销售额（元）",
    "q1_sales": "Q1销售额（元）",
    "q2_sales": "Q2销售额（元）",
    "sales_q1": "Q1销售额（元）",
    "sales_q2": "Q2销售额（元）",
    "quarter": "季度",
    "门店_SKU": "门店 - SKU",
}


def _normalize_y_arg(y: Any) -> str | list[str] | None:
    if y is None:
        return None
    if isinstance(y, list):
        cols = [str(c).strip() for c in y if str(c).strip()]
        return cols or None
    text = str(y).strip()
    if not text:
        return None
    # Model sometimes passes "['q1_sales', 'q2_sales']" as a string
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return _normalize_y_arg(parsed)
        except json.JSONDecodeError:
            pass
    return text


def _infer_plot_columns(
    df: pd.DataFrame,
    x: str | None,
    y: str | list[str] | None,
    color: str | None,
) -> tuple[str | None, str | list[str] | None, str | None, str | None]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    def pick(*candidates: str) -> str | None:
        for name in candidates:
            if name in df.columns:
                return name
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    if not x:
        x = pick("store_name", "product_name", "category", "region", "store_id", "product_id")
    if color and color not in df.columns:
        color = None
    if not color:
        # Only auto-group by true series columns (avoid coloring by region accidentally).
        color = pick("quarter", "channel_code")

    if y is None:
        # Prefer long-format value column
        y = pick("sales_amount", "sales", "refund_rate_pct", "refund_rate", "qty", "quantity")
        # Wide Q1/Q2 comparison
        q_cols = [c for c in cols if re.search(r"(q1|q2).*(sales|amount)|sales.*(q1|q2)", c, re.I)]
        if len(q_cols) >= 2:
            y = q_cols[:2]
            color = None
        elif y is None:
            numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != x]
            if numeric:
                y = numeric[0]

    hint = None
    if x is None or y is None:
        hint = f"无法自动推断绘图列。可用列: {cols}。请显式传入 x/y。"
    return x, y, color, hint


def _axis_label(col: str) -> str:
    if not col:
        return ""
    if col in _AXIS_LABELS:
        return _AXIS_LABELS[col]
    lower = col.lower()
    if lower.endswith("_pct") or lower.endswith("_percent"):
        return col.replace("_pct", "（%）").replace("_percent", "（%）")
    if "sales" in lower or "amount" in lower or "销额" in col or "销售额" in col:
        return f"{col}（元）" if "元" not in col else col
    if "rate" in lower or "率" in col:
        return col if "（%）" in col or "%" in col else f"{col}（%）" if "rate" in lower else col
    return col


def _is_percent_column(col: str) -> bool:
    lower = (col or "").lower()
    return (
        lower.endswith("_pct")
        or lower.endswith("_percent")
        or "rate" in lower
        or "率" in (col or "")
    )


_FINISH_ANSWER_KEYS = (
    "answer_markdown",
    "answer",
    "answerMarkdown",
    "final_answer",
    "markdown",
    "content",
    "summary",
    "text",
    "message",
    "结论",
)


def _stringify_answer_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        nested = _extract_finish_answer(value)
        if nested:
            return nested
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value).strip()
    if isinstance(value, (list, tuple)):
        parts = [_stringify_answer_value(v) for v in value]
        return "\n".join(p for p in parts if p).strip()
    return str(value).strip()


def _extract_finish_answer(args: dict[str, Any]) -> str:
    """Accept common finish argument aliases / nested shapes from the LLM."""
    if not isinstance(args, dict) or not args:
        return ""
    for key in _FINISH_ANSWER_KEYS:
        if key in args:
            text = _stringify_answer_value(args.get(key))
            if text:
                return _unwrap_answer_markdown_json(text)
    # Case-insensitive key match
    lower_map = {str(k).lower(): v for k, v in args.items()}
    for key in _FINISH_ANSWER_KEYS:
        if key.lower() in lower_map:
            text = _stringify_answer_value(lower_map[key.lower()])
            if text:
                return _unwrap_answer_markdown_json(text)
    # Single-value object: {"anything": "markdown..."}
    if len(args) == 1:
        text = _stringify_answer_value(next(iter(args.values())))
        if text:
            return _unwrap_answer_markdown_json(text)
    return ""


def _unwrap_answer_markdown_json(text: str) -> str:
    """If the model nested a whole finish JSON into the answer string, unwrap it."""
    s = (text or "").strip()
    if not s.startswith("{") or "answer_markdown" not in s:
        return s
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        return s
    if isinstance(parsed, dict):
        inner = _extract_finish_answer(parsed)
        if inner and inner != s:
            return inner
    return s


def _recover_finish_from_raw(raw_args: str) -> str:
    """Best-effort extract markdown when tool arguments JSON is malformed."""
    raw = (raw_args or "").strip()
    if not raw or raw in {"{}", "[]", "null", '""'}:
        return ""
    # Whole payload is a JSON string
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()
        if isinstance(parsed, dict):
            return _extract_finish_answer(parsed)
    except json.JSONDecodeError:
        pass
    # Broken JSON but contains answer_markdown": "....
    for pat in (
        r'"answer_markdown"\s*:\s*"(.*)"\s*\}\s*$',
        r'"answer"\s*:\s*"(.*)"\s*\}\s*$',
        r"'answer_markdown'\s*:\s*'(.*)'\s*\}\s*$",
    ):
        m = re.search(pat, raw, flags=re.DOTALL)
        if m:
            frag = m.group(1)
            try:
                return json.loads(f'"{frag}"')
            except json.JSONDecodeError:
                return (
                    frag.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .strip()
                )
    # Raw markdown / plain text arguments
    if raw.startswith("{") and "answer" in raw.lower():
        return ""
    return raw.strip().strip('"')


def _recover_sql_from_args(args: dict[str, Any], raw_args: str = "") -> str:
    """Pull a non-empty SQL string from common keys or broken tool JSON."""
    if isinstance(args, dict):
        for key in ("sql", "query", "SQL", "statement"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    raw = (raw_args or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _recover_sql_from_args(parsed, "")
        if isinstance(parsed, str) and re.match(r"(?is)^\s*(WITH|SELECT)\b", parsed):
            return parsed.strip()
    except json.JSONDecodeError:
        pass
    m = re.search(
        r'"(?:sql|query|SQL|statement)"\s*:\s*"(.*?)"\s*(?:,|\})',
        raw,
        flags=re.DOTALL,
    )
    if m:
        frag = m.group(1)
        try:
            return json.loads(f'"{frag}"').strip()
        except json.JSONDecodeError:
            return (
                frag.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .strip()
            )
    m = re.search(r"(?is)\b((?:WITH|SELECT)\b[\s\S]+)$", raw)
    if m:
        sql = m.group(1).strip().rstrip('"').rstrip("'").rstrip("}")
        if len(sql) > 20:
            return sql
    return ""


def _should_stack_bars(df: pd.DataFrame, x: str, color: str | None, title: str) -> bool:
    """Stack only for composition charts (e.g. SKU share within a store).

    Never stack Q1/Q2 series comparisons (color == 系列 / quarter).
    """
    if not color or color not in df.columns:
        return False
    color_l = str(color).lower()
    # Melted wide Q1/Q2 or explicit quarter => always grouped side-by-side
    if color in {"系列", "quarter"} or "quarter" in color_l or color_l in {"q1", "q2"}:
        return False
    if any(k in color_l for k in ("product", "sku", "category", "商品", "品类")):
        return True
    title_l = title or ""
    if any(k in title_l for k in ("结构", "占比", "构成")) and "Q1" not in title_l and "Q2" not in title_l:
        return True
    try:
        avg_parts = df.groupby(x)[color].nunique().mean()
        return float(avg_parts) >= 2.0 and color_l not in {"系列"}
    except Exception:  # noqa: BLE001
        return False


def _ensure_store_sku_axis(df: pd.DataFrame, x: str | None, y: Any) -> tuple[pd.DataFrame, str | None]:
    """If both store and product exist, prefer『门店 - SKU』so product names stay visible.

    Store-level charts normally have no product_name column. When product_name is
    present (SKU drill-down), never plot with x=store_name alone — that hides SKUs
    and can mis-aggregate Q1/Q2 bars.
    """
    if "store_name" not in df.columns or "product_name" not in df.columns:
        return df, x
    multi_y = isinstance(y, list) and len(y) >= 2
    looking_at_products = x in {None, "product_name", "product_id", "门店_SKU"}
    # Any frame that carries product_name is SKU-grain for our agent workflows.
    if multi_y or looking_at_products or x == "store_name":
        out = df.copy()
        out["门店_SKU"] = out["store_name"].astype(str) + " - " + out["product_name"].astype(str)
        return out, "门店_SKU"
    return df, x


def _friendly_series_name(name: str) -> str:
    lower = (name or "").lower()
    if "q1" in lower:
        return "Q1"
    if "q2" in lower:
        return "Q2"
    return name


def _color_label(col: str) -> str:
    return _AXIS_LABELS.get(col, col)


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get live SQLite table/column schema and row counts.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a read-only SELECT (or WITH ... SELECT) against SQLite. "
                "Result is stored as the latest dataframe for plotting. "
                "IMPORTANT: result_name is ONLY an in-memory label for later plot_chart/"
                "analyze_dataframe — it is NOT a SQL table. Never SELECT FROM a previous result_name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Single read-only SQL statement",
                    },
                    "result_name": {
                        "type": "string",
                        "description": "Optional name to store this result for later plot/analyze",
                    },
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_chart",
            "description": (
                "Create a bar/line/pie chart from a stored query result. "
                "For『各门店对比图』, ALWAYS plot the full-store result. "
                "Q1/Q2 comparison options: "
                "(A) long format: x=store_name, y=sales_amount, color=quarter; "
                "(B) wide format: x=store_name, y=['q1_sales','q2_sales'] — tool melts automatically. "
                "If x/y omitted or wrong, tool tries to infer from available columns. "
                "title MUST be a full self-contained caption (what the chart shows + key takeaway in parentheses); "
                "the tool auto-prefixes『图N：』by generation order and embeds it ON the chart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "pie"],
                    },
                    "x": {"type": "string", "description": "X / category column"},
                    "y": {
                        "description": (
                            "Y column name, or list of numeric columns for grouped comparison "
                            "(e.g. q1_sales and q2_sales)."
                        ),
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "Chart caption without『图N』prefix (tool adds it). "
                            "Example:『2025年Q1与Q2门店销额对比（星河路店和科技园店销额显著增长）』"
                        ),
                    },
                    "color": {
                        "type": "string",
                        "description": "Optional grouping/color column for grouped bar/line",
                    },
                    "threshold_y": {
                        "type": "number",
                        "description": (
                            "Optional: highlight bars with y > threshold "
                            "(e.g. 5 for refund_rate_pct). Still plot ALL rows."
                        ),
                    },
                    "result_name": {
                        "type": "string",
                        "description": "Named result from run_sql; default latest",
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_dataframe",
            "description": (
                "Return summary stats / top rows for a stored query result to support analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result_name": {"type": "string"},
                    "sort_by": {"type": "string"},
                    "ascending": {"type": "boolean"},
                    "top_n": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Finish the task with a final Chinese markdown answer for the user. "
                "REQUIRED: pass a non-empty string in answer_markdown (the full user-facing conclusion). "
                "Do not call finish with empty arguments. "
                "Charts already show『图N：说明』on the image; in text only cite『见图1』『见图2』— "
                "do NOT re-list a『图表展示』block that can mismatch figure numbers. "
                "For『是否存在低毛利SKU放量』questions: give a clear 是/否 per store based on THIS query's numbers; "
                "if a SKU grew >=1x ((Q2-Q1)/Q1>=1), you MUST call it 显著放量 — never contradict your own multiples. "
                "Do not assume 科技园店 never has 放量."
                "If the user asked for charts, you MUST have already called plot_chart successfully."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer_markdown": {
                        "type": "string",
                        "description": "Final Chinese Markdown conclusion shown to the user (required, non-empty).",
                    },
                    "answer": {
                        "type": "string",
                        "description": "Alias of answer_markdown (accepted for compatibility).",
                    },
                },
                "required": ["answer_markdown"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class ToolArtifacts:
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
    latest_name: str | None = None
    figures: list[tuple[str, Figure]] = field(default_factory=list)
    figure_fingerprints: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    final_answer: str | None = None
    logs: list[str] = field(default_factory=list)


def _sku_coverage_warning(df: pd.DataFrame, title: str = "") -> str | None:
    """Warn when a store-SKU frame looks cherry-picked (incomplete low-margin drill-down)."""
    if "store_name" not in df.columns or "product_name" not in df.columns:
        return None
    title_l = title or ""
    sku_intent = any(k in title_l for k in ("SKU", "低毛利", "放量", "商品"))
    n_stores = int(df["store_name"].nunique())
    n_products = int(df["product_name"].nunique())
    n_rows = int(len(df))
    if n_stores < 1 or n_products < 1:
        return None
    # Always relevant when both dims exist with multi-y style grain
    if not sku_intent and n_products < 2:
        return None

    per_store = df.groupby("store_name")["product_name"].nunique()
    counts = {str(k): int(v) for k, v in per_store.items()}
    # Balanced coverage (e.g. 2+2 for margin<30% on this dataset) is OK — not cherry-picking.
    if n_stores >= 2 and int(per_store.min()) == int(per_store.max()) and int(per_store.min()) >= 2:
        return None
    if n_stores >= 2 and per_store.min() <= 2 and per_store.max() >= 4:
        return (
            f"提示：各店 SKU 覆盖不均衡 {counts}。低毛利下钻应列出阈值内全部门店-SKU，"
            "禁止只挑少数商品（易漏判某店放量）。请放宽/修正 SQL 后重查再画。"
        )
    if n_stores >= 2 and n_rows < n_stores * 3 and int(per_store.min()) < 2:
        return (
            f"提示：仅 {n_rows} 行门店-SKU（{n_stores} 家店）。"
            "请确认已覆盖各店毛利率低于阈值的全部 SKU（Q1∪Q2 并集 + 外层过滤），"
            "勿 LIMIT、勿点名抽样。"
        )
    if sku_intent and n_products <= 2 and n_stores >= 2 and int(per_store.min()) < 2:
        return (
            f"提示：标题像低毛利/SKU 分析但总共只有 {n_products} 个商品。"
            "通常应包含更多低毛利 SKU；请检查是否抽样过窄。"
        )
    return None


def _sql_error_hint(error: str, sql: str = "") -> str:
    """Actionable fix hints so the model retries correctly instead of empty/broken SQL."""
    err = (error or "").lower()
    text = sql or ""
    if (
        "misuse of aggregate" in err
        or "aggregate function used in where" in err
        or ("where" in text.lower() and "sum(" in text.lower())
    ):
        return (
            "聚合不能写在 WHERE 里（工具已拒收）。正确写法：CTE 内 GROUP BY 算出 q1_margin/q2_margin，"
            "JOIN 后在外层 WHERE q1_margin < 0.3 OR q2_margin < 0.3；"
            "或 GROUP BY ... HAVING (SUM(毛利)/NULLIF(SUM(销额),0)) < 0.3。"
            "请直接提交完整正确 SQL，不要空 sql。"
        )
    if "no such table" in err:
        return (
            "result_name 不是数据库表。请把上次查询写成同一个 WITH 子查询，"
            "或重新写完整 SELECT，禁止 FROM 上次的 result_name。"
        )
    if "no such column" in err:
        return "检查 JOIN 别名与列名是否来自已声明的表（fs/st/p/fr），不要用不存在的别名。"
    if "sql is empty" in err:
        return "禁止空 sql；请提交完整 WITH/SELECT 重试。"
    if "missing multiply" in err or "quantityfs" in text.lower() or "quantityp." in text.lower():
        return (
            "销额公式缺少乘号 *。必须写 fs.quantity * fs.sale_price 与 fs.quantity * p.unit_cost，"
            "不要写成 quantityfs.sale_price（乘号常被 Markdown/复制吃掉）。"
        )
    if "refund rate denominator" in err or "refund_rate" in err and "denominator" in err:
        return (
            "退损率分母写错了。禁止 FROM fact_refund JOIN fact_sales 后同时 SUM 退款与销额。"
            "请用双 CTE：sales 按 order_date 汇总全店销额，refunds 按 refund_date 汇总退款，"
            "再 refund_amt/sales_amt*100 得到 refund_rate_pct（本数据通常个位数%，不会是 70%+）。"
        )
    if "implausible refund_rate" in err:
        return (
            "结果退损率不合理（本零售数据通常 <15%）。多半是分母用了退款订单销额。"
            "请改用 Schema Pack 中的门店退损率双 CTE 模板重跑，再画图/下钻。"
        )
    return "请根据 error 修正后重试 run_sql（必须带非空完整 SQL）。"


def _implausible_refund_rate_error(df: pd.DataFrame) -> str | None:
    """Reject store-level refund-rate results that are almost certainly wrong."""
    if df is None or df.empty:
        return None
    cols = {str(c).lower(): c for c in df.columns}
    pct_col = cols.get("refund_rate_pct")
    rate_col = cols.get("refund_rate")
    values: list[float] = []
    if pct_col is not None:
        series = pd.to_numeric(df[pct_col], errors="coerce").dropna()
        values.extend(float(x) for x in series.tolist())
        # treat as percent points
        if values and max(values) > 30.0:
            return (
                "implausible refund_rate: "
                f"max({pct_col})={max(values):.2f} (>30). "
                "Denominator must be full-store sales, not refunded-order sales."
            )
    if rate_col is not None:
        series = pd.to_numeric(df[rate_col], errors="coerce").dropna()
        vals = [float(x) for x in series.tolist()]
        if not vals:
            return None
        # fraction form (0.08) or mistaken percent-in-rate-column (8.1 / 82)
        mx = max(vals)
        if mx <= 1.5 and mx > 0.30:
            return (
                "implausible refund_rate: "
                f"max({rate_col})={mx:.4f} (>0.30). "
                "Denominator must be full-store sales, not refunded-order sales."
            )
        if mx > 30.0:
            return (
                "implausible refund_rate: "
                f"max({rate_col})={mx:.2f} looks like a broken percent. "
                "Use dual-CTE refund_amt / sales_amt."
            )
    return None


class ToolRuntime:
    def __init__(self, charts_dir: Path | None = None):
        self.artifacts = ToolArtifacts()
        self.charts_dir = charts_dir or CHARTS_DIR
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "get_schema": self._get_schema,
            "run_sql": self._run_sql,
            "plot_chart": self._plot_chart,
            "analyze_dataframe": self._analyze,
            "finish": self._finish,
        }
        handler = handlers.get(name)
        if not handler:
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"}, ensure_ascii=False)
        try:
            return handler(arguments)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    def _get_schema(self, _args: dict[str, Any]) -> str:
        text = get_schema_text()
        return json.dumps({"ok": True, "schema": text}, ensure_ascii=False)

    def _run_sql(self, args: dict[str, Any]) -> str:
        sql = _recover_sql_from_args(args if isinstance(args, dict) else {}, "")
        name = (args.get("result_name") or "").strip() or f"result_{len(self.artifacts.dataframes) + 1}"
        if not sql:
            return json.dumps(
                {
                    "ok": False,
                    "error": "SQL is empty",
                    "hint": (
                        "禁止提交空 sql。请直接写出完整 WITH/SELECT（含筛选与聚合），"
                        "例如退款原因下钻："
                        "SELECT store_name, refund_reason, SUM(refund_amount) ... "
                        "GROUP BY 1,2。不要先发 {} 再补 SQL。"
                    ),
                },
                ensure_ascii=False,
            )

        result = run_sql(sql)
        self.artifacts.logs.append(f"run_sql[{name}]: {sql[:200]}")

        if not result.ok:
            err = result.error or "SQL failed"
            hint = _sql_error_hint(err, sql)
            return json.dumps(
                {"ok": False, "error": err, "hint": hint},
                ensure_ascii=False,
            )

        if result.dataframe is not None:
            bad_rate = _implausible_refund_rate_error(result.dataframe)
            if bad_rate:
                hint = _sql_error_hint(bad_rate, sql)
                return json.dumps(
                    {"ok": False, "error": bad_rate, "hint": hint},
                    ensure_ascii=False,
                )
            self.artifacts.dataframes[name] = result.dataframe
            self.artifacts.latest_name = name
            self.artifacts.tables.append((name, result.dataframe.copy()))

        payload = {
            "ok": True,
            "result_name": name,
            "columns": result.columns,
            "row_count": result.row_count,
            "empty": result.empty,
            "preview": result.preview,
        }
        if result.empty:
            payload["message"] = (
                "Query returned 0 rows. Consider relaxing filters (date/region/channel) "
                "or verify field values via get_schema / a broader SELECT."
            )
        elif result.dataframe is not None:
            cov = _sku_coverage_warning(result.dataframe, name + " " + sql[:120])
            if cov:
                payload["warning"] = cov
                payload["message"] = cov
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _resolve_df(self, result_name: str | None) -> tuple[str, pd.DataFrame]:
        name = (result_name or "").strip() or self.artifacts.latest_name
        if not name or name not in self.artifacts.dataframes:
            raise ValueError("No query result available. Call run_sql first.")
        return name, self.artifacts.dataframes[name]

    def _plot_chart(self, args: dict[str, Any]) -> str:
        chart_type = (args.get("chart_type") or "bar").lower()
        if chart_type in {"chart", "auto", ""}:
            chart_type = "bar"
        x = args.get("x") or None
        y = _normalize_y_arg(args.get("y"))
        title = (args.get("title") or "").strip()
        color = args.get("color") or None
        threshold_y = args.get("threshold_y")
        name, df = self._resolve_df(args.get("result_name"))

        if df.empty:
            return json.dumps({"ok": False, "error": "Cannot plot empty dataframe"}, ensure_ascii=False)

        # Recover when model passes missing/wrong columns.
        need_infer = (
            not x
            or y is None
            or (isinstance(y, str) and y not in df.columns)
            or (isinstance(y, list) and any(c not in df.columns for c in y))
        )
        if need_infer:
            # Keep valid multi-y if all exist; otherwise re-infer.
            y_for_infer = y if isinstance(y, list) and all(c in df.columns for c in y) else None
            if isinstance(y, str) and y in df.columns:
                y_for_infer = y
            x, y, color, hint = _infer_plot_columns(
                df,
                x if x in df.columns else None,
                y_for_infer,
                color,
            )
            if hint and (x is None or y is None):
                return json.dumps(
                    {
                        "ok": False,
                        "error": hint,
                        "hint": (
                            "宽表对比请用 y=['q1_sales','q2_sales']；"
                            "长表对比请用 x=store_name, y=sales_amount, color=quarter。"
                        ),
                    },
                    ensure_ascii=False,
                )

        if color and color not in df.columns:
            color = None

        plot_df = df.copy()
        plot_df, x = _ensure_store_sku_axis(plot_df, x, y)

        # Wide -> long for multi-y grouped bars (Q1 vs Q2).
        if isinstance(y, list):
            missing = [c for c in y if c not in plot_df.columns]
            if missing:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"Columns not found: {missing}. Available: {list(df.columns)}",
                    },
                    ensure_ascii=False,
                )
            if not x or x not in plot_df.columns:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"x column required for multi-y chart. Available: {list(plot_df.columns)}",
                    },
                    ensure_ascii=False,
                )
            # Keep store/product dims if present so labels stay clear after melt
            id_vars = [x]
            for extra in ("store_name", "product_name", "product_id", "category"):
                if extra in plot_df.columns and extra not in id_vars and extra != x:
                    # Don't add if it would duplicate rows incorrectly — only when constant per x
                    if plot_df.groupby(x)[extra].nunique().max() <= 1:
                        id_vars.append(extra)
            y_cols = list(y)
            plot_df = plot_df.melt(
                id_vars=id_vars,
                value_vars=y,
                var_name="系列",
                value_name="数值",
            )
            plot_df["系列"] = plot_df["系列"].map(_friendly_series_name)
            color = "系列"
            y = "数值"
            chart_type = "bar" if chart_type == "pie" else chart_type
            melt_source_cols = y_cols
        else:
            melt_source_cols = []

        assert isinstance(y, str)
        if x not in plot_df.columns or y not in plot_df.columns:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"Columns not found. Available: {list(df.columns)}",
                    "hint": "Check x/y against available columns, or omit them to auto-infer.",
                },
                ensure_ascii=False,
            )

        if not title or title.lower() in {"chart", "图", "图表"}:
            # Avoid generic "Chart" duplicates; build a readable default once.
            if x == "门店_SKU":
                title = "低毛利SKU销额对比（按门店-商品）"
            else:
                title = f"{_axis_label(x)} - {_axis_label(y)}对比"
        # Strip any model-supplied 图N prefix; renumber by successful plot order.
        title_body = re.sub(r"^图\s*\d+\s*[：:]\s*", "", title).strip() or title
        if title_body in {"门店名称 - 数值对比", "门店名称 - 销售额（元）对比"} and x == "门店_SKU":
            title_body = "低毛利SKU销额对比（按门店-商品）"

        coerced = pd.to_numeric(plot_df[y], errors="coerce")
        if coerced.notna().any():
            plot_df[y] = coerced

        if melt_source_cols:
            if any(_is_percent_column(c) for c in melt_source_cols):
                y_label = "比率（%）"
                percent_like = True
            elif any("sales" in c.lower() or "amount" in c.lower() or "销" in c for c in melt_source_cols):
                y_label = "销售额（元）"
                percent_like = False
            else:
                y_label = "数值"
                percent_like = False
        else:
            y_label = _axis_label(y) if y != "数值" else "销售额（元）"
            percent_like = _is_percent_column(y)
        x_label = _axis_label(x)
        if (
            percent_like
            and not str(y).lower().endswith(("_pct", "_percent"))
            and not melt_source_cols  # already percent-aware for melted multi-y
            and plot_df[y].notna().any()
            and float(plot_df[y].max()) <= 1.5
        ):
            plot_df[y] = plot_df[y] * 100.0
            if "（%）" not in y_label:
                y_label = f"{y_label}（%）" if y_label else "比率（%）"
            if threshold_y is not None:
                try:
                    thr_raw = float(threshold_y)
                    if thr_raw <= 1.5:
                        threshold_y = thr_raw * 100.0
                except (TypeError, ValueError):
                    pass

        hover_y = f"%{{y:,.2f}}{'%' if percent_like else ''}"
        color_label = _color_label(color) if color else ""

        if threshold_y is not None and chart_type == "bar" and color is None:
            try:
                thr = float(threshold_y)
                plot_df["是否超阈值"] = plot_df[y].apply(
                    lambda v: "超过阈值" if pd.notna(v) and float(v) > thr else "未超过阈值"
                )
                color = "是否超阈值"
                color_label = "是否超阈值"
            except (TypeError, ValueError):
                threshold_y = None

        stack_bars = chart_type == "bar" and _should_stack_bars(plot_df, x, color, title_body)
        # Q1/Q2 melted series must be grouped, never stacked
        if color == "系列":
            stack_bars = False

        if stack_bars and color:
            totals = plot_df.groupby(x)[y].transform("sum")
            plot_df = plot_df.copy()
            plot_df["_占比"] = (plot_df[y] / totals.replace(0, pd.NA) * 100).fillna(0)

        # Placeholder; replaced with 图N：… after duplicate check / before append.
        chart_title = title_body

        if chart_type == "bar":
            bar_kwargs: dict[str, Any] = {
                "data_frame": plot_df,
                "x": x,
                "y": y,
                "color": color,
                "title": chart_title,
                "barmode": "stack" if stack_bars else "group",
                "labels": {x: x_label, y: y_label, **({color: color_label} if color else {})},
            }
            if stack_bars and "_占比" in plot_df.columns:
                bar_kwargs["custom_data"] = ["_占比"]
            fig = px.bar(**bar_kwargs)
            if stack_bars and color:
                fig.update_traces(
                    hovertemplate=(
                        f"{x_label}=%{{x}}<br>"
                        f"{color_label}=%{{fullData.name}}<br>"
                        f"{y_label}={hover_y}<br>"
                        f"占比=%{{customdata[0]:.1f}}%<extra></extra>"
                    )
                )
            elif color:
                fig.update_traces(
                    hovertemplate=(
                        f"{x_label}=%{{x}}<br>"
                        f"{color_label}=%{{fullData.name}}<br>"
                        f"{y_label}={hover_y}<extra></extra>"
                    )
                )
            else:
                fig.update_traces(
                    hovertemplate=(
                        f"{x_label}=%{{x}}<br>"
                        f"{y_label}={hover_y}<extra></extra>"
                    )
                )
        elif chart_type == "line":
            fig = px.line(
                plot_df,
                x=x,
                y=y,
                color=color,
                title=chart_title,
                markers=True,
                labels={x: x_label, y: y_label, **({color: color_label} if color else {})},
            )
            if color:
                fig.update_traces(
                    hovertemplate=(
                        f"{x_label}=%{{x}}<br>"
                        f"{color_label}=%{{fullData.name}}<br>"
                        f"{y_label}={hover_y}<extra></extra>"
                    )
                )
            else:
                fig.update_traces(
                    hovertemplate=(
                        f"{x_label}=%{{x}}<br>"
                        f"{y_label}={hover_y}<extra></extra>"
                    )
                )
        elif chart_type == "pie":
            fig = px.pie(plot_df, names=x, values=y, title=chart_title)
            fig.update_traces(
                hovertemplate=(
                    f"{x_label}=%{{label}}<br>"
                    f"{y_label}=%{{value:,.2f}}（%{{percent}}）<extra></extra>"
                )
            )
        else:
            return json.dumps(
                {"ok": False, "error": f"Unsupported chart_type: {chart_type}"},
                ensure_ascii=False,
            )

        fig.update_layout(template="plotly_white", title_x=0.5, title_font_size=14)
        if stack_bars:
            fig.update_layout(legend_title_text=color_label or "系列")
        if chart_type in {"bar", "line"}:
            y_tickformat = ",.2f" if percent_like else ",.0f"
            fig.update_yaxes(
                title_text=y_label,
                tickformat=y_tickformat,
                separatethousands=True,
                exponentformat="none",
                showexponent="none",
            )
            fig.update_xaxes(title_text=x_label)
            if threshold_y is not None:
                try:
                    fig.add_hline(
                        y=float(threshold_y),
                        line_dash="dash",
                        line_color="#d62728",
                        annotation_text=f"阈值 {float(threshold_y):g}",
                        annotation_position="top left",
                    )
                except (TypeError, ValueError):
                    pass

        chart_id = f"{chart_type}_{uuid.uuid4().hex[:8]}"
        out_path = self.charts_dir / f"{chart_id}.html"

        fingerprint = "|".join(
            [
                name or "",
                chart_type,
                str(x),
                str(y),
                str(color),
                title_body,
                str(sorted(plot_df.columns.tolist())),
                str(len(plot_df)),
                str(round(float(plot_df[y].sum()), 2)) if y in plot_df.columns else "",
            ]
        )
        if fingerprint in self.artifacts.figure_fingerprints:
            return json.dumps(
                {
                    "ok": True,
                    "skipped_duplicate": True,
                    "title": title_body,
                    "message": "与已生成图表重复，已跳过，无需再次 plot_chart。",
                },
                ensure_ascii=False,
            )

        fig_idx = len(self.artifacts.figures) + 1
        numbered_title = f"图{fig_idx}：{title_body}"
        # Keep full caption on the chart so UI text cannot drift from figure numbers.
        fig.update_layout(
            title={
                "text": numbered_title,
                "x": 0.5,
                "xanchor": "center",
            }
        )
        fig.write_html(str(out_path), include_plotlyjs="cdn")

        self.artifacts.figures.append((numbered_title, fig))
        self.artifacts.figure_fingerprints.append(fingerprint)
        self.artifacts.logs.append(f"plot_chart: {numbered_title} ({chart_type}) from {name}")

        warning = None
        cov = _sku_coverage_warning(df, title_body)
        if cov:
            warning = cov
        # Only warn for true "all stores" charts; skip when analysis is intentionally filtered.
        filtered_intent = any(
            k in title_body
            for k in ("超过", "增长", "下降", "筛选", "高退损", "Top", "TOP", "最好")
        )
        title_asks_all_stores = any(k in title_body for k in ("各门店", "每家门店", "全部门店"))
        n_cats = plot_df[x].nunique() if x in plot_df.columns else len(plot_df)
        if title_asks_all_stores and not filtered_intent and x in ("store_name", "store_id"):
            try:
                from .db import connect

                with connect() as conn:
                    n_stores = conn.execute("SELECT COUNT(*) FROM dim_store").fetchone()[0]
                if int(n_cats) < int(n_stores):
                    store_warn = (
                        f"提示：标题含「各门店」但图中仅 {n_cats} 家门店（库中共 {n_stores} 家）。"
                        "若用户要全量对比请重查全部门店；若只分析筛出的门店则无需重画。"
                    )
                    warning = f"{warning} {store_warn}".strip() if warning else store_warn
            except Exception:  # noqa: BLE001
                pass

        payload = {
            "ok": True,
            "chart_id": chart_id,
            "title": numbered_title,
            "chart_type": chart_type,
            "from_result": name,
            "used_columns": {"x": x, "y": y, "color": color},
            "row_count_plotted": int(len(plot_df)),
            "path": str(out_path),
            "message": "Chart created and will be shown in the UI.",
        }
        if warning:
            payload["warning"] = warning
            payload["message"] = warning
        return json.dumps(payload, ensure_ascii=False)

    def _analyze(self, args: dict[str, Any]) -> str:
        name, df = self._resolve_df(args.get("result_name"))
        work = df.copy()
        sort_by = args.get("sort_by")
        if sort_by and sort_by in work.columns:
            work = work.sort_values(sort_by, ascending=bool(args.get("ascending", False)))
        top_n = int(args.get("top_n") or 20)
        preview = work.head(top_n)
        numeric = work.select_dtypes(include="number")
        summary = {}
        if not numeric.empty:
            summary = numeric.describe().round(4).to_dict()
        return json.dumps(
            {
                "ok": True,
                "result_name": name,
                "row_count": len(work),
                "columns": list(work.columns),
                "preview": preview.where(pd.notnull(preview), None).to_dict(orient="records"),
                "numeric_summary": summary,
            },
            ensure_ascii=False,
            default=str,
        )

    def _finish(self, args: dict[str, Any]) -> str:
        answer = _extract_finish_answer(args if isinstance(args, dict) else {})
        if not answer:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "answer_markdown is required and must be non-empty. "
                        "请再次调用 finish，参数形如："
                        '{"answer_markdown":"### 分析结论\\n..."}'
                    ),
                    "received_keys": sorted(str(k) for k in (args or {}).keys())
                    if isinstance(args, dict)
                    else [],
                },
                ensure_ascii=False,
            )
        self.artifacts.final_answer = answer
        return json.dumps({"ok": True, "finished": True}, ensure_ascii=False)
