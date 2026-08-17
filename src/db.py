"""SQLite connection and read-only SQL execution."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


@dataclass
class QueryResult:
    ok: bool
    columns: list[str]
    row_count: int
    preview: list[dict[str, Any]]
    dataframe: pd.DataFrame | None
    error: str | None = None
    empty: bool = False


def ensure_db(db_path: Path | None = None) -> Path:
    path = db_path or DB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path}. Run: python scripts/import_excel.py"
        )
    return path


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = ensure_db(db_path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def validate_readonly_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("SQL is empty")
    if ";" in cleaned:
        raise ValueError("Only a single SQL statement is allowed")
    if _FORBIDDEN.search(cleaned):
        raise ValueError("Only read-only SELECT queries are allowed")
    # Allow WITH ... SELECT and plain SELECT
    head = cleaned.lstrip("(").lstrip()
    if not re.match(r"(?is)^(WITH|SELECT)\b", head):
        raise ValueError("SQL must start with SELECT or WITH")
    agg_where = _find_aggregate_in_where(cleaned)
    if agg_where:
        raise ValueError(agg_where)
    missing_mul = _find_missing_multiply(cleaned)
    if missing_mul:
        raise ValueError(missing_mul)
    refund_trap = _find_refund_rate_denominator_trap(cleaned)
    if refund_trap:
        raise ValueError(refund_trap)
    return cleaned


def _find_missing_multiply(sql: str) -> str | None:
    """Catch quantity*sale_price written as quantityfs.sale_price (star dropped)."""
    if re.search(r"\bquantity\s*(fs|p)\.", sql, flags=re.IGNORECASE):
        return (
            "Invalid SQL: missing multiply operator '*'. "
            "Write 'fs.quantity * fs.sale_price' and 'fs.quantity * p.unit_cost', "
            "never 'fs.quantityfs.sale_price' / 'fs.quantityp.unit_cost'."
        )
    if re.search(r"\bquantity(fs|p)\.", sql, flags=re.IGNORECASE):
        return (
            "Invalid SQL: missing multiply operator '*'. "
            "Write 'fs.quantity * fs.sale_price' and 'fs.quantity * p.unit_cost'."
        )
    return None


def _find_refund_rate_denominator_trap(sql: str) -> str | None:
    """Reject refund-rate SQL that uses refunded-order sales as denominator.

    Anti-pattern: one SELECT driven by fact_refund JOIN fact_sales that projects
    both SUM(refund_amount) and SUM(quantity*sale_price-discount).
    """
    for from_m in re.finditer(r"(?is)\bfrom\s+fact_refund\b", sql):
        before = sql[: from_m.start()]
        select_matches = list(re.finditer(r"(?is)\bselect\b", before))
        select_list = None
        for sel in reversed(select_matches):
            between = before[sel.end() :]
            # Skip SELECTs that already have their own FROM (belong to earlier clauses).
            if re.search(r"(?is)\bfrom\b", between):
                continue
            select_list = between
            break
        if select_list is None:
            continue
        if not re.search(r"refund_amount", select_list, flags=re.IGNORECASE):
            continue
        if not re.search(r"sale_price|discount_amount", select_list, flags=re.IGNORECASE):
            continue
        rest = sql[from_m.end() : from_m.end() + 400]
        if re.search(r"(?is)\bjoin\s+fact_sales\b", rest[:250]):
            return (
                "Invalid SQL: refund rate denominator trap. "
                "Do NOT compute refund_rate from FROM fact_refund JOIN fact_sales "
                "while SUM(refund_amount) and SUM(quantity*sale_price-discount) together — "
                "that uses only refunded-order sales as denominator (often 70%+). "
                "Use two CTEs: sales aggregated by store on order_date, refunds aggregated "
                "by store on refund_date, then refund_amt / sales_amt."
            )
    return None


def _find_aggregate_in_where(sql: str) -> str | None:
    """Catch WHERE SUM(...)/AVG(...) before SQLite raises misuse of aggregate."""
    for m in re.finditer(r"\bWHERE\b", sql, flags=re.IGNORECASE):
        rest = sql[m.end() :]
        stop = re.search(r"\b(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT)\b", rest, flags=re.IGNORECASE)
        chunk = rest[: stop.start()] if stop else rest
        agg = re.search(r"\b(SUM|AVG|COUNT|MAX|MIN)\s*\(", chunk, flags=re.IGNORECASE)
        if not agg:
            continue
        # Allow WHERE x IN (SELECT SUM(...)) / WHERE EXISTS (SELECT ...)
        if re.search(r"\bSELECT\b", chunk[: agg.start()], flags=re.IGNORECASE):
            continue
        return (
            "Invalid SQL: aggregate function used in WHERE. "
            "Move the filter to HAVING after GROUP BY, or compute the metric in the CTE "
            "and filter the alias in an outer WHERE. "
            "Example: GROUP BY store_name, product_name HAVING "
            "(SUM(sales-cost)/NULLIF(SUM(sales),0)) < 0.3"
        )
    return None


def run_sql(sql: str, db_path: Path | None = None, preview_rows: int = 30) -> QueryResult:
    try:
        safe_sql = validate_readonly_sql(sql)
    except ValueError as exc:
        return QueryResult(
            ok=False,
            columns=[],
            row_count=0,
            preview=[],
            dataframe=None,
            error=str(exc),
        )

    try:
        with connect(db_path) as conn:
            df = pd.read_sql_query(safe_sql, conn)
    except Exception as exc:  # noqa: BLE001 - surface DB errors to agent
        return QueryResult(
            ok=False,
            columns=[],
            row_count=0,
            preview=[],
            dataframe=None,
            error=f"SQL execution error: {exc}",
        )

    if df.empty:
        return QueryResult(
            ok=True,
            columns=list(df.columns),
            row_count=0,
            preview=[],
            dataframe=df,
            empty=True,
        )

    preview = df.head(preview_rows).where(pd.notnull(df.head(preview_rows)), None)
    return QueryResult(
        ok=True,
        columns=list(df.columns),
        row_count=len(df),
        preview=preview.to_dict(orient="records"),
        dataframe=df,
        empty=False,
    )


def get_schema_text(db_path: Path | None = None) -> str:
    with connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        lines: list[str] = []
        for (name,) in tables:
            lines.append(f"TABLE {name}")
            cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
            for col in cols:
                # cid, name, type, notnull, dflt_value, pk
                pk = " PRIMARY KEY" if col[5] else ""
                lines.append(f"  - {col[1]} {col[2] or 'TEXT'}{pk}")
            n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            lines.append(f"  rows: {n}")
            lines.append("")
    return "\n".join(lines).strip()
