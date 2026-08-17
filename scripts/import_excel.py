"""Import Excel files into SQLite (dim_store / dim_product / fact_sales / fact_refund)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "retail.db"

FILE_TO_TABLE = {
    "store_info.xlsx": "dim_store",
    "product_info.xlsx": "dim_product",
    "sales_order.xlsx": "fact_sales",
    "refund_record.xlsx": "fact_refund",
}

DATE_COLUMNS = {
    "dim_store": ["open_date"],
    "fact_sales": ["order_date"],
    "fact_refund": ["refund_date"],
}


def find_excel(name: str, search_dirs: list[Path]) -> Path:
    for d in search_dirs:
        candidate = d / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing {name}. Place it in project root or data/. Searched: "
        + ", ".join(str(d) for d in search_dirs)
    )


def normalize_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def import_all(db_path: Path, excel_dirs: list[Path] | None = None) -> dict[str, int]:
    search = excel_dirs or [ROOT, ROOT / "data"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for filename, table in FILE_TO_TABLE.items():
            path = find_excel(filename, search)
            df = pd.read_excel(path)
            df = normalize_dates(df, DATE_COLUMNS.get(table, []))
            df.to_sql(table, conn, index=False, if_exists="replace")
            counts[table] = len(df)

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_sales_store ON fact_sales(store_id);
            CREATE INDEX IF NOT EXISTS idx_sales_product ON fact_sales(product_id);
            CREATE INDEX IF NOT EXISTS idx_sales_date ON fact_sales(order_date);
            CREATE INDEX IF NOT EXISTS idx_sales_channel ON fact_sales(channel_code);
            CREATE INDEX IF NOT EXISTS idx_refund_order ON fact_refund(order_id);
            CREATE INDEX IF NOT EXISTS idx_refund_date ON fact_refund(refund_date);
            """
        )
        conn.commit()

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Import retail Excel files into SQLite")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    try:
        counts = import_all(args.db)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {args.db}")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
