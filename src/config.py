"""Project paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "retail.db"
CHARTS_DIR = ROOT / "charts"
DOCS = {
    "store": ROOT / "store_info.md",
    "product": ROOT / "product_info.md",
    "sales": ROOT / "sales_order.md",
    "refund": ROOT / "refund_record.md",
    "terms": ROOT / "business_terms.md",
}


def load_env() -> None:
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=True)
        return
    except ImportError:
        pass

    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def llm_settings() -> dict:
    load_env()
    api_key = (
        os.getenv("QINIU_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if not api_key:
        raise RuntimeError("QINIU_API_KEY is missing. Set it in .env")

    # Dual intervals: fast for single Q&A, stable for batch / frequent asks.
    # Legacy LLM_MIN_INTERVAL_S maps to stable if the new keys are absent.
    legacy = float(os.getenv("LLM_MIN_INTERVAL_S", "6"))
    fast = float(os.getenv("LLM_FAST_MIN_INTERVAL_S", "0"))
    stable = float(os.getenv("LLM_STABLE_MIN_INTERVAL_S", str(legacy)))
    mode = (os.getenv("LLM_PACE_MODE") or "auto").strip().lower()
    if mode not in {"auto", "fast", "stable"}:
        mode = "auto"

    return {
        "api_key": api_key,
        "base_url": os.getenv("QINIU_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://api.qnaigc.com/v1",
        "model": os.getenv("QINIU_MODEL")
        or os.getenv("LLM_MODEL")
        or "deepseek-v3",
        "fast_interval_s": fast,
        "stable_interval_s": stable,
        "pace_mode": mode,
        # backward-compatible alias used by older call sites
        "min_interval_s": stable if mode == "stable" else fast,
        "provider": os.getenv("LLM_PROVIDER", "qiniu"),
    }
