"""Persist multi-conversation chat history for the Streamlit UI."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR

HISTORY_PATH = DATA_DIR / "chat_history.json"

_TITLE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("退损",), "门店退损情况分析"),
    (("毛利", "放量"), "毛利率与低毛利SKU分析"),
    (("毛利",), "门店毛利率分析"),
    (("季度", "对比"), "季度销售额对比"),
    (("动销", "SKU"), "华东战区动销SKU"),
    (("销售额", "柱状"), "门店销售额查询"),
    (("销售额",), "门店销售额查询"),
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def summarize_title(first_user_text: str, *, max_len: int = 22) -> str:
    """Short DeepSeek-style title from the first user message (no LLM)."""
    text = re.sub(r"\s+", "", (first_user_text or "").strip())
    if not text:
        return "新对话"
    for keys, title in _TITLE_RULES:
        if all(k in text for k in keys):
            return title[:max_len]
    # Prefer clause before first punctuation
    cut = re.split(r"[，。；;！!？?、]", text, maxsplit=1)[0].strip() or text
    if len(cut) <= max_len:
        return cut
    return cut[: max_len - 1] + "…"


def _serialize_figures(figures: list | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in figures or []:
        if isinstance(item, dict) and "figure" in item:
            out.append(item)
            continue
        title, fig = item
        if fig is None:
            continue
        if hasattr(fig, "to_json"):
            out.append({"title": title, "figure": fig.to_json()})
        elif isinstance(fig, dict):
            out.append({"title": title, "figure": json.dumps(fig, ensure_ascii=False)})
    return out


def _deserialize_figures(payload: list | None) -> list[tuple[str, Any]]:
    if not payload:
        return []
    try:
        import plotly.io as pio
    except ImportError:
        return []
    out: list[tuple[str, Any]] = []
    for item in payload:
        title = (item or {}).get("title") or ""
        raw = (item or {}).get("figure")
        if not raw:
            continue
        try:
            if isinstance(raw, str):
                fig = pio.from_json(raw)
            else:
                fig = pio.from_json(json.dumps(raw))
            out.append((title, fig))
        except Exception:  # noqa: BLE001
            continue
    return out


def _serialize_tables(tables: list | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in tables or []:
        if isinstance(item, dict) and "data" in item:
            out.append(item)
            continue
        name, df = item
        if df is None:
            continue
        frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        out.append(
            {
                "name": name,
                "columns": [str(c) for c in frame.columns.tolist()],
                "data": json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso")),
            }
        )
    return out


def _deserialize_tables(payload: list | None) -> list[tuple[str, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame]] = []
    for item in payload or []:
        name = (item or {}).get("name") or "table"
        cols = (item or {}).get("columns") or []
        data = (item or {}).get("data") or []
        df = pd.DataFrame(data)
        if cols and list(df.columns) != cols:
            df = df.reindex(columns=cols)
        out.append((name, df))
    return out


def serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages or []:
        role = msg.get("role") or "assistant"
        item: dict[str, Any] = {"role": role, "content": msg.get("content") or ""}
        if role == "assistant":
            item["process_steps"] = msg.get("process_steps") or []
            item["figures"] = _serialize_figures(msg.get("figures"))
            item["tables"] = _serialize_tables(msg.get("tables"))
        out.append(item)
    return out


def deserialize_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages or []:
        role = msg.get("role") or "assistant"
        item: dict[str, Any] = {"role": role, "content": msg.get("content") or ""}
        if role == "assistant":
            item["process_steps"] = msg.get("process_steps") or []
            item["figures"] = _deserialize_figures(msg.get("figures"))
            item["tables"] = _deserialize_tables(msg.get("tables"))
        out.append(item)
    return out


def new_conversation() -> dict[str, Any]:
    now = _now_iso()
    return {
        "id": uuid.uuid4().hex[:12],
        "title": "新对话",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def default_store() -> dict[str, Any]:
    conv = new_conversation()
    return {"current_id": conv["id"], "conversations": [conv]}


def load_store(path: Path | None = None) -> dict[str, Any]:
    target = path or HISTORY_PATH
    if not target.exists():
        store = default_store()
        save_store(store, target)
        return store
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        store = default_store()
        save_store(store, target)
        return store
    if not isinstance(data, dict):
        return default_store()
    convs = data.get("conversations")
    if not isinstance(convs, list) or not convs:
        return default_store()
    current = data.get("current_id")
    ids = {c.get("id") for c in convs if isinstance(c, dict)}
    if current not in ids:
        data["current_id"] = convs[0].get("id")
    return data


def save_store(store: dict[str, Any], path: Path | None = None) -> None:
    target = path or HISTORY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_conversation(store: dict[str, Any], conv_id: str | None) -> dict[str, Any] | None:
    for conv in store.get("conversations") or []:
        if isinstance(conv, dict) and conv.get("id") == conv_id:
            return conv
    return None


def upsert_conversation(store: dict[str, Any], conv: dict[str, Any]) -> None:
    convs: list[dict[str, Any]] = list(store.get("conversations") or [])
    for i, existing in enumerate(convs):
        if existing.get("id") == conv.get("id"):
            convs[i] = conv
            store["conversations"] = convs
            return
    convs.insert(0, conv)
    store["conversations"] = convs


def sync_current_messages(
    store: dict[str, Any],
    *,
    conv_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write live messages into the current conversation and refresh title."""
    conv = get_conversation(store, conv_id) or new_conversation()
    conv["id"] = conv_id
    conv["messages"] = serialize_messages(messages)
    conv["updated_at"] = _now_iso()
    first_user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    # Do not overwrite a manually renamed title.
    if first_user and not conv.get("title_locked"):
        conv["title"] = summarize_title(str(first_user))
    elif not conv.get("title"):
        conv["title"] = "新对话"
    if not conv.get("created_at"):
        conv["created_at"] = conv["updated_at"]
    upsert_conversation(store, conv)
    store["current_id"] = conv_id
    save_store(store)
    return conv


def start_new_conversation(store: dict[str, Any]) -> dict[str, Any]:
    """Archive is automatic (current already in store); switch to a blank thread."""
    conv = new_conversation()
    upsert_conversation(store, conv)
    store["current_id"] = conv["id"]
    # Keep newest-first ordering
    store["conversations"] = sorted(
        store.get("conversations") or [],
        key=lambda c: c.get("updated_at") or c.get("created_at") or "",
        reverse=True,
    )
    save_store(store)
    return conv


def switch_conversation(store: dict[str, Any], conv_id: str) -> dict[str, Any] | None:
    conv = get_conversation(store, conv_id)
    if not conv:
        return None
    store["current_id"] = conv_id
    save_store(store)
    return conv


def delete_conversation(store: dict[str, Any], conv_id: str) -> dict[str, Any]:
    """Remove a conversation. If it was current, switch to newest remaining (or new blank)."""
    convs = [c for c in (store.get("conversations") or []) if c.get("id") != conv_id]
    was_current = store.get("current_id") == conv_id
    if not convs:
        blank = new_conversation()
        store["conversations"] = [blank]
        store["current_id"] = blank["id"]
        save_store(store)
        return blank
    convs = sorted(
        convs,
        key=lambda c: c.get("updated_at") or c.get("created_at") or "",
        reverse=True,
    )
    store["conversations"] = convs
    if was_current:
        store["current_id"] = convs[0]["id"]
    save_store(store)
    return get_conversation(store, store["current_id"]) or convs[0]


def rename_conversation(store: dict[str, Any], conv_id: str, new_title: str) -> bool:
    """Rename a conversation. Empty title → False, keep original name."""
    title = (new_title or "").strip()
    if not title:
        return False
    conv = get_conversation(store, conv_id)
    if not conv:
        return False
    conv["title"] = title
    conv["title_locked"] = True
    conv["updated_at"] = _now_iso()
    upsert_conversation(store, conv)
    save_store(store)
    return True


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now()


def group_conversations(
    conversations: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group like DeepSeek: 今天 / 昨天 / 7天内 / 30天内 / YYYY-MM."""
    now = now or datetime.now()
    today = now.date()
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []

    def add(label: str, conv: dict[str, Any]) -> None:
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(conv)

    for conv in conversations or []:
        if not isinstance(conv, dict):
            continue
        # Skip empty drafts that are not current — still show current empty as 新对话
        ts = _parse_ts(conv.get("updated_at") or conv.get("created_at"))
        d = ts.date()
        if d == today:
            add("今天", conv)
        elif d == today - timedelta(days=1):
            add("昨天", conv)
        elif d >= today - timedelta(days=7):
            add("7 天内", conv)
        elif d >= today - timedelta(days=30):
            add("30 天内", conv)
        else:
            add(ts.strftime("%Y-%m"), conv)

    return [(label, buckets[label]) for label in order]
