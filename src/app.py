"""Streamlit Web UI for the retail data analysis agent."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import DataAnalysisAgent
from src.chat_store import (
    delete_conversation,
    deserialize_messages,
    get_conversation,
    group_conversations,
    load_store,
    rename_conversation,
    start_new_conversation,
    switch_conversation,
    sync_current_messages,
)
from src.config import DB_PATH, load_env
from src.db import ensure_db


EXAMPLE_QUERIES = [
    "查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。",
    "查询 2025 年上半年华东战区即时零售渠道动销最好的 3 个 SKU，按销额排序，并给出每个 SKU 的销量、销额和所属品类。",
    "查询各门店 2025 年上半年的退损情况，画出全部门店退损率对比图；再找出退损率超过 5% 的门店，并分析这些高退损门店的主要退款原因。",
    "比较每家门店 2025 年第一季度和第二季度的销售额，找出第二季度销售额比第一季度增长超过 10% 的门店，并生成两个季度销售额对比图。",
    "找出 2025 年第二季度销额比第一季度增长超过 10%，但毛利率下降的门店。进一步分析这些门店是否存在低毛利 SKU 放量导致整体毛利率下降，并生成合适的图表。",
]


_FAKE_LINK_RE = re.compile(
    r"\[?\s*查看图表\s*\]?\s*(\([^)]*\))?|"
    r"\[([^\]]+)\]\([^)]*查看图表[^)]*\)|"
    r"（?查看图表）?",
    re.IGNORECASE,
)


def _dedupe_figures(figures: list) -> list:
    """Keep last figure per title to avoid identical duplicates in UI."""
    by_title: dict[str, Any] = {}
    order: list[str] = []
    for title, fig in figures or []:
        key = str(title or id(fig))
        if key not in by_title:
            order.append(key)
        by_title[key] = (title, fig)
    return [by_title[k] for k in order]


def _ensure_database() -> None:
    if DB_PATH.exists():
        return
    st.info("首次启动：正在将 Excel 导入 SQLite…")
    from scripts.import_excel import import_all

    counts = import_all(DB_PATH)
    st.success("导入完成：" + ", ".join(f"{k}={v}" for k, v in counts.items()))


def _clean_answer(text: str) -> str:
    cleaned = _FAKE_LINK_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _user_wants_chart_ui(query: str) -> bool:
    q = query or ""
    return any(k in q for k in ("图", "画", "chart", "plot", "可视化"))


def _scroll_to_latest(anchor_id: str | None = None, *, mode: str = "once") -> None:
    """Scroll toward the latest chat turn — but never fight the user.

    mode:
      - start: new user turn; re-enable auto-scroll and briefly follow
      - follow: during live generation; scroll only if user hasn't scrolled up
      - once: single nudge (e.g. after finish); no long retry chain

    Previous bug: long setTimeout retries (up to ~4s) kept yanking the viewport
    back to bottom while the user was reading earlier content.
    """
    token = re.sub(r"[^a-zA-Z0-9_-]", "_", anchor_id or "bottom")
    nonce = int(st.session_state.get("_scroll_nonce", 0)) + 1
    st.session_state["_scroll_nonce"] = nonce
    mode = mode if mode in {"start", "follow", "once"} else "once"
    # start/follow: short catch-up only; once: immediate only
    if mode == "start":
        delays_js = "[0, 80, 200]"
        reset_pause = "true"
    elif mode == "follow":
        delays_js = "[0, 60]"
        reset_pause = "false"
    else:
        delays_js = "[0]"
        reset_pause = "false"

    st.html(
        f"""
        <div data-scroll-token="{token}-{nonce}" data-scroll-mode="{mode}"
             style="height:0;width:0;overflow:hidden;margin:0;padding:0;"></div>
        <script>
        (function() {{
          const mode = "{mode}";
          const delays = {delays_js};
          const resetPause = {reset_pause};

          if (!window.__chatScrollGuardInstalled) {{
            window.__chatScrollGuardInstalled = true;
            window.__chatScrollTimers = window.__chatScrollTimers || [];
            window.__userPausedAutoScroll = false;

            function pauseAutoScroll() {{
              window.__userPausedAutoScroll = true;
              (window.__chatScrollTimers || []).forEach(function(id) {{
                clearTimeout(id);
              }});
              window.__chatScrollTimers = [];
            }}

            // User scrolls up / wheels up / touch-drags → stop auto follow.
            window.addEventListener("wheel", function(e) {{
              if (e.deltaY < 0) pauseAutoScroll();
            }}, {{ passive: true }});
            window.addEventListener("touchmove", function() {{
              pauseAutoScroll();
            }}, {{ passive: true }});

            document.addEventListener("scroll", function(e) {{
              const t = e.target;
              if (!t || t === document || t === document.documentElement) return;
              if (typeof t.scrollTop === "number" && t.dataset) {{
                const prev = Number(t.dataset._prevScrollTop || t.scrollTop);
                if (t.scrollTop + 2 < prev) pauseAutoScroll();
                t.dataset._prevScrollTop = String(t.scrollTop);
              }}
            }}, true);
          }}

          if (resetPause) {{
            window.__userPausedAutoScroll = false;
          }}

          // Cancel any pending retries from earlier scroll injections.
          (window.__chatScrollTimers || []).forEach(function(id) {{ clearTimeout(id); }});
          window.__chatScrollTimers = [];

          function pickScrollables() {{
            const sel = [
              '[data-testid="stMain"]',
              '[data-testid="stAppViewContainer"] section.main',
              'section.main',
              '[data-testid="stAppViewContainer"]',
              '.main',
            ];
            const out = [];
            for (const s of sel) {{
              document.querySelectorAll(s).forEach(function(el) {{
                if (el && out.indexOf(el) < 0) out.push(el);
              }});
            }}
            document.querySelectorAll("div").forEach(function(el) {{
              try {{
                const stl = window.getComputedStyle(el);
                const oy = stl.overflowY;
                if (
                  (oy === "auto" || oy === "scroll" || oy === "overlay") &&
                  el.scrollHeight > el.clientHeight + 8
                ) {{
                  if (out.indexOf(el) < 0) out.push(el);
                }}
              }} catch (e) {{}}
            }});
            if (document.scrollingElement) out.push(document.scrollingElement);
            return out;
          }}

          function scrollOnce() {{
            if (window.__userPausedAutoScroll) return;
            try {{
              const msgs = document.querySelectorAll('[data-testid="stChatMessage"]');
              if (!msgs.length) return;
              // On a new turn, keep the user bubble fully visible under the toolbar.
              // Do NOT force scrollTop=scrollHeight (that hides the first line).
              let target = msgs[msgs.length - 1];
              if (mode === "start" && msgs.length >= 2) {{
                target = msgs[msgs.length - 2];
              }}
              target.scrollIntoView({{
                behavior: mode === "once" ? "auto" : "smooth",
                block: mode === "start" ? "start" : "nearest",
                inline: "nearest"
              }});
              if (mode === "start") {{
                // Extra offset beyond scroll-margin-top (sticky header + brand bar).
                const offset = 96;
                pickScrollables().forEach(function(el) {{
                  try {{ el.scrollTop = Math.max(0, el.scrollTop - offset); }} catch (e) {{}}
                }});
                try {{ window.scrollBy(0, -offset); }} catch (e) {{}}
              }}
            }} catch (e) {{}}
          }}

          delays.forEach(function(ms) {{
            const id = setTimeout(scrollOnce, ms);
            window.__chatScrollTimers.push(id);
          }});
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
        width="content",
    )


_STOP_REPLY = "分析已停止。可修改问题后重新提问。"

# IMPORTANT: While ScriptRequestType.STOP is pending, ANY st.session_state
# get/set calls SafeSessionState yield_callback → raises StopException.
# Cancel intent must live outside session_state until STOP is cleared.
_cancel_by_session: dict[str, bool] = {}


def _script_session_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        return str(getattr(ctx, "session_id", None) or "default")
    except Exception:  # noqa: BLE001
        return "default"


def _latch_cancel() -> None:
    _cancel_by_session[_script_session_id()] = True


def _clear_cancel_latch() -> None:
    _cancel_by_session.pop(_script_session_id(), None)


def _cancel_latched() -> bool:
    return bool(_cancel_by_session.get(_script_session_id()))


def _native_stop_pending() -> bool:
    """True when Streamlit script_requests is in terminal STOP state.

    Reads only script_requests — never touches st.session_state (would yield).
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime.scriptrunner_utils.script_requests import ScriptRequestType

        ctx = get_script_run_ctx()
        if ctx is None:
            return False
        requests = getattr(ctx, "script_requests", None)
        if requests is None:
            return False
        with requests._lock:  # noqa: SLF001 — cooperative cancel with Streamlit Stop
            return requests._state == ScriptRequestType.STOP  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return False


def _streamlit_stop_requested() -> bool:
    """True when user clicked chat_input stop / toolbar Stop / cancel latch.

    Must not touch st.session_state while native STOP may be pending.
    """
    if _cancel_latched():
        return True
    if _native_stop_pending():
        _latch_cancel()
        return True
    return False


def _is_streamlit_stop_exc(exc: BaseException) -> bool:
    """StopException inherits BaseException (not Exception) — must detect by name/type."""
    name = type(exc).__name__
    if name in {"StopException", "ScriptControlException"}:
        return True
    try:
        from streamlit.runtime.scriptrunner_utils.exceptions import StopException

        return isinstance(exc, StopException)
    except Exception:  # noqa: BLE001
        return False


def _is_streamlit_rerun_exc(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name == "RerunException":
        return True
    try:
        from streamlit.runtime.scriptrunner_utils.exceptions import RerunException

        return isinstance(exc, RerunException)
    except Exception:  # noqa: BLE001
        return False


def _clear_native_stop_for_rerun() -> bool:
    """STOP is terminal for request_rerun() and blocks session_state access.

    Clear it before any st.session_state / widget / st.rerun() calls.
    Cancel intent must already be latched via ``_latch_cancel()``.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime.scriptrunner_utils.script_requests import ScriptRequestType

        ctx = get_script_run_ctx()
        requests = getattr(ctx, "script_requests", None) if ctx else None
        if requests is None:
            return False
        with requests._lock:  # noqa: SLF001
            if requests._state == ScriptRequestType.STOP:  # noqa: SLF001
                requests._state = ScriptRequestType.CONTINUE  # noqa: SLF001
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _append_stopped_assistant(*, steps: list | None = None) -> bool:
    """Persist stop reply to messages + disk.

    Caller must clear native STOP first (session_state yields otherwise).
    ``steps=None`` means do not overwrite an existing stop message's steps.
    """
    msgs = list(st.session_state.get("messages") or [])
    clean_steps = (
        [{k: v for k, v in ev.items() if k != "figure"} for ev in steps]
        if steps is not None
        else None
    )
    if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("content") == _STOP_REPLY:
        if clean_steps is not None:
            msgs[-1] = {
                **msgs[-1],
                "process_steps": clean_steps,
            }
            st.session_state.messages = msgs
            _persist_current_conversation()
        return False
    if not msgs or msgs[-1].get("role") != "user":
        return False
    msgs.append(
        {
            "role": "assistant",
            "content": _STOP_REPLY,
            "figures": [],
            "tables": [],
            "process_steps": clean_steps or [],
        }
    )
    st.session_state.messages = msgs
    _persist_current_conversation()
    return True


def _commit_stop_and_rerun(*, steps: list | None = None) -> None:
    """Safe stop teardown: latch → clear STOP → persist → full-page rerun."""
    _latch_cancel()
    _clear_native_stop_for_rerun()
    try:
        _append_stopped_assistant(steps=steps)
    finally:
        _clear_cancel_latch()
        st.session_state.pop("_agent_cancel", None)
    try:
        st.rerun()
    except BaseException as exc:  # noqa: BLE001
        if _is_streamlit_rerun_exc(exc):
            raise
        if _is_streamlit_stop_exc(exc):
            return
        raise


def _repair_incomplete_turn() -> bool:
    """If a previous run was killed mid-flight, close the orphan user turn."""
    return _append_stopped_assistant()

def _init_chat_state() -> None:
    """Load multi-conversation store once per browser session."""
    if "chat_store" not in st.session_state:
        st.session_state.chat_store = load_store()
    store = st.session_state.chat_store
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = store.get("current_id")
    if "messages" not in st.session_state:
        conv = get_conversation(store, st.session_state.current_conv_id)
        st.session_state.messages = deserialize_messages((conv or {}).get("messages"))


def _persist_current_conversation() -> None:
    store = st.session_state.chat_store
    conv_id = st.session_state.current_conv_id
    sync_current_messages(store, conv_id=conv_id, messages=st.session_state.messages)
    st.session_state.chat_store = store


def _activate_conversation(conv_id: str) -> None:
    store = st.session_state.chat_store
    conv = switch_conversation(store, conv_id)
    if not conv:
        return
    st.session_state.chat_store = store
    st.session_state.current_conv_id = conv_id
    st.session_state.messages = deserialize_messages(conv.get("messages"))
    st.session_state.pop("_scroll_to_bottom", None)
    st.session_state.pop("_agent_cancel", None)


def _render_sidebar() -> None:
    store = st.session_state.chat_store
    current_id = st.session_state.current_conv_id
    current_msgs = st.session_state.get("messages") or []

    if st.button("新建对话", width="stretch", type="primary"):
        if not current_msgs:
            st.toast("当前对话为空，无需新建")
        else:
            _persist_current_conversation()
            conv = start_new_conversation(st.session_state.chat_store)
            st.session_state.current_conv_id = conv["id"]
            st.session_state.messages = []
            st.session_state.pop("_scroll_to_bottom", None)
            st.session_state.pop("_agent_cancel", None)
            st.rerun()

    st.header("示例问题")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        if st.button(f"示例 {i}", key=f"ex_{i}", width="stretch"):
            st.session_state["prefill"] = q
            _request_scroll_to_bottom()

    st.divider()
    st.caption("历史对话")
    convs = list(store.get("conversations") or [])
    convs = sorted(
        convs,
        key=lambda c: c.get("updated_at") or c.get("created_at") or "",
        reverse=True,
    )
    visible = []
    for conv in convs:
        cid = conv.get("id")
        msgs = conv.get("messages") or []
        # Show current even if empty (DeepSeek-style live slot); hide other blanks.
        if not msgs and cid != current_id:
            continue
        visible.append(conv)

    if not visible:
        st.caption("暂无记录，提问后会出现在这里")
    else:
        for label, group in group_conversations(visible):
            st.markdown(
                f"<div style='font-size:0.75rem;opacity:0.65;margin:0.55rem 0 0.2rem;'>{label}</div>",
                unsafe_allow_html=True,
            )
            for conv in group:
                cid = conv.get("id") or ""
                title = (conv.get("title") or "新对话").strip() or "新对话"
                is_current = cid == current_id
                # Current session stays in the list (highlighted). Still clickable so it
                # doesn't look "broken"; click is a no-op when already active.
                open_col, menu_col = st.columns([5, 1], gap="small")
                with open_col:
                    label_txt = f"{'● ' if is_current else ''}{title}"
                    if st.button(
                        label_txt,
                        key=f"conv_{cid}",
                        width="stretch",
                        type="primary" if is_current else "secondary",
                    ):
                        if not is_current:
                            _activate_conversation(cid)
                            st.rerun()
                with menu_col:
                    with st.popover("⋯"):
                        if st.button(
                            "重命名",
                            key=f"rename_open_{cid}",
                            width="stretch",
                            type="secondary",
                        ):
                            st.session_state["rename_cid"] = cid
                            st.session_state["rename_draft"] = title
                            st.rerun()
                        if st.button(
                            "删除",
                            key=f"del_{cid}",
                            width="stretch",
                            type="secondary",
                        ):
                            if cid != current_id:
                                _persist_current_conversation()
                            nxt = delete_conversation(st.session_state.chat_store, cid)
                            st.session_state.chat_store = load_store()
                            st.session_state.current_conv_id = st.session_state.chat_store.get(
                                "current_id"
                            )
                            cur = get_conversation(
                                st.session_state.chat_store,
                                st.session_state.current_conv_id,
                            )
                            st.session_state.messages = deserialize_messages(
                                (cur or nxt or {}).get("messages")
                            )
                            st.session_state.pop("_agent_cancel", None)
                            st.session_state.pop("rename_cid", None)
                            st.toast(f"已删除：{title}")
                            st.rerun()

                if st.session_state.get("rename_cid") == cid:
                    draft = st.text_input(
                        "新标题",
                        value=st.session_state.get("rename_draft", title),
                        key=f"rename_input_{cid}",
                        label_visibility="collapsed",
                        placeholder="输入新标题（不能为空）",
                    )
                    ok_col, cancel_col = st.columns(2)
                    with ok_col:
                        if st.button("确定", key=f"rename_ok_{cid}", width="stretch"):
                            new_title = (draft or "").strip()
                            if not new_title:
                                st.toast("标题不能为空，仍保持原名")
                            elif rename_conversation(
                                st.session_state.chat_store, cid, new_title
                            ):
                                st.session_state.chat_store = load_store()
                                st.session_state.pop("rename_cid", None)
                                st.session_state.pop("rename_draft", None)
                                st.toast(f"已重命名为：{new_title}")
                                st.rerun()
                            else:
                                st.toast("重命名失败，仍保持原名")
                    with cancel_col:
                        if st.button("取消", key=f"rename_cancel_{cid}", width="stretch"):
                            st.session_state.pop("rename_cid", None)
                            st.session_state.pop("rename_draft", None)
                            st.rerun()


def _request_scroll_to_bottom() -> None:
    st.session_state["_scroll_to_bottom"] = True


def _maybe_scroll_to_bottom(*, force: bool = False) -> None:
    if force or st.session_state.pop("_scroll_to_bottom", False):
        _scroll_to_latest("pending", mode="start")



def _render_process_steps(steps: list[dict[str, Any]], *, expanded: bool = False) -> None:
    if not steps:
        return
    with st.expander("Agent 分析过程", expanded=expanded):
        for i, ev in enumerate(steps, 1):
            text = ev.get("text") or ev.get("type") or ""
            tool = ev.get("tool")
            prefix = f"**{i}.** "
            if tool:
                st.markdown(f"{prefix}`{tool}` — {text}")
            else:
                st.markdown(f"{prefix}{text}")


def _figure_title_body(title: str) -> str:
    return re.sub(r"^图\s*\d+\s*[：:]\s*", "", title or "").strip()


def _rewrite_figure_refs(answer: str, numbered: list[tuple[int, str, Any]]) -> str:
    """Normalize vague『下图为』refs into 图N labels."""
    text = answer
    for idx, title, _fig in numbered:
        body = _figure_title_body(title)
        if not body:
            continue
        patterns = [
            rf"下图为[「「\"']?{re.escape(body)}[」」\"']?",
            rf"如下图[（(]?[「\"']?{re.escape(body)}[」」\"']?[）)]?",
        ]
        if title and title != body:
            patterns.extend(
                [
                    rf"下图为[「「\"']?{re.escape(title)}[」」\"']?",
                    rf"如下图[（(]?[「\"']?{re.escape(title)}[」」\"']?[）)]?",
                ]
            )
        for pat in patterns:
            text = re.sub(pat, f"如图{idx}所示", text)
        for bare in (f"「{body}」", f"「{title}」"):
            if bare in text and f"图{idx}" not in text.split(bare)[0][-8:]:
                text = text.replace(f"- {bare}", f"- 见图{idx}")
                text = text.replace(f"* {bare}", f"* 见图{idx}")
    # Drop redundant「图表展示」checklist — captions already live on the charts.
    text = re.sub(
        r"^[ \t]*图表展示[：:].*?(?=\n[ \t]*业务建议|\n[ \t]*#{1,3}\s|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _render_answer_with_figures(
    answer: str,
    figures: list,
    *,
    key_prefix: str,
) -> None:
    """Render answer, then charts in 图1→图2→… order.

    Charts are NOT interleaved by whatever order the model wrote『见图N』—
    that previously put 图2 above 图1 when the text mentioned 图2 first.
    """
    answer = _clean_answer(answer)
    remaining = _dedupe_figures(list(figures or []))
    if not remaining:
        if answer:
            st.markdown(answer)
        return

    numbered: list[tuple[int, str, Any]] = [
        (i + 1, title, fig) for i, (title, fig) in enumerate(remaining)
    ]
    answer = _rewrite_figure_refs(answer, numbered)
    if answer:
        st.markdown(answer)

    for idx, title, fig in numbered:
        display_title = (title or "").strip() or f"图{idx}"
        if not re.match(rf"^图\s*{idx}\b", display_title):
            body = _figure_title_body(display_title) or display_title
            display_title = f"图{idx}：{body}"
        try:
            fig.update_layout(
                title={"text": display_title, "x": 0.5, "xanchor": "center"},
                title_font_size=14,
            )
        except Exception:  # noqa: BLE001
            pass
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_f{idx}")


def _render_message(msg: dict[str, Any], *, key_prefix: str) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return
        _render_process_steps(msg.get("process_steps") or [], expanded=False)
        _render_answer_with_figures(
            msg.get("content") or "",
            msg.get("figures") or [],
            key_prefix=key_prefix,
        )
        for i, (name, df) in enumerate(msg.get("tables") or []):
            with st.expander(f"数据表：{name}", expanded=False):
                st.dataframe(df, width="stretch")


def main() -> None:
    st.set_page_config(page_title="自然语言数据分析 Agent", layout="wide")
    load_env()
    st.markdown(
        """
        <style>
        /* Brand on Streamlit top bar — always dark bar + light text (light theme
           inherits dark text, which made the title / Deploy / ⋮ invisible) */
        header[data-testid="stHeader"] {
            background-color: #0e1117 !important;
        }
        header[data-testid="stHeader"]::before {
            content: "自然语言数据分析 Agent";
            position: absolute;
            left: 3.75rem; /* clear sidebar toggle */
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            white-space: nowrap;
            pointer-events: none;
            z-index: 1;
            color: #fafafa !important;
            opacity: 0.95;
        }
        /* Force light chrome on dark header (Deploy + ⋮ stay readable in Light theme) */
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] a,
        header[data-testid="stHeader"] [kind],
        header[data-testid="stHeader"] [data-testid="stToolbarActions"],
        header[data-testid="stHeader"] [data-testid="stMainMenu"] {
            color: #fafafa !important;
        }
        header[data-testid="stHeader"] button:hover,
        header[data-testid="stHeader"] a:hover {
            color: #ffffff !important;
            background-color: rgba(250, 250, 250, 0.12) !important;
        }
        header[data-testid="stHeader"] svg {
            fill: #fafafa !important;
            stroke: #fafafa !important;
            color: #fafafa !important;
        }
        header[data-testid="stHeader"] span,
        header[data-testid="stHeader"] p,
        header[data-testid="stHeader"] label {
            color: #fafafa !important;
        }
        /* Keep content below sticky header (1rem was too small → first line clipped) */
        div.block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 1rem !important;
            max-width: 1100px;
        }
        div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
        }
        div[data-testid="stChatMessage"] {
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            gap: 0.3rem !important;
            margin: 0 !important;
            overflow: visible !important;
            /* scrollIntoView(block=start) must clear sticky header */
            scroll-margin-top: 4.75rem !important;
        }
        div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            padding-top: 0.35rem !important;
            padding-bottom: 0.35rem !important;
            overflow: visible !important;
        }
        div[data-testid="stStatus"] {
            margin: 0 !important;
        }
        .stMarkdown h3 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.3rem !important;
            font-size: 1.15rem !important;
        }
        div[data-testid="stExpander"] {
            margin: 0.1rem 0 !important;
        }
        /* Hide Streamlit toolbar Stop / running widget — chat_input stop is enough */
        [data-testid="stStatusWidget"],
        [data-testid="stStatusWidget"] button {
            display: none !important;
        }
        /* History ⋯ popover: keep default chevron; shrink menu chrome */
        section[data-testid="stSidebar"] [data-testid="stPopover"] > div > button {
            min-height: 2rem !important;
            height: 2rem !important;
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
        }
        div[data-testid="stPopoverBody"],
        div[data-baseweb="popover"] > div {
            min-width: 7.5rem !important;
            padding: 0.35rem !important;
        }
        /* Two independent compact buttons with clear separation */
        div[data-testid="stPopoverBody"] button,
        div[data-baseweb="popover"] button {
            min-height: 1.85rem !important;
            height: 1.85rem !important;
            padding: 0.2rem 0.55rem !important;
            font-size: 0.84rem !important;
            line-height: 1.2 !important;
            margin: 0 !important;
        }
        div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
        div[data-baseweb="popover"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Fallback: newer Streamlit host chrome labels the control as plain "Stop".
    # Never hide the chat_input stop button (stChatInputStopButton).
    st.html(
        """
        <script>
        (function() {
          function hideToolbarStop() {
            document.querySelectorAll("button, [role='button']").forEach(function(el) {
              if (el.closest('[data-testid="stChatInput"]') ||
                  el.getAttribute("data-testid") === "stChatInputStopButton") {
                return;
              }
              const t = (el.textContent || "").replace(/\\s+/g, " ").trim();
              if (t === "Stop") {
                el.style.setProperty("display", "none", "important");
              }
            });
          }
          hideToolbarStop();
          if (!window.__hideStopObserver) {
            window.__hideStopObserver = new MutationObserver(hideToolbarStop);
            window.__hideStopObserver.observe(document.documentElement, {
              childList: true,
              subtree: true
            });
          }
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
        width="content",
    )

    try:
        _ensure_database()
        ensure_db()
    except Exception as exc:  # noqa: BLE001
        st.error(f"数据库不可用：{exc}")
        st.stop()

    _init_chat_state()

    with st.sidebar:
        _render_sidebar()

    _repair_incomplete_turn()

    # No fixed height — avoids a large empty black band under short replies.
    chat_box = st.container(border=False)
    conv_key = st.session_state.current_conv_id or "main"
    with chat_box:
        for idx, msg in enumerate(st.session_state.messages):
            _render_message(msg, key_prefix=f"hist_{conv_key}_{idx}")

    # After replaying history, also jump via main-document JS when requested.
    _maybe_scroll_to_bottom()

    prefill = st.session_state.pop("prefill", None)
    # DeepSeek-style: send button becomes stop (circle+square) while the script runs.
    user_input = st.chat_input(
        "用自然语言描述你的查询或分析需求…",
        submit_mode="stop",
    )
    query = prefill or user_input
    if not query:
        return

    _request_scroll_to_bottom()
    _clear_cancel_latch()
    st.session_state.pop("_agent_cancel", None)
    st.session_state.messages.append({"role": "user", "content": query})
    _persist_current_conversation()
    with chat_box:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            anchor = f"latest_{len(st.session_state.messages)}"
            _scroll_to_latest(anchor, mode="start")

            progress_slot = st.empty()
            with progress_slot.container():
                status = st.status("Agent 分析中…", expanded=True)
            live_steps: list[dict[str, Any]] = []
            event_count = 0
            stop_ui_done = False

            def _mark_stopping() -> None:
                """Latch cancel; clear STOP before any session_state / widget I/O."""
                nonlocal stop_ui_done
                if stop_ui_done:
                    return
                stop_ui_done = True
                _latch_cancel()
                # Critical: clear STOP before touching session_state or status.
                _clear_native_stop_for_rerun()
                _append_stopped_assistant(steps=live_steps)
                try:
                    status.update(label="正在停止…", state="error", expanded=True)
                    status.write("已收到停止请求，正在结束当前步骤…")
                except BaseException:  # noqa: BLE001
                    try:
                        progress_slot.empty()
                        st.caption("✗ 正在停止…")
                    except BaseException:  # noqa: BLE001
                        pass

            def on_event(event: dict[str, Any]) -> None:
                nonlocal event_count
                if _streamlit_stop_requested():
                    _mark_stopping()
                    return
                live_steps.append(event)
                event_count += 1
                text = event.get("text") or ""
                tool = event.get("tool")
                try:
                    if tool:
                        status.write(f"`{tool}` — {text}")
                    elif text:
                        status.write(text)
                    if event_count in {1, 3, 6} or (
                        event_count > 6 and event_count % 4 == 0
                    ):
                        _scroll_to_latest(f"{anchor}_step{event_count}", mode="follow")
                except BaseException as exc:  # noqa: BLE001 — Stop mid-write / scroll
                    if _is_streamlit_rerun_exc(exc):
                        raise
                    _mark_stopping()
                    return

            def _persist_assistant(
                content: str,
                *,
                figures: list | None = None,
                tables: list | None = None,
                steps: list | None = None,
            ) -> None:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "figures": _dedupe_figures(figures or []),
                        "tables": (tables or [])[-3:],
                        "process_steps": [
                            {k: v for k, v in ev.items() if k != "figure"}
                            for ev in (steps or live_steps)
                        ],
                    }
                )
                _persist_current_conversation()

            def _finish_progress(label: str, *, ok: bool) -> None:
                """Collapse live status (keep steps inside) instead of wiping it."""
                try:
                    status.update(
                        label=f"✓ {label}" if ok else f"✗ {label}",
                        state="complete" if ok else "error",
                        expanded=False,
                    )
                except BaseException:  # noqa: BLE001
                    try:
                        progress_slot.empty()
                        st.caption(f"{'✓' if ok else '✗'} {label}")
                        _render_process_steps(live_steps, expanded=False)
                    except BaseException:  # noqa: BLE001
                        pass

            def _finalize_stopped() -> None:
                _commit_stop_and_rerun(steps=live_steps)

            try:
                agent = DataAnalysisAgent(max_steps=10, pace_mode="auto")
                result = agent.run(
                    query,
                    on_event=on_event,
                    should_cancel=_streamlit_stop_requested,
                )
            except BaseException as exc:  # noqa: BLE001 — StopException is BaseException
                if _is_streamlit_rerun_exc(exc):
                    raise
                if _is_streamlit_stop_exc(exc) or _streamlit_stop_requested():
                    _finalize_stopped()
                    return
                if isinstance(exc, Exception):
                    if _streamlit_stop_requested() or "Stop" in type(exc).__name__:
                        _finalize_stopped()
                        return
                    _finish_progress("失败", ok=False)
                    st.error(str(exc))
                    _persist_assistant(f"执行失败：{exc}", steps=live_steps)
                    st.rerun()
                    return
                raise

            if result.error == "cancelled" or _cancel_latched():
                _finalize_stopped()
                return

            _finish_progress(f"完成（{result.steps} 步）", ok=True)

            answer_text = _clean_answer(result.answer)
            if not re.match(r"^#{1,3}\s*分析结论", answer_text):
                st.markdown("### 分析结论")
            _render_answer_with_figures(
                answer_text,
                result.figures,
                key_prefix=f"final_{anchor}",
            )
            if _user_wants_chart_ui(query) and not result.figures:
                st.warning("本次未生成图表。可点击侧边栏示例或重新提问，要求 Agent 必须出图。")

            for name, df in result.tables[-3:]:
                with st.expander(f"数据表：{name}", expanded=False):
                    st.dataframe(df, width="stretch")

            _persist_assistant(
                _clean_answer(result.answer),
                figures=result.figures,
                tables=result.tables,
                steps=result.process_steps or live_steps,
            )
            st.rerun()

if __name__ == "__main__":
    main()
