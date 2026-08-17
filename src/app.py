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
              if (msgs.length) {{
                msgs[msgs.length - 1].scrollIntoView({{
                  behavior: "smooth",
                  block: "nearest",
                  inline: "nearest"
                }});
              }}
              pickScrollables().forEach(function(el) {{
                try {{
                  el.scrollTop = el.scrollHeight;
                }} catch (e) {{}}
              }});
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

    st.title("自然语言数据分析 Agent")

    try:
        _ensure_database()
        ensure_db()
    except Exception as exc:  # noqa: BLE001
        st.error(f"数据库不可用：{exc}")
        st.stop()

    with st.sidebar:
        st.header("调用节奏")
        pace_label = st.radio(
            "LLM 限速模式",
            options=["自动(推荐)", "快速(单次问答)", "稳定(批量验收)"],
            index=0,
            help=(
                "快速：间隔 0s，响应更快；稳定：间隔 6s，不易 429；"
                "自动：默认快速，触发限速后本会话自动切到稳定。"
            ),
        )
        pace_mode = {
            "自动(推荐)": "auto",
            "快速(单次问答)": "fast",
            "稳定(批量验收)": "stable",
        }[pace_label]

        st.header("示例问题")
        for i, q in enumerate(EXAMPLE_QUERIES, 1):
            if st.button(f"示例 {i}", key=f"ex_{i}", width="stretch"):
                st.session_state["prefill"] = q
                _request_scroll_to_bottom()
        st.divider()
        if st.button("清空对话", width="stretch"):
            st.session_state.messages = []
            st.session_state.pop("_scroll_to_bottom", None)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Fixed-height chat pane; autoscroll=False so Streamlit won't keep pinning
    # to bottom after the user starts reading earlier messages.
    chat_box = st.container(height=720, autoscroll=False, border=False)
    with chat_box:
        for idx, msg in enumerate(st.session_state.messages):
            _render_message(msg, key_prefix=f"hist_{idx}")

    # After replaying history, also jump via main-document JS when requested.
    _maybe_scroll_to_bottom()

    prefill = st.session_state.pop("prefill", None)
    user_input = st.chat_input("用自然语言描述你的查询或分析需求…")
    query = prefill or user_input
    if not query:
        return

    _request_scroll_to_bottom()
    st.session_state.messages.append({"role": "user", "content": query})
    with chat_box:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            anchor = f"latest_{len(st.session_state.messages)}"
            _scroll_to_latest(anchor, mode="start")

            status = st.status("Agent 分析中…", expanded=True)
            live_steps: list[dict[str, Any]] = []
            event_count = 0

            def on_event(event: dict[str, Any]) -> None:
                nonlocal event_count
                live_steps.append(event)
                event_count += 1
                text = event.get("text") or ""
                tool = event.get("tool")
                if tool:
                    status.write(f"`{tool}` — {text}")
                elif text:
                    status.write(text)
                # Follow only while generating; stops immediately if user scrolls up.
                if event_count in {1, 3, 6} or (event_count > 6 and event_count % 4 == 0):
                    _scroll_to_latest(f"{anchor}_step{event_count}", mode="follow")

            try:
                agent = DataAnalysisAgent(max_steps=10, pace_mode=pace_mode)
                result = agent.run(query, on_event=on_event)
                status.update(
                    label=f"完成（{result.steps} 步 · {pace_label}）",
                    state="complete",
                )
            except Exception as exc:  # noqa: BLE001
                status.update(label="失败", state="error")
                st.error(str(exc))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"执行失败：{exc}",
                        "figures": [],
                        "tables": [],
                        "process_steps": live_steps,
                    }
                )
                _scroll_to_latest(f"{anchor}_end", mode="once")
                return

            st.markdown("---")
            answer_text = _clean_answer(result.answer)
            # Avoid duplicate「分析结论」heading when model already wrote it.
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

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": _clean_answer(result.answer),
                    "figures": _dedupe_figures(result.figures),
                    "tables": result.tables[-3:],
                    "process_steps": [
                        {k: v for k, v in ev.items() if k != "figure"}
                        for ev in (result.process_steps or live_steps)
                    ],
                }
            )
            _scroll_to_latest(f"{anchor}_end", mode="once")


if __name__ == "__main__":
    main()
