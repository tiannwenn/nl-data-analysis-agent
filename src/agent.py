"""ReAct-style data analysis agent with tool calling."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import ChatLLM, PaceMode
from .schema_pack import build_schema_pack
from .tools import (
    TOOL_DEFINITIONS,
    ToolRuntime,
    _extract_finish_answer,
    _recover_finish_from_raw,
    _recover_sql_from_args,
)


SYSTEM_TEMPLATE = """你是零售业务数据分析 Agent。用户用中文提出查询/分析/画图需求。
你必须结合下方 Schema Pack（含 Markdown 字段说明与业务术语）理解问题，再调用工具完成任务。

工作方式：
1. 先理解查询对象、时间范围、指标、筛选条件；术语优先查 business_terms 映射。
2. 用 run_sql 生成并执行只读 SQL（可多步：全量查询 → 画图 → 筛选下钻 → 原因分析）。
3. 需要图表时调用 plot_chart（bar/line/pie）；x/y 必须使用 run_sql 返回的**精确列名**。
4. 若 plot_chart 或 run_sql 返回 ok=false，根据 error 修正后重试，不要声称已成功画图。
5. 对结果做简要业务分析，最后必须调用 finish，且 **answer_markdown 必须是非空中文 Markdown 结论**（不要空参数、不要只写在普通回复里）。
6. 查询为空、字段不确定、SQL 报错时：解释原因并尝试修正；仍不行则如实告知，不要编造数据。
7. **只要用户要求「图表/画图/对比图/生成合适的图表」**，finish 前必须至少成功画 1 张图；优先简单分组柱状图，不要把方案做复杂。
8. 「销额增长 + 毛利率下降 + 低毛利 SKU 放量」推荐最短路径（约 3～4 轮即可完成，减少试错）：
   1) 一个 SQL 找出增长>阈值且毛利率下降的门店（宽表含 q1/q2 销额与毛利率）；
   2) 先画**门店级** Q1/Q2 销额对比（图1），title 写清结论；
   3) **一个** SQL 对上述门店做低毛利 SKU **全量下钻**（禁止只挑几个商品）：
      - CTE 内按 store_name+product_name 算出 Q1/Q2 销额与毛利率（只 GROUP BY，禁止 WHERE SUM）；
      - 用 Q1/Q2 键的并集（FULL：UNION 两端键再 LEFT JOIN），避免只在一季有销售的 SKU 被 INNER JOIN 丢掉；
      - 外层过滤：`WHERE COALESCE(q1_margin,q2_margin) < 阈值`（如 0.3），保留阈值内**全部**门店-SKU；
      - 结果列必须含 store_name, product_name, q1_sales, q2_sales（缺季用 0）；
      - **禁止** LIMIT、按品类随便抽、只写星河路激光打印机等「点名抽样」。
   4) 确认结果中**每个目标门店都有多行 SKU**（若某店只有 1～2 个而另一店很多，先检查 SQL 再画图）；
      再画低毛利 SKU 对比（图2），title 按本次数据写清谁放量；然后 finish。
   说明「放量」用同一 SKU 的 Q1→Q2 销额对比。失败后立刻按 hint 改 SQL，禁止空 sql。

「是否存在低毛利 SKU 放量导致整体毛利率下降」（必须遵守）：
- **逐店、按本次查询数字判定**，禁止套用「某店通常不放量」的先验；星河路店与科技园店用同一套规则。
- **下钻必须全面**：对筛出的每家门店，分析/作图须覆盖毛利率低于阈值的**全部** SKU（或这些店在 Q1∪Q2 的全部 SKU 再标低毛利），
  不得只列举叙事好看的两三个商品；结论须基于该全量结果。
- 「显著放量」判定（满足任一即成立）：
  1) 同一低毛利 SKU：(Q2-Q1)/Q1 >= 1（即文中「增长 ≥ 1 倍」/ 至少翻倍，Q2/Q1 ≥ 2）；或
  2) 该 SKU 的 Q2−Q1 增量占该门店销额增量的主要部分。
- 小幅增长（未翻倍）且非增量主体 → **不算**显著放量，可写「未见显著放量」。
- **禁止自相矛盾**：若已写出「增长 3 倍 / 4.9 倍 / 18 倍」等（≥1 倍），同一门店结论必须是「存在显著放量」，
  禁止再写「未翻倍」「未见显著放量」。数字与定性必须一致。
- **「最高/最低/其次」必须按数值比大小，禁止按表格行序叙事**（SQL 常按店名排序，第一行不一定最大）：
  写「A 最高，达到 X，其次是 B（Y）」时，必须 X ≥ Y；若 Y 更大，最高的是 B。
  错误：天府店 POS 最高（43.40万），其次星河路店（59.93万）← 59.93>43.40，星河路才是最高。
  正确：星河路店 POS 最高（59.93万），其次科技园店（52.44万）。
  写完后必须回头核对：定性排序 = 数字从大到小（或从小到大）。
- 存在显著放量且毛利率很低时，可归因「低毛利 SKU 放量拖累整体毛利率」；
  仅当该店**所有**低毛利 SKU 均未达上述阈值时，才写「不足以归因于低毛利 SKU 放量」。
- 业务建议须与上述结论一致。
- **原因分析建议（必须遵守）**：若某门店 Top 原因占比接近（差距 ≤3 个百分点，或同为「并列第一」），
  建议须**同时覆盖这些并列主因**，禁止只挑其中一个写行动项。
  例：星河路店「不符合预期」≈40%、「质量问题」≈40% → 应同时写优化商品描述/预期管理，以及质量管控；
  不得只写其中一条。
- **金额时间顺序（极易写反，必须遵守）**：写「增长/下降（A → B）」时，A 必须是 **Q1（前期）**，B 必须是 **Q2（后期）**。
  - 正确：增长 89.64%（Q1 41.49万元 → Q2 78.69万元）
  - 错误：增长 89.64%（78.69万元 → 41.49万元）← 箭头方向与「增长」矛盾，禁止

图表与筛选分离（必须遵守）：
- 用户要求「各门店 / 每家门店…对比图」或「比较每家门店…再找出阈值」（示例3/4）时，图表必须含**全部**相关门店；阈值只用于文字名单与下钻，**禁止**把筛选子集当成「各门店对比图」的唯一数据源。
- 用户明确说「找出…的门店，比较**这些门店**并生成对比图」（如下降门店、增长且毛利率下降再下钻）时：图**只画筛出的门店**即可；title **禁止**写「各门店/每家门店/全部门店」，应写「下降门店」「目标门店」或点名店名，且与结果集行数一致。
- 例外补充：问题主线是「增长>10%**且**毛利率下降 + 低毛利 SKU 放量」（示例5），门店级图也可只画筛出店；title 同样不要写「各门店」。
- 典型正确流程（退损率题，示例3）：
  1) run_sql 用 Schema Pack 中的**双 CTE 模板**查全部门店退损率（含 refund_rate_pct）；
     **禁止** `FROM fact_refund JOIN fact_sales` 后同时 SUM 退款与销额再相除（分母会变成退款订单销额，出现 70%+ 假数据）。
  2) plot_chart 用该全量结果画对比图，可设 threshold_y=5 高亮超阈值门店；
  3) 再文字列出超过阈值的门店，并另查这些门店的退款原因（按 store_name + refund_reason 汇总）。
  本数据上半年正确量级约：科技园≈8%、星河路≈5.7%，其余门店更低；若出现 70%+ 必须重跑 SQL，不得写入结论。
- 典型正确流程（Q1/Q2 销额对比 + 找出增长>10%，示例4）：
  1) 一个 SQL 查**全部门店** Q1/Q2 销额与增长率（不要 WHERE growth>0.1 后再画图）；
  2) plot_chart 用该全量结果画两季度对比图（宽表 y=['q1_sales','q2_sales'] 或长表 color=quarter）；
     title 可写「2025年各门店Q1与Q2销售额对比（星河路店和科技园店增长超10%）」等，图上必须仍有全部门店；
  3) 文字再点名增长超过 10% 的门店及具体数字。
  **禁止**只画增长>10% 的两家店（那不是「每家门店」对比图）。
- 典型正确流程（找出 Q2 相对 Q1 **下降**的门店，比较这些门店并画图）：
  1) SQL 筛出 growth<0 的门店（本数据通常仅天府店）；
  2) plot_chart **只用该筛选结果**；title 例：`2025年Q2相对Q1销售额下降门店对比（天府店下降16.56%）`——**不要**写「各门店」；
  3) 正文说明下降门店及数字；无需为了凑「全部门店」再画一张全量图。
- title 必须与结果集范围一致：图上几家店，标题就按几家/按「下降门店」表述；禁止「标题写各门店、数据只有 1 家」。
- 若 plot_chart 因标题误写「各门店」被警告：题目属示例3/4（要全量）→ 用全量结果重画；题目属「只比筛出店」→ **改 title 后**用同一筛选结果再画即可，正文只引用最终正确的图，不要无意义再画一张全量图。
- 同一张图不要重复调用 plot_chart；若返回 skipped_duplicate，直接继续分析即可。
- 两季度对比图推荐：宽表 `x=store_name, y=['q1_sales','q2_sales']`；或长表 `x=store_name, y=sales_amount, color=quarter`。

图表展示约定（必须遵守）：
- 禁止在回答里写「查看图表」伪链接或 Markdown 链接。
- plot_chart 的 title 必须是**完整图面说明**（主题 + 括号内关键结论），例如：
  - `2025年各门店Q1与Q2销售额对比（星河路店和科技园店增长超10%）`（示例4：图含全部门店）
  - `2025年Q2相对Q1销售额下降门店对比（天府店下降16.56%）`（只比筛出店：勿写「各门店」）
  - `2025年Q1与Q2门店销额对比（星河路店和科技园店销额显著增长）`（示例5：可只画筛出的门店）
  - `2025年Q1与Q2低毛利SKU销额对比（按本次数据写清哪家店、哪个SKU放量；有放量写放量，无则写未见）`
  工具会按生成顺序自动加「图1：」「图2：」并写在图内；**不要**自己在 title 里写「图1」「图2」。
- 多张图以 plot_chart 成功顺序为准（第 1 次成功画图=图1，以此类推）；文字里只需「见图1」「见图2」引用。
- **禁止**再写一段「图表展示：图1……图2……」清单（易与图面编号错位）；图意以图内标题为准。
- 禁止连续写多句「下图为……」却不标注图号。
- 界面按图号把图表插到对应文字后面。

SQL 易错点（必须避免）：
- JOIN 后只能使用已定义的表别名，例如 `fact_sales fs` / `dim_store st` / `fact_refund fr`，不要写不存在的 `s.store_id`。
- `run_sql` 的 `result_name` 只是会话内存结果名，**不是**数据库表；后续不能写 `FROM q1_sales`，应把完整子查询写进同一个 WITH/SELECT。
- 比较 Q1/Q2 时，用一个 SQL（WITH q1 AS (...), q2 AS (...)）一次算完，不要依赖上一次查询的虚表名。
- **绝不要调用 sql 为空的 run_sql**；若上一轮失败，直接提交改好的完整 SQL。
- 画完图后若还要下钻：下一轮 run_sql 必须立刻带上完整 SELECT/WITH，不要先发空参数占位。
- **禁止 WHERE SUM(...)/AVG(...)**（SQLite: misuse of aggregate）。过滤聚合结果用 HAVING，或先聚合再在外层 WHERE 列名。
- 成交销额必须写 `fs.quantity * fs.sale_price - fs.discount_amount`（乘号 `*` 不能丢）。
- 退损率：销额 CTE（按 order_date）与退款 CTE（按 refund_date）分开汇总后再除；工具会拒收「退款表驱动且同时汇总销额」的错误写法。
- 两季度对比图推荐：宽表 `x=store_name, y=['q1_sales','q2_sales']`；或长表 `x=store_name, y=sales_amount, color=quarter`。
- SKU 放量对比：SQL 必须含 store_name 与 product_name；画图时即使传入 x=store_name，
  工具也会自动改成「门店 - SKU」横轴。禁止只用 product_name 把两家店混在一根柱里。
  plot_chart 请显式传 x 与 y（如 y=['q1_sales_amount','q2_sales_amount']），勿留空。
- 文字里引用图号时按生成顺序写（先画的是图1）；不要颠倒写成先见图2后见图1。

约束：
- 禁止写操作 SQL。
- 不要编造不存在的表/字段/数值。
- 单次任务尽量在有限工具调用内完成。
- 回答使用简洁中文，关键数字可保留表格摘要。
- 凡写「从 X → Y」的销额/毛利率对比，必须 X=Q1、Y=Q2（或「上期→本期」），并与前面的「增长/下降」语义一致。

{schema_pack}
"""


@dataclass
class AgentResult:
    answer: str
    figures: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    process_steps: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    error: str | None = None


EventCallback = Callable[[dict[str, Any]], None]
ProgressCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]


def _user_wants_chart(query: str) -> bool:
    q = query or ""
    keys = ("图", "画", "chart", "plot", "可视化")
    return any(k in q.lower() if k.isascii() else k in q for k in keys)


class DataAnalysisAgent:
    def __init__(
        self,
        max_steps: int = 8,
        llm: ChatLLM | None = None,
        pace_mode: PaceMode | None = None,
    ):
        self.max_steps = max_steps
        self.llm = llm or ChatLLM(pace_mode=pace_mode)

    def run(
        self,
        user_query: str,
        on_progress: ProgressCallback | None = None,
        on_event: EventCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AgentResult:
        schema_pack = build_schema_pack()
        system = SYSTEM_TEMPLATE.format(schema_pack=schema_pack)
        runtime = ToolRuntime()
        process_steps: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ]

        cancel_latched = False

        def emit(event: dict[str, Any]) -> None:
            nonlocal cancel_latched
            process_steps.append(event)
            if on_event:
                try:
                    on_event(event)
                except BaseException as exc:  # noqa: BLE001 — Streamlit Stop is BaseException
                    # Allow cooperative cancel to finish; never swallow rerun.
                    name = type(exc).__name__
                    if name == "RerunException":
                        raise
                    if name in {"StopException", "ScriptControlException"}:
                        cancel_latched = True
                        return
                    raise
            text = event.get("text")
            if text and on_progress:
                on_progress(text)

        def cancelled() -> bool:
            nonlocal cancel_latched
            if cancel_latched:
                return True
            if not should_cancel:
                return False
            try:
                if should_cancel():
                    cancel_latched = True
                    return True
            except BaseException as exc:  # noqa: BLE001 — session_state yield under STOP
                name = type(exc).__name__
                if name == "RerunException":
                    raise
                if name in {"StopException", "ScriptControlException"}:
                    cancel_latched = True
                    return True
                raise
            return False

        def cancel_result(step: int) -> AgentResult:
            emit({"type": "warn", "step": step, "text": "用户已停止分析"})
            return AgentResult(
                answer="分析已停止。可修改问题后重新提问。",
                figures=runtime.artifacts.figures,
                tables=runtime.artifacts.tables,
                logs=runtime.artifacts.logs,
                process_steps=process_steps,
                steps=max(0, step),
                error="cancelled",
            )

        for step in range(1, self.max_steps + 1):
            if cancelled():
                return cancel_result(step - 1)
            if step == 1:
                llm_text = (
                    f"第 {step}/{self.max_steps} 轮：模型理解问题，规划并开始执行"
                    "（写 SQL / 画图 / 下钻 / 输出结论）…"
                )
            else:
                llm_text = (
                    f"第 {step}/{self.max_steps} 轮：模型根据上一轮工具结果决定下一步"
                    "（改 SQL / 画图 / 下钻 / 输出结论）…"
                )
            emit({"type": "llm", "step": step, "text": llm_text})
            pool = ThreadPoolExecutor(max_workers=1)
            fut = pool.submit(self.llm.chat, messages, TOOL_DEFINITIONS, 0.0)
            try:
                resp = None
                while True:
                    if cancelled():
                        # Do not wait for the in-flight HTTP call; UI can show 已停止.
                        pool.shutdown(wait=False, cancel_futures=True)
                        return cancel_result(step)
                    try:
                        resp = fut.result(timeout=0.35)
                        break
                    except FuturesTimeout:
                        continue
            except Exception as exc:  # noqa: BLE001
                pool.shutdown(wait=False, cancel_futures=True)
                emit({"type": "error", "step": step, "text": f"模型调用失败：{exc}"})
                return AgentResult(
                    answer=f"模型调用失败：{exc}",
                    figures=runtime.artifacts.figures,
                    tables=runtime.artifacts.tables,
                    logs=runtime.artifacts.logs,
                    process_steps=process_steps,
                    steps=step,
                    error=str(exc),
                )
            else:
                pool.shutdown(wait=False)

            if cancelled():
                return cancel_result(step)

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                text = (msg.content or "").strip()
                if not text:
                    text = "未能生成有效回答，请换一种问法重试。"
                emit({"type": "answer", "step": step, "text": "模型直接给出结论（未调用 finish）"})
                return AgentResult(
                    answer=text,
                    figures=runtime.artifacts.figures,
                    tables=runtime.artifacts.tables,
                    logs=runtime.artifacts.logs,
                    process_steps=process_steps,
                    steps=step,
                )

            finished = False
            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        args = {"answer_markdown": str(args)} if name == "finish" else {}
                except json.JSONDecodeError:
                    args = {}
                    if name == "finish":
                        recovered = _recover_finish_from_raw(raw_args)
                        if recovered:
                            args = {"answer_markdown": recovered}
                        else:
                            tool_result = json.dumps(
                                {
                                    "ok": False,
                                    "error": (
                                        "finish 参数 JSON 无效。请用合法 JSON 重试："
                                        '{"answer_markdown":"### 分析结论\\n..."}'
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                }
                            )
                            emit(
                                {
                                    "type": "tool_error",
                                    "step": step,
                                    "tool": name,
                                    "text": f"工具 {name} 参数 JSON 无效",
                                }
                            )
                            continue
                    else:
                        tool_result = json.dumps(
                            {"ok": False, "error": f"Invalid JSON arguments: {raw_args[:200]}"},
                            ensure_ascii=False,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            }
                        )
                        emit(
                            {
                                "type": "tool_error",
                                "step": step,
                                "tool": name,
                                "text": f"工具 {name} 参数 JSON 无效",
                            }
                        )
                        continue

                # finish: tolerate aliases / empty tool args + assistant content fallback
                if name == "finish":
                    answer = _extract_finish_answer(args)
                    if not answer:
                        answer = _recover_finish_from_raw(raw_args)
                    if not answer and (msg.content or "").strip():
                        answer = str(msg.content).strip()
                    if answer:
                        args = {"answer_markdown": answer}

                # run_sql: recover sql from aliases / broken JSON before dispatch
                if name == "run_sql":
                    sql = _recover_sql_from_args(args if isinstance(args, dict) else {}, raw_args)
                    if sql:
                        args = {**(args if isinstance(args, dict) else {}), "sql": sql}
                    else:
                        tool_result = json.dumps(
                            {
                                "ok": False,
                                "error": "SQL is empty",
                                "hint": (
                                    "本次 run_sql 未带 sql。请立刻重试，arguments 必须形如："
                                    '{"sql":"SELECT ...","result_name":"..."}；'
                                    "禁止空对象 {} 或 sql:\"\"。"
                                ),
                            },
                            ensure_ascii=False,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            }
                        )
                        emit(
                            {
                                "type": "tool_end",
                                "step": step,
                                "tool": name,
                                "text": "run_sql 失败：SQL is empty（已跳过空调用）",
                            }
                        )
                        continue

                detail = _tool_preview(name, args)
                emit({"type": "tool_start", "step": step, "tool": name, "text": detail})
                figs_before = len(runtime.artifacts.figures)

                # Block finish when charts are required but none produced yet.
                if (
                    name == "finish"
                    and _user_wants_chart(user_query)
                    and not runtime.artifacts.figures
                ):
                    tool_result = json.dumps(
                        {
                            "ok": False,
                            "error": (
                                "用户要求生成图表，但尚未成功 plot_chart。"
                                "请用已有查询结果画一张简单对比图即可，例如："
                                "低毛利 SKU 的 Q1/Q2 销额对比（x=product_name, y=['q1_sales','q2_sales']），"
                                "或门店 Q1/Q2 销额对比。画完后再 finish。"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )
                    emit(
                        {
                            "type": "tool_end",
                            "step": step,
                            "tool": name,
                            "text": "finish 被拒绝：尚未生成图表，需先 plot_chart",
                        }
                    )
                    runtime.artifacts.final_answer = None
                    continue

                tool_result = runtime.dispatch(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
                summary = _tool_result_summary(name, tool_result)
                event: dict[str, Any] = {
                    "type": "tool_end",
                    "step": step,
                    "tool": name,
                    "text": summary,
                }
                if name == "plot_chart" and len(runtime.artifacts.figures) > figs_before:
                    title, fig = runtime.artifacts.figures[-1]
                    event["figure"] = {"title": title, "fig": fig}
                emit(event)

                if cancelled():
                    return cancel_result(step)

                if name == "finish" and runtime.artifacts.final_answer:
                    finished = True

            if finished and runtime.artifacts.final_answer:
                emit({"type": "done", "step": step, "text": "分析完成，正在整理结论与图表…"})
                if cancelled():
                    return cancel_result(step)
                return AgentResult(
                    answer=runtime.artifacts.final_answer,
                    figures=runtime.artifacts.figures,
                    tables=runtime.artifacts.tables,
                    logs=runtime.artifacts.logs,
                    process_steps=process_steps,
                    steps=step,
                )

        emit({"type": "warn", "text": "达到步数上限，尝试根据已有结果自动补图并汇总…"})
        if (
            _user_wants_chart(user_query)
            and not runtime.artifacts.figures
            and runtime.artifacts.dataframes
        ):
            auto_title = "低毛利SKU或门店指标对比（自动补图）"
            auto_res = runtime.dispatch(
                "plot_chart",
                {"title": auto_title, "chart_type": "bar"},
            )
            emit(
                {
                    "type": "tool_end",
                    "tool": "plot_chart",
                    "text": _tool_result_summary("plot_chart", auto_res),
                }
            )

        if runtime.artifacts.final_answer:
            answer = runtime.artifacts.final_answer
        elif runtime.artifacts.tables:
            last_name, last_df = runtime.artifacts.tables[-1]
            preview = last_df.head(12).to_string(index=False)
            chart_note = (
                f"\n\n已根据结果集自动生成图表（见图1）。"
                if runtime.artifacts.figures
                else "\n\n未能自动生成图表。"
            )
            answer = (
                "### 分析结论\n\n"
                f"已达到工具轮次上限，基于最后查询结果（`{last_name}`）汇总如下：\n\n"
                f"```\n{preview}\n```\n"
                f"{chart_note}\n"
                "若需更完整结论，可缩小问题范围后重试。"
            )
        else:
            answer = "未能在限定步数内完成分析，请简化问题后重试。"

        return AgentResult(
            answer=answer,
            figures=runtime.artifacts.figures,
            tables=runtime.artifacts.tables,
            logs=runtime.artifacts.logs,
            process_steps=process_steps,
            steps=self.max_steps,
            error="max_steps_reached",
        )


def _tool_preview(name: str, args: dict[str, Any]) -> str:
    if name == "run_sql":
        sql = (args.get("sql") or "").strip().replace("\n", " ")
        if len(sql) > 160:
            sql = sql[:160] + "…"
        return f"执行 SQL：{sql}"
    if name == "plot_chart":
        return (
            f"绘制{args.get('chart_type', 'chart')}图："
            f"{args.get('title') or ''}（x={args.get('x')}, y={args.get('y')}）"
        )
    if name == "get_schema":
        return "读取数据库表结构"
    if name == "analyze_dataframe":
        return f"分析结果集：{args.get('result_name') or '最新查询'}"
    if name == "finish":
        return "汇总最终结论"
    return f"执行工具：{name}"


def _tool_result_summary(name: str, tool_result: str) -> str:
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return f"{name} 完成"
    if not data.get("ok", True):
        return f"{name} 失败：{data.get('error') or tool_result[:120]}"
    if name == "run_sql":
        empty = data.get("empty")
        n = data.get("row_count", 0)
        return f"SQL 完成：{n} 行" + ("（空结果）" if empty else "")
    if name == "plot_chart":
        if data.get("skipped_duplicate"):
            return f"图表重复，已跳过：{data.get('title') or ''}"
        warn = data.get("warning")
        if warn:
            return f"图表已生成，但有警告：{warn}"
        return f"图表已生成：{data.get('title') or ''}"
    if name == "finish":
        return "结论已生成"
    return f"{name} 完成"
