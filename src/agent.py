"""ReAct-style data analysis agent with tool calling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import ChatLLM, PaceMode
from .schema_pack import build_schema_pack
from .tools import TOOL_DEFINITIONS, ToolRuntime, _extract_finish_answer, _recover_finish_from_raw


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
- 存在显著放量且毛利率很低时，可归因「低毛利 SKU 放量拖累整体毛利率」；
  仅当该店**所有**低毛利 SKU 均未达上述阈值时，才写「不足以归因于低毛利 SKU 放量」。
- 业务建议须与上述结论一致。
- **金额时间顺序（极易写反，必须遵守）**：写「增长/下降（A → B）」时，A 必须是 **Q1（前期）**，B 必须是 **Q2（后期）**。
  - 正确：增长 89.64%（Q1 41.49万元 → Q2 78.69万元）
  - 错误：增长 89.64%（78.69万元 → 41.49万元）← 箭头方向与「增长」矛盾，禁止

图表与筛选分离（必须遵守）：
- 用户要求「各门店…对比图 / 画出各门店…」且问题本身是全量对比时，图表应含全部相关门店。
- 若问题是「找出满足条件的门店再分析」（如增长>10%且毛利率下降），图可以只画筛出的门店，不必强行画全部门店。
- 「找出超过某阈值」用于文字结论与下钻；不要把筛选子集当成「全量各门店对比」除非用户明确要求。
- 典型正确流程（退损率题）：
  1) run_sql 用 Schema Pack 中的**双 CTE 模板**查全部门店退损率（含 refund_rate_pct）；
     **禁止** `FROM fact_refund JOIN fact_sales` 后同时 SUM 退款与销额再相除（分母会变成退款订单销额，出现 70%+ 假数据）。
  2) plot_chart 用该全量结果画对比图，可设 threshold_y=5 高亮超阈值门店；
  3) 再文字列出超过阈值的门店，并另查这些门店的退款原因（按 store_name + refund_reason 汇总）。
  本数据上半年正确量级约：科技园≈8%、星河路≈5.7%，其余门店更低；若出现 70%+ 必须重跑 SQL，不得写入结论。
- 同一张图不要重复调用 plot_chart；若返回 skipped_duplicate，直接继续分析即可。
- 两季度对比图推荐：宽表 `x=store_name, y=['q1_sales','q2_sales']`；或长表 `x=store_name, y=sales_amount, color=quarter`。

图表展示约定（必须遵守）：
- 禁止在回答里写「查看图表」伪链接或 Markdown 链接。
- plot_chart 的 title 必须是**完整图面说明**（主题 + 括号内关键结论），例如：
  - `2025年Q1与Q2门店销额对比（星河路店和科技园店销额显著增长）`
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
    ) -> AgentResult:
        schema_pack = build_schema_pack()
        system = SYSTEM_TEMPLATE.format(schema_pack=schema_pack)
        runtime = ToolRuntime()
        process_steps: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ]

        def emit(event: dict[str, Any]) -> None:
            process_steps.append(event)
            if on_event:
                on_event(event)
            text = event.get("text")
            if text and on_progress:
                on_progress(text)

        for step in range(1, self.max_steps + 1):
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
            try:
                resp = self.llm.chat(messages, tools=TOOL_DEFINITIONS, temperature=0.0)
            except Exception as exc:  # noqa: BLE001
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

                if name == "finish" and runtime.artifacts.final_answer:
                    finished = True

            if finished and runtime.artifacts.final_answer:
                emit({"type": "done", "step": step, "text": "分析完成，正在整理结论与图表…"})
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
