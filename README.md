# 自然语言数据分析 Agent

支持用自然语言完成零售数据查询、分析与绘图。数据说明 Markdown 与业务术语注入模型上下文，Agent 自动生成并执行只读 SQL。

## 技术栈

- **SQLite**：`data/retail.db`（由 Excel 导入）
- **Streamlit**：Web 对话界面
- **七牛 DeepSeek-V3**：OpenAI 兼容 API（`.env` 配置；支持快速/稳定双限速）
- **Plotly**：柱状图 / 折线图 / 饼图

## 整体设计思路

1. **Schema Pack**：启动时组装表 DDL + `store_info.md` / `product_info.md` / `sales_order.md` / `refund_record.md` / `business_terms.md`，注入 System Prompt。术语（战区、即时零售、SKU、销额、退损率等）优先按 `business_terms.md` 映射到字段与条件，而不是字段名相似度猜测。
2. **ReAct + Tools**：模型通过 `get_schema` / `run_sql` / `plot_chart` / `analyze_dataframe` / `finish` 多步完成复杂问题（如退损原因下钻、毛利率与 SKU 结构分析）。
3. **只读 SQL 层**：拦截写操作；空结果与执行错误回传模型改写；指标约定使用 `NULLIF` 防除零。
4. **Web UI**：Streamlit 展示结论、图表与中间结果表；侧边栏可一键填入题目示例。

## 数据表

| Excel | 表名 |
|---|---|
| store_info.xlsx | dim_store |
| product_info.xlsx | dim_product |
| sales_order.xlsx | fact_sales |
| refund_record.xlsx | fact_refund |

关联：`store_id` / `product_id` / `order_id`。

核心指标：成交销额 = `quantity * sale_price - discount_amount`；退损率 = 退款金额 / 成交销额；毛利率依赖 `unit_cost`。

## 快速开始

```bash
pip install -r requirements.txt
# 配置 .env（QINIU_*；限速见下方双模式参数）
python scripts/import_excel.py
streamlit run src/app.py
```

### LLM 限速双模式

| 参数 | 含义 | 默认 |
|---|---|---|
| `LLM_FAST_MIN_INTERVAL_S` | 快速模式间隔（秒） | `0` |
| `LLM_STABLE_MIN_INTERVAL_S` | 稳定模式间隔（秒） | `6` |
| `LLM_PACE_MODE` | `auto` / `fast` / `stable` | `auto` |

- **快速**：适合单次问答  
- **稳定**：适合批量验收 / 频繁提问  
- **自动**：默认按快速跑；一旦遇到 429，本会话自动改用稳定间隔  

Web 侧边栏也可切换这三种模式（会覆盖 `.env` 的 `LLM_PACE_MODE`）。

## 目录

```
.env
requirements.txt
scripts/
  import_excel.py      # Excel → SQLite
  e2e_examples.py      # 题目示例 1–5 批量跑
  smoke_test.py        # SQL 冒烟测试
src/
  config.py
  db.py
  schema_pack.py
  llm.py
  tools.py
  agent.py
  app.py               # Streamlit Web（侧边栏也是这 5 道题）
data/retail.db         # 导入后生成
charts/                # 画图 HTML 缓存（可删，会再生）
```
