# refund_record.xlsx 数据说明

`refund_id`：退款记录编号；`order_id`：原订单编号；`refund_date`：退款日期；`refund_quantity`：退款数量；`refund_amount`：退款金额（元）；`refund_reason`：退款原因。

业务人员有时将退款相关损失简称为“退损”。

统一约定：

- `退损金额 = refund_amount`
- `退损率 = 退款金额 / 成交销额`
- `净销额 = 成交销额 - 退款金额`

统计某时间段“退损”时，以 `refund_date` 判断退款发生时间；对应时段的成交销额以 `fact_sales.order_date` 判断。

**重要**：退损率的分母是该门店该时段的**全部成交销额**，不是「发生过退款的订单」的销额。
计算时应分别汇总 `fact_refund` 与 `fact_sales` 再相除，禁止从 `fact_refund JOIN fact_sales` 的明细上同时 `SUM` 两边再做比率。

