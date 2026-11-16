# MarketDailyRisk 历史修复

计算版本：`market-risk-grouped-shift-v2`。

生产 pandas 实现按 `secid, date（如果存在）, minute` 排序，在证券／日期内 shift。
不把另一只股票的收盘价作为前值，不自动填充缺失分钟。重复分钟或混合日期输入报错。
原有市值权重分母、分钟 >=1 的收益统计、开收盘 <=5 或 >=42 边界及单位保持不变。
这里没有同时改为“仅有效收益参与权重分母”。

Polars 独立计算分钟收益、市场加权收益、样本标准差（ddof=1）、无偏偏度、
开收盘成交占比、平均成交规模（包括无 num_trades 时的资金流分档估算）。
重算逐日比较市场分钟收益和四项最终指标，rtol=1e-9、atol=1e-12。
两边共同产生非有限最终指标仍判定失败。

在服务器项目根目录手动运行全历史重算：

```bash
.venv/bin/python scripts/8_fixes/5_rebuild_market_daily_risk.py
```

默认从最早支持的分钟日期（或更早的已有风险记录）开始，到最新已确认且有分钟源数据的日期。
历史记录早于现存分钟源时会报缺源，不能悄悄缩短并宣称全历史修复。
默认结束日期之后的已有记录会保留并单列 `preserved_later_dates`，这部分不算已验证。
可用 `--start` / `--end` 明确指定范围。

只验证并生成候选，不替换数据库：

```bash
.venv/bin/python scripts/8_fixes/5_rebuild_market_daily_risk.py --validate-only
```

ScriptTool 任务名为 `rebuild_market_daily_risk`，也可从 Run Scripts 选择。

每次独立写入 `results/market_risk_rebuild/<run_id>/`：

- `backup/`：旧风险文件原样备份和摘要中的 SHA256。
- `candidate/`：已验证的完整候选风险表。
- `reports/`：版本、日期、进度／失败原因、pandas/Polars 对照和新旧历史差异。

所有请求日期通过检查之后才生成数据库同目录的临时文件，使用项目 Save/Load 回读验证，
再原子替换。若计算期间原风险文件被日常更新修改，拒绝覆盖，需在闲时重新执行。
缺源、指标不一致或非有限结果均保留失败报告，原数据库不变。
报告复用 artifact-exchange，备份和候选数据不对外发布。脚本不重训模型、不更新旧模型缓存。
历史重算由操作者执行；代码修复本身不会自动改写既有风险库。

代码组织：pandas／Polars 实现在 `src/data/update/custom/market_daily_risk.py`，
可通过 `MarketDailyRiskUpdater.calculate(date, backend="pandas"|"polars")` 调用。
重算、逐日校验、备份和替换逻辑全部在 `5_rebuild_market_daily_risk.py`。
