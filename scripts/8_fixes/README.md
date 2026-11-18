# 手动数据修复

三个入口均可在 CLI 的 Run Scripts → 8 Fixes 中发现。不会修改 worklist、启动模型训练或调用历史覆盖率审计流程。日志、执行差异和失败摘要通过普通 ScriptTool 邮件与 artifact exchange 投递；备份、原始数据及模型权重不投递。

## 从脚本6的 data 阶段失败恢复（新版脚本7）

新版7改为接续恢复入口，文件名为保持 CLI/历史命令兼容仍叫
`7_backfill_factor_sequence_pre2017.py`。**不再要求完成7后再跑6。**
分钟风险历史尚未按新版重算时，先单独完成5；已完成不重复。

本次使用新的恢复 run ID，原始脚本6目录保持不变：

```sh
.venv/bin/python scripts/8_fixes/7_backfill_factor_sequence_pre2017.py \
  --source-run rebuild_20261116T064711Z_a33e87ad \
  --run-id recovery_20261117_01 --dry-run

.venv/bin/python scripts/8_fixes/7_backfill_factor_sequence_pre2017.py \
  --source-run rebuild_20261116T064711Z_a33e87ad \
  --run-id recovery_20261117_01
```

CLI 的 Run Scripts → 8 Fixes 也提供同样参数。默认 source_run 是本次失败任务；
其他来源必须显式指定。仅接受数据合同兼容、除 data 外全部步骤通过的 rebuild。
`dry-run` 校验来源并写执行计划，不扫描完整缺口、不执行计算。

### 执行顺序

1. 从旧检查点读取固定截止日（本次 **20261113**），不追随日常更新扩大范围。
2. 逐个检查旧 dtop/FF3 日文件的完成记录及 SHA256；仍匹配的直接登记复用，
   不重算、不为未修改的复用文件另做一份备份。文件变更或缺失则重算该项。
3. 按完整交易日历检查33个原始因子，补算缺失/损坏的日文件；正常文件跳过。
   默认覆盖2011至固定截止日，包含2017年前五日步长遗留的空位，并检查后续年份。
   `--end` 可缩小补缺区间，但不改变下游固定截止日；不应通过缩短区间绕过源缺失。
   Affiliate 因子仍由其底层来源提供，缺少来源时报错，不伪造文件。
4. 重新生成 high_level_switch、相关 daily/weekly stats、依赖合成权重及合成结果，
   然后构建三个数据块。不能直接复用补缺前的 stats/合成权重。
5. 输出独立版本数据、报告、完成/失败邮件。只有最后完整校验通过才记 completed。

源文件存在但个股为 NaN 的分析师等真实缺失仍保留，不把它视为待补文件。
旧运行的124,292个 passed步骤不是全部直接迁移：只采用验证过的因子日文件，
下游按新历史刷新。原运行保持 failed，不篡改为成功。

### 与 daily_update 并行和断点续跑

新版7采用持久化 **逐文件首次读取快照**：只复制实际读到的原始数据库文件到
`results/factor_sequence_fixes/<新run ID>/raw_snapshot/`，保存 SHA256 和采集时间。
使用真实副本，不使用会被原地写入影响的硬链接。每次复制检查源是否在复制期间变化，
最多重试三次。后续读取及续跑使用私有副本，底层 pandas/polars 批量读取在该作用域
关闭线程/lazy加速，避免绕过快照。

因此，日常追加新日期、重写整表或修改无关原始文件不会触发最终 data 的全库指纹检查。
即使已采集的源之后被历史修订，本次仍明确使用其冻结版本；采用修订应另开运行。
新接触的源按首次读取时采集，**这不是所有文件在同一时刻的数据库事务快照**。
旧运行没有原始数据副本，无法重新证明其历史来源一致；此限制和继承来源均进入
`recovery_source.json` 和数据块 manifest，而不是删除一致性检查后宣称完全追溯。

失败后用同一条命令、相同新 run ID 续跑。不要删除 `checkpoint.sqlite` 或
`raw_snapshot/`。已完成输出和副本仍校验 SHA256；代码/参数变化、输出损坏或快照丢失
会明确拒绝续跑。单个阶段失败不抹掉之前完成记录。快照可能占用较多磁盘，磁盘不足
会报错；不要在恢复期间清理它。原始副本、检查点和备份不会被投递到共享报告包。

仍使用项目 `update_factors` 锁，防止因子更新同时改写因子/stats；原始数据日常采集
不需要全程停用。此流程不启动历史审计、不训练模型、不修改 worklist。

### 日志

脚本6/7都将 `stock_factor.* is empty` 的逐条提示改为汇总计数，保存到
`reports/empty_factor_reads.json`。不屏蔽来源错误、损坏、全NaN等其他警告或异常。
保留按日期批次、stats年段、合成进度和数据块阶段的日志。

脚本6仍是从头语义重建入口，原有全库来源检查保留；本次恢复请使用新版7，
不要再次用6创建新的全量任务。

## 修改规则和影响

### 缺失

- 分析师、预期和 PEAD 的缺失不填经济零值、不统一前向填充。最终模型数值占位零与 valid 分离。
- dtop 仅在当日估值源非空、字段和证券标识合法时补经济零值。取消旧的“整日有效股票少时回取一年最近分红率”的回退。无限值和整日源故障报错。
- dtop 文件保留 `dtop__source_observed`、`dtop__source_filled`。标准化及 stats 使用补零后的值，但来源标记仍保留。
- 这是明确的因子经济语义变更，因此 dtop、相关 stats、事件合成权重和数据块都必须刷新。

### 分组

相同值不拆分。在不同值之间选择最靠近十分位累计人数的切点，重复切点去重；不足 `ceil(N/50)` 的组向因子均值最近的相邻组合并，距离相同优先低组。实际最高减最低，不足两组则 spread 无效。方向先应用，因子／收益缺失均不参与分组。

factorperf、常规分组 stats 和事件合成共用此规则；固定十组／按证券顺序拆并列值的旧结果不再等价。常规组合回测的持仓分组不变。年度分组评价使用每日真实极端组；原本事后比较最高／最低两侧的评价方式保留。事件合成保留 `top` 模式，只有多空模式改正“第二组当底组”的错误。

分组版本写入 stats 参数与结果；新版数据的 `groups.jsonl` 记录每日因子的实际各组人数和失效原因。旧 stats 文件仅替换受影响日期，其他日期保留其原版本标记。

### FF3

只修改 FF3 拟合路径，不改变全项目 apply_ols。按证券使用 X/Y 共同有效观测回归，四参数至少五个观测且满秩；零分母无效。保留原来的非中心化 R² 分母。没有有效回归的残差动量不再因空求和变成零。所有在 behavior_ff3 模块注册的 ff_* 因子均刷新，而不只 ff_r2_1m。

固定历史日期及最后确认日输出 `ff3_<date>.csv`，包含有效样本数、失效原因及独立伪逆解的系数差。原文件与新版有效数／变更数写入 `differences.jsonl`；旧文件备份可追溯。服务器上的历史覆盖突变是否由旧文件版本导致，仍需运行这些真实数据比较后确认。

### 高位切换事件

先读取完整预热历史、排序及交易日对齐，再截取输出；冲突重复日期报错。缺行情或滚动窗口不成熟保留 NaN 和 `source_valid/source_reason`，不能当成未触发。市场因子覆盖写入改为确实保留新记录。

行业实际历史成员关系不可得：当前行业列表中的每个行业，从其首次行情后第 244 个交易日起纳入要求集合；纳入后缺失行情会使事件未知，整个行业源缺失也不会作为零贡献。此限制写入新数据 manifest。原阈值（三个行业、243 日新高及 ATR60）不变。`event_sources.csv` 逐日区分市场预热／缺失、行业预热／缺失与有效。

## 新数据合同及存储

- 新数据放在 `data/Interim/factor_sequence_repaired/<版本>/`，不覆盖旧 bundle，也不改已有训练配置。
- factorseq 全视图保留原十类通道，另存 dtop 两个来源标记；compact/slim 股票视图仍是 `value/valid/gap/never` 共 132 通道。
- factorperf 为 33 个原因子的 IC、spread，各自附 valid，共 132 通道。
- factormarket 为三事件标签／年龄、market_up、八项风险指标，各自附 valid，共 30 通道。
- 事件年龄遇到源中断重置为未知，直到再次触发。风险指标只用此前 120 日、至少 60 个有效观测标准化。
- 使用 `.npy` 磁盘映射、32 日批量及跨批状态，统计与标准化不依赖训练 minibatch。加载支持新版格式，旧版 NPZ 仍可读取。
- `manifest.json` 最后写入作为完成标记；保存来源、代码、规则、特征顺序和文件校验。占位零在统计结束后才编码。
- 不调用旧版全量 `audit` 工作流；执行期必要的输入验证、独立回归校验、输出校验和差异记录仍保留。

## 输出及故障处理

执行摘要在 `results/factor_sequence_fixes/<run_id>/reports/`；检查点为同目录 `checkpoint.sqlite`；私有备份在 `backups/`，包含原路径及 SHA256 映射。`reports/dataset.json` 给出最终数据路径。

源数据缺少、输出无效、计算异常会停止后续依赖阶段，保存失败记录并由 ScriptTool 通知。不得用缩短股票池、补零整个文件或跳过失败阶段来伪装成功。重算可能需要大量时间和磁盘空间；本地自动测试通过不代表服务器真实历史已重算完成。

## 本地验证

2026-11-17 恢复改造：新增9项测试覆盖 SHA256 复用、损坏拒绝、快照续跑、复制竞争、
pandas/polars 批量读取隔离、日志作用域，以及补缺→stats/合成→数据块的端到端模拟。
全仓库 Ruff 通过；本次生产代码与新增测试 Pyright 为0 errors/0 warnings；并行读取相关3项回归通过。
完整因子时序测试78项中77项通过，唯一未通过的是本地 worklist 已将所有 fit 项注释后，
旧 schedule 配置测试仍断言三个任务必须启用。本次未重新启用用户队列。
所有通知均模拟，没有执行服务器历史重算，也不将本地测试当作真实数据验收。

```sh
.venv/bin/ruff check .
.venv/bin/pyright --pythonpath .venv/bin/python src scripts/8_fixes
.venv/bin/python -m unittest discover -s tests -p 'test_factor_sequence*.py' -q
.venv/bin/python -m unittest discover -s tests -p test_market_daily_risk.py -q
```

## All-history missing-date backfill (script 8)

```sh
.venv/bin/python scripts/8_fixes/8_backfill_all_factor_history.py --dry-run
.venv/bin/python scripts/8_fixes/8_backfill_all_factor_history.py
```

This manual script scans all updatable registered factors from the configured history
start through the latest confirmed trading date. Optional `--start`, `--end`, and
`--batch-size` (default 32) narrow the range or statistics batch size. It preserves
each calculator's configured update frequency; it does not expand five-day factors
to daily history. Affiliate values remain owned by their upstream data sources.
Daily/weekly statistics cover missing eligible dates of all existing factor files.

It processes dependencies by level, completes each factor's values and statistics,
and logs factor names and every target date before calculation, followed by completion
or failure. Project `parallel()` calls, including nested loads and process requests,
execute serially only inside this script's context. Third-party native math library
threads are not changed. Ordinary updates retain their configured concurrency.
Daily automatic factor/statistics discovery remains restricted to the latest two years.

The script uses the existing `update_factors` lock, bypasses daily empty-job TTL caches,
and never requests overwrite. Successful output is retained; rerunning scans remaining
gaps. A missing result after calculation raises an error and stops dependent work.
Dry run performs no updates; statistics gaps created by future value writes will only
appear during the actual run. This repairs missing dates, not stale definitions,
corrupt files, or missing upstream source data; use scripts 6/7 for semantic rebuilds.

## Microcap 400 historical repair (script 9)

```sh
# Recompute the complete index history, validate staged outputs, then install.
.venv/bin/python scripts/8_fixes/9_repair_microcap_400.py

# Perform the same calculations and validation without changing live data.
.venv/bin/python scripts/8_fixes/9_repair_microcap_400.py --validate-only

# Optional bounded repair; records outside this interval are preserved.
.venv/bin/python scripts/8_fixes/9_repair_microcap_400.py --start 20261118 --end 20261118
```

The default return interval is 20050101 through the latest confirmed trading date,
not the date the bug was committed: historical backfills may also have written zero
returns. The script retains the existing portfolio timing (return at D uses the
portfolio at D-1, whose stock selection uses D-2 inputs). It validates holdings,
weight totals, complete finite security returns, and agreement with the repaired
production calculation. Genuine zero returns are permitted; missing returns are not.

Run in an idle data-maintenance window, after raw Tushare history has been restored.
The script shares the `daily_update` script lock, but this does not exclude every
manual downloader, preprocessing job, or already-running model process. Stop those
writers/readers before installation. It does not download missing history or repair
raw prices. Missing daily quotes, incomplete holding returns, or existing AdjPrice
inconsistent with raw quotes stop the run and identify the date; repair that upstream
data before rerunning. In particular, rebuild 20261118 AdjPrice if it still contains
forward-filled whole-market prices from the outage.

All candidates are prepared before installation:

- Full replacement of the selected microcap return dates, preserving other records.
- Full reconstruction of `market` fit and predict dumps using the candidate index,
  with the microcap log-return channel and serialized files verified.
- Invalidation of legacy ModuleData `data.pt` caches whose metadata explicitly
  includes `market`; other data types, including `factormarket`, are untouched.

`market` is not a HistNormTasks key and has no separate historical norm file to rebuild.
No model training, checkpoint edits, inference, predictions, trades, or backtest
recomputation is invoked. Old model results are retained; any process holding market
inputs in memory must be restarted before later use.

Outputs are under `results/microcap_repair/<run_id>/`. `reports/` contains recomputed
returns, old/new comparisons, and a summary; only this directory is registered for
artifact delivery. `staged/` and `backups/` remain local. Output hashes are checked
before installation. Ordinary installation failures roll back changed files.
`transaction.json` lists every target and its backup and records installing/complete/
rolled_back status. A hard kill or power loss during installation cannot execute
rollback: keep other jobs stopped, restore the listed targets from backups (remove
new targets marked `existed: false`), and rerun. Do not mistake an `installing` journal
for a completed repair. The whole history is recomputed on a rerun; there is no
per-date resume shortcut that could reuse stale source inputs.

Microcap repair defaults to 2005-01-01 through the latest confirmed trading day.
The first return uses holdings at 2004-12-31; the existing one-trading-day lag
for constituent selection therefore also requires 2004-12-30 valuation data.
The equal-weight constituent count is min(400, floor(listed SSE/SZSE stocks / 2)),
counting ST names in the market total but excluding them from selection. Selection
also requires a holding-date quote with status=1 and a finite positive close;
suspended stocks and stocks without confirmed trading quotes are excluded.
This does not use next-day status and cannot prevent a holding from halting later. If fewer
eligible stocks are available, use all of them; an empty selection fails.
Run script 9 over the full default range to replace the previous methodology
and rebuild market inputs; existing model weights are not retrained.

When a holding has a missing/nonfinite return, script 9 prints both endpoint
records for that security and saves `reports/missing_returns_<start>_<end>.csv`.
Evidence includes weights, returned returns, row presence/counts, daily close,
adjustment factor, status, volume, adjusted close, missing columns and read errors.
`status=0` is a suspension flag; an absent quote alone does not establish a halt.
Diagnostics never fill missing returns, remove holdings, or permit installation
after a failed return validation.

Close-return queries now use DATAVENDOR's lazily cached `list_day_price` table
when the interval starts on/before a stock's listing date and ends on/after it.
The reference is listing-day `preclose * adjfactor`, not listing-day close; the
actual factor is used (missing factors follow the existing fallback of 1).
`get_returns_block` applies this only to listing-day close returns. Open/VWAP
retain their previous calculations. Missing references remain NaN and are retried;
`DATAVENDOR.clear_all()` clears both listing and return caches.
