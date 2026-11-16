# 手动数据修复

三个入口均可在 CLI 的 Run Scripts → 8 Fixes 中发现。不会修改 worklist、启动模型训练或调用历史覆盖率审计流程。日志、执行差异和失败摘要通过普通 ScriptTool 邮件与 artifact exchange 投递；备份、原始数据及模型权重不投递。

## 运行顺序

在项目根目录执行。分钟风险历史尚未按新版重算时先运行 5；已经完成则不必重复。

```sh
.venv/bin/python scripts/8_fixes/5_rebuild_market_daily_risk.py
.venv/bin/python scripts/8_fixes/7_backfill_factor_sequence_pre2017.py --run-id factorseq_backfill_01
.venv/bin/python scripts/8_fixes/6_rebuild_factor_sequence_dependencies.py --run-id factorseq_repair_01
```

先查看计划，不修改因子数据：

```sh
.venv/bin/python scripts/8_fixes/7_backfill_factor_sequence_pre2017.py --dry-run
.venv/bin/python scripts/8_fixes/6_rebuild_factor_sequence_dependencies.py --dry-run
```

失败后使用原命令、相同 run ID 续跑。代码、原始数据文件的大小／修改时间或已完成输出的 SHA256 发生变化时拒绝复用旧检查点，应使用新 run ID。第二个脚本可以复用第一个脚本已校验的 dtop、FF3 和事件结果；stats、合成因子及数据块重新按依赖构建。每次邮件／投递使用独立 attempt ID，避免失败后续跑与旧包冲突。

`--start` 默认 20110101；补全的 `--end` 默认 20161230，修复的 `--end` 默认服务器最新确认日。`--batch-size` 默认 32，控制数据块构建批量。两个脚本使用 `update_factors` 锁，不与普通因子更新并发。风险脚本保留原来的安全替换校验。

**范围说明：**补全脚本仅对 2017 年前缺失／损坏的普通因子文件补算，但新版 dtop／FF3 的定义修复必须延续至冻结的最新确认日，才能构建一致的完整新数据版本。因此它还会完成该语义刷新，以及事件、stats、合成因子和数据块刷新。已有正常的其他因子文件不会覆盖。`end` 是本次补缺区间终点，不是滚动状态、事件权重和相关下游修订的截断点。窄范围运行仍要求更早输入已满足新版合同；尤其旧 dtop 文件缺少来源标记时会明确失败，推荐首次使用完整默认区间。

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

本轮执行并通过：因子时序相关 64 项 unittest、风险重算 6 项 unittest；并列值分组的独立检查；全仓库 Ruff；`src` 与三个修复脚本的 Pyright（指定项目 `.venv/bin/python`，0 errors / 0 warnings）。邮件、投递和重算替换测试使用临时目录／模拟发送，没有执行服务器重算。

```sh
.venv/bin/ruff check .
.venv/bin/pyright --pythonpath .venv/bin/python src scripts/8_fixes
.venv/bin/python -m unittest discover -s tests -p 'test_factor_sequence*.py' -q
.venv/bin/python -m unittest discover -s tests -p test_market_daily_risk.py -q
```
