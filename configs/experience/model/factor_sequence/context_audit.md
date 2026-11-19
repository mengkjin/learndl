# Contextual 输入与结果审计（不训练）

入口：`scripts/0_check/factor_sequence_coverage.py`，普通 ScriptTool，任务名
`factor_sequence_coverage`。不修改 worklist、训练配置、样本筛选或填充规则。

在服务器项目根目录运行：

```bash
.venv/bin/python scripts/0_check/factor_sequence_coverage.py
```

默认 20110101 至 `CALENDAR.updated()`，额外加载预热。CLI 的 Run Scripts 中也可选择
该脚本。`--start YYYYMMDD --end YYYYMMDD` 可限定报告期；`--no-email` 同时禁止邮件和
对外投递，适合本机检查。脚本不持有训练锁，但全期分钟数据审计有显著磁盘 I/O，建议在闲时执行。

结果保存在 `results/factor_sequence/context_audit_<UTC>_<UUID>/`，每次独立目录。
邮件沿用 TaskResultMail，包含摘要和完整日度 CSV ZIP；发布先于邮件，服务器保留原件。
机器关闭邮件会明确记录 skipped，不能视作已发送。部分失败也保存结果并发送不完整摘要。

## 输入合同

`feature_dictionary.json` 保存版本、指纹和有序 66/15 通道定义。候选视图不注册为训练块。
`factorperf_values.csv` 仅有 33 原因子的日 IC、百分点 spread；
`factormarket_values.csv` 仅有三类事件的 flag/age、market_up、8 项风险 z-score。
未知值保留 NaN，独立导出 validity。原始 330/2454 通道缓存不修改。
风险 z-score 使用前 120 日（排除当日）、至少 60 个有效观测、总体标准差。
常量参考窗口的标准差为零时仍为未知，不制造有效零值。

## 输出与解释

- `coverage_daily.csv`：独立上市 91 交易日／退市股票池，完整样本比例、缺失数量分布、60 日窗口。
- `coverage_factors.csv`：逐日逐因子覆盖、源状态和去掉该因子的完整率增量。
- `source_status.csv`：文件版本 SHA256、未开始覆盖、文件缺失／空文件／损坏。
- `coverage_strata.csv`：当日行业、规模、成交状态分层。`traded_proxy` 是日成交额>0，
  不是完整可买卖性规则（涨跌停、停牌等没有可靠状态时不冒充可交易样本）。
- `factorperf_windows.csv`、`factormarket_windows.csv`、`context_joint_windows.csv`：
  各块及联合 120 日完整性；历史不足为未知。
- `events_annual.csv`、`risk_maturity.csv`：事件可用／触发／连续段、风险来源及标准化成熟情况。
- `risk_quality.csv`：实际权重覆盖、底层无效比例、独立参考、当前算法及存储结果。
  分组分钟收益与相同生产分母的参考另列，避免混淆跨证券 shift 和分母覆盖两种影响。
- `horizon_lags.csv`：固定 lag=0/1/5/10/20/60 的描述性关系。目标是已有日度记录中的
  10 日 label 与当前来源因子截面的 IC；不自动选最优 lag，也不声称排除来源修订。
- `diagnostics/`：三个明确的 `@2` 根目录、数据／标签身份检查、逐日和年度配对 IC、
  分支尺度、覆盖率分层、逐 epoch 指标和 lifecycle/archive/frozen-fold 对照。
- `weight_probes.json`：最早／中间／最晚 checkpoint 的首个 OOS 日期，最多 256 只
  按 secid 排序的已记录股票；严格加载旧权重、使用旧完整情境输入并核对预测复现。
  NPZ 解压到临时磁盘映射以限制内存，磁盘不足明确失败。不会构造新训练样本。
  梯度为该小批 MSE 梯度；扰动是将整个窗口固定为首个状态，不是重训消融。
- `report_recovery.json`、`mail_status.json`：精确实验的回执重试／报告补投递与邮件状态。
  已 published 的包不会重复投递；补投递在模型生命周期锁内冻结 results，绝不恢复训练。

## 必须保留的验收边界

1. 八项风险不是统一市值加权；旧版分钟收益的全表 shift 存在跨证券串联风险。
   生产计算已另行修正；此审计仍只核对，不修订风险库。历史修订使用手动重算入口。`training_gate` 保持未批准；需单独决定修复和新版本重算范围。
2. 完整源到达时间、财报披露版本、已被上游填零的真缺失无法由每日文件恢复。
3. 配置 weight decay 不是优化器 param-group 状态证据。缺少训练 checkpoint/停止日志时，
   实际优化器状态及停止原因继续标待核查；不从最佳 epoch 或 IC 猜测。
4. `completed` 仅表示日度审计循环完成。候选训练准入看 `training_gate`，模型诊断的缺失项
   看 `diagnostics.pending`；它们不等于自动批准新实验。
5. 根目录固定为 `factorseq_residual@factorseq_v2_{snapshot,temporal,contextual}@2`。
   不自动选择“最新”目录。若服务器身份不同，应明确更新调用的 roots 后再执行。
6. 发布成功仅由 published 回执证明；远端归档需接收端 receipt。服务器无法由本地目录
   推断 Mac 已收到。空投递会打印 unregistered 数量与原因。

离线验证：

```bash
MPLCONFIGDIR=/tmp/learndl-mpl-fs .venv/bin/python -m unittest discover -s tests -p 'test_factor_sequence*.py' -q
.venv/bin/python -m unittest discover -s tests/automation -p test_artifact_exchange.py -q
```

服务器缺少原始数据或真实 `@2` 权重时不能完成生产验收。不得以人工测试数据的成功替代它。
