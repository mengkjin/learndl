# 首版验证记录（2026-10-30）

## 已验证

- `tests/test_factor_sequence.py`：16 项通过，CPU 人工数据。
- 覆盖截面缩尾/掩码/常量、成熟期、事件确认、前缀不变、分块和增量一致性、
  manifest 拒绝不匹配和损坏、公共数据单份存储及真实 ModuleData 加载路径、
  五模型梯度/优化器更新/磁盘保存加载、FiLM 恒等初始化、历史情境因果性、
  缺失诊断、实际配置接入、标签区间清除、预测存储路径和比较样本一致性。
- `git diff --check` 无空白错误。

回归范围为 `test_label_sources.py`、`test_rnn_hidden_standardize.py`、
`test_15mraw_training.py`：共 25 项，23 项通过，2 项因既有配置缺失失败：

1. `test_schedule_and_dump_identity`：找不到 `gru_day_lag0` schedule。
2. `test_schedule_loading_and_baseline_equivalence`：找不到
   `configs/schedule/current/gru_day_new_std.yaml`。

这两个文件在当前工作区及 HEAD 都不存在；本次没有创建替代 schedule 或修改测试使其绕过。
因此不能称为“全部回归通过”。

## 真实数据状态

结果目录：`results/factor_sequence/`。

- `factorseq_v1_20261030_01`：build 失败。部分 btop 文件使用命名 secid 索引、
  其他文件使用 secid 列，原批量加载合并时丢失索引。已在新适配器中改为逐文件整理后合并，
  该问题已解决；旧失败记录不覆盖，补充说明保存为 `source_diagnosis_followup.json`。
- `factorseq_v1_20261030_02`：build 失败，适配器误将 risk_lncap 当作普通导出文件。
  已修正为保留 affiliate 因子的底层读取路径。
- `factorseq_v1_20261030_03`：build 失败，发现 **mom_1m 的 4 个文件没有证券代码**：
  20241101、20241115、20241129、20241213。文件只有 mom_1m 一列及 RangeIndex，
  无法从文件自身可靠地恢复股票对应关系。
  完整文件路径见 `source_diagnosis.json`，最终读取门禁复核见 `source_gate_final.json`。

真实 audit、smoke、compare 均未运行，五模型的真实训练/样本外成绩尚未产生。
没有修改股票池、删因子、跳过坏日期或用合成数据替代真实闭环。
源文件补齐可验证的证券代码后，应新建 run ID 重新 build；当前没有可复用的真实 bundle。
其他尚未读取完成的源文件也可能有问题，清单不代表全部源数据已经审计完成。

## 首版限制

- 首次真实闭环未通过，不能据此比较 M1/M2 的预测效果。
- 分块构建采用历史前缀重放以保证正确性；股票块尚未做内存受限的流式实现。
- 来源为每日快照；不能证明历史数据库无事后修订，也不能恢复底层披露/更新时间。
- 成本后收益需要后续真实持仓/成交评价，当前不会从 10 日标签生成伪日净值。
