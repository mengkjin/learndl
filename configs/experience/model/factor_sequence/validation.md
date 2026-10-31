# 首版验证记录（2026-10-30）

## 服务器失败报告与 CLI 修复（2026-10-31）

收到 run `fs_server_20261031T114646Z_49d0f6` 的 suite.json 和 result_report.zip。
服务器在 offline 阶段终止：23 项测试中 3 项报错，均为 pandas Series.clip 不接受 max 参数。
发生位置为 bundle 的审计分组和 pipeline 的评价分组；np.floor(Series) 仍返回 Series。
恢复为 clip(upper=9)，其余数据逻辑不变。尚未运行 build、audit 或真实训练。

脚本未出现在 Run Scripts 的原因为头部普通注释被当作 YAML 解析，header 被禁用。
现已用空行结束 YAML 元数据，并登记运行参数。CLI 直接调用 main(**kwargs)，
因此另将 argparse 移到独立入口，避免解析外层 CLI 的参数；运行失败会明确抛错。

修复后 24 项离线测试全部通过，包括真实脚本发现、CLI 动态加载、参数传递及失败后通知。
测试中的训练与邮件均为 mock，未启动真实服务器训练或发送邮件。
更新服务器代码后重新进入 Run Scripts，使用新 run ID 重跑；旧失败记录保留。

## 门控模型与服务器入口

新增 `factorseq_gate` 和 `scripts/4_train/6_factor_sequence_verify.py`。
本次离线测试共 23 项通过：增加门控恒等初始化、股票状态差异、通道保留与因果性、
2017 起点对应的滚动切分与边界清除、服务器最早源日期探测、跨段汇总完整性、
阶段成功/失败编排及邮件打包。邮件调用使用 mock，未实际发信；编排测试使用 mock，
不作为真实服务器训练通过证据。脚本帮助与 `git diff --check` 通过。

服务器 2017 年至最新的真实训练尚未执行，最早因子日期与成熟标签终点由服务器运行时确定。
下文 16 项测试和五模型的描述是此前验证记录。

## 源文件修复复核（2026-10-31 UTC）

用户授权使用因子计算器重算并覆盖缺少证券代码的文件后，已扫描冻结因子池中
31 个普通股票因子的全部 18,723 个已有日度文件；另 2 个 affiliate 因子不属于该导出类型。
证券代码列和命名索引均视为有效来源，不按行号推断证券代码。

仅发现原先的 4 个 mom_1m 文件有同类问题，均已通过 `calc_and_deploy` 严格校验后
重算覆盖，并重新读取检查证券代码非空、唯一及有效值数量：

- 20241101：5,363 行，5,362 个有效因子值。
- 20241115：5,366 行，5,365 个有效因子值。
- 20241129：5,372 行，5,371 个有效因子值。
- 20241213：5,377 行，5,373 个有效因子值。

原文件备份、前后 SHA256、逐项状态及可复现修复脚本保存在
`results/factor_sequence/factorseq_repair_20261031/`。修复后的 16 项离线测试全部通过。
下文首次构建失败记录保留为历史，不能视为这 4 个文件仍未修复。

修复后的真实构建 `factorseq_v1_20261031_after_repair` 已通过（约 153 秒），新数据版本为
`d12a3cbee32fb5972f11fbd6`。日期范围 20220505–20250422，共 721 个交易日、5,557 只股票；
factorseq / factorperf / factormarket 分别为 330 / 2454 / 26 个通道，公共块保持单股票轴。

最终真实审计 `factorseq_v1_20261031_verified` 已通过（约 86 秒）：独立核对了 37 组日度表现、
231 组事件表现，覆盖三类事件。本次还修复了两处
审计实现问题：压缩数组在循环中重复解压，以及先取最早事件再过滤时间窗导致事件核对为空。
新增回归断言验证每个大型数组只读取一次、窗口外早期事件不会占用核对名额；16 项测试仍全部通过。

额外尝试的 smoke（`factorseq_v1_20261031_verify`）在训练前的数据加载阶段失败：
日志参数 `vb_level` 被误传给不接受该参数的标签加载方法。已将该参数交给预处理器初始化，
并用严格签名 mock 加强加载测试；本轮没有完成修复后的真实模型训练，不能登记 smoke/compare 通过。
第一次审计因重复解压而主动中止，旧阶段失败记录保留；最终审计状态以 `verified` run 为准。

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

## 非阻塞 autofix 发布验证（2026-10-31）

`20261031_factor_sequence_verify_01` 清单验证通过；24 项因子测试、3 项一次性请求测试通过。
扩展运行 idle_worklist/channels/autofix 的 99 项原有测试，出现 5 个失败、16 个错误、1 个跳过；
使用修改前 HEAD 的 idle_worklist/runtime 代码对照，失败数量与测试名称完全一致，
包括已过时的 installed_config mock 和监督状态断言；不将这些既有失败记为通过。
本机未运行真实服务器任务，Git 发布后由服务器领取请求。

## 首段训练初始化失败与修复（2026-10-31）

服务器 `_01` 的离线、discover、build、audit、plan 已通过。数据版本
`7f15876de75cdb51189de190`，历史 20110104–20261030；当前标签块成熟终点为 20260723，
计划 37 段、222 次训练。smoke_000 首个 MLP 在 StepLR 初始化失败，尚未完成训练。

根因是扁平配置只覆盖已知叶子：scheduler.name 改为 step 后，旧 base_lr/step_size_up
仍存在，新 step_size/gamma 未进入配置。现仅在显式覆盖改变调度器类型时替换其完整子树，
同类型覆盖行为保持不变。24 项离线测试通过；增强的配置测试以实际 ModelConfig/Optimizer
对六个模型执行梯度、optimizer.step 和 scheduler.step，并核验旧 cycle 配置仍可初始化。
此结果不冒充真实 GPU 训练通过。发布 `_02` 新请求继续验证。
