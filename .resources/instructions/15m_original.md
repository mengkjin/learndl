# 15m_original 实验

基线为 dd51a3cbf48c20470e518c06c9c74f3c005fb234。网络与该提交一致；保留当前 Global2Top 重试纠错及空数据保护。实验不覆盖生产 15m，也不加入常规每日预处理或 worklist。

## 执行

Autofix `20261021_create_15m_original_train_03` 构建 fit/predict，保存回读并核验后提交独立的 `gru_15m_original_rtn` 训练请求。维护完成且 GPU 空闲时，现有 idle worker 执行该请求。相同请求 ID 幂等，已有 worklist 及其他模型的执行版本不变。Git 同步并由服务器拉取之前，不会在服务器执行。

任务支持 `dry_run=true` 和可选 `snapshot_manifest`。报告保存在 `PATH.interim/autofix_backups/15m_original_<时间>_<ID>/report.json`。失败不提交训练请求；若第二份缓存失败，第一份可能已保存，原实验缓存备份在报告目录中。没有跨两个缓存的事务回滚。

## 数据来源与限制

历史备份必须有明确的基线版本声明和内容摘要，不能仅凭备份目录时间推断来源。旧版 VWAP Autofix 的报告没有记录原数据生成版本，默认跳过这些未验证的备份；这是刻意的限制。

可提供 JSON 快照清单：顶层 `baseline_commit` 等于上述 SHA、`historical_snapshot` 为 true；`blocks` 包含每个 frame 的 `key: "15m"`、`frame: "fit"` 或 `"predict"`、`backup`（mmap 目录路径）、`sha256`。相对路径相对于清单文件。摘要由 `create_15m_original.py` 的 `audit(block)["sha256"]` 生成，包含证券、日期、特征坐标和数值。声明必须来自已核实的历史运行记录；重新计算摘要本身不能证明历史来源。

没有可靠快照的日期，从 `trade_ts/min`（支持原有备用源）按冻结的 dd51 聚合公式重算；不通过当前 15min 的 VWAP 反推旧值。保留原求和 dtype、直接 amount/volume、旧缺失值处理。随后执行 dd51 预处理、上市时间掩码与 float32 转换。原有及转换产生的 Inf 保留在实验缓存里，训练输入仍按原有流程转 NaN、筛选和填充。

fit/predict 均允许目标范围首尾缺源。以已核实快照日期与 min/day 共同可用日期的并集确定实际首尾，仅在这个区间检查交易日连续性。内部缺口仍报错，不自动从最后缺口后开始，不用已修复的 15min 猜测旧值。构建后先检查日期轴和内部全空日期，再裁去全空首尾；全空特征及 fit 无有效训练窗口仍失败。报告保存目标范围、实际来源范围、排除日期及最终保存范围，预检失败也保存诊断（dry-run 只输出，不写报告）。不强制补齐 2007 年或最新目标日。快照未覆盖部分单独标注为重算。当前分钟、日频、标签、交易日历、依赖版本与历史运行可能不同，因此只有已验证的快照区间可称历史缓存一致，不能声称整个历史环境逐位复现。

报告逐日记录各特征 Inf/NaN、与当前 15m 的差异，以及 30 日输入窗口有效数量。该数量不含标签过滤和分期切分，不当作实际训练样本数。实际训练会另外保存每期标签过滤后的样本数到模型目录 `snapshot/15m_original/fit.<模型日期>.json`，仍明确是 train/valid 切分之前的数量。无历史样本数记录时不能推算旧值。

## 训练结果对比

沿用标准训练日志、每期指标快照和 basic_test 输出。明确指定旧运行和新运行，避免误选同名模型的最新目录：

```sh
python -m src.res.model.util.original_comparison \
  --baseline-log /path/to/old.html --candidate-log /path/to/new.html \
  --baseline-model /path/to/old_model --candidate-model /path/to/new_model \
  --output /path/to/comparison.json
```

训练摘要按共同模型期比较轮数、最终 Trial 重试编号、NaN redo 编号、TrainIC、ValidIC、实际有效样本数。测试从两份 `snapshot/basic_test/test_by_date.feather` 按模型编号、模型日期、submodel、测试日期取交集，再按指标的有限值配对比较 RankIC 和头部表现；不直接比较时间范围不同的总均值。`top5pct` 是收益 zscore，不是百分比收益。缺失文件或指标明确输出 unavailable/null，不以其他期或输入窗口数替代。

网络源码、配置和公式的基线一致性，以及异常输入、缓存读写、任务失败不启动训练、独立请求幂等均有回归测试。

## 第二次任务的启动修正

旧 ID `_01` 已停用，新 ID `_02` 用于重新尝试。已确认的失败原因是脚本遗漏 `if __name__ == '__main__': main()`：进程仅加载定义就以 0 退出，因此没有 ScriptTool 结果和通知接管。现已补齐入口，并让静态清单验证拒绝缺少 main 调用的脚本。Matplotlib 缓存警告不是此次失败原因；子进程缓存改为沙箱内的 `/tmp/learndl-matplotlib`。数据预检不再导入 `src.res.model`：Python 会先执行父包的 `__init__.py`，旧写法因此连带初始化完整模型及其他算法依赖。窗口检查使用固定 30 日、step=1 的等价张量计算，回归测试与训练侧 DataOperator 对照，并在全新 Python 进程中禁止导入模型包验证启动隔离。实际训练请求仍在数据验证完成后提交。

Git 更新邮件的 Follow-up 原来只识别依赖文件和 systemd 定义，`none detected` 不表示 Autofix 没有任务，也不是启动异常。现在 Autofix 清单或脚本变动会明确提示等待独立 worker 验证，并注明 Git 通知不能证明 Autofix 已启动/完成。更新通知与任务结果通知仍是两个独立阶段。


## 第三次任务与结果通知

旧 `_02` 已停用，新 `_03` 使用上述首尾规则。如果 20100914、20101013、20101020 在服务器仍是内部源缺口，新任务会记录具体来源并停止，不请求训练。原 `_02` 在预检处失败，没有开始数据重算。结果邮件的队列现在由 watchdog 每 60 秒检查；沿用发送端的跨进程锁，锁忙跳过，避免与其他进程同时重试同一封邮件。此检查仅重试通知，不重新执行失败的 Autofix。
