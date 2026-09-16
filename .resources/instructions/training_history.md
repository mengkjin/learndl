# 训练历史与 worklist 完成记录

`configs/schedule/worklist.yaml` 支持 `fit`、`resume` 和 `force`。
`force` 缺省为 `false`；`resume`、`force` 必须是 YAML 布尔值。

```yaml
fit:
  - gru_day_new_rtn
resume: false
force: false
```

## Worklist 执行规则

- 已成功、训练目录存在、worklist 和对应 schedule 文件未变：跳过，包括 `resume: true`。
- worklist 原始字节或该文件最近 Git commit 变化：列表中的所有项目重新执行。注释、空格、顺序也算变化。
- 对应 schedule YAML 的原始字节或该文件最近 Git commit 变化：该项目重新执行。共享目录中的 schedule 同样比较原始内容。
- 未登记、失败、中断或上次仍停留在 running：按 `resume` 执行。
- 记录中的训练目录已不存在：改为新训练（`resume: false`）。
- `force: true`：忽略上述完成判断，严格按 `resume` 执行；不会绕过正在运行的 worklist 项目锁。
- 普通代码提交、数据更新、时间流逝不触发重训。不再使用原来的 7 天目录创建时间判断。

配置变化决定是否进入执行；`resume` 决定执行方式。恢复已登记的训练时，明确使用记录中的实际目录。
`resume: true` 继续该目录中保存的配置与成果；若要按修改后的配置从头训练，应设 `resume: false`。
首次登记时不会推断旧目录已经成功：`resume: true` 先按 schedule 名称跨模型模块、目录编号查找旧目录，再加载其中保存的配置；唯一候选自动使用，多个候选在 CLI 中明确选择，无候选则报错。非交互运行存在多个候选时明确报错，要求显式提供 `base_path`，不会默认选择第一个或最新目录。普通训练与 short-test 目录分开查找；`false` 分配新目录。
目录选择立即写入训练历史，并同步到 worklist 状态；之后恢复准确复用该目录。保存的 model、schedule、algo 配置缺失或损坏时明确报错，不回退到当前配置。旧训练的原始 Git 节点仍不可推断，新记录描述本次恢复使用的代码。
非 short-test 的新训练也不会清空已有的失败/空目录，而是分配未占用的目录编号。
恢复训练前检查保存的模型文件；缺失或无法读取时报错，不自动降级为新训练。

每个项目结束后立即保存状态。训练期间 Git Pull 或配置编辑不会把新配置误记为已经完成。
项目失败后继续执行后续项目，最后汇总报错；中断信号直接退出。

## 两套独立的本机记录

均使用 `PATH.lc_machine`，即 `.local_resources/<机器名>/`，不会进入 Git。

1. `schedule_worklist/<schedule 名称的 SHA256>.json`：worklist 专用的最近执行状态、原因、force/resume、实际模型目录、worklist/schedule 原文与摘要、文件最近 Git commit、对应训练 run ID。不把手动训练的成功直接当作 worklist 已完成。
2. `training_history/<run_id>/run.json`：公共训练生命周期的每次执行记录，覆盖 CLI、脚本和 API 经 `BaseTrainer` 运行的手动训练、schedule、恢复训练、模型更新；测试流程也记录，并用 `kind`/`stages` 区分。

统一历史包含请求参数、实际目录、有效 model/schedule/algo 配置（包括 override）、开始/结束 UTC 时间、成功/失败/中断状态、异常、进程身份、机器、Python/主要训练包版本、Git HEAD/tree/branch、工作区状态和结束时 HEAD。
有未提交的已跟踪代码/配置改动时，另存 `working_tree.patch`（相对开始时 HEAD，包含暂存与未暂存改动）。
Git 信息和有效配置用于追溯；记录不备份训练数据和模型权重，也不能仅凭 commit 恢复未跟踪文件。

SIGKILL/断电无法执行退出逻辑，磁盘记录可能保留 `running`。worklist 下次获得锁后将其视作未完成。
读取统一历史时可用 `TrainingRun.read(path)`，它通过 PID 和进程创建时间把已退出进程标为 `interrupted`。

记录以临时文件、fsync 和原子替换落盘；在训练目录选定后立即更新，不等训练结束才登记目录。
