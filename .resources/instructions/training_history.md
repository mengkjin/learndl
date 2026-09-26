# 训练历史与 worklist 完成记录

`configs/schedule/worklist.yaml` 支持 `fit`、`resume`、`force` 和 `rerun_mark`。
`force` 缺省为 `false`；`rerun_mark` 缺省或 null 在邮件中显示为 `none`；`resume`、`force` 必须是 YAML 布尔值。

```yaml
fit:
  - gru_day_new_rtn
resume: false
force: false
rerun_mark: 0
```

## Worklist 执行规则

- 已成功、训练目录存在、worklist 和对应 schedule 文件未变：跳过，包括 `resume: true`。
- worklist 原始字节或该文件最近 Git commit 变化：列表中的所有项目重新执行。注释、空格、顺序也算变化。
- 对应 schedule YAML 的原始字节或该文件最近 Git commit 变化：该项目重新执行。共享目录中的 schedule 同样比较原始内容。
- 未登记、失败、中断或上次仍停留在 running：按 `resume` 执行。
- 记录中的训练目录已不存在：改为新训练（`resume: false`）。
- `force: true`：忽略上述完成判断，严格按 `resume` 执行；不会绕过正在运行的 worklist 项目锁。
- `rerun_mark`：纯象征字段，不参与完成判断。改它的值（例如 `0` → `1`）会改变 worklist 字节，形成新版本并触发自动重训；这是在模型代码已改、配置未改时请求再跑一遍的预期做法。Watchdog 闲时训练邮件会带 `Rerun mark: <值>`。
- 普通代码提交、数据更新、时间流逝不触发重训。不再使用原来的 7 天目录创建时间判断。

配置变化决定是否进入执行；`resume` 决定执行方式。恢复已登记的训练时，明确使用记录中的实际目录。
`resume: true` 继续该目录中保存的配置与成果；若要按修改后的配置从头训练，应设 `resume: false`。
首次登记时不会推断旧目录已经成功：`resume: true` 先按 schedule 名称跨模型模块、目录编号查找旧目录，再加载其中保存的配置；唯一候选自动使用，多个候选在 CLI 中明确选择，无候选则报错。非交互运行存在多个候选时明确报错，要求显式提供 `base_path`，不会默认选择第一个或最新目录。普通训练与 short-test 目录分开查找；`false` 分配新目录。
目录选择立即写入训练历史，并同步到 worklist 状态；之后恢复准确复用该目录。保存的 model、schedule、algo 配置缺失或损坏时明确报错，不回退到当前配置。旧训练的原始 Git 节点仍不可推断，新记录描述本次恢复使用的代码。
非 short-test 的新训练也不会清空已有的失败/空目录，而是分配未占用的目录编号。
恢复训练前检查保存的模型文件；缺失或无法读取时报错，不自动降级为新训练。

每个项目结束后立即保存状态。训练期间 Git Pull 或配置编辑不会把新配置误记为已经完成。
项目失败后继续执行后续项目，最后汇总报错；中断信号直接退出。

Watchdog 每次闲时 worklist 检查都会先核对所有 schedule 名称，即使当前有训练或显存繁忙。
名称必须是 schedule YAML 文件名去掉 `.yaml`，例如 `gru_mincr`，不能填写其数据类型 `mincr`。
无法找到配置时，在 watchdog 状态的 `missing_schedules` 中列出名称，并发送
`Watchdog Idle Worklist - configuration error` 邮件，包含缺失名称、查找目录和 worklist 的 Git 版本。
同一 worklist 版本、同一组缺失名称只通知一次；通知持久化到现有邮件队列，发送失败会在后续
watchdog 轮询重试，重启不会丢失。其他有效 schedule 仍可按闲时条件执行；修正名称或补上配置后
自动重新检查。此错误不创建训练记录，也不计作训练失败，不会阻止修复后的首次训练。

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


## Hidden 数据源必须固定模型

`input.hidden.types` 推荐写完整的 `模块@模型名@目录编号@参数编号@子模型`，例如：

```yaml
input:
  type: hidden
  hidden.types:
    - gru@gru_day_new_rtn@1@0@best
```

这里目录编号 `1` 指 `models/nn/gru@gru_day_new_rtn`（没有编号后缀）；
目录编号 `2` 指 `models/nn/gru@gru_day_new_rtn@2`。参数编号从 `0` 开始，
上面的引用读取该目录 `archive/0/<训练日期>/best/state_dict.pt`。
如需第 2 次训练的结果，必须显式写 `gru@gru_day_new_rtn@2@0@best`。

旧式 `gru@gru_day_new_rtn@0@best` 为兼容保存的配置仍可使用，但严格指向
目录编号 1、参数编号 0，不会自动选择唯一候选或最新目录。
模块、模型名、参数编号、子模型必须填写。目录、保存的配置、参数编号或
所选子模型的 checkpoint 缺失时直接报错；交互和非交互训练行为一致。
每个 hidden 来源使用自己选定参数编号下的训练日期，不借用其他参数编号的日期。

`gru_day_lgbm` 和 `gru_day_xgboost` 现均固定使用
`gru@gru_day_new_rtn@0@best`（未带编号后缀的目录，参数编号 0）。更新配置不会改变已经启动的训练；若恢复旧
训练，仍以其保存的配置为准。要采用新的来源，请使用新的训练配置启动。
