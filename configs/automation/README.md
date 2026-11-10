# Project scheduling

## Module and configuration boundaries

`src/automation` owns the watchdog, CLI recovery and scheduling control (workers,
installation, admission, recovery, Git updates and autofix). `src/api/task_monitor`
owns the monitor UI, queries and output cache. Its former control modules are
compatibility entrypoints only; add all new control behavior under `src/automation`.
General mail, script locks and task database helpers remain in their existing packages.

This directory is the single editable source for non-training automation settings:
`tasks.yaml` registers tasks, `updates.yaml` / `backfills.yaml` set calendar times,
`maintenance.yaml` configures watchdog workers, `hosts/` supplies host overrides,
and `autofix.yaml` declares one-shot repairs. See [autofix operations](AUTOFIX.md).
Model/worklist/RL configurations and `.secret` machine settings stay in place.

The old built-in `--config-dir runs/scheduling` resolves to this directory;
custom config directories retain their original meaning. Old Python entrypoints
and shell install commands remain supported, including services installed before
this reorganization. Apply publishes new startup commands and bootstrap bytes.
Service names, scheduler versions, runtime directories, request identities and
lock files do not change. Do not copy, clear or relocate ledgers during this upgrade.

Run the installer **as the ordinary server account**, from the deployed project.
It uses sudo only to install the fixed project units and reload/enable timers.
Do not run the complete installer with sudo. Apply enables timers; enabled idle training can start on a subsequent watchdog check.

```bash
bash runs/install_schedules.sh plan --host mengkjin-server
bash runs/install_schedules.sh apply --host mengkjin-server
bash runs/install_schedules.sh status --host mengkjin-server
bash runs/install_schedules.sh rollback --host mengkjin-server
```

`runs/install_watchdog.sh` remains a compatibility entrypoint to this installer.
Its old `--remove-legacy-cron` option no longer deletes cron lines. Comment those
lines yourself and inspect a plan before applying.

## Removed shared-folder command queue

The legacy `ScheduledTask` / `TaskScheduler` API and its notebook example have
been removed without a compatibility layer. Daily update no longer prints or
executes commands from `PATH.share_folder/task_schedule`; it still synchronizes
shared model schedule configurations with `SharedSync.sync()`.

Use the supported entrypoints instead:

- Automatic model training: add schedule names to `configs/schedule/worklist.yaml`;
  watchdog's `idle_worklist` job queues eligible training for the independent worker.
- Manual training: use the existing single schedule model or schedule worklist
  entrypoint (`ScheduleModel` / `CarryOutScheduleWorkList`).
- Periodic scripts: register entrypoints and configure tasks/plans through this
  unified scheduler, as described below.

Existing `.await` / `.done` files are left untouched and are no longer consumed
by the updated daily script. They are not automatically executed, migrated, or
deleted: arbitrary shell commands cannot be converted directly into a model
worklist. Review any remaining server files and external callers separately;
imports of the removed API must be updated. This removal requires no timer,
cron, or watchdog configuration changes. Already-running Python processes keep
their loaded code until they exit; subsequent processes use the updated code.

## Configuration

- `tasks.yaml`: registered entrypoint, arguments, lock group, overlap policy,
  optional timeout (`12h`, `72h`, or `null`). Entry points are registered in code;
  configuration cannot supply arbitrary executable paths or shell commands.
- `updates.yaml`, `backfills.yaml`: named plans with `task`, `days`, and `times`.
  `days` accepts `daily` or a weekday list such as `[sat]`. `times` is a list of
  quoted `HH:MM` strings. Alternatively `every_minutes` is a positive divisor of
  1440 and cannot be combined with days/times. Plans represent local wall time.
- `maintenance.yaml`: watchdog tick, registered jobs, enabled flags, intervals,
  watched service names and bounded cache prewarming options. An explicitly empty
  job registry stays disabled. Nonempty older snapshots gain missing `cli_recovery`
  and `email_outbox` defaults; explicit per-job disables remain effective.
- `hosts/<hostname>.yaml`: timezone, default backend (`systemd` or `cron`), include
  list and explicit task overrides (`enabled`, `backend`, `days`, `times`). Duplicate
  IDs/keys are errors. Override time lists replace the old list completely.

Example override:

```yaml
overrides:
  daily_update:
    times: ['18:45', '23:20']
  rcquant_sec_backfill:
    backend: cron
```

The shipped host timezone is `Asia/Shanghai`; verify the server timezone before
installation. Apply detects it with timedatectl. Cron cannot silently emulate a
different timezone. For offline planning only:

```bash
bash runs/install_schedules.sh plan --host mengkjin-server \
  --crontab-file /path/to/exported-crontab.txt --server-timezone Asia/Shanghai
```

## Cron coexistence

Uncommented legacy cron for the exact script/arguments takes priority. The
installer subtracts its time slots and installs only missing times. Comments
do not count as coverage. Consequently commenting an old cron line and applying
moves those times to the configured backend. Restoring that line and applying
removes the corresponding managed times again. Extra legacy times remain active.

Only the `BEGIN LEARNDL MANAGED` block is rewritten. Non-project jobs, including
OneDrive checks, are not owned or changed. Unsupported date expressions, command
wrappers, ambiguous argument/environment/timezone combinations and duplicates
defer affected tasks. Supported legacy wrappers are the project's absolute bash
script paths, optionally preceded by bash/sh. Direct Python commands are reported
as ambiguous for explicit review. Existing standalone watchdog/old maintenance
cron must be commented out before installing the watchdog trigger.

Manual cron edits can create duplicates until the next apply. A shared execution
lock prevents overlap but does not promise exactly-once execution of two sequential
triggers. Calendar tasks do not use a durable business queue; idle worklist training uses the separate queue described below.

## Supervision and timeouts

Only commands launched through the installed runner have timeout protection.
Existing direct cron and direct manual script invocations are not retroactively
supervised. `runner run <task-id>` can be used manually after installation and is
supervised just like a generated cron entry.

Daily update and forced daily update have a 12-hour limit; weekly update has a
72-hour limit. Time starts after obtaining the execution lock and covers the
whole script including daily update's internal scheduler. Continued log output
does not reset the deadline. Each run snapshots its limit; later apply affects
new runs. Forced daily update waits up to 30 minutes for the same lock used by
normal daily update; ordinary overlapping invocations skip without error mail.

Watchdog persists termination intent, sends TERM, and after a 30-second grace
period sends KILL on a subsequent check. This is polling, not a hard real-time
deadline. Disabling/stopping watchdog prevents enforcement; the Monitor reports
missing/stale timeout heartbeats. PID creation time, ownership and session/cgroup
membership are checked before signalling. On systemd the dedicated service cgroup
also captures children that leave the original session; on cron intentionally
detached descendants cannot be guaranteed recoverable. The runner stays alive
and holds its lock until the supervised children exit.

Confirmed timeout exits are `killed` with `TIMEOUT` metadata. Watchdog sends one
durable timeout event (or a failure event followed by recovery), with delivery
retry. Python error mail and generic killed mail do not duplicate a timeout event.
SMTP cannot guarantee exactly-once delivery if a crash happens between delivery
and recording success.

## Deployment and recovery

Installed JSON snapshots, run/event SQLite data, transaction journal and rollback
backup live under the machine's `PATH.runtime/scheduling`. Plan changes no system
configuration. Apply checks crontab again before writing, stages/verifies units,
pauses affected timers and preserves running business processes. A running legacy
script can defer conversion until it exits. Repeated apply is idempotent.

If installation fails, inspect `transaction.json`; rerun apply to reconcile the
partial installation, or rollback. Rollback refuses to overwrite externally edited
cron. Removed units are moved to `.learndl-disabled` backups, not deleted. Running
work is not undone by rollback. A cron edit simultaneous with `crontab -` cannot
be made atomic; avoid editing cron during apply.

Pulling Git changes does not publish schedule settings: apply publishes them.
It **can** change script code in a live working directory. Use reviewed deployment
versions and keep credentials outside Git. Do not grant arbitrary passwordless
sudo to the task user or execute unreviewed installer code as root.

Service checks should list persistent services only, not scheduled oneshot units.
The unified installer is the sole source of service definitions; it generates
the active units from the installed configuration.

## Idle worklist training

`maintenance.yaml` enables `idle_worklist` on CUDA Linux servers. Re-run installer
`plan` and `apply` on the server after pulling this change; an existing installed
snapshot does not pick up new YAML settings automatically.

Every 900 seconds watchdog checks shared `FitLock` occupancy and **cuda:0 memory
used / total < 20%** (not GPU compute utilization). Unknown GPU status defers work.
Inference/tests hold no fit lock but their GPU memory still counts. The shared
lock covers each trainer's fit stage, including waiting for the NN lock; data
preparation and evaluation do not hold it. This detects cooperating project
trainers, not external programs that do not take this lock.

`FitLock` now allows unlimited simultaneous holders for NN and Boost. `FitLockNN`
retains the old exclusive NN behavior and the `train_fit.lock` filename. Inference
CPU/GPU fallback consults `FitLockNN`. The preference section is `fit_lock_nn`;
legacy preference keys remain fallbacks. Existing NN processes are detectable
through the old lock; an already-running old Boost process cannot acquire the
new shared lock retroactively. Let those processes finish before enabling idle
training. OS process exit (including SIGKILL) releases the lock without cleanup.

Watchdog queues at most one schedule, in worklist order. A separate
`learndl-idle-worklist.service`/timer consumes that request and starts a fresh
Python process for exactly one schedule. It has its own cgroup and no watchdog
1 GB memory cap. The cron backend has a separate minutely worker trigger. The
worker rechecks fit locks, GPU memory and configuration before launch. Completion
does not immediately start another schedule: the next 15-minute poll decides.
Manual training arriving after launch does not interrupt the automatic task;
NN fitting remains serialized by `FitLockNN`.

Success uses the existing `schedule_worklist` record, and actual training also
writes independent `training_history` provenance. `force: true` runs only once
for a given successful configuration version in automatic mode; manual worklist
execution retains its force behavior. Raw worklist edits (including comments,
reordering, or `rerun_mark`) and edits to a schedule create a new version.
`rerun_mark` is symbolic: bump it when model code changed but the schedule YAML
did not. Watchdog idle-training mail includes `Rerun mark: <value>` (`none` if
absent). Unrelated training code or data changes do not trigger reruns. Completion
uses status and configuration version only; deleting or renaming a recorded model
directory does not invalidate completion. Use a rerun mark or explicit reset to
request another attempt.

For automatic `resume: true`, the trainer scans current directories for the same
schedule and chooses the highest numeric suffix with a matching effective training
definition and valid saved artifacts (the unsuffixed directory is index 1).
Creation times and worklist audit paths do not influence selection. Matching
includes model/algo, input, labels, date grid/window, loss and optimizer settings;
Callback, logging, device, compilation and execution-range settings are excluded.
Invalid or incompatible directories are logged and skipped. With no matching run,
the trainer safely creates an unused directory without clearing existing ones.
`resume: false` requests a fresh run. Explicit manual paths and interactive choices
keep their existing behavior.

Abnormal-exit retries also delegate directory selection to the trainer. Legacy
`model_path`/`resume_model_path` fields remain audit data, not directory bindings.
Retry identity, configuration version, stop suppression, cooldown, memory admission
and retry limits remain enforced.

Ordinary script errors and explicit interruptions are not retried for that version;
recoverable abnormal exits follow the bounded retry policy above. Other eligible schedules continue. A configuration version change or explicit
reset makes it eligible again. Reset preserves past run records:

```bash
.venv/bin/python -m src.automation.scheduling.idle_worklist status
.venv/bin/python -m src.automation.scheduling.idle_worklist reset SCHEDULE_NAME
```

There is no total training duration limit. By default **12 hours without stdout
or stderr output** triggers TERM; KILL follows after a 30-second grace period on
a later watchdog tick if necessary. Business output renews the deadline; watchdog
heartbeats do not. A repeatedly logging application deadlock cannot be detected
by this policy. Only the verified automatic session/cgroup is targeted.

The training script sends its own success and handled-error reports, including
its usual attachments. Watchdog sends the start notification and acts as a
fallback for abnormal exits (signals, timeout, reboot, launcher failure, or a
Python failure whose script error email was not delivered). A successful SMTP
send records an error-mail receipt, preventing a duplicate watchdog error report.
A success report cannot suppress a later abnormal failure. Successful completion
and deferred/skipped work do not generate watchdog finish mail for new workers.
For rolling upgrades, workers record `script_email_enabled: true` at launch.
Older runs without that marker retain watchdog success mail unless a script
success-mail receipt exists, so a training already running with `email=False`
keeps its result notification without restart.

Watchdog notifications are durable and retry failed delivery. They include the
schedule, version/Git file commits, log path and available training-history IDs.
After server restart, abnormal-exit notification is delivered when watchdog
resumes. No idle-poll or watchdog lifecycle mail is sent by this feature.

Queue, attempts, notifications and logs live under `PATH.runtime/scheduling`
(`runs.sqlite`, `idle_worklist/*.log`). To stop new automatic starts, set
`idle_worklist.enabled: false` and apply. Running work finishes and remains
supervised; do not disable `task_timeouts` while automatic work is running.

## Automatic master updates

`git_auto_update` runs at its configured interval (currently 60 seconds), on server platforms only.
With scheduler version 2 it is attempted before new data work; older installations
retain Git-last behavior until installer apply. Enable/configure it in
`maintenance.yaml` and publish that configuration with installer plan/apply.
Each check has a 45-second Git command budget; errors wait until the next poll.
Only the existing `origin` remote's `master` is fetched. A successful update queues one email for the independent mail worker containing the host,
update time, before/after commit hashes, changed files and required follow-up.
Failed delivery is retried by the mail worker from a persistent outbox
in `runs.sqlite`; already delivered commit transitions are not mailed again.
Unchanged, deferred and failed pulls do not produce success mail.
Local changes (staged, unstaged, or non-ignored untracked files) generate a separate
English `Watchdog Git Update Paused` notice with host, repository, observation time,
and up to 100 Git status entries. No file contents are included. This check runs
before fetch, so the notice does not imply that a remote update is available.
One durable notice is queued per continuously observed dirty episode; changing
log files or worker restarts do not generate more notices. A clean observation
resets this suppression, allowing a later dirty episode to notify. Failed mail
retries retain the original snapshot. Generated files are never deleted or stashed
automatically. Move them outside the checkout or configure appropriate ignore rules
(after reviewing whether any are already tracked). These notices use the existing
mail worker and require no timer reinstall after pulling the code.
 SMTP cannot
promise exactly-once delivery if the process dies after sending but before saving
its receipt. Normal checks send no additional email. Inspect `PATH.runtime/task_watchdog/state.json`, job
`git_auto_update`, for outcome, before/after hashes, changed files and follow-up
requirements. A failed fetch/update also makes the watchdog invocation fail.

Automatic replacement requires the current branch to be `master`, no tracked,
staged or untracked changes (ignored runtime files do not count), no in-progress
Git operation, and a fast-forward with no local-only commits. It uses fetch plus
`merge --ff-only` to the exact fetched hash. It never resets, cleans, rebases,
stashes, changes branches or force-updates the working tree. Git hooks are not
executed and a newly tracked path cannot overwrite an ignored local file.

Running project training/evaluation pipelines hold a shared update lock for their
whole recorded lifecycle, including loading and final saves. This does not
serialize training or change FitLock/NN rules. The automatic updater acquires the
exclusive side of that lock. Active managed tasks, existing FitLock holders and
live training-history records also defer checkout, covering older recorded
pipelines that have not acquired the new lock. A malformed/unreadable record
blocks automatic updates until investigated. Existing training without any record
or lock cannot be reliably detected. Let such old processes finish before enabling.

A fresh watchdog process uses the new code on the next tick. Existing CLI, web,
prediction or other long-running processes are not restarted; imports performed
before a process acquires the lifecycle guard can still belong to the old code.
This is an in-place update, not an atomic versioned deployment. Manual Git commands
and external file editors do not cooperate with the update lock; avoid using them
concurrently. Killing Git during checkout can leave a partial/dirty checkout that
requires inspection; no automatic reset/rollback is attempted.

Dependencies are not installed and systemd settings are not applied automatically.
Changes to dependency/runtime files or scheduling definitions are recorded as
follow-up items. New master code can contain bugs or require newer dependencies;
it can even prevent the next watchdog from starting, removing timeout and mail
coverage until repaired. Use a tested master and retain an external way to inspect
systemd failures. Updating worklist/schedule contents also invalidates completion
versions and can cause future idle training to run again, per worklist policy.

## Autofix maintenance

`autofix` is an explicitly installed watchdog job and independent systemd worker. Requests are declared in `autofix.yaml`. Installation requires Bubblewrap on Linux and initializes a protected, project-external ledger; ordinary code pulls do not initialize it. Admission follows the installed scheduler version described below. See [Autofix configuration and operations](AUTOFIX.md) for one-attempt IDs, script opt-in, notification ownership, sandbox prerequisites, and installation/rollback instructions.

## RL 实验：Git 配置变更后空闲运行一次

`configs/strategy/rl_experiment.yaml` 是完整配置入口。`mode: single` 使用
`single` 参数；`mode: walk_forward` 使用 `walk_forward` 参数。`enabled` 只控制
自动调度，手动脚本仍可执行。`rerun_mark` 是重跑标记：修改并提交即可在下一次
空闲时重新执行；成功、普通失败、崩溃或重启都不会自动重跑同一版本。

将代码和配置提交并同步到服务器后，以普通服务器账号执行一次：

```bash
bash runs/install_schedules.sh plan --host mengkjin-server
bash runs/install_schedules.sh apply --host mengkjin-server
```

检查 plan 后再 apply；apply 会启用定时器，首次发现已启用的配置也会排队。
后续只需提交 YAML 修改，不必反复安装服务。watchdog 的 `rl_experiment` 作业默认
120 秒检查一次，复用 `learndl-idle-worklist.service`，不在 watchdog 中训练。
如需停用，提交 `enabled: false`，或关闭 maintenance 中的 RL 作业并重新 apply。
禁用只取消尚未启动的请求，不中断正在进行的实验。

Git 检查每条命令最多等待 10 秒，并禁用可选的索引写锁。超时表示暂时无法核实配置，
状态记为 `waiting for git`，按 RL 检查间隔重试（默认 120 秒），不发送配置错误邮件。
不会使用旧缓存启动 RL；尚未启动的旧队列会延后，正在运行的实验不受影响，普通模型
worklist 仍可参与调度。YAML 内容或参数确实无效时仍按原规则发送配置错误通知。

```bash
python -m src.automation.scheduling.idle_worklist rl-status
python scripts/4_train/3_train_rl_experiment.py --config_path configs/strategy/rl_experiment.yaml
```

版本由该文件最后修改的 commit 与内容指纹组成；无关 Git 提交不会触发。
配置必须已经提交，服务器存在本地修改时暂停启动。服务器忙碌期间仅执行最终同步
到本地的最新版，不重放未观察到的中间 Git 历史。运行中的任务使用保存的配置副本，
结束后再考虑新版本。RL 和模型 worklist 同时等待时轮流执行，启动前检查训练锁、
维护门禁及 GPU 显存占用低于 20%；12 小时无日志输出才超时，不限制总运行时长。

自动输出位于配置结果根目录的 `automatic/<请求编号>` 子目录。原始 YAML、解析
后的 JSON、Git 来源及结果位置保存在 runtime/scheduling/rl_experiment 下；实验包的
配置中也包含来源信息。季度实验必须已有兼容行情快照；默认 `stage: all`。如果要
跨手动调用继续 `screen`/`quarters` 阶段，请使用已有 JSON 及其固定输出目录；新的
自动请求刻意使用新目录，不继承另一版本的筛选结果。

正常完成或 Python 异常由任务发送一封 HTML 日志和 ZIP/JSON 附件邮件。
watchdog 发送一次启动通知，脚本接手执行后的收尾邮件由脚本负责；启动准备或导入失败
由 watchdog 提醒。`no_email: true` 关闭这些通知。运行详情及失败记录可在上述 status 中查看。手动运行不消耗 Git 自动请求。


## Pending email retry

`email_outbox` runs every 60 seconds through the existing watchdog. It retries
only published entries in `PATH.email_outbox`; it does not generate alerts or
consume other subsystems' notification queues. No timer reinstall is needed for
an older nonempty watchdog snapshot: the missing job defaults to enabled.

Retries take the same exclusive `.lock` as ordinary email sending, without waiting.
If a sender is active the job skips that tick. Reading, SMTP delivery and removal
of each queued message all occur under the lock. Hidden staging directories and
`dead/` are ignored. A tick processes at most 20 entries with a 20-second budget
checked between messages; in-flight SMTP is allowed to complete. Network failures
retain messages for the next tick; SMTP rejections retain the existing dead-letter
policy. Watchdog state records attempts, sent and remaining counts, lock contention
and errors. SMTP acceptance followed by a crash before local deletion can still
result in a later duplicate; this is not an exactly-once delivery protocol.

## 邮件大小、积压恢复与告警

邮件默认以 **30,000,000 字节的完整 SMTP MIME** 为上限（含 Base64、正文、邮件头和 CRLF）。
单个附件或分卷最多 **20,000,000 字节**；分卷先完成压缩，再根据附件容量和 MIME 剩余容量计算最少卷数，均分各卷（大小相差最多 1 字节）。例如压缩后 45 MB 分为三卷、每卷 15 MB。已生成的队列分卷保持原样，避免影响发送回执。
长正文转为完整 UTF-8 文件；过大 HTML/PDF 先打 ZIP；仍超限时生成标准 `.7z.001` 分卷。
每封编号邮件独立记录成功状态，下载全部分卷到同一目录，用 7-Zip 或 Keka 打开第一卷。
普通 ZIP 可直接解压。相关 HTML 的内嵌资源保存在 MHTML 中，需支持 MHTML 的阅读器。

服务器须能在 watchdog 的 PATH 中找到 `7zz` 或 `7z`；工具缺失时大邮件会被保留并隔离，
其他小邮件仍会发送。安装工具后显式重新入队即可。代码更新不需要重新安装 watchdog。

只读检查（不发送邮件、不创建队列目录）：

```bash
.venv/bin/python -m src.proj.util.web.mail_queue inspect
```

输出包含压缩工具、原邮件大小、每卷结果、重试时间、连接冷却、通知发送状态及文件位置。
现有旧队列无需手动转换；下一轮 watchdog 会自动处理。小邮件优先于超大历史邮件。
转换使用暂存目录和原子发布，发送成功后原 `.eml`、元数据、分卷及回执存入 `.archive/`。
失败隔离到 `dead/`；备份和隔离区不会自动删除，注意长期磁盘占用。

修正工具/配置问题后，将隔离批次重新入队（不立即发送，保留成功分卷回执）：

```bash
.venv/bin/python -m src.proj.util.web.mail_queue retry BATCH_ID
```

外部调用方自行管理重试的 `.external/` 回执使用以下命令重置未成功分卷，然后由原调用方重试：

```bash
.venv/bin/python -m src.proj.util.web.mail_queue retry RECEIPT_ID --external
```

单封提交失败延后 1、5、15、60 分钟重试，5 次后隔离；明确永久拒绝立即隔离。
连续两次连接失败或一次认证失败触发全局冷却，不消耗未提交邮件的失败次数。
失败超过 5 分钟仍未恢复时发送小型摘要，隔离事件进入下一轮通知；通知受冷却和发送预算约束。
相同错误不重复通知，变化最多每 30 分钟合并一次；已告警问题恢复后发送恢复摘要。
警告自身失败只延后重试，不产生递归警告。SMTP 完全不可用时先查看本地检查输出。

开发接口：普通 `Email.send` 由共享队列负责重试。外部重试使用
`queue_on_failure=False, delivery_id="稳定事件ID"`，或给 `send_with_smtplib` 传 `delivery_id`；
回执只由调用方重试，不进入 watchdog 的自动发送列表。相同 ID 的内容必须不变。
旧接口没有传 ID 时，超限邮件使用内容指纹复用回执，小邮件仍视为独立调用。
底层 `_deliver` 只接受上限内的单封邮件。已有实验报告账本仍可恢复，新报告复用统一打包模块。
SMTP 接受与本地回执落盘之间仍有极小的崩溃重复窗口，不能保证严格 exactly-once。


## 启动优先级与独立辅助服务（调度版本 2）

本节描述已有 v2 安装的行为；当前安装器启用下文的 v3 通道规则。
无需卸载旧 watchdog；仅拉取 Git 或重启 watchdog 不会启用新版服务结构。
`apply` 保留 autofix ID 账本、训练记录和定时任务队列；`rollback` 恢复此前安装快照，
不清空执行历史。未 apply 的旧快照继续使用旧调度方式。

启动顺序为：

1. 到期的 `git_auto_update` 优先获得安全空档。
2. `email_outbox`、`cli_recovery`、`task_monitor_cache` 在各自独立服务中运行，不占数据维护许可，也不等待彼此完成。
3. 到期日常任务及补跑优先，其次 autofix，最后自动训练/RL（含一次性 A/B 实验）。

优先级约束新任务的准入，不中断正在执行的受管任务，不拆开一个任务内部的多个模型。
定时任务仍保留原有组内锁与普通任务共享许可；autofix 独占维护许可。
训练与 RL 保留原有轮换、GPU 空闲条件及恢复策略。连续的高优先级日常任务可以推迟 autofix；
连续排队的训练不能越过已发现的 autofix。

独立 timer 同时唤醒时，在同一准入锁下做最终检查。训练领取前重新读取 autofix 清单，
不依赖 watchdog 已经扫描或 `pending` 文件已经出现；排队中的训练记录不算正在执行。
日常任务先登记计划时间槽，再申请许可，同一时间槽重复触发不会再执行。
等待不消耗 autofix 的一次尝试，失败重跑仍必须新增 ID。

Git 网络失败、工作区有修改等未改动代码的失败，记录后允许继续使用现有代码。
Git 因任务仍运行而等待时保留优先权，辅助工作仍可运行；当进入代码锁排空阶段时，
停止新的辅助执行，已有辅助执行有硬超时。尝试替换代码后当前 watchdog 不再派发任务，
下一轮新进程核对目标 commit、分支与工作区。结果不明会保持暂停，不自动 reset 或清理。
修复运行期间不自动替换代码。受管 worker 由项目外的引导文件先取得代码读锁，再导入业务模块。

辅助服务分别是：

- `learndl-aux-email-outbox.service`：邮件重试与调度通知投递；最大运行 120 秒。
- `learndl-aux-cli-recovery.service`：已注册 CLI 的恢复检查；最大运行 120 秒。
- `learndl-aux-task-monitor-cache.service`：缓存维护；最大运行时间为配置的 `max_seconds + 30`，最低 60 秒。

各服务有独立 timer、单实例锁和状态文件。CLI 注册/会话条件、缓存间隔及邮件重试间隔仍有效。
关闭 `email_outbox` job 只关闭普通脚本邮件 outbox 重试，调度告警投递服务仍保留。
主 watchdog 只登记事件；普通邮件仍使用原来的脚本邮件队列，调度事件由独立通知队列负责，
不把失败调度邮件再次加入脚本队列。SMTP 在发送成功但回执未保存的崩溃窗口仍可能重复投递。

查询 `runs/install_schedules.sh status` 可看到调度版本、Git 状态、日常/autofix 队列、各辅助服务状态。
辅助状态位于 `PATH.runtime/scheduling/auxiliary/`，运行状态 `running` 若对应服务已经失败，表示进程中断；
下一次到期检查会重试辅助工作。训练 `status` 的 `blocked_by`、autofix `status` 的 worker observation
说明等待原因。服务日志使用 `journalctl -u <服务名>` 查看。

### Linux 上线验收

在测试实例、空 autofix 清单下执行安装器 plan/apply；不要为了测试改写生产修复 ID。
检查 status 出现 `scheduler_version: 3`，数据 lane、Git 与辅助 timer 都启用，并检查各自 journal。
运行下面的临时夹具验收（不会启动真实训练、修复或部署生产服务）：

```bash
LEARNDL_SYSTEMD_ACCEPTANCE=1 .venv/bin/python -m unittest src.automation.tests.test_priority_linux -v
```

该测试要求 Linux、可用的 `systemd-run --user` 和 Bubblewrap；创建的临时测试单元自动回收。
另运行 `python -m unittest src.automation.tests.test_priority -v` 验证优先级、Git 核验和后台通知边界。
生产启用后观察训练与独立数据任务并行、阻塞型 autofix 等待和超时释放，以及 Git 等待代码使用者结束的行为，
不要手工重置账本来制造验收场景。本机 macOS 测试不代表 Linux/systemd 隔离验收已经通过。

## 调度 v3：通道与维护

完整执行 scheduling installer `plan --diff` / `apply` 才会启用 v3。仅拉取代码不会改变旧安装的任务编排。
日更新、周更新各有 `learndl-lane-<task>.service`，原日历服务只登记请求；各 lane 每分钟读取自己的队列。
日更普通与强制入口共享自身互斥锁。同类任务运行期间的新请求标记 `skipped_overlap`，不补跑；维护期间同类未执行请求保留最新一个。
自动研究仍使用单一 worker，完整生命周期占用研究通道；手动训练不受该队列限制，但保留资源锁和代码保护。
数据 worker 在可用内存低于 10% 时暂缓启动并记录原因；一个任务的暂缓不阻塞其他 lane。

任务配置支持 `channel: data | independent`（默认 data）以及 `window: rcquant_evening`。
RCQuant 配置为 independent，保持北京时间 21:01；当天日更成功前等待，每分钟重试，23:59 到达已有下载截止点后不再启动，跨日请求过期。
不同时间的同日 RCQuant 触发归一到同一个请求。日更内嵌补全与独立补全使用同一份每类型、每交易日的跨进程下载锁。
取得锁后重新检查原始缓存；转换失败只重做转换。样本下载 `first_n > 0` 使用隔离缓存，不写全量数据库。
原始数据及 RCQuant 派生数据均通过临时文件原子发布；远端响应后、本地保存前崩溃仍可能导致重试请求。

Autofix 顶层可设置 `blocking: none | research | research_and_data`，默认 none；autofix 本身始终串行。
实际执行时阻塞只覆盖指定业务通道，不影响独立 RCQuant。等待本身不预占业务通道。
每次申请等待最多 3600 秒，随后仅该请求冷却 900 秒；其他请求继续竞争。原始等待时间保留，同一等待每天最多提醒一次。阻塞型 autofix 实际执行最多 3600 秒（更短 timeout 有效），超时先 TERM、30 秒后 KILL，
确认子进程全部退出才释放通道，记录 timed_out，不重跑已尝试 ID。非阻塞任务保留原 timeout。
源码、配置、依赖和调度控制文件只读，仅 data/results/models/.local_resources/.logs/.cache 业务目录可写；
不自动推断业务数据冲突，需由作者选择 blocking 类型。更改已有任务声明仍要求新 ID；未声明 blocking 的历史记录不会因此重跑。

Git 由独立 `learndl-git-update.service` 检查，fetch 无需等待业务任务。
确认有可应用的新版本才登记维护请求；等待时日程、研究和辅助任务继续运行。
同时具备执行条件时优先安装、其次 Git、再其次阻塞型 autofix；未就绪请求不拦截其他可执行工作。
等待上限和冷却同上；代码切换必须取得独占代码锁，独立任务也参与保护。
新进程核验切换结果后开放任务；失败核验不会因等待超时而自动放行。
watchdog 在 Git 等待期间继续检查任务生命周期，避免已结束的进程记录阻塞维护；实际代码切换时同样遵守代码读锁。
v3 Git 检查结果位于 `PATH.runtime/scheduling/git-channel.json`，失败由独立 Git 服务返回非零退出码。
普通终端 `git pull` 不遵守项目代码锁，由操作者判断风险。已导入的 Python 对象通常继续执行旧代码；
后续动态导入、子进程、新读取的配置可能混用新版本。涉及接口/锁协议/共享数据格式不兼容，或同步升级依赖和迁移数据时，
应等待受影响任务退出。仅文档或无关模块变动通常风险较低；`--ff-only` 不保证运行期间代码版本一致。

`installer status` 展示维护状态及最近 100 个日历请求的计划时间、启动延迟、等待原因和最终状态。
日志与记录保留；升级前须解决旧日历服务冲突，避免新旧 worker 同时派发。实际 Linux systemd/Bubblewrap 行为需部署后验证。

### Monitor 展示没有 ScriptTool 的受管任务

Task Monitor 的查询现在合并脚本 `task_records`、调度 `scheduling/runs.sqlite` 与
项目外 autofix 账本的执行记录。自动训练、定时任务和 autofix 的受管任务组即使没有
ScriptTool，也显示为 `watchdog` 来源，提供原始阶段、主 PID、详情中的子进程 PID 和监督日志。
这是按受管任务组展示，并不把操作系统的每个子进程都复制为独立任务；watchdog 心跳类
短检查仍沿用健康状态展示。未接入这些受管记录的外部程序不属于自动发现范围。

调度任务与脚本通过已有 task_links 且同 PID、同逻辑状态时去重；子脚本与父任务组的 PID
不同则分别保留。活动任务不受“最近一天”等时间窗口截断。PID 存活校验同时比较进程创建时间，
PID 已复用或退出显示 stale，Monitor 不负责修改调度状态。受管组不通过普通脚本 Kill 接口终止。
账本缺失时不创建，所有查询只读；已有 `_02` 可直接显示，无需重新训练。
代码同步后需让 Monitor 服务载入新代码；这不改变现有 Git/训练互斥规则，也不需重装调度单元。

本次完整 Monitor 回归 55 项通过（包含需要宿主进程读取权限的内存采样测试）；
新增覆盖无脚本数据库、旧长任务、分页计数、同进程去重、子进程保留、PID 复用、autofix 历史及只读查询。


## 在线发布与安装分级

`plan --diff` 输出 `impact.mode`、原因及通道范围；`config` 表示普通在线配置，
`compatible` 表示明确审阅过的框架兼容升级，`exclusive` 表示必须排空相关任务。
`groups: null` 为全局，数组为受影响通道。首次安装及旧安装没有兼容协议记录时需要一次独占窗口。
源文件 SHA256 的变化默认不兼容；只有 `framework_compatibility.json`（相对源码路径映射到旧/新 SHA256 二元数组列表）明确登记并通过测试的
旧/新哈希对才可兼容发布，不自动信任文件名、版本号相同或“改动很小”。

时间、周期、任务启停、任务优先级和后续启动的超时配置可在线发布。禁用生命周期监督或更换未知执行环境仍需独占。
任务 `priority` 为整数，默认 0；只在共享同一脚本锁的多个就绪日程请求之间择优，数值越大越优先，不改变独立通道并行。
运行中的任务保留原参数。已经登记的请求保存配置快照；禁用任务会取消其未启动请求并记录原因。
更改锁组/入口等执行策略时，相关旧请求先完成，避免旧请求与新锁组交错执行。
安装不终止正在运行的 oneshot service；只更新必要的定时器和未来启动配置。

`generations/<sha256>.json` 保存不可变快照，`installed.json` 原子切换为指向快照的相对符号链接，旧读取接口不变。
短暂发布锁协调队列登记、配置切换和定时器激活；补登发布期间经过的受管触发点，以原请求 ID 去重，
不补跑发布前的历史时段，不接管外部 cron。运行账本、锁文件和已成功请求 ID 均不清空。
配置发布不顺带重写 bootstrap/policy；部署框架变化才通过对应升级路径替换。

```bash
bash runs/install_schedules.sh apply --host mengkjin-server   # 在线发布，或立即尝试独占；忙则退出
bash runs/install_schedules.sh defer --host mengkjin-server   # 忙时后台等待，10 秒重试，最长 24 小时
bash runs/install_schedules.sh force --host mengkjin-server   # 预览、确认受管进程，再受控停止并安装
```

等待安装不暂停业务。提交后代码/配置/单元内容或 Git HEAD 改变、主机重启、授权过期均取消旧审批，需重新预览。
`force` 对普通在线配置直接发布；需要独占时使用后台一次性安装服务，关闭 SSH 不取消已确认的排空操作。
它只允许停止预览中的受管进程树：重新核验 PID、创建时间和会话/cgroup，先 TERM，30 秒后 KILL，排空及取得许可最多 120 秒。
新出现的受管任务需要重新预览；不杀手工任务、未知进程、Git 或其他安装进程，也不删除锁来绕过存活子进程。
若仍有非受管代码使用者阻塞，则安装失败，保留现有锁和产物。

强制停止写入独立 `installation_stops` 记录，保留安装请求 ID，阻止自动恢复和已派生的排队重试。
自然完成的任务保留原结果；需要恢复时手动决定或提交新的显式请求。
安装前失败恢复准入；事务失败尝试回滚，回滚失败或安装中断进入 `repair_required`，需显式检查和回滚。
待安装超过一小时发送提醒，同一等待状态每天最多一次；通知失败仍由独立邮件通道重试。

本机验证：`python -m unittest discover -s src/automation/tests`。
Linux 验收使用上文 `LEARNDL_SYSTEMD_ACCEPTANCE=1` 命令；本机测试通过不代表已经部署或完成 Linux 隔离验收。


### Maintenance delay diagnostics

Git checks retain their 60-second interval, with no admission cooldown after an
hour of waiting. Existing Git cooldown records resume admission attempts on the
next poll. Only one delay notice is generated per Git request, across subsequent
days and worker restarts. Existing notification timestamps count as already notified.
Completion or cancellation clears the request, so a new request may notify again.
Delivery failures still retry the original queued notice. This policy is loaded by
new workers after a code pull; no watchdog reinstall or timer change is required.

New Git, blocking-autofix and installation delay notices use English evidence snapshots.
They identify the waiting operation, UTC observation/wait/start times, elapsed durations,
last admission failure, and matching managed task identities/log paths. Active tasks are
explicitly labelled candidates, not proven lock owners. PID creation time is checked;
a dead root does not imply all descendants have exited. Missing/unreadable ledgers are
reported without creating or repairing them. Collection has a two-second query/iteration
budget and a 100-record cap; the body shows ten candidates, with larger snapshots attached
as JSON. SMTP retries retain the captured observation rather than attributing historical
blocking to newly running tasks. Legacy queued notices retain their original limited detail.


### Research notification ownership

Automatic RL (including resume) and factor-sequence research send one durable
watchdog start notice per run. Delivery retries retain the same event ID and body;
`no_email` still disables notifications. Flat request metadata is supported as well
as Git revision mappings. Previously suppressed historical start events are not replayed.
Once the worker hands execution to the script entry point, result mail belongs to
the script, including runtime errors and failed/deferred SMTP delivery. Watchdog
continues supervising the process but does not generate a second completion email,
including for later termination. Failures before that handoff (such as script import,
entry-point validation or launch preparation failures) retain watchdog failure mail.
Successful legacy runs do not receive duplicate result mail; older failed runs without
handoff evidence retain the existing receipt-based fallback. Model-training and autofix
notification policies are unchanged. Pulling this code updates new worker behavior;
no systemd timer or watchdog reinstall is required.
