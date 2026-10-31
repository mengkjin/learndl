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
absent). Unrelated training code or data changes do not. Missing recorded
directories remain eligible for new training. `resume: true` with no recoverable checkpoint is an error. Unrecorded
legacy directories resume the newest candidate automatically, by creation log
time, falling back to saved `model.yaml` modification time and then model index;
manual use still prompts when ambiguous. `resume: false` creates a new directory.

A failed, killed or interrupted automatic attempt is not retried for that version;
other eligible schedules continue. A configuration version change or explicit
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
Unchanged, deferred and failed pulls do not produce success mail. SMTP cannot
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
未送达的任务结果、进程被杀等由 watchdog 兜底；`no_email: true` 同时关闭这两种
通知。运行详情及失败记录可在上述 status 中查看。手动运行不消耗 Git 自动请求。


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
阻塞只覆盖指定业务通道，不影响独立 RCQuant。申请等待最多 3600 秒，超时开放通道、告警并冷却 900 秒；
冷却期间其他维护请求不能接力封锁。阻塞型 autofix 实际执行最多 3600 秒（更短 timeout 有效），超时先 TERM、30 秒后 KILL，
确认子进程全部退出才释放通道，记录 timed_out，不重跑已尝试 ID。非阻塞任务保留原 timeout。
源码、配置、依赖和调度控制文件只读，仅 data/results/models/.local_resources/.logs/.cache 业务目录可写；
不自动推断业务数据冲突，需由作者选择 blocking 类型。更改已有任务声明仍要求新 ID；未声明 blocking 的历史记录不会因此重跑。

Git 由独立 `learndl-git-update.service` 检查，fetch 无需等待业务任务。
确认有可应用的新版本才预约全局维护；可优先于尚在等待的 autofix，但不抢占已经运行的任务。
等待上限和冷却同上；代码切换必须取得独占代码锁，独立任务也参与保护。
新进程核验切换结果后开放任务；失败核验不会因等待超时而自动放行。
watchdog 在 Git 等待期间继续检查任务生命周期，避免已结束的进程记录阻塞维护；实际代码切换时同样遵守代码读锁。
v3 Git 检查结果位于 `PATH.runtime/scheduling/git-channel.json`，失败由独立 Git 服务返回非零退出码。
普通终端 `git pull` 不遵守项目代码锁，应在安全空档操作。

`installer status` 展示维护状态及最近 100 个日历请求的计划时间、启动延迟、等待原因和最终状态。
日志与记录保留；升级前须解决旧日历服务冲突，避免新旧 worker 同时派发。实际 Linux systemd/Bubblewrap 行为需部署后验证。
