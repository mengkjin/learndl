# Project scheduling

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

## Configuration

- `tasks.yaml`: registered entrypoint, arguments, lock group, overlap policy,
  optional timeout (`12h`, `72h`, or `null`). Entry points are registered in code;
  configuration cannot supply arbitrary executable paths or shell commands.
- `updates.yaml`, `backfills.yaml`: named plans with `task`, `days`, and `times`.
  `days` accepts `daily` or a weekday list such as `[sat]`. `times` is a list of
  quoted `HH:MM` strings. Alternatively `every_minutes` is a positive divisor of
  1440 and cannot be combined with days/times. Plans represent local wall time.
- `maintenance.yaml`: watchdog tick, registered jobs, enabled flags, intervals,
  watched service names and bounded cache prewarming options. Empty/disabled job
  lists remain disabled; they do not fall back to the built-in defaults.
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
The legacy static watchdog templates in `runs/systemd` are reference material;
the unified installer generates the active units from the installed configuration.

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
.venv/bin/python -m src.api.task_monitor.scheduling.idle_worklist status
.venv/bin/python -m src.api.task_monitor.scheduling.idle_worklist reset SCHEDULE_NAME
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

`git_auto_update` runs every 300 seconds, on server platforms only, at the end of
watchdog after health checks and notification delivery. Enable/configure it in
`maintenance.yaml` and publish that configuration with installer plan/apply.
Each check has a 45-second Git command budget; errors wait until the next poll.
Only the existing `origin` remote's `master` is fetched. A successful update queues and immediately attempts one email containing the host,
update time, before/after commit hashes, changed files and required follow-up.
Failed delivery is retried on subsequent watchdog ticks from a persistent outbox
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
