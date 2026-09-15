# Project scheduling

Run the installer **as the ordinary server account**, from the deployed project.
It uses sudo only to install the fixed project units and reload/enable timers.
Do not run the complete installer with sudo. No business task is started by apply.

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
triggers. The installer does not create a durable business task queue.

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
