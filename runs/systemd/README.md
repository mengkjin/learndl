# Learndl system watchdog

This watchdog is intentionally installed as a **system** service that runs as
the Learndl account. Its process therefore lives in `system.slice`, outside the
desktop `user@UID.service` cgroup that `systemd-oomd` may kill.

It runs once per minute. Its internal job registry schedules each operation at
its own interval, so no additional cron jobs or timers are needed for short,
high-frequency maintenance. The bundled jobs:

- finalizes task-database records whose process disappeared (every minute);
- checks named systemd services and alerts once per outage (every minute);
- retries pending alert emails (every minute);
- prewarms monitor output caches, removes safely cached crash logs, and applies
  cache retention (every 15 minutes, capped at 20 crash logs per pass).

It does not restart training automatically. Training may have side effects, so
automatic restart should only be added after its resume operation is confirmed
to be idempotent.

## Install

From the project root on the Ubuntu server:

```bash
bash runs/install_schedules.sh plan --host mengkjin-server
bash runs/install_schedules.sh apply --host mengkjin-server
```

The unified installer and configuration format are documented in
[scheduling/README.md](../scheduling/README.md). Edit `watchdog.units` in
`runs/scheduling/maintenance.yaml` to monitor persistent services, then apply:

```bash
bash runs/install_schedules.sh apply --host mengkjin-server
```

Re-run apply after changing configuration or project location. Existing legacy
cron remains untouched; comment out old watchdog/maintenance cron before moving
its trigger to systemd. `--remove-legacy-cron` no longer deletes entries.

## Verify

```bash
systemctl status learndl-watchdog.timer
sudo systemctl start learndl-watchdog.service
journalctl -u learndl-watchdog.service --since today --no-pager
bash runs/install_watchdog.sh --verify
```

The persistent watchdog state, job heartbeats, alert retry state, job durations,
and job errors are stored below the machine-specific runtime directory in
`task_watchdog/state.json`. The Monitor page reads this file without changing
task state.

## Add a short maintenance job

Register it in `src/api/task_monitor/watchdog.py` through `default_jobs()`.
Add its allowed ID to the scheduling configuration registry and configure its
interval/enabled flag in `runs/scheduling/maintenance.yaml`, then apply.
Each job must be idempotent, bounded (both item count and elapsed work), and
return JSON-safe statistics. The one-minute watchdog timer invokes due jobs;
their individual intervals live in the persistent state. A job failure is
recorded without preventing the lifecycle and alert jobs from running.

Jobs that are long-running, externally destructive, or unsafe to repeat should
remain their own systemd service/timer pair rather than joining the watchdog.

## Supervise persistent Learndl processes

Do not launch production services as children of WezTerm, Cursor, or the
interactive CLI. Give each service its own system unit and use a bounded restart
policy such as:

```ini
[Unit]
StartLimitIntervalSec=15min
StartLimitBurst=3

[Service]
User=mengkjin
WorkingDirectory=/home/mengkjin/workspace/learndl
ExecStart=/home/mengkjin/workspace/learndl/.venv/bin/python -m your.module
Restart=on-failure
RestartSec=30s
```

Run long-lived or side-effectful scheduled jobs as systemd service/timer pairs
instead of keeping them under an interactive terminal. Short, idempotent
maintenance should register with the watchdog. Jobs that are not safe to repeat
should alert on failure but should not use `Restart=on-failure`.

This local watchdog cannot report a complete host, power, or network outage.
For that case, use a dead-man heartbeat checked from another machine or service.
Task execution errors handled by Python / `BackendTaskRecorder` are intentionally
not watchdog alerts: their normal task error email is the single notification.
Watchdog task alerts are reserved for unexpected process loss (`killed`) and
installer-owned task timeouts. Daily/forced daily runs have a 12-hour limit and
weekly runs a 72-hour limit. Direct legacy cron is not timeout-protected.
# CLI recovery and reconstruction diagnostics

The `cli_recovery` watchdog job restores the registered **DirectCall Hub only**
on Linux/WezTerm. It never reruns a failed reconstruction. Install/update the
schedules as the ordinary desktop user:

```bash
bash runs/install_schedules.sh plan --diff
bash runs/install_schedules.sh apply
bash runs/install_schedules.sh status
```

Open the CLI once from the server's remote desktop after updating the code.
This records the desktop session and opts that project into recovery. `/quit`
or a normal menu exit clears recovery intent; closing the window or killing
the process leaves it enabled. A registered live Hub is not opened twice by
recovery. Explicitly opened additional menus still work and leave recovery
ownership with the registered Hub.
`/reload` hands off registration to the new process.

The installer also manages a project-specific file under
`~/.config/autostart/learndl-cli-*.desktop`. It refreshes desktop credentials
on the next graphical login, but never opts an unregistered project in.
Its contents participate in installation preview, status and rollback.
Recovery checks a live X11/Wayland login session, falling back to the previously
captured local display socket when `XDG_SESSION_ID` is absent or its type is
unspecified (common in remote desktops). It also requires a functioning
`systemd --user` manager; no display number is guessed. A disconnected but
still-running remote desktop can retain its session; a logged-out session
waits until login. The system watchdog dispatches a separate user service,
so finishing the watchdog cannot terminate the recovered window.

```bash
uv run --frozen python -m src.api.task_monitor.cli_recovery status
uv run --frozen python -m src.api.task_monitor.cli_recovery pause
uv run --frozen python -m src.api.task_monitor.cli_recovery resume
```

Failed launches back off; three failures pause recovery and create a retryable
email notification. Correct the desktop/user-bus problem before `resume`.
The normal check interval is 60 seconds plus watchdog runtime.

Each confirmed CLI reconstruction creates a Monitor task with source
`cli-reconstruct` and a unique virtual `_interactive/reconstruct-*.py` label
(not a runnable pipeline script). `both` records fit and predict separately.
Task artifacts live in `PATH.runtime/interactive_runs/<run-id>/`:

- `run.json`: inputs, dates, revision, host, process identity and sampler mode.
- `output.log`: continuously flushed Python stdout/stderr and traceback.
- `memory.jsonl`: independent two-second process-tree, system, swap and cgroup samples.
- `evidence.json`: bounded kernel/oomd journal evidence after an unexpected exit.
- `pending.json` / `notified.json`: durable failure notification delivery state.

The sampler tries a separate user service. If unavailable, it uses a detached
process and records that it may still share the terminal's cgroup. Missing
journal permissions and unsupported metrics are reported as unavailable,
not evidence that no OOM occurred. An ancestor cgroup OOM or whole-machine
failure can still kill both the worker and sampler; watchdog attempts to
recover journal evidence afterwards. Buffered native-library output is not
guaranteed to survive SIGKILL.

Failure emails include a bounded log tail and artifact paths, not multi-GB
attachments. Success is recorded without sending email. Failed deliveries
are retried by watchdog. Closed logs are retained for 30 days; live runs and
unsent failure notices are never pruned. Existing scheduled script notification
ownership is unchanged. This release does not optimize the reconstruction
algorithm, restart tasks, or provide computation checkpoints.

Server acceptance: close/kill a disposable Hub, confirm exactly one replacement;
verify `/quit` stays closed; log out/in with a pending recovery; kill a tiny
reconstruction and verify Monitor history, persisted output and one failure
email. Test small inputs before the full minc/mincr rebuild. GUI/user-service
behavior requires Linux acceptance even when local mocked tests pass.

Recovery launch output is appended to `runtime/cli_recovery/recovery.log`.
The state file records the latest status, check time and recovery unit name.
Existing nonempty watchdog job configurations automatically include CLI recovery
unless it is explicitly disabled; a CLI must still register recovery intent.
