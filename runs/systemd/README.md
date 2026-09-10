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
bash runs/install_watchdog.sh --remove-legacy-cron
```

To monitor persistent system services too, provide a comma- or space-separated
list during installation:

```bash
LEARNDL_WATCHDOG_UNITS="learndl-app.service learndl-worker.service" \
  bash runs/install_watchdog.sh
```

Re-run the installer whenever this list, the project location, or the systemd
templates change. `--remove-legacy-cron` is explicit because it edits the
current user's crontab; use it once to remove the old monitor-maintenance cron
entry after upgrading.

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
