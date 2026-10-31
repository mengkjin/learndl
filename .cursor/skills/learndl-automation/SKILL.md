---
name: learndl-automation
description: >-
  Develop and troubleshoot learndl watchdog, automatic scheduling, worker channels,
  autofix, Git maintenance, CLI recovery and scheduling installation. Use for automation
  control or configuration changes; not model training algorithms or monitor UI layout.
---

# Learndl automation control

## Find the implementation

- `src/automation/watchdog.py`: bounded lifecycle inspection and dispatch.
- `src/automation/cli_recovery.py`: recovery of explicitly registered desktop CLI sessions.
- `src/automation/scheduling/`: calendar lanes, automatic research queue, admission,
  process supervision, autofix, Git updates and transactional installation.
- `src/automation/paths.py`: checkout and configuration locations. Runtime-state
  paths remain owned by the existing stores; do not derive them from module depth.
- `configs/automation/`: task registration, schedules, maintenance, hosts and
  `autofix.yaml`. Read `README.md` for operations and `AUTOFIX.md` for repair requests.

Monitor UI, queries and output caching stay under `src/api/task_monitor`.
Former control modules there only forward to `src/automation`; never add another
implementation to a compatibility module. Shared mail, logging and script locks
remain in `src/proj`; the task database remains in `src/api/util/backend`.
Training worklists, model schedules and RL configurations are not automation
configuration files and retain their existing locations.

## Preserve operational identities

- Daily/weekly/RCQuant lanes are independent; automatic research is single-lane
  across its full process lifecycle. Consult `channels.py` for maintenance scope.
- Blocking autofix execution and maintenance admission are separately bounded;
  do not release a lease while surviving child processes still hold it.
- Managed launchers acquire the code read lock before business imports. Git
  checkout needs exclusive access and subsequent fresh-process verification.
- Successful one-shot request IDs never rerun because of a rename or config move.
  Preserve declaration fingerprints, runtime directories, service names, ledgers,
  lock paths and notification IDs when reorganizing modules.
- Compatibility imports must resolve to the same module/state objects, especially
  maintenance context variables. Keep old installed startup commands working until
  explicit installer apply replaces them.

## Validate and deploy

Run relevant tests in `src/automation/tests`; for monitor interactions also run
`src/api/task_monitor/tests`. Migration coverage is in `test_migration.py`.
Tests use temporary ledgers and mocked notifications, never real production IDs.
Linux isolation/service tests are opt-in; macOS passes do not verify systemd/Bubblewrap.

Use `bash runs/install_schedules.sh plan --host <host> --diff`, then the authorized
`apply` and `status` workflow. A code pull or watchdog restart does not publish
configuration or replace the installed bootstrap. Do not initialize a missing
production ledger as a repair shortcut. Keep rollback and old snapshot validation.

This tracked skill is maintained in `.cursor/skills`; synchronize the local
`.agents/skills` discovery copy when editing it. Source and operator documentation
remain authoritative for current options rather than duplicating their schemas here.
