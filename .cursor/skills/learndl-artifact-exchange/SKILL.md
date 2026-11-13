---
name: learndl-artifact-exchange
description: >-
  Integrate Syncthing artifact delivery when creating or changing learndl research
  scripts, model training or analysis outputs; troubleshoot publication, local
  archival and experiment-panel sources. Excludes autofix sandbox changes.
---

# Research artifacts and local archival

Read [operator and API examples](../../../runs/artifact_exchange/README.md) before adding a producer.
Implementation: `src/automation/artifact_exchange/`; panel discovery: `src/api/learndl_monitor/experiments.py`.

## Producer contract

- Choose a stable research `category`, an explicit `experiment` identity, and a unique `run_id`.
  Model training uses the full model name, not just architecture or a filename.
- Register pure analysis directories with `ArtifactSpec`. With no `members`, all ordinary
  files are copied, regardless of extension. Use `destination` only for a batch-relative directory.
  Never register a whole model root, raw dataset, checkpoint tree or weight-containing bundle.
- A batch groups all roles of one category/experiment/run. Reports go into `results/`,
  TensorBoard into `tensorboard/*.zip`, SQLite backup snapshots into `optuna/*.zip`.
  Custom research categories are supported; do not make reports/tensorboard/optuna top-level types.
- Use an existing `AutoRunTask.register_artifact(spec)` for normal task completion; use
  `research.publish_safe(...)` for independent scripts and completed failure diagnostics.
  Flush asynchronous saves and close writers before publication. Never infer completion from file age.
- Model trainers already use `research.model_run` inside their lifecycle lock. Extend that
  collector rather than publishing again from each report callback. It snapshots the complete
  model `rslt()` directory and co-locates captured Optuna/TensorBoard artifacts.
- Preserve original research errors and SMTP behavior when delivery fails. Successful package
  handoff removes only the associated analysis attachments, retaining logs/execution reports.
  Do not rewrite historical mail queues. In autofix contexts skip external delivery entirely.

## Receiver and source contract

Syncthing shares `$HOME/workspace/Syncthing`; this feature consumes only completed packages
under its `artifact_exchange/` child. Ordinary files placed there are not ready notifications:
manual materials must also use the publish CLI. Keep state and temporary files outside the share.

Archive under `results/<category>/<source_host>__<experiment>/<batch>/`. Use the configured
stable source_host, not a fresh Wi-Fi-dependent hostname. Server originals remain; only a checked,
committed package with durable receipt is removed. Existing protocol v1 packets and legacy panel
ZIPs remain supported. Do not move historical directories when adding a new research producer.

The experiment panel advertises only TensorBoard/Optuna payloads identified by committed
source.json manifests; a report ZIP is not an experiment. Reuse the importer and shared capacity limits.

## Verification

Test a mixed-role round trip, source retention, shared-package removal, two hosts with the same
experiment, disabled/autofix behavior and publication failure. For panel changes verify both old
and new ZIP discovery/import. Use temporary paths and mocked notifications, never send test email.

Maintain this tracked skill in `.cursor/skills` and synchronize `.agents/skills` for local discovery.
