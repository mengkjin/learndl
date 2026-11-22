---
name: learndl-base-modules
description: >-
  Routes agents to learndl base packages under `src/proj` (paths, DB, calendar, logging, Proj facade)
  and `src/func` (tensor/factor ops, numpy/pandas helpers). Use when editing or importing `proj` or
  `func`, configuring environments, logging, trade calendars, or shared numerics—not `src/res`
  strategy/factor pipelines or `src/data` loaders.
---

# learndl base libraries (`proj` / `func`)

Use this file as a route map only; behavior and APIs live in source and module docstrings.

## Project logging convention

- Do not add direct `print(...)` calls to application code or scripts. For plain
  console output use `from src.proj import Logger` and `Logger.stdout(...)`.
  Reuse an existing task/bound Logger when available; use its warning/error or
  other level methods when the message needs that meaning.
- Preserve flushing, line endings, output capture and machine-readable output.
  Use `vb_level=0` for required CLI results, explicit diagnostics and supervised
  progress that must survive ordinary verbosity filtering. `Logger.stdout(file=...)`
  takes a path, not an open stream; use output redirection for an existing stream.
- Keep imports lazy where needed to avoid circular imports or changing bootstrap
  behavior. Automation code that runs before project initialization/code-lock
  admission uses the standard-library Logger in `src/automation/console.py`.
- Direct `print` is appropriate only in tests/fixtures intentionally exercising
  third-party stdout/stderr capture. File serialization is not console logging.
- Before finishing Python changes, inspect direct `print` calls in changed files
  and convert ordinary diagnostics to the appropriate Logger.

## Router table

| Need | Open first |
|------|------------|
| Proj singleton, cwd, instance registry | `src/proj/env/variable/proj.py`, `src/proj/env/variable/ins.py` |
| Paths, data roots, config roots | `src/proj/env/path.py`, `src/proj/env/machine.py` |
| Logging, LogFile, file logging | `src/proj/log/logger.py`, `src/proj/log/logfile.py`, `src/proj/env/variable/files.py` |
| Trade calendar, offsets, business days | `src/proj/calendar/calendar.py`, `trade_date.py`, `basic.py` |
| data interface | `src/proj/db/interface.py`, `io/saver.py`, `io/loader.py` |
| Default configs (factor, model, trading, …) | `src/proj/env/constant/conf/` |
| HTTP, device, email, SQLite, misc | matching modules under `src/proj/util/` |
| Process pools, proxy/cache callers | `src/proj/util/proxy/` |
| Script locks, script lifecycle, CLI helpers | `src/proj/util/script/` |
| Watchdog, scheduling, autofix, installation | `src/automation/`; use `learndl-automation` skill |
| Factor tensors, rolling, neutralize, rank | `src/func/tensor.py` |
| Index align, fillna, `index_merge` | `src/func/basic.py` |
| NumPy/pandas transforms, OLS, cov | `src/func/transform.py` |
| Torch metrics, IC, weighted corr | `src/func/metric.py` |
| Symmetric orthogonalization | `src/func/linalg.py` |
| Package exports | `src/proj/__init__.py`, `src/func/__init__.py` |

## `src/proj` layout

- **`env/`**: `MACHINE`, `PATH`, `CONST`, `Proj` (machine and directory conventions).
- **`log/`**: `Logger`, `LogFile`; prefer `src.proj.Logger` in app code.
- **`cal/`**: `CALENDAR`, `TradeDate`, etc.—start here for session/backtest dates.
- **`db/`**: low-level access and mappers; not the same as `src/data` vendor loaders.
- **`core/`**: project logging controls, output helpers and trading types.
- **`util/`**: general utilities—**not** `src/func`.
- **`util/func/`**: engineering helpers (parallel, disk, shell)—**not** the math package `src/func`.

## Lightweight shared utilities

- `src/shared/file_ops.py`: atomic file publication and directory sync.
- `src/shared/resource/memory.py`: process memory measurements, cgroup headroom and budget calculation. Data block allocation policy lives in `src/data/util/memory_guard.py`.
- `src/shared/duration.py`, `dynamic_import.py`, `cached_properties.py`, `singleton.py`, `filtered.py`: standalone timing, module loading and Python primitives.

## `src/func` layout

- **`tensor.py`**: largest module; `TsRoller`, `process_factor`, `neutralize_*`, `rank`, `corrwith`, `ts_*`.
- **`basic.py`**: `DIV_TOL`, `allna`, index/`intersect_*` helpers.
- **`transform.py`**: NumPy/pandas/statsmodels-oriented.
- **`metric.py`**: torch-oriented metrics, `ic` / `rankic` variants.
- **`linalg.py`**: `symmetric_orth` (torch and numpy).

## Easy to confuse

- **`src/func`**: shared quant math (`from func import ...` / `import func.tensor`).
- **`src/proj/util/func`**: project utilities; unrelated despite the name.

## Outside this skill

- Pipelines and business logic: mostly `src/data`, `src/res/`.
- Longer operator docs: `resources/instructions/` (e.g. `uv_instruction.md`).

## How to use

1. Pick a row from the router table, open the file(s), read the module docstring at the top.
2. This copy lives under `.cursor/skills/` so Cursor can discover it as a project skill.
