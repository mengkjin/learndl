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

## Router table

| Need | Open first |
|------|------------|
| Proj singleton, cwd, instance registry | `src/proj/env/variable/proj.py`, `src/proj/env/variable/ins.py` |
| Paths, data roots, config roots | `src/proj/env/path.py`, `src/proj/env/machine.py` |
| Logging, LogFile, file logging | `src/proj/log/logger.py`, `src/proj/log/logfile.py`, `src/proj/env/variable/files.py` |
| Trade calendar, offsets, business days | `src/proj/calendar/calendar.py`, `trade_date.py`, `basic.py` |
| Raw DB, data interface, mmap | `src/proj/db/data_interface.py`, `memory_map.py`, `code_mapper.py` |
| Default configs (factor, model, trading, …) | `src/proj/env/constant/conf/` |
| HTTP, device, email, SQLite, misc | matching modules under `src/proj/util/` |
| Process pools, proxy/cache callers | `src/proj/util/proxy/` |
| Script locks, scheduling, CLI helpers | `src/proj/util/script/` |
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
- **`core/`**: `Silence`, `Duration`, `singleton`, small shared abstractions.
- **`util/`**: general utilities—**not** `src/func`.
- **`util/func/`**: engineering helpers (parallel, disk, shell)—**not** the math package `src/func`.

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
