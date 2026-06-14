# API Layer — `src/api/`

**Purpose:** Unified public surface for the learndl project. Merges the former
`src/api/` facades, `src/call/` direct operations, and `src/interactive/`
Streamlit app into one package with four sub-areas.

**Depends on:** [[project_infra]], [[data_pipeline]], [[factor_engine]], [[nn_models]], [[trading]]

---

## Package Layout

```
src/api/
├── pkgs/                 # Research API facades (stable script/import surface)
│   ├── data.py           # DataAPI
│   ├── factor.py         # FactorAPI
│   ├── model.py          # ModelAPI
│   ├── trading.py        # TradingAPI
│   ├── summary.py        # SummaryAPI
│   ├── update.py         # UpdateAPI
│   ├── notification.py   # NotificationAPI
│   ├── dashboard.py      # DashboardAPI, TSBoardAPI, OptunaDBAPI
│   └── test.py           # TestAPI
├── calls/                # DirectCall operations (formerly src/call/)
│   ├── app.py            # LaunchApp, KillAndRebootApp
│   ├── computer.py       # Machine / environment helpers
│   ├── data.py           # Data pipeline one-offs (e.g. ReconstructPreprocessedData)
│   ├── files.py          # Git, model archive, log cleanup, WezTerm config
│   ├── research.py       # Schedule training lists, config validation
│   └── source_code.py    # Dependency checks, code hygiene
├── interactive/          # Streamlit app (see [[interactive_app]])
│   ├── launch.py         # App entry point
│   ├── pages/            # Intro pages (home, task_queue, api_console, …)
│   ├── util/             # Session control, navigation, control panel, quick calls
│   └── frontend/         # App-specific UI (logo, style, yaml editor)
└── util/                 # Shared API infrastructure
    ├── backend/          # ScriptRunner, TaskDatabase, TaskQueue, BackendTaskRecorder
    ├── st_frontend/      # Reusable Streamlit widgets (ParamInputsForm, YAML editor, …)
    ├── contract.py       # APIEndpoint discovery, [API Interaction] schema
    ├── direct_call.py    # DirectCall ABC
    └── wrapper.py        # wrap_update / print_update_records for pkgs
```

---

## Import Conventions

| Use case | Import |
|----------|--------|
| Pipeline scripts & notebooks | `from src.api.pkgs import DataAPI, ModelAPI, …` |
| One-off CLI / UI quick actions | `from src.api.calls.files import ArchiveCurrentModel` |
| Launch Streamlit from shell | `uv run launch.py` → `src.api.calls.app.LaunchApp` |
| API console / endpoint adapter | `from src.api.util import APIEndpoint` |

> **Migration note:** Old paths `from src.api.data import DataAPI` and
> `from src.api.model import ModelAPI` are replaced by `from src.api.pkgs import …`.

---

## `pkgs/` — Research Facades

Thin wrappers over `src/data/`, `src/res/`, and related internals. Exported from
`src.api.pkgs`:

| Module | Class | Role |
|--------|-------|------|
| `data.py` | `DataAPI` | Data download, preprocessing, risk-model update |
| `factor.py` | `FactorAPI` | Factor calculate / update / test / normalize |
| `model.py` | `ModelAPI` | Train, schedule, test, predict |
| `trading.py` | `TradingAPI` | Tracking update, backtest, analyze |
| `summary.py` | `SummaryAPI` | Account and model performance summaries |
| `update.py` | `UpdateAPI` | Orchestrated multi-step update pipeline |
| `notification.py` | `NotificationAPI` | Email / alert dispatch |
| `dashboard.py` | `DashboardAPI`, `TSBoardAPI`, `OptunaDBAPI` | TensorBoard, Optuna, dashboards |

Methods decorated with a `[API Interaction]:` docstring block (see `util/contract.py`)
can be discovered and run from the interactive **API Console** page.

---

## `calls/` — DirectCall Operations

Subclass `DirectCall` (`util/direct_call.py`) for imperative, often interactive
operations launched from the UI quick-call buttons or `DirectCall.go()`:

```python
from src.api.calls.app import LaunchApp
LaunchApp.go()   # opens Streamlit via Shell.open
```

| Module | Examples |
|--------|----------|
| `app.py` | `LaunchApp`, `KillAndRebootApp` |
| `files.py` | `ArchiveCurrentModel`, `ClearOutdatedCatcherLogs`, `ReplaceWeztermConfig` |
| `research.py` | `CarryOutScheduleModelList`, `CheckAllConfigFiles` |
| `data.py` | `ReconstructPreprocessedData` |
| `source_code.py` | `CheckDependencyVersion`, `CheckCodeIssues` |
| `computer.py` | Machine-specific helpers |

---

## `util/` — Shared Infrastructure

### `util/backend/`

Moved from the former `src/interactive/backend/`. Used by both the Streamlit app
and API-adapter scripts.

| Class | File | Role |
|-------|------|------|
| `PathItem` | `script.py` | Script/folder discovery under `PATH.scpt` |
| `ScriptHeader` | `script.py` | YAML header parsed from script comment block |
| `ScriptRunner` | `script.py` | Build tasks, preview commands, expose params |
| `TaskDatabase` | `task.py` | SQLite persistence for tasks |
| `TaskQueue` | `task.py` | In-memory ordered task list with filters |
| `TaskItem` | `task.py` | Single subprocess task (run, kill, status) |
| `BackendTaskRecorder` | `recorder.py` | Capture stdout/stderr of subprocesses |

### `util/contract.py` — API Endpoint Contract

Parses `[API Interaction]:` YAML blocks in docstrings. Enforces roles, risk
level, lock numbers, platform disables, execution time, and memory usage.
`APIEndpoint.iter_with_schema(exposed=True)` drives the API Console browser.

### `util/st_frontend/`

Reusable Streamlit primitives shared by the interactive app and endpoint forms:
`ParamInputsForm`, `YAMLFileEditor`, `FilePreviewer`, `SacBoundButton`, etc.

---

## Entry Points

| Command | Target |
|---------|--------|
| `uv run launch.py` | `LaunchApp` → `streamlit run src/api/interactive/launch.py` |
| `uv run streamlit run src/api/interactive/launch.py` | Direct Streamlit entry |
| `scripts/.core/0.run_api_endpoint.py` | In-process / queued execution of `APIEndpoint` callables |

---

## Design Patterns

| Pattern | Where |
|---------|-------|
| Facade | `pkgs/*` — stable surface over `src/res/` and `src/data/` |
| DirectCall | `calls/*` — imperative ops with `go()` / `run()` |
| Endpoint contract | `[API Interaction]` docstrings + `APIEndpoint` discovery |
| Singleton | `SessionControl` (`SC`), `get_cached_task_db` in interactive app |
| `@ScriptTool` bridge | Scripts run via `TaskItem` → same queue as interactive UI |
