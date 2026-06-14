# Interactive App — `src/api/interactive/`

**Purpose:** Streamlit-based web UI for managing and monitoring the learndl ML
pipeline. Users discover scripts, configure parameters, submit tasks to a
SQLite-backed queue, browse/run `src.api.pkgs` endpoints, and view live/historical
run reports.

**Depends on:** [[api_layer]], [[project_infra]]

---

## Entry Points

```bash
uv run launch.py
# or
uv run streamlit run src/api/interactive/launch.py
```

`launch.py` validates the working directory, optionally starts an auto-refresh
timer, then calls `page_setup()` which assembles navigation and runs the active
page.

---

## Module Structure

```
src/api/interactive/
├── launch.py              # App entry point (calls page_setup)
├── pages/                 # Intro / utility pages
│   ├── home.py
│   ├── developer_info.py
│   ├── config_editor.py
│   ├── task_queue.py
│   ├── api_console.py     # Browse & run [API Interaction] endpoints
│   └── script_structure.py
├── util/                  # App logic
│   ├── session_control.py # SessionControl singleton (SC)
│   ├── navigation.py      # page_config, top_navigation, page_setup
│   ├── page.py            # Intro/script page registry, auto-generated stubs
│   ├── script_detail.py   # Per-script detail view (params, history, report)
│   ├── api_adapter.py     # stAPIEndpoint — API Console UI
│   ├── style.py           # Custom CSS injection
│   ├── control_panel/     # Top action bar (run, git-pull, refresh, popovers)
│   ├── quick_calls/       # QuickCallButton subclasses → src.api.calls.*
│   └── components/        # Task queue, param forms, intro panels
└── frontend/              # App-specific UI assets
    ├── logo.py
    ├── style.py
    └── components/        # yaml_editor, file_previewer, widgets, …
```

Shared backend and reusable widgets live under `src/api/util/` (see [[api_layer]]):

- `util/backend/` — `ScriptRunner`, `TaskDatabase`, `TaskQueue`, `BackendTaskRecorder`
- `util/st_frontend/` — `ParamInputsForm`, `YAMLFileEditor`, `FilePreviewer`, …

---

## Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `SessionControl` (`SC`) | `util/session_control.py` | Owns task queue, script runners, param cache, click handlers |
| `ScriptRunner` | `util/backend/script.py` | Runnable script facade; `build_task`, `preview_cmd`, header/param access |
| `ScriptHeader` | `util/backend/script.py` | YAML header from script comment block |
| `TaskDatabase` | `util/backend/task.py` | SQLite wrapper: task/queue CRUD |
| `TaskQueue` | `util/backend/task.py` | In-memory ordered `TaskItem` dict with filters |
| `TaskItem` | `util/backend/task.py` | Single task: status, cmd, PID, run/kill |
| `BackendTaskRecorder` | `util/backend/recorder.py` | Captures subprocess stdout/stderr |
| `ControlPanel` | `util/control_panel/panel.py` | Horizontal action bar on every page |
| `QuickCallButton` | `util/quick_calls/basic.py` | Base for buttons that invoke `src.api.calls.*` |
| `stAPIEndpoint` | `util/api_adapter.py` | Streamlit wrapper over `APIEndpoint` for API Console |

---

## Intro Pages

| Page key | Module | Purpose |
|----------|--------|---------|
| `home` | `pages/home.py` | Tutorial, system info, quick links |
| `developer_info` | `pages/developer_info.py` | Session state / queue JSON inspector |
| `config_editor` | `pages/config_editor.py` | YAML editor for model/algo configs |
| `task_queue` | `pages/task_queue.py` | Full task queue with filters and inline reports |
| `api_console` | `pages/api_console.py` | Browse and run exposed `pkgs` endpoints |
| `script_structure` | `pages/script_structure.py` | Script tree navigator |

Script detail pages are auto-generated stubs under `pages/_<group>_<script>.py`
(created by `page.make_script_detail_file` on first access).

---

## Data Flow

```
Script discovery (PathItem.iter_folder under PATH.scpt)
        │
        ▼
page.py — script_pages()           ← caches st.Page objects per script key
        │
        ▼
navigation.py — page_setup()       ← MPA navigation + sidebar links
        │
        ▼
pages/_<group>_<script>.py        ← auto-generated stub → show_script_detail()
        │
        ▼
script_detail.py
 ├── show_task_history()          ← historical TaskItem list
 ├── show_param_settings()        ← ParamInputsForm + YAML editor
 └── show_report_main()           ← live/completed report, file previewer
        │  (on Run click)
        ▼
SessionControl.click_script_runner_run()
  └── ScriptRunner.build_task()   ← creates TaskItem in TaskDatabase
      └── TaskItem.run_script()   ← subprocess + BackendTaskRecorder
```

**API Console flow:** `api_adapter.stAPIEndpointList` discovers exposed endpoints
via `APIEndpoint.iter_with_schema`, renders parameter forms, and dispatches
through `scripts/.core/0.run_api_endpoint.py` (in-process or queued).

**Quick calls:** `util/quick_calls/buttons.py` subclasses `QuickCallButton` to
invoke `src.api.calls.*` (e.g. `LaunchApp`, `CheckAllConfigFiles`,
`ArchiveCurrentModel`, `DashboardAPI`).

---

## Design Patterns

| Pattern | Where |
|---------|-------|
| **Singleton** | `SessionControl` (`SC`), `get_cached_task_db` (`@st.cache_resource`) |
| **Decorator** | `queue_refresh_trigger`, `BackendTaskRecorder.__call__` |
| **Factory** | `ScriptRunner.from_key`, `PathItem.from_path`, `TaskItem.create` |
| **Registry** | `QuickCallButtonMeta.registry` for quick-call button subclasses |
| **Streamlit fragments** | `@st.fragment(run_every=…)` for backend queue polling |

---

## External Dependencies

| Dependency | Used for |
|------------|----------|
| `streamlit` | All UI rendering |
| `streamlit-autorefresh` | Optional periodic page refresh |
| `streamlit-antd-components` | Sidebar navigation, expanders |
| SQLite (via `DBConnHandler`) | `TaskDatabase` persistence |
| `src.proj.Options` | Dynamic dropdowns (`options_cache.json`) |
| `src.proj.Const.Pref` | `configs/preference/interactive.yaml` |
| `src.proj.MACHINE` | Platform detection (git-pull, file-open) |
| `src.proj.PATH` | Script root, config paths |

---

## Notes

- Regenerate script stub pages after adding/removing scripts: use the ↺ refresh
  button (`ControlRefreshInteractiveButton`) or `remake_all_script_detail_files()`.
- Hidden from the app: scripts/dirs starting with `.` or `_`, and `scripts/_miscellaneous/`.
- The legacy `main/` subtree under `interactive/` is unused; canonical paths are
  `util/` and `pages/`.
