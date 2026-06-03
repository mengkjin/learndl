# Interactive App — `src/interactive`

**Purpose:** Streamlit-based web UI for managing and monitoring the learndl ML
pipeline.  Users discover scripts, configure parameters, submit tasks to a
SQLite-backed queue, and view live/historical run reports — all from a browser.

**Depends on:** `src/proj` (PATH, MACHINE, Logger, Options, CONST), SQLite
(via `DBConnHandler`), Streamlit, `streamlit-autorefresh`.

---

## Entry Point

```bash
uv run streamlit run src/interactive/main/launch.py
```

`launch.py` validates the working directory, optionally starts an
auto-refresh timer, then calls `page_setup()` which assembles navigation and
runs the active page.

---

## Module Structure

```
src/interactive/
├── backend/
│   ├── recorder.py      # BackendTaskRecorder — captures stdout/stderr of subprocesses
│   ├── script.py        # PathItem, ScriptHeader, ScriptParamInput, ScriptRunner
│   └── task.py          # TaskDatabase (SQLite), TaskQueue, TaskItem
├── frontend/
│   ├── frontend.py      # ActionLogger, FilePreviewer, YAMLFileEditor, ColoredText, helpers
│   ├── logo.py          # SVG logo / banner generation
│   ├── param_cache.py   # ParamCache — persists widget values across reruns
│   ├── param_input.py   # WidgetParamInput, ParamInputsForm — dynamic parameter form
│   └── style.py         # CustomCSS — injects custom Streamlit CSS
└── main/
    ├── launch.py         # App entry point
    ├── util/
    │   ├── control.py        # SessionControl (SC), ControlPanel, ControlPanelButton subclasses
    │   ├── navigation.py     # page_config, top_navigation, custom_sidebar_navigation, page_setup
    │   ├── page.py           # intro_pages, script_pages, make_script_detail_file, print_page_header
    │   └── script_detail.py  # show_script_detail and component show_* functions
    └── pages/
        ├── home.py           # Tutorial, system info, pending-features banner
        ├── developer_info.py # Session state / queue JSON inspector
        ├── config_editor.py  # YAML file editor for model/algo configs
        ├── task_queue.py     # Full task queue with filters and inline reports
        └── _<group>_<script>.py  # 22 auto-generated wrapper pages (one per script)
```

---

## Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `ScriptRunner` | `backend/script.py` | Represents a runnable script; provides `build_task`, `preview_cmd`, header/param access |
| `ScriptHeader` | `backend/script.py` | YAML header parsed from a script file (`## params:`, `## disabled:`, etc.) |
| `ScriptParamInput` | `backend/script.py` | Schema for a single script parameter (type, title, validation) |
| `TaskDatabase` | `backend/task.py` | SQLite wrapper: task/queue CRUD, backup/restore |
| `TaskQueue` | `backend/task.py` | In-memory ordered dict of `TaskItem`; filter, sort, latest helpers |
| `TaskItem` | `backend/task.py` | Single task record: status, cmd, PID, exit files, run/kill methods |
| `BackendTaskRecorder` | `backend/recorder.py` | Context manager / decorator that captures subprocess output into a `TaskItem` |
| `ParamInputsForm` | `frontend/param_input.py` | Builds Streamlit widgets from `ScriptParamInput` schemas; validates and returns param dict |
| `YAMLFileEditor` | `frontend/frontend.py` | Singleton per key; shows a text-area YAML editor with load/validate/save |
| `SessionControl` | `main/util/control.py` | Dataclass singleton (`SC`) — owns task queue, script runners, param cache, all click handlers |
| `ControlPanel` | `main/util/control.py` | Horizontal action bar: run, latest-task, refresh, git-pull buttons + settings popover |

---

## Data Flow

```
Script discovery (PathItem.iter_folder)
        │
        ▼
page.py — script_pages()          ← caches st.Page objects per script key
        │
        ▼
navigation.py — page_setup()      ← assembles MPA navigation + sidebar links
        │
        ▼
main/pages/_<group>_<script>.py   ← auto-generated stub, calls show_script_detail()
        │
        ▼
script_detail.py
 ├── show_script_task_selector()  ← renders historical TaskItem list with pagination
 ├── show_param_settings()        ← ParamInputsForm + optional YAML editor/previewer
 └── show_report_main()           ← live/completed report, exit files, file previewer
        │  (on Run click)
        ▼
SessionControl.click_script_runner_run()
  └── ScriptRunner.build_task()   ← creates TaskItem in TaskDatabase
      └── TaskItem.run_script()   ← launches subprocess, BackendTaskRecorder captures output
```

---

## Design Patterns

| Pattern | Where |
|---------|-------|
| **Singleton** | `SessionControl` (module-level `SC`), `YAMLFileEditor._instances`, `get_cached_task_db` (`@st.cache_resource`) |
| **Decorator** | `universal_action` (refreshes queue after every callback), `clear_and_show` (clears placeholder before re-render), `BackendTaskRecorder.__call__` |
| **Factory** | `ScriptRunner.from_key`, `PathItem.from_path`, `TaskItem.create` |
| **Strategy** | `ControlPanelButton` ABC + concrete subclasses (run, latest, refresh, git-pull) |
| **Streamlit fragments** | `@st.fragment` / `@st.cache_resource` used in backend for performance |

---

## External Dependencies

| Dependency | Used for |
|------------|---------|
| `streamlit` | All UI rendering |
| `streamlit-autorefresh` | Optional periodic page refresh |
| SQLite (via `DBConnHandler`) | `TaskDatabase` persistence |
| `src.proj.Options` | Reading/refreshing project options on refresh |
| `src.proj.CONST.Pref` | App preferences (`page_title`, `navigation_position`, `auto_refresh_interval`) |
| `src.proj.MACHINE` | Platform detection (git-pull availability, file-open command) |
| `src.proj.PATH` | Script discovery root, config paths |

---

## Notes

- Script wrapper pages (`_*.py`) are auto-generated by `make_script_detail_file`
  at first access.  Run `ControlRefreshInteractiveButton.refresh_all()` (the ↺
  button in the UI) or call `remake_all_script_detail_files()` to regenerate
  after adding/removing scripts.
- See `TODO_interactive.md` at the project root for known improvement areas.
