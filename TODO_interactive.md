# TODO — `src/interactive` Improvement List

Identified during the documentation pass (2026-04-15). Grouped by layer.
Items are ordered roughly by impact; none are blockers.

---

## Backend

### `backend/task.py`
- **Busy-wait polling loop**: `wait_until_completion` sleeps 1 s per iteration.
  Consider using `asyncio` or `threading.Event` to avoid burning a thread. **may use watchdog**
- **No DB indexes**: the SQLite task table has no indexes on `script` or
  `status` columns; full-table scans become slow as the database grows. **already add**
- **Full queue always loaded**: `TaskQueue` loads every row on each rerun.
  Introduce server-side pagination at the DB level for large histories.
- **`check_killed` heuristic**: relies on a crash-protector file pattern —
  could produce false negatives if the file is not cleaned up correctly.
- **Ad-hoc status strings**: task status transitions use bare string literals
  (`'running'`, `'complete'`, `'error'`, `'killed'`).  A formal `Enum` or
  state-machine would make invalid transitions detectable at type-check time.

### `backend/recorder.py`
- **`parse_kwargs` uses `eval()`**: type-coercion of kwargs values is done via
  `eval()`.  If kwargs ever originate from untrusted input this is a security
  risk.  Replace with a typed parser or `ast.literal_eval` where possible. **already done**

### `backend/script.py`
- **Duplicated `format_path` logic**: `PathItem.format_path` and
  `ScriptRunner.format_path` share the same implementation; extract a shared
  helper to avoid drift. **already done**
- **Misleading existing docstrings**: the original `build_task` and
  `preview_cmd` docstrings said "return exit code" — they actually return
  `TaskItem` / `str` respectively (fixed in the doc pass).

---

## Frontend

### `frontend/frontend.py`
- **`YAMLFileEditor` singleton leak**: instances are keyed on an arbitrary
  string and stored in a class-level dict with no cleanup on page change.
  Long-running sessions accumulate stale `YAMLFileEditorState` objects. **no need to change**

### `frontend/param_input.py`
- **Fragile command-string parsing**: `ParamInputsForm.cmd_to_param_values`
  restores parameter values by regex-parsing the rendered command string.
  If the command format changes the restoration silently fails. **no need to change**

### `frontend/param_cache.py`
- **Partial overlap with `st.session_state`**: `ParamCache` manually mirrors
  some of what Streamlit's session state already provides.  Consider unifying
  to reduce duplication. **no need to change**

### `frontend/logo.py`
- **Runtime font download**: fonts are fetched from `dafont.com` at startup.
  This creates a network dependency and is fragile in air-gapped environments.
  Bundle the font files in the repo or cache them locally on first download. **already doing that**

---

## Main

### `main/util/control.py`
- **Queue refresh on every callback**: `universal_action` calls
  `TaskQueue.refresh()` after every single UI callback, even for lightweight
  actions.  Batch or debounce the refresh for better performance. **that is exactly what i plan for**
- **`ControlGitClearPullButton` platform check**: `MACHINE.platform_coding`
  is evaluated via a string comparison on `MACHINE.name`.  A proper capability
  flag (e.g. `MACHINE.supports_git_pull: bool`) would be more robust. **too much**

### `main/util/script_detail.py`
- **Typo in function name**: `direclty_open_file` → `directly_open_file`.
  The typo is preserved for backwards compatibility with existing call sites
  inside this module, but should be fixed with a deprecation alias. **altered**

### `main/util/page.py`
- **Side effect at module load**: `make_script_detail_file` (called from
  `script_pages`) generates `.py` files when the module is first imported.
  This is an import-time side effect that complicates testing and cold starts. **done**

### `main/util/navigation.py` / `main/util/control.py`
- **Hard-coded magic constants**: page size 500, queue timeout 20 s, max queue
  100 are scattered as literals.  Move them to `configs/` or `CONST.Pref`.

### `main/pages/` (generated wrappers)
- **22 near-identical page files**: each wrapper is a 7-line stub.  A single
  dynamic page that reads `script_key` from URL query params would eliminate
  all generated files and the `make_script_detail_file` side effect. **done**

---

## Cross-cutting

- **`Any` overuse**: many return types are annotated `Any` or left unannotated;
  tighten with concrete types as Streamlit's stubs improve.
- **`ActionLogger` not wired up**: `ActionLogger` is defined and imported but
  all call sites are commented out.  Either wire it in or remove the dead code. **done**
- **Session state key collisions**: bare string keys (`'choose-task-page'`,
  `'task-filter-status'`, etc.) are duplicated across `task_queue.py` and
  `script_detail.py`.  A central registry or `NamedTuple` of key constants
  would prevent silent collisions. 
