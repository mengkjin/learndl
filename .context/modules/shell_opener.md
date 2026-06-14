# Shell Opener — `src/proj/util/shell/`
**Purpose:** Cross-platform terminal launcher (`Shell.open`, `Shell.run`) and per-OS opener backends (WezTerm, cmd, cmux, …).
**Key source paths:** `src/proj/util/shell/`
**Depends on:** `src/proj/env` (preferences), `src/proj/log`

---

## Public API

```python
from src.proj.util.shell import Shell

Shell.open(cmd, done_action="pause", cwd=..., title=..., new_on="tab")
Shell.open_py("scripts/foo.py", title="foo.py")
Shell.run(cmd)  # background, no terminal window
```

- `cmd` should be an **argv list** on Windows when possible; `to_shell_string()` uses `subprocess.list2cmdline`.
- `compose_with_done_action()` appends pause/exit snippets (`pause` → `timeout /t -1` on Windows).

Preferences: `Const.Pref.shell_opener` / `configs/preference/shell.yaml`.

---

## Windows WezTerm spawn (2025-06)

**Source:** `src/proj/util/shell/windows/wezterm/open.py`

### Symptom

`Shell.open` from Streamlit quick-call buttons failed in WezTerm on Windows:

- `wezterm cli set-tab-title`: `unexpected argument 'Logger''`
- `python -c`: `SyntaxError: unterminated string literal`

### Root causes

1. **`cmd.exe` does not treat `'` as string delimiters** (unlike bash). Single-quoted title / `-c` payloads are split into bogus tokens.
2. **Argv list vs manual terminal input differ.** `Popen([..., "cmd.exe", "/k", inner])` without `shell=True` lets WezTerm/CreateProcess rebuild the command line; quoting often diverges from typing the same line in a terminal.
3. **`subprocess.list2cmdline` on the full `/k` payload is wrong** for this use case: it wraps `inner` as one argument and emits backslash-escaped quotes (`\"`), which does not match what works when pasted into cmd.
4. **Bare `/k {inner}` without an outer quote pair** lets the parent `cmd.exe` (when `shell=True`) split on `&` before WezTerm sees the full chain.

### Working pattern

Launch via **`popen_detached(cmdline_string, shell=True)`** with a manually built line:

```
wezterm cli spawn --cwd E:\workspace\learndl -- cmd.exe /k "wezterm cli set-tab-title "Test Logger" & uv run python -c "from src.proj import Logger;Logger.test_logger()" & …"
```

Rules:

| Piece | Rule |
|-------|------|
| Outer `/k` payload | One pair of `"` around the **entire** inner chain |
| Title / `python -c` | Normal cmd double quotes inside the outer pair; **no** `\"` backslash escaping |
| `cwd` | `_win_cmd_token()` — quote only when path has spaces or metacharacters |
| WezTerm flags | `windows_detached_process=False`, `windows_create_no_window=False` on `popen_detached` |

Helper: `_wezterm_shell_cmdline()` in `windows/wezterm/open.py`.

### What we tried and rejected

| Approach | Why it failed |
|----------|----------------|
| Wrap title in `'…'` | cmd treats `'` literally → split arguments |
| `title.replace('"', "'")` + single-quote wrap | Same |
| `Popen(argv_list)` without shell | Differs from terminal typing; quoting lost |
| `list2cmdline` on full argv including `inner` | Produces `\"` escapes; WezTerm/cmd reject it |
| `/k {inner}` with no outer quotes | Parent shell splits on `&` |

### Helpers — `src/proj/util/shell/util/commands.py`

- `_win_cmd_quote(s)` — always double-quote; internal `"` → `""` (cmd rules).
- `_win_cmd_token(s)` — quote only when needed (paths without spaces stay bare).
- `to_shell_string()` — on Windows always `list2cmdline`.

### Debug

Set `SHELL_OPENER_DEBUG_WEZTERM=1` for GUI socket discovery logs (`discover_wezterm_gui_socket`).

---

## Related files

| File | Role |
|------|------|
| `shell/shell.py` | `Shell` facade |
| `shell/util/process.py` | `popen_detached` (str → `shell=True`), `spawn_native` |
| `shell/windows/cmd_terminal/open.py` | `start cmd /c` via `popen_detached_shell_windows` |
| `shell/util/pausing.py` | `done_action` suffixes |
| `src/api/interactive/util/quick_calls/basic.py` | Builds `["uv", "run", "python", "-c", …]` argv |
