# Rollback: interactive script pages (callable `st.Page`)

**Date:** 2026-04-16  
**Intent:** Script detail navigation no longer depends on generated `pages/_*.py` stubs; each script is a `st.Page(callable, url_path=...)`.

## Files touched

| File | Change |
|------|--------|
| [src/interactive/main/util/page.py](../../src/interactive/main/util/page.py) | `st.Page` from callable + `url_path`; remove stub generation from `script_pages`; `remake_all_script_detail_files` deletes legacy `_*.py` and clears `app_script_pages`; add `re`, `script_detail_url_path`, `_script_detail_page_callable` |
| [src/interactive/main/pages/task_queue.py](../../src/interactive/main/pages/task_queue.py) | `st.switch_page(get_script_page(...)['page'])` instead of `runs_page_url` string |
| [src/interactive/main/util/control.py](../../src/interactive/main/util/control.py) | `GlobalScriptLatestTaskButton`: lazy-import `get_script_page` + `st.switch_page(meta['page'])` |

## How to roll back (restore stub-file behaviour)

1. Revert the above files to the previous revision (e.g. `git checkout HEAD~1 -- <paths>` or your known-good commit).
2. Regenerate stubs if needed: from a branch that still has `make_script_detail_file` wired in `script_pages`, run the app once or call `remake_all_script_detail_files()` after restoring its old implementation (unlink + regenerate).
3. Hard-refresh the Streamlit app so `st.session_state['app_script_pages']` repopulates from disk paths.

## Notes

- `TaskItem.page_url` / `ScriptRunner.page_url` / `runs_page_url()` remain in the codebase for string keys and any external references; navigation now uses `get_script_page(k)['page']` for `st.switch_page`.
- Old bookmarks pointing at file-based URLs may differ from new `url_path` (single segment `_script_<slug>`, no `/` — Streamlit API restriction); use in-app navigation after upgrade.
