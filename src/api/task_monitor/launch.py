"""Standalone Streamlit interface for Learndl task monitoring."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from src.api.task_monitor.core import (
    PageCursor, TaskMonitorRepository, TaskSnapshot, TaskStatus, TimeWindow,
)
from src.api.task_monitor.output import OutputCache, OutputStyle, parse_live_output

PAGE_SIZE = 50
MAX_HTML_PREVIEW_BYTES = 4 * 1024 * 1024
STATUS_OPTIONS: tuple[TaskStatus, ...] = ('running', 'complete', 'error', 'killed')
WINDOW_LABELS: dict[str, TimeWindow] = {
    'History': 'history',
    'Last day': 'last_day',
    'Last week': 'last_week',
    'Last month': 'last_month',
}
STYLE_LABELS: dict[str, OutputStyle] = {'Text only': 'text', 'Colored text': 'colored'}


def _time(value: float | None) -> str:
    return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S') if value else '—'


def _duration(task: TaskSnapshot) -> str:
    end = task.end_time if task.end_time is not None else datetime.now().timestamp()
    seconds = max(0, int(end - task.effective_start))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _colored_document(content: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{ margin: 0; background: #111827; color: #e5e7eb; font: 12px/1.45 Monaco, Menlo, 'Ubuntu Mono', monospace; }}
.output {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 10px; }}
.output-line {{ margin: 0 0 2px; }}
.meta {{ color: #9ca3af; }}
.muted {{ color: #6b7280; font-style: italic; }}
.stderr {{ border-left: 2px solid #f87171; background: rgba(248, 113, 113, 0.08); padding-left: 6px; }}
.stderr .meta {{ color: #fca5a5; }}
</style></head><body><div class="output">{content}</div></body></html>"""


def _render_output(content: str, style: OutputStyle, *, height: int = 560) -> None:
    if style == 'colored':
        components.html(_colored_document(content), height=height, scrolling=True)
    else:
        st.code(content, language=None)


def _render_full_html(path: Path) -> None:
    if not path.is_file():
        st.caption(f'Raw output file is no longer available: {path}')
        return
    size = path.stat().st_size
    with st.expander('Full HTML report and download'):
        if size <= MAX_HTML_PREVIEW_BYTES:
            try:
                components.html(path.read_text(encoding='utf-8', errors='replace'), height=600, scrolling=True)
            except OSError as exc:
                st.caption(f'Unable to preview output: {exc}')
        else:
            st.caption(f'The {size / 1024 / 1024:.1f} MB report exceeds the 4 MB inline preview limit.')
        try:
            with path.open('rb') as source:
                st.download_button(
                    'Download full HTML report', data=source, file_name=path.name,
                    mime='text/html', key=f'download-{path}',
                )
        except OSError:
            pass


def _output_candidates(task: TaskSnapshot) -> list[Path]:
    paths = list(task.exit_files)
    if task.crash_log is not None and task.crash_log not in paths:
        paths.insert(0, task.crash_log)
    return paths


def _render_task_detail(task: TaskSnapshot, style: OutputStyle) -> None:
    st.subheader(task.script)
    st.caption(f'Task ID: `{task.task_id}`')
    st.code(task.cmd, language='shell')
    columns = st.columns(4)
    columns[0].metric('Status', task.display_status.upper())
    columns[1].metric('Started', _time(task.effective_start))
    columns[2].metric('Duration', _duration(task))
    columns[3].metric('PID', str(task.pid) if task.pid is not None else '—')
    st.caption(
        f'Source: `{task.source or "unknown"}` · Background: {"yes" if task.is_background else "no"} '
        f'· Ended: {_time(task.end_time)}'
    )
    if task.display_status == 'stale':
        st.warning('The recorded PID is no longer alive. The system watchdog has not reconciled this task yet.')
    if task.exit_message:
        st.info(task.exit_message)
    if task.exit_error:
        st.error(task.exit_error)

    candidates = _output_candidates(task)
    if not candidates:
        st.info('No output file is recorded for this task.')
        return
    labels = [f'{path.name} — {path}' for path in candidates]
    selected_label = st.selectbox('Output file', labels, key=f'output-source-{task.task_id}')
    source = candidates[labels.index(selected_label)]
    st.subheader('Output')

    if task.is_active:
        content = parse_live_output(source, style)
        if content:
            _render_output(content, style)
            if source.is_file():
                st.caption(f'Live tail · updated {_time(source.stat().st_mtime)} · no disk cache')
        else:
            st.info('No live output is available yet.')
        return

    cache = OutputCache()
    page_key = f'output-page-{task.task_id}-{style}-{source}'
    requested_page = st.session_state.get(page_key)
    cached_page = cache.load_page(task.task_id, source, style, requested_page)
    if cached_page is None:
        st.info('The output source and its parsed cache are unavailable.')
    elif cached_page.total_pages == 0:
        st.info('The selected output contains no text records.')
    else:
        if cached_page.total_pages > 1:
            selected_page = st.number_input(
                'Output page', min_value=1, max_value=cached_page.total_pages,
                value=cached_page.page + 1, step=1, key=f'{page_key}-input',
            ) - 1
            if selected_page != cached_page.page:
                st.session_state[page_key] = selected_page
                cached_page = cache.load_page(task.task_id, source, style, selected_page) or cached_page
        _render_output(cached_page.content, style)
        st.caption(
            f'{cached_page.record_count} parsed records · page {cached_page.page + 1}/{cached_page.total_pages} '
            f'· {"cache hit" if cached_page.cache_hit else "cache created"}'
        )
    if source.suffix.lower() in ('.html', '.htm'):
        _render_full_html(source)


def _reset_paging_if_filters_changed(signature: tuple[object, ...]) -> None:
    if st.session_state.get('monitor-filter-signature') != signature:
        st.session_state['monitor-filter-signature'] = signature
        st.session_state['monitor-cursors'] = [None]
        st.session_state.pop('monitor-selected-task', None)


def _render_dashboard() -> None:
    st.title('Learndl Monitor')
    st.caption('Read-only task status and on-demand output. Lifecycle reconciliation is performed by the system watchdog.')
    with st.expander('Filters & results', expanded=False):
        filter_columns = st.columns((2, 1, 1))
        selected_statuses = filter_columns[0].multiselect(
            'Status', STATUS_OPTIONS, default=list(STATUS_OPTIONS),
        )
        window_label = filter_columns[1].selectbox('Start time', list(WINDOW_LABELS), index=0)
        style_label = filter_columns[2].radio('Output style', list(STYLE_LABELS), index=0)
        statuses = set(selected_statuses)
        window = WINDOW_LABELS[window_label]
        style = STYLE_LABELS[style_label]
        signature = (tuple(selected_statuses), window)
        _reset_paging_if_filters_changed(signature)
        st.caption(f'Filters: {", ".join(selected_statuses) or "none"} · {window_label} · {style_label}')
        repository = TaskMonitorRepository()
        cursors: list[PageCursor | None] = st.session_state.setdefault('monitor-cursors', [None])
        page_number = len(cursors)
        page = repository.list_tasks(statuses=statuses, window=window, cursor=cursors[-1], limit=PAGE_SIZE)
        counts = repository.status_counts(statuses=statuses, window=window)
        lifecycle_health = repository.watchdog_health('task_lifecycle', stale_after_seconds=3 * 60)
        cache_health = repository.watchdog_health('task_monitor_cache', stale_after_seconds=30 * 60)
        st_autorefresh(interval=3000 if counts['running'] else 30000, key='monitor-refresh')

        if lifecycle_health.stale:
            message = 'No successful watchdog lifecycle heartbeat was found.' if lifecycle_health.last_run is None else f'Last lifecycle run: {_time(lifecycle_health.last_run)}.'
            st.warning(f'{message} Check learndl-watchdog.timer; task states may be stale.')
        elif cache_health.stale:
            st.warning(
                f'Task lifecycle is healthy, but monitor-cache maintenance is stale '
                f'(last success: {_time(cache_health.last_run)}).'
            )
        else:
            st.caption(
                f'Watchdog healthy · lifecycle {_time(lifecycle_health.last_run)} · '
                f'cache {_time(cache_health.last_run)}'
            )

        metrics = st.columns(4)
        metrics[0].metric('Running', counts['running'])
        metrics[1].metric('Complete', counts['complete'])
        metrics[2].metric('Error', counts['error'])
        metrics[3].metric('Killed', counts['killed'])
        st.caption(f'{page.total} matching tasks · page {page_number} · up to {PAGE_SIZE} rows per page')

        if not page.tasks:
            st.info('No tasks match the selected filters.')
        else:
            rows = [
                {
                    'Status': task.display_status.upper(),
                    'Background': task.is_background,
                    'Script': task.script,
                    'Started': _time(task.effective_start),
                    'Duration': _duration(task),
                    'Source': task.source or 'unknown',
                    'PID': task.pid,
                    'Task ID': task.task_id,
                }
                for task in page.tasks
            ]
            event = st.dataframe(
                rows, hide_index=True, use_container_width=True, on_select='rerun',
                selection_mode='single-row', key=f'monitor-table-{signature}-{page_number}',
            )
            nav = st.columns((1, 1, 6))
            if nav[0].button('Previous', disabled=page_number == 1, use_container_width=True):
                cursors.pop()
                st.session_state.pop('monitor-selected-task', None)
                st.rerun()
            if nav[1].button('Next', disabled=page.next_cursor is None, use_container_width=True):
                assert page.next_cursor is not None
                cursors.append(page.next_cursor)
                st.session_state.pop('monitor-selected-task', None)
                st.rerun()

            selection = getattr(event, 'selection', None)
            selected_rows = getattr(selection, 'rows', []) if selection is not None else []
            if selected_rows:
                st.session_state['monitor-selected-task'] = page.tasks[selected_rows[0]].task_id
    selected_task_id = st.session_state.get('monitor-selected-task')
    if selected_task_id:
        task = repository.get_task(selected_task_id)
        if task is not None:
            st.divider()
            _render_task_detail(task, style)
    else:
        st.caption('Select one table row to load its details and output.')


st.set_page_config(page_title='Learndl Monitor', layout='wide')
_render_dashboard()
