"""Standalone Streamlit interface for Learndl task monitoring."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

import psutil

import streamlit as st

from src.api.task_monitor.core import (
    PageCursor, TaskMonitorRepository, TaskSnapshot, TaskStatus, TimeWindow,
)
from src.api.task_monitor.actions import KillTarget, kill_task, kill_unavailable_reason, prepare_kill
from src.api.task_monitor.resources import ResourceSnapshot, sample_resources
from src.api.task_monitor.output import OutputCache, OutputStyle, parse_live_output

PAGE_SIZE = 50
MAX_HTML_PREVIEW_BYTES = 4 * 1024 * 1024
STATUS_OPTIONS: tuple[TaskStatus, ...] = ('complete', 'error', 'killed')
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
        _render_isolated_html(_colored_document(content), height=height)
    else:
        st.code(content, language=None, height=height)


def _render_isolated_html(document: str, *, height: int) -> None:
    # st.iframe's own sandbox permits scripts/same-origin. Keep report content
    # inside a second, strictly sandboxed frame and escape the srcdoc attribute.
    wrapper = (f'<iframe title="Task output" sandbox="" srcdoc="{escape(document, quote=True)}" '
               f'style="width:100%;height:{height - 8}px;border:0"></iframe>')
    st.iframe(wrapper, height=height, width='stretch')


def _render_full_html(path: Path) -> None:
    if not path.is_file():
        st.caption(f'Raw output file is no longer available: {path}')
        return
    size = path.stat().st_size
    with st.expander('Full HTML report and download'):
        if size <= MAX_HTML_PREVIEW_BYTES:
            try:
                _render_isolated_html(path.read_text(encoding='utf-8', errors='replace'), height=600)
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

    _render_task_output(task, style)


def _render_task_output(task: TaskSnapshot, style: OutputStyle) -> None:
    candidates = _output_candidates(task)
    if not candidates:
        st.info('No output file is recorded for this task.')
        return
    labels = [f'{path.name} — {path}' for path in candidates]
    selected_label = st.selectbox('Output file', labels, key=f'output-source-{"running" if task.is_active else "finished"}-{task.task_id}')
    source = candidates[labels.index(selected_label)]
    st.subheader('Output')

    if task.is_active:
        content = parse_live_output(source, style)
        if content:
            _render_output(content, style)
            try:
                st.caption(f'Live tail · updated {_time(source.stat().st_mtime)} · no disk cache')
            except OSError:
                st.caption('Output file is no longer available.')
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
    if st.session_state.get('finished-filter-signature') != signature:
        st.session_state['finished-filter-signature'] = signature
        st.session_state['finished-cursors'] = [None]
        st.session_state.pop('finished-selected-task', None)


def _render_finished() -> None:
    _render_header()
    with st.expander('Filters & results', expanded=False):
        filter_columns = st.columns((2, 1, 1))
        st.session_state.setdefault('finished-statuses', list(STATUS_OPTIONS))
        selected_statuses = filter_columns[0].multiselect(
            'Status', STATUS_OPTIONS, key='finished-statuses',
        )
        window_label = filter_columns[1].selectbox('Start time', list(WINDOW_LABELS), index=0, key='finished-window')
        style_label = filter_columns[2].radio('Output style', list(STYLE_LABELS), index=0, key='finished-style')
        statuses = set(selected_statuses)
        window = WINDOW_LABELS[window_label]
        style = STYLE_LABELS[style_label]
        signature = (tuple(selected_statuses), window)
        _reset_paging_if_filters_changed(signature)
        st.caption(f'Filters: {", ".join(selected_statuses) or "none"} · {window_label} · {style_label}')
        repository = TaskMonitorRepository()
        cursors: list[PageCursor | None] = st.session_state.setdefault('finished-cursors', [None])
        page_number = len(cursors)
        page = repository.list_tasks(statuses=statuses, window=window, cursor=cursors[-1], limit=PAGE_SIZE)
        counts = repository.status_counts(statuses=statuses, window=window)
        metrics = st.columns(3)
        metrics[0].metric('Complete', counts['complete'])
        metrics[1].metric('Error', counts['error'])
        metrics[2].metric('Killed', counts['killed'])
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
                rows, hide_index=True, width='stretch', on_select='rerun',
                selection_mode='single-row',
                key=f'finished-table-{signature}-{page_number}-{tuple(task.task_id for task in page.tasks)}',
            )
            nav = st.columns((1, 1, 6))
            if nav[0].button('Previous', disabled=page_number == 1, width='stretch'):
                cursors.pop()
                st.session_state.pop('finished-selected-task', None)
                st.rerun()
            if nav[1].button('Next', disabled=page.next_cursor is None, width='stretch'):
                assert page.next_cursor is not None
                cursors.append(page.next_cursor)
                st.session_state.pop('finished-selected-task', None)
                st.rerun()

            selection = getattr(event, 'selection', None)
            selected_rows = getattr(selection, 'rows', []) if selection is not None else []
            if selected_rows and selected_rows[0] < len(page.tasks):
                st.session_state['finished-selected-task'] = page.tasks[selected_rows[0]].task_id
    selected_task_id = st.session_state.get('finished-selected-task')
    if selected_task_id:
        task = repository.get_task(selected_task_id)
        if task is not None and not task.is_active:
            st.divider()
            _render_task_detail(task, style)
    else:
        st.caption('Select one table row to load its details and output.')

    _render_health(repository)
    _auto_refresh(30000, 'finished-refresh')


def _render_health(repository: TaskMonitorRepository) -> None:
    lifecycle_health = repository.watchdog_health('task_lifecycle', stale_after_seconds=3 * 60)
    cache_health = repository.watchdog_health('task_monitor_cache', stale_after_seconds=30 * 60)
    timeout_health = repository.watchdog_health('task_timeouts', stale_after_seconds=3 * 60)
    for label, health in [('Lifecycle', lifecycle_health), ('Cache maintenance', cache_health), ('Timeout protection', timeout_health)]:
        if health.payload.get('disabled'):
            st.caption(f'{label}: disabled')
    if timeout_health.stale:
        st.warning('Timeout protection heartbeat is missing or stale; task runtime limits may not be enforced.')

    if lifecycle_health.stale:
        message = 'No successful watchdog lifecycle heartbeat was found.' if lifecycle_health.last_run is None else f'Last lifecycle run: {_time(lifecycle_health.last_run)}.'
        st.warning(f'{message} Check learndl-watchdog.timer; task states may be stale.')
    elif cache_health.stale:
        st.warning(
            f'Task lifecycle is healthy, but monitor-cache maintenance is stale '
            f'(last success: {_time(cache_health.last_run)}).'
        )
    elif not timeout_health.stale and not all(health.payload.get('disabled') for health in (lifecycle_health, cache_health, timeout_health)):
        st.caption(
            f'Watchdog healthy · lifecycle {_time(lifecycle_health.last_run)} · '
            f'cache {_time(cache_health.last_run)}'
        )


@st.cache_data(ttl=3, show_spinner=False)
def _resources() -> ResourceSnapshot:
    from src.proj import PATH
    return sample_resources(PATH.main)


def _usage(value: tuple[float, float, float] | None) -> str:
    if value is None:
        return 'Unavailable'
    used, total, percent = value
    return f'{used / 1024**3:.1f} / {total / 1024**3:.1f} GiB ({percent:.1f}%)'


def _percent(value: float | None) -> str:
    return 'Unavailable' if value is None else f'{value:.1f}%'


def _resource_row(label: str, value: str) -> None:
    label_column, value_column = st.columns((2, 3))
    label_column.markdown(f'**{label}**')
    value_column.write(value)


def _render_header() -> None:
    title, system = st.columns((5, 1), vertical_alignment='center')
    title.title('Learndl Monitor')
    with system.popover('System', icon=':material/device_thermostat:', width='stretch'):
        snapshot = _resources()
        st.caption(f'Host resources · sampled {_time(snapshot.sampled_at)}')
        _resource_row('Memory', _usage(snapshot.memory))
        _resource_row('CPU', _percent(snapshot.cpu_percent))
        _resource_row('Project disk', _usage(snapshot.disk))
        for gpu in snapshot.gpus:
            st.markdown(f'**GPU {gpu.index} · {gpu.name}**')
            used = 'Unavailable' if gpu.used_mib is None else f'{gpu.used_mib / 1024:.1f} GiB'
            total = 'Unavailable' if not gpu.total_mib else f'{gpu.total_mib / 1024:.1f} GiB'
            _resource_row('GPU memory', f'{used} / {total} ({_percent(gpu.memory_percent)})')
            _resource_row('GPU compute utilization', _percent(gpu.utilization))
        for error in snapshot.errors:
            st.caption(error)


def _auto_refresh(interval: int, key: str) -> None:
    from streamlit_autorefresh import st_autorefresh
    if not st.session_state.get('monitor-kill-target'):
        st_autorefresh(interval=interval, key=key)


def _clear_kill() -> None:
    for key in ('monitor-kill-target', 'monitor-kill-token', 'monitor-kill-busy'):
        st.session_state.pop(key, None)


def _start_kill(task: TaskSnapshot) -> None:
    try:
        target = prepare_kill(task)
    except (ValueError, psutil.Error) as exc:
        st.session_state['monitor-action-result'] = ('error', str(exc))
        return
    st.session_state['monitor-kill-target'] = target
    st.session_state['monitor-kill-token'] = uuid4().hex
    st.session_state['monitor-kill-busy'] = False


def _queue_kill(checkbox_key: str) -> None:
    if st.session_state.get(checkbox_key) and st.session_state.get('monitor-kill-target'):
        st.session_state['monitor-kill-busy'] = True


@st.dialog('Kill running script?', on_dismiss=_clear_kill)
def _confirm_kill(target: KillTarget, token: str) -> None:
    st.warning('This will terminate the selected script process.')
    st.write(target.script)
    st.caption(f'Task ID: {target.task_id} · PID: {target.pid} · Started: {_time(target.started)}')
    busy = st.session_state.get('monitor-kill-busy', False)
    checkbox_key = f'kill-confirm-{token}'
    confirmed = st.checkbox('I confirm that I want to terminate this task.',
                            value=False, key=checkbox_key, disabled=busy)
    cancel, execute = st.columns(2)
    if cancel.button('Cancel', key=f'kill-cancel-{token}', disabled=busy):
        _clear_kill()
        st.rerun()
    execute.button('Confirm Kill', key=f'kill-execute-{token}', type='primary',
                   disabled=not confirmed or busy, on_click=_queue_kill, args=(checkbox_key,))
    if busy and st.session_state.get('monitor-kill-token') == token:
        # Consume this confirmation before any side effects, including repeated widget events.
        _clear_kill()
        with st.spinner('Terminating process…'):
            result = kill_task(target)
        st.session_state['monitor-action-result'] = (
            'success' if result.status == 'success' else 'error', result.message,
        )
        if result.status == 'success':
            st.session_state['finished-cursors'] = [None]
            st.session_state.pop('finished-selected-task', None)
        st.rerun()


def _pin_task(task_id: str) -> None:
    st.session_state['running-pinned-task'] = task_id


def _follow_latest() -> None:
    st.session_state.pop('running-pinned-task', None)


def _running_tasks(repository: TaskMonitorRepository) -> list[TaskSnapshot]:
    tasks: list[TaskSnapshot] = []
    cursor = None
    while True:
        page = repository.list_tasks(statuses={'running'}, window='history', cursor=cursor, limit=PAGE_SIZE)
        tasks.extend(page.tasks)
        if page.next_cursor is None:
            return tasks
        cursor = page.next_cursor


def _render_running() -> None:
    _render_header()
    if result := st.session_state.pop('monitor-action-result', None):
        level, message = result
        getattr(st, level)(message)
    repository = TaskMonitorRepository()
    tasks = _running_tasks(repository)
    pinned = st.session_state.get('running-pinned-task')
    if pinned and not any(task.task_id == pinned for task in tasks):
        _follow_latest()
        pinned = None
        st.info('The selected task has finished or is no longer available. Following the latest task.')
    selected = pinned or (tasks[0].task_id if tasks else None)
    left, right = st.columns((1, 3), gap='large')
    with left:
        st.subheader(f'Running ({len(tasks)})')
        st.button('Follow latest', on_click=_follow_latest, disabled=pinned is None,
                  key='running-follow', width='stretch')
        with st.container(height=640, border=False):
            if not tasks:
                st.info('No running tasks.')
            for task in tasks:
                with st.container(border=True):
                    st.markdown(f'**{Path(task.script).name}**')
                    st.caption(f'{task.display_status.upper()} · PID {task.pid or "—"}')
                    st.caption(f'Started {_time(task.effective_start)} · {_duration(task)}')
                    view, kill = st.columns(2)
                    view.button('Output', key=f'view-{task.task_id}', on_click=_pin_task,
                                args=(task.task_id,), type='primary' if task.task_id == selected else 'secondary',
                                help=f'{task.script} · {task.task_id}', width='stretch')
                    reason = kill_unavailable_reason(task)
                    kill.button('Kill', key=f'kill-{task.task_id}', disabled=reason is not None,
                                help=reason or 'Review and confirm termination', on_click=_start_kill,
                                args=(task,), width='stretch')
                    if reason:
                        st.caption(reason)
    with right:
        style_label = st.radio('Output style', list(STYLE_LABELS), horizontal=True, key='running-style')
        st.caption('Pinned task' if pinned else 'Following latest task')
        task = repository.get_task(selected) if selected else None
        if task is not None and task.is_active:
            st.subheader(Path(task.script).name)
            st.caption(f'PID {task.pid or "—"} · Started {_time(task.effective_start)} · {_duration(task)}')
            _render_task_output(task, STYLE_LABELS[style_label])
            with st.expander('Task details'):
                st.write(task.script)
                st.caption(f'Task ID: {task.task_id} · Source: {task.source or "unknown"}')
                st.code(task.cmd, language='shell')
                if task.display_status == 'stale':
                    st.warning('The recorded PID has exited; waiting for watchdog reconciliation.')
        elif selected:
            st.info('This task has just finished. The next refresh will select another running task.')
        else:
            st.info('Output will appear here when a task starts.')
    _render_health(repository)
    _auto_refresh(3000, 'running-refresh')
    if target := st.session_state.get('monitor-kill-target'):
        _confirm_kill(target, st.session_state['monitor-kill-token'])


def _render_dashboard() -> None:
    # Preserve filters even when their page's widgets are not rendered this run.
    for key in ('finished-statuses', 'finished-window', 'finished-style', 'running-style'):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    page = st.navigation([
        st.Page(_render_running, title='Running', icon=':material/play_circle:', default=True),
        st.Page(_render_finished, title='Finished', icon=':material/task_alt:', url_path='finished'),
    ], position='top')
    if page.title != 'Running':
        _clear_kill()
    page.run()


if __name__ == '__main__':
    st.set_page_config(page_title='Learndl Monitor', layout='wide')
    _render_dashboard()
