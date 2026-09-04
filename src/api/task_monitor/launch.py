"""Standalone Streamlit entry point for monitoring Learndl script executions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.api.task_monitor.core import FINISHED_STATUSES, TaskSnapshot, TaskMonitorRepository, read_live_log

REFRESH_SECONDS = 3
LOOKBACK_HOURS = 24
MAX_HTML_PREVIEW_BYTES = 4 * 1024 * 1024


def _time(value: float | None) -> str:
    return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S') if value else '—'


def _duration(task: TaskSnapshot) -> str:
    start = task.start_time or task.create_time
    end = task.end_time if task.end_time is not None else datetime.now().timestamp()
    seconds = max(0, int(end - start))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _status_label(task: TaskSnapshot) -> str:
    background = ' · BACKGROUND' if task.is_background else ''
    return f'{task.display_status.upper()}{background}'


def _render_html_output(path: Path) -> None:
    if not path.is_file():
        st.caption(f'Output file is no longer available: {path}')
        return
    if path.stat().st_size > MAX_HTML_PREVIEW_BYTES:
        st.caption(f'HTML output is {path.stat().st_size / 1024 / 1024:.1f} MB; download it instead of previewing it here.')
    else:
        try:
            components.html(path.read_text(encoding='utf-8', errors='replace'), height=600, scrolling=True)
        except OSError as exc:
            st.caption(f'Unable to read output: {exc}')
    try:
        st.download_button('Download HTML output', data=path.read_bytes(), file_name=path.name, mime='text/html', key=f'download-{path}')
    except OSError:
        pass


def _render_task(task: TaskSnapshot) -> None:
    icon = '🟠' if task.is_active else ('🔴' if task.display_status in {'error', 'killed', 'stale'} else '🟢')
    with st.expander(f'{icon} {_status_label(task)} — {task.script}', expanded=task.is_active):
        st.caption(f'Task ID: `{task.task_id}`')
        st.code(task.cmd, language='shell')
        left, middle, right = st.columns(3)
        left.metric('Started', _time(task.start_time or task.create_time))
        middle.metric('Duration', _duration(task))
        right.metric('PID', str(task.pid) if task.pid is not None else '—')
        st.caption(f'Source: `{task.source or "unknown"}` · Ended: {_time(task.end_time)}')
        if task.display_status == 'stale':
            st.warning('The task database says this task is active, but its recorded PID no longer exists. This monitor does not alter the record.')
        if task.is_active:
            st.subheader('Live crash-protector output')
            output = read_live_log(task.crash_log)
            if output:
                st.code(output, language=None)
                if task.crash_log is not None:
                    st.caption(f'Updated: {_time(task.crash_log.stat().st_mtime)} · {task.crash_log}')
            else:
                st.info('No live crash-protector output is available yet.')
            return

        if task.exit_code is not None:
            st.caption(f'Exit code: {task.exit_code}')
        if task.exit_message:
            st.info(task.exit_message)
        if task.exit_error:
            st.error(task.exit_error)
        if task.crash_log is not None:
            st.warning('A crash-protector file remains after this completed task. It may indicate abnormal cleanup; it is shown only as diagnostic evidence.')
            st.subheader('Recovered crash output')
            st.code(read_live_log(task.crash_log), language=None)
            try:
                st.download_button(
                    'Download recovered crash log', data=task.crash_log.read_bytes(),
                    file_name=task.crash_log.name, mime='text/markdown', key=f'crash-{task.task_id}',
                )
            except OSError:
                pass
        html_outputs = [path for path in task.exit_files if path.suffix.lower() in {'.html', '.htm'}]
        if html_outputs:
            st.subheader('Completed output')
            for path in html_outputs:
                _render_html_output(path)
        else:
            st.caption('No completed HTML output was recorded for this task.')


def _render_dashboard() -> None:
    tasks = TaskMonitorRepository().list_tasks(lookback_hours=LOOKBACK_HOURS)
    active = [task for task in tasks if task.is_active]
    finished = [task for task in tasks if task.status in FINISHED_STATUSES]
    background = [task for task in active if task.is_background]
    failures = [task for task in finished if task.status in {'error', 'killed'}]

    st.title('Learndl Monitor')
    st.caption('Read-only task status and output. Refreshes every 3 seconds; shows active tasks and tasks completed in the last 24 hours.')
    metrics = st.columns(4)
    metrics[0].metric('Active', len(active))
    metrics[1].metric('Background active', len(background))
    metrics[2].metric('Finished (24h)', len(finished))
    metrics[3].metric('Failed / killed (24h)', len(failures))
    if not tasks:
        st.info('No active tasks or tasks completed in the last 24 hours are recorded in the local task database.')
        return
    for task in tasks:
        _render_task(task)


st.set_page_config(page_title='Learndl Monitor', layout='wide')
st.fragment(run_every=REFRESH_SECONDS)(_render_dashboard)()
