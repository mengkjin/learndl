"""Read-only remote task browser; all input comes from the local Syncthing copy."""
from __future__ import annotations

import time

import streamlit as st

from .remote import CHUNK_BYTES
from .remote_reader import RemoteCache
from .output import sanitize_inline_html


def render():
    from src.automation.artifact_exchange.core import Config
    from .launch import _auto_refresh, _time
    root = Config.load().shared_root / 'runtime'
    cache = st.session_state.setdefault('remote-cache', RemoteCache())
    hosts = {}
    for path in sorted(root.glob('*/index.json')):
        packet = cache.document(path)
        if packet and packet.get('version') in (1, 2):
            hosts[path.parent.name] = (path.parent, packet)
    st.subheader('Remote Running')
    if not hosts:
        st.info('Waiting for server snapshots in Syncthing.')
        _auto_refresh(5000, 'remote-refresh')
        return
    host = st.selectbox('Server', list(hosts), format_func=lambda key: hosts[key][1]['host'])
    directory, packet = hosts[host]
    heartbeat = cache.document(directory / 'heartbeat.json') if packet['version'] == 2 else packet
    sampled = (heartbeat or {}).get('sampled_at', 0)
    cache.prune(directory, {task['task_id'] for task in packet['tasks']})
    st.caption(f"Sampled {_time(sampled)}")
    if time.time() - sampled > 60:
        st.warning('Snapshot is out of date. Showing the last received state.')
    active = [t for t in packet['tasks'] if t['status'] in ('starting', 'running')]
    finished = [t for t in packet['tasks'] if t['status'] not in ('starting', 'running')]
    left, right = st.columns((1, 3))
    with left:
        recent = st.checkbox('Recently finished', key=f'remote-finished-{host}')
        tasks = finished if recent else active
        tasks = sorted(tasks, key=lambda t: t.get('start_time') or 0, reverse=True)
        st.caption(f"{'Recently finished' if recent else 'Running'} ({len(tasks)})")
        selected = st.selectbox('Task', range(len(tasks)),
                                format_func=lambda i: tasks[i]['script'],
                                key=f'remote-task-{host}-{recent}') if tasks else None
    with right:
        if selected is None:
            st.info('No tasks in this snapshot.')
        else:
            task = tasks[selected]
            if packet['version'] == 2:
                manifest = cache.document(directory / 'tasks' / task['key'] / 'manifest.json')
                if not manifest:
                    st.info('Task details are still syncing.')
                    _auto_refresh(5000, 'remote-refresh')
                    return
                task = dict(task, **manifest)
            if st.button('Reload logs', key=f"remote-reload-{host}-{task['key']}"):
                cache.clear_task(directory, task['task_id'])
            st.subheader(task['script'])
            st.caption(f"{task['status']} · Started {_time(task.get('start_time'))} · Last output {_time(task.get('output_at'))}")
            if task.get('phase'):
                st.caption(task['phase'])
            if task.get('pid_alive') is False:
                st.warning('Server reports that the recorded process has exited. Awaiting lifecycle reconciliation.')
            if task.get('gap'):
                st.warning('The source log changed or became unavailable. Some output may be missing.')
            if not task['complete']:
                st.caption(f"History received by server publisher: {task['offset']} / {task['size']} bytes")
            if task.get('expires_at'):
                st.caption(f"Available until {_time(task['expires_at'])}")
            follow = st.checkbox('Follow latest', value=True, key=f"remote-follow-{host}-{task['key']}")
            if follow:
                text = task.get('preview') if packet['version'] == 1 else cache.render(directory, task, max(0, task['offset'] - 8192), task['offset'])
                if text is None:
                    st.info('Latest output is still syncing.')
                else:
                    st.code((sanitize_inline_html(text, 'text') if packet['version'] == 1 else text) or 'No output yet.', language=None)
            else:
                generations = task.get('previous', []) + [{'generation': task['generation'], 'offset': task['offset']}]
                generation = st.selectbox('Log generation', range(len(generations)),
                                          index=len(generations) - 1, key=f"remote-generation-{host}-{task['key']}")
                advertised = dict(task, **generations[generation])
                count = (advertised['offset'] + CHUNK_BYTES - 1) // CHUNK_BYTES
                if count:
                    number = st.number_input('Output volume', min_value=1, max_value=count, value=1,
                                             key=f"remote-volume-{host}-{task['key']}-{generation}")
                    text = cache.render(directory, advertised, (int(number) - 1) * CHUNK_BYTES, min(int(number) * CHUNK_BYTES, advertised['offset']))
                    if text is None:
                        st.info('This output volume is still syncing.')
                    else:
                        st.code(text, language=None)
                else:
                    st.info('No history received yet.')
    _auto_refresh(5000, 'remote-refresh')
