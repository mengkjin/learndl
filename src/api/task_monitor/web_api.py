"""Read models and confirmed operator actions for the native Task Center."""

from __future__ import annotations

import math
import secrets
import threading
import time
from dataclasses import asdict

from .actions import kill_task, kill_unavailable_reason, prepare_kill
from .core import TaskFilters, TaskMonitorRepository
from .output import OutputCache, format_live_output, parse_live_output
from .remote import CHUNK_BYTES, task_key
from .remote_reader import RemoteCache
from .resources import sample_resources

__all__ = ['TaskCenter']

ACTIVE = {'starting', 'running'}
TERMINAL = {'complete', 'error', 'killed'}
WINDOWS = {'history', 'last_day', 'last_week', 'last_month'}
HISTORY_BYTES = 64 * 1024


def task_summary(task):
    return dict(task_id=task.task_id, script=task.script, status=task.display_status,
                start_time=task.effective_start, end_time=task.end_time, pid=task.pid,
                source=task.source, background=task.is_background, phase=task.managed_phase)


def output_sources(task):
    files = list(task.exit_files)
    if task.crash_log is not None and task.crash_log not in files:
        files.insert(0, task.crash_log)
    return files


def date_bounds(params, field):
    lower = float(params[f'{field}_from']) if f'{field}_from' in params else None
    upper = float(params[f'{field}_before']) if f'{field}_before' in params else None
    if any(value is not None and not math.isfinite(value) for value in (lower, upper)):
        raise ValueError('Invalid date boundary')
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError('Start and end of date range are out of order')
    return lower, upper


class TaskCenter:
    def __init__(self, repository=None, shared_root=None):
        self.repository = repository if repository is not None else TaskMonitorRepository()
        self.shared_root = shared_root
        self.remote_cache = RemoteCache()
        self.remote_lock = threading.Lock()
        self.resource_lock = threading.Lock()
        self.resources = None
        self.confirmations = {}
        self.confirmation_lock = threading.Lock()

    def health(self):
        jobs = {'task_lifecycle': 180, 'task_monitor_cache': 1800, 'task_timeouts': 180}
        return {name: asdict(self.repository.watchdog_health(name, seconds))
                for name, seconds in jobs.items()}

    def resource_snapshot(self):
        from src.proj import PATH
        with self.resource_lock:
            if self.resources is None or time.time() - self.resources.sampled_at >= 3:
                self.resources = sample_resources(PATH.main)
            return asdict(self.resources)

    def local_list(self, mode, params):
        cursor = None
        if mode == 'running':
            tasks = []
            while True:
                page = self.repository.list_tasks(statuses={'running'}, window='history', cursor=cursor, limit=50)
                tasks.extend(task_summary(task) for task in page.tasks)
                cursor = page.next_cursor
                if not cursor:
                    return dict(tasks=tasks, health=self.health())
        statuses = set(filter(None, params.get('statuses', 'complete,error,killed').split(',')))
        window = params.get('window', 'history')
        if not statuses <= TERMINAL or window not in WINDOWS:
            raise ValueError('Invalid task filter')
        if 'cursor_time' in params:
            cursor = (float(params['cursor_time']), params['cursor_id'])
        filters = TaskFilters(params.get('script', '').strip(),
                              date_bounds(params, 'started'), date_bounds(params, 'ended'))
        page = self.repository.list_tasks(statuses=statuses, window=window, cursor=cursor, limit=50, filters=filters)
        return dict(tasks=[task_summary(task) for task in page.tasks], total=page.total,
                    next_cursor=page.next_cursor,
                    counts=self.repository.status_counts(statuses=statuses, window=window, filters=filters),
                    health=self.health())

    def local_task(self, task_id):
        task = self.repository.get_task(task_id)
        if task is None:
            raise LookupError('Task no longer available')
        return task

    def local_detail(self, task_id):
        task = self.local_task(task_id)
        return dict(task_summary(task), cmd=task.cmd, child_pids=task.child_pids,
                    exit_message=task.exit_message, exit_error=task.exit_error,
                    active=task.is_active,
                    kill_reason=kill_unavailable_reason(task) if task.is_active else None,
                    files=[dict(name=path.name, path=str(path), html=path.suffix.lower() in {'.html', '.htm'})
                           for path in output_sources(task)])

    def local_output(self, params, style):
        task = self.local_task(params['task'])
        source = self.source(task, int(params.get('source', '0')))
        if task.is_active:
            return {'content': parse_live_output(source, style)}
        page = int(params['page']) if 'page' in params else None
        cached = OutputCache().load_page(task.task_id, source, style, page)
        return asdict(cached) if cached else {'content': '', 'unavailable': True}

    @staticmethod
    def source(task, index):
        paths = output_sources(task)
        if not 0 <= index < len(paths):
            raise LookupError('Output file no longer available')
        return paths[index]

    def remote_hosts(self):
        if self.shared_root is None:
            from src.automation.artifact_exchange.core import Config
            self.shared_root = Config.load().shared_root
        hosts = {}
        for path in sorted((self.shared_root / 'runtime').glob('*/index.json')):
            packet = self.remote_cache.document(path)
            if isinstance(packet, dict) and packet.get('version') in (1, 2):
                hosts[path.parent.name] = (path.parent, packet)
        return hosts

    def remote_list(self):
        with self.remote_lock:
            hosts = []
            for identity, (root, packet) in self.remote_hosts().items():
                tasks = packet['tasks']
                self.remote_cache.prune(root, {task['task_id'] for task in tasks})
                heartbeat = self.remote_cache.document(root / 'heartbeat.json') if packet['version'] == 2 else packet
                hosts.append(dict(id=identity, name=packet['host'], sampled_at=(heartbeat or {}).get('sampled_at'),
                                  tasks=[{name: value for name, value in task.items() if name in
                                          {'task_id', 'script', 'status', 'start_time', 'end_time', 'pid_alive', 'phase'}}
                                         for task in tasks]))
            return hosts

    def remote_task(self, host, task_id):
        associated = self.remote_hosts().get(host)
        if associated is None:
            raise LookupError('Server snapshot no longer available')
        root, packet = associated
        task = next((row for row in packet['tasks'] if row['task_id'] == task_id), None)
        if task is None:
            raise LookupError('Remote task no longer available')
        if packet['version'] == 2:
            manifest = self.remote_cache.document(root / 'tasks' / task_key(task_id) / 'manifest.json')
            if not manifest:
                return root, task, True
            task = {**manifest, **task}
        return root, task, False

    def remote_detail(self, params):
        with self.remote_lock:
            _, task, syncing = self.remote_task(params['host'], params['task'])
            return {**{name: value for name, value in task.items() if name not in {'segments', 'log', 'preview'}},
                    'syncing': syncing}

    def remote_output(self, params, style):
        with self.remote_lock:
            root, task, syncing = self.remote_task(params['host'], params['task'])
            if syncing:
                return dict(syncing=True)
            if 'generation' in params:
                histories = task.get('previous', []) + [task]
                history = next((row for row in histories if row['generation'] == int(params['generation'])), None)
                if history is None:
                    raise LookupError('Log generation no longer available')
                task = {**task, **history}
            offset = task['offset']
            end = min(offset, max(0, int(params.get('before', offset))))
            if 'start' in params:
                start = min(end, max(end - HISTORY_BYTES * 2, int(params['start']), 0))
            else:
                start = max(0, end - HISTORY_BYTES)
                if start:
                    # Begin history pages at a line boundary when possible.
                    probe = self.remote_cache.read(root, task, start, min(end, start + 8192))
                    if probe is not None and (newline := probe.find(b'\n')) >= 0:
                        start += newline + 1
            if not end:
                return dict(content='', start=0, end=0, syncing=False)
            content = self.remote_cache.render(root, task, start, end, style, compact=True)
            if content is None and 'before' not in params:
                # A manifest may arrive ahead of older segments. Keep showing any available tail.
                if task.get('segments') is not None:
                    starts = [part['start'] for part in task['segments']
                              if part['generation'] == task['generation'] and start < part['start'] < end]
                else:
                    starts = range((start // CHUNK_BYTES + 1) * CHUNK_BYTES, end, CHUNK_BYTES)
                for position in sorted(starts):
                    content = self.remote_cache.render(root, task, position, end, style, compact=True)
                    if content is not None:
                        return dict(content=content, start=position, end=end, syncing=True)
                if 'generation' not in params and task.get('preview'):
                    return dict(content=format_live_output(task['preview'], style), start=start, end=end, syncing=True)
            return dict(content=content, start=start, end=end, syncing=content is None)

    def reload_remote(self, params):
        with self.remote_lock:
            root, _, _ = self.remote_task(params['host'], params['task'])
            self.remote_cache.clear_task(root, params['task'])
        return {'ok': True}

    def prepare_termination(self, task_id):
        target = prepare_kill(self.local_task(task_id))
        token = secrets.token_urlsafe(24)
        with self.confirmation_lock:
            now = time.monotonic()
            self.confirmations = {key: item for key, item in self.confirmations.items() if item[0] > now}
            if len(self.confirmations) >= 128:
                raise ValueError('Too many pending confirmations')
            self.confirmations[token] = (now + 120, target)
        return dict(token=token, task_id=target.task_id, script=target.script, pid=target.pid,
                    started=target.started, managed=target.managed_identity is not None)

    def confirm_termination(self, token):
        with self.confirmation_lock:
            item = self.confirmations.pop(token, None)
        if item is None or item[0] <= time.monotonic():
            raise ValueError('Confirmation expired. Select the task again.')
        return asdict(kill_task(item[1]))
