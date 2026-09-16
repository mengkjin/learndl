"""Validated configuration shared by the installer, runner and watchdog."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ENTRYPOINTS = {
    'daily_update': 'scripts/1_autorun/0_daily_update.py',
    'weekly_update': 'scripts/1_autorun/1_weekly_update.py',
    'rcquant_sec_backfill': 'scripts/1_autorun/5_rcquant_sec_backfill.py',
}
DAYS = ('sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat')
JOB_NAMES = {'task_lifecycle', 'task_timeouts', 'systemd_unit_probe', 'task_monitor_cache', 'idle_worklist'}
IDENTIFIER = re.compile(r'^[a-z][a-z0-9_]*$')


def runtime_dir() -> Path:
    from src.proj import PATH
    return PATH.runtime / 'scheduling'


def duration(value: Any) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(s|m|h|d)', str(value))
    if not match or float(match[1]) <= 0:
        raise ValueError(f'Invalid positive duration: {value!r}')
    return float(match[1]) * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[match[2]]


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently changing a schedule."""


def _mapping(loader: UniqueLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f'Duplicate configuration key: {key}')
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict:
    value = yaml.load(path.read_text(), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f'Expected mapping: {path}')
    return value


def _keys(value: dict, allowed: set[str]) -> None:
    if extra := set(value) - allowed:
        raise ValueError(f'Unknown configuration keys: {extra}')


def schedule_slots(schedule: dict) -> set[tuple[int, int]]:
    """Return (weekday, minute-of-day) in cron's Sunday-zero convention."""
    _keys(schedule, {'task', 'days', 'times', 'every_minutes'})
    if 'every_minutes' in schedule:
        interval = schedule['every_minutes']
        if type(interval) is not int or interval <= 0 or 1440 % interval:
            raise ValueError('every_minutes must be a positive divisor of 1440')
        if 'times' in schedule or 'days' in schedule:
            raise ValueError('every_minutes cannot be combined with days/times')
        return {(day, minute) for day in range(7) for minute in range(0, 1440, interval)}
    days = schedule.get('days', 'daily')
    if days != 'daily' and (not isinstance(days, list) or not days):
        raise ValueError('days must be daily or a nonempty weekday list')
    weekdays = range(7) if days == 'daily' else [DAYS.index(day) for day in days]
    times = schedule.get('times', [])
    if not isinstance(times, list) or not times:
        raise ValueError('times must be a nonempty list')
    minutes = []
    for value in times:
        if not isinstance(value, str) or not re.fullmatch(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]', value):
            raise ValueError(f'Invalid time: {value!r}')
        hour, minute = map(int, value.split(':'))
        minutes.append(hour * 60 + minute)
    return {(day, minute) for day in weekdays for minute in minutes}


def load_config(directory: Path, host: str) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', host):
        raise ValueError('Invalid host name')
    settings = read_yaml(directory / 'hosts' / f'{host}.yaml')
    _keys(settings, {'timezone', 'default_backend', 'include', 'overrides'})
    ZoneInfo(settings['timezone'])
    if settings['default_backend'] not in {'cron', 'systemd'}:
        raise ValueError('default_backend must be cron or systemd')
    tasks_doc = read_yaml(directory / 'tasks.yaml')
    _keys(tasks_doc, {'tasks'})
    tasks = tasks_doc['tasks']
    schedules: dict = {}
    watchdog: dict | None = None
    for name in settings['include']:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Include escapes configuration directory')
        document = read_yaml(path)
        _keys(document, {'schedules', 'watchdog'})
        for key, value in document.get('schedules', {}).items():
            if key in schedules:
                raise ValueError(f'Duplicate schedule: {key}')
            schedules[key] = value
        if 'watchdog' in document:
            if watchdog is not None:
                raise ValueError('Duplicate watchdog configuration')
            watchdog = document['watchdog']
    for key, override in settings.get('overrides', {}).items():
        if key not in tasks:
            raise ValueError(f'Unknown task override: {key}')
        _keys(override, {'backend', 'enabled', 'times', 'days'})
        tasks[key].update({k: v for k, v in override.items() if k in {'backend', 'enabled'}})
        time_override = {k: v for k, v in override.items() if k in {'times', 'days'}}
        for schedule in schedules.values():
            if schedule['task'] == key and time_override:
                schedule.pop('every_minutes', None)
                schedule.update(time_override)
    for key, task in tasks.items():
        if not IDENTIFIER.fullmatch(key):
            raise ValueError(f'Invalid task ID: {key}')
        _keys(task, {'entrypoint', 'args', 'lock_group', 'overlap', 'wait_timeout_seconds', 'timeout', 'backend', 'enabled'})
        if task['entrypoint'] not in ENTRYPOINTS or not IDENTIFIER.fullmatch(task['lock_group']):
            raise ValueError(f'Invalid entrypoint/lock for {key}')
        if task['overlap'] not in {'skip', 'wait'}:
            raise ValueError(f'Invalid overlap policy for {key}')
        if not isinstance(task['args'], list) or not all(isinstance(arg, str) and '\n' not in arg for arg in task['args']):
            raise ValueError('args must be a list of single-line strings')
        # Only documented, typed options may cross the privileged installation boundary.
        allowed_args = {
            'daily_update': [[], ['--forfeit_if_done', 'False'], ['--forfeit_if_done', 'True']],
            'rcquant_sec_backfill': [[], ['--max_days', 'none']],
        }.get(task['entrypoint'], [[]])
        if task['args'] not in allowed_args:
            raise ValueError(f'Unsupported arguments for {key}')
        task['timeout_seconds'] = duration(task.pop('timeout', None))
        task.setdefault('wait_timeout_seconds', 1800)
        if type(task['wait_timeout_seconds']) is not int or task['wait_timeout_seconds'] <= 0:
            raise ValueError('wait_timeout_seconds must be positive')
        task.setdefault('backend', settings['default_backend'])
        task.setdefault('enabled', True)
        if task['backend'] not in {'cron', 'systemd'} or type(task['enabled']) is not bool:
            raise ValueError('Invalid task backend/enabled')
    for key, schedule in schedules.items():
        if not IDENTIFIER.fullmatch(key) or schedule['task'] not in tasks:
            raise ValueError(f'Invalid schedule: {key}')
        schedule_slots(schedule)
    if watchdog is None:
        raise ValueError('A watchdog configuration is required')
    _keys(watchdog, {'tick_seconds', 'jobs', 'units'})
    tick = watchdog['tick_seconds']
    if type(tick) is not int or tick < 60 or tick % 60 or 3600 % tick:
        raise ValueError('tick_seconds must be a whole-minute divisor of 3600')
    if not isinstance(watchdog['units'], list) or any(not re.fullmatch(r'[A-Za-z0-9_.@:-]+', unit) for unit in watchdog['units']):
        raise ValueError('Invalid systemd unit list')
    if any(unit.startswith('learndl-schedule-') or unit in {'learndl-watchdog.service', 'learndl-idle-worklist.service'} for unit in watchdog['units']):
        raise ValueError('Scheduled oneshot units cannot be probed as persistent services')
    for key, job in watchdog['jobs'].items():
        if key not in JOB_NAMES:
            raise ValueError(f'Unknown maintenance job: {key}')
        _keys(job, {'enabled', 'interval_seconds', 'max_sources', 'max_seconds'} |
              ({'max_gpu_memory_percent', 'progress_timeout_seconds'} if key == 'idle_worklist' else set()))
        if key == 'idle_worklist':
            percent = job.get('max_gpu_memory_percent', 20)
            limit = job.get('progress_timeout_seconds', 43200)
            if type(percent) not in (int, float) or not 0 < percent <= 100:
                raise ValueError('max_gpu_memory_percent must be in (0,100]')
            if type(limit) is not int or limit <= 0:
                raise ValueError('progress_timeout_seconds must be positive')
            if job.get('enabled') and not watchdog['jobs'].get('task_timeouts', {}).get('enabled'):
                raise ValueError('idle_worklist requires task_timeouts supervision')
        if type(job['enabled']) is not bool or type(job['interval_seconds']) is not int or job['interval_seconds'] < tick or job['interval_seconds'] % 60:
            raise ValueError(f'Invalid maintenance interval/enabled: {key}')
        for option in ('max_sources', 'max_seconds'):
            if option in job and (type(job[option]) is not int or job[option] <= 0):
                raise ValueError(f'Invalid {option}')
    return {'version': 1, 'host': host, 'timezone': settings['timezone'], 'default_backend': settings['default_backend'],
            'tasks': tasks, 'schedules': schedules, 'watchdog': watchdog}


def installed_config() -> dict | None:
    path = runtime_dir() / 'installed.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())
