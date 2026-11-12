"""Worklist completion policy; general training history lives separately."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import portalocker
import yaml

from src.proj import PATH
from src.res.model.util.training_history import file_revision, now, write_json

__all__ = ['WorklistState', 'AutomaticDeferred']


class AutomaticDeferred(Exception):
    """No automatic training was attempted; reconsider on a later poll."""



class WorklistState:
    def __init__(self, schedule_name: str):
        self.root = PATH.lc_machine / 'schedule_worklist'
        key = hashlib.sha256(schedule_name.encode()).hexdigest()
        self.path = self.root / f'{key}.json'
        self.schedule_name = schedule_name

    @staticmethod
    def _options(value: Mapping) -> dict[str, Any]:
        from src.res.model.util.lifecycle import ModelLifecycle
        options = {key: value[key] for key in ('resume', 'force', 'short_test', 'start', 'end') if key in value}
        for key in ('resume', 'force', 'short_test'):
            if key in options and type(options[key]) is not bool:
                raise ValueError(f'worklist {key} must be a YAML boolean')
        from datetime import datetime
        for key in ('start', 'end'):
            if key in options:
                if type(options[key]) is not int:
                    raise ValueError(f'worklist {key} must be a YYYYMMDD integer')
                try:
                    datetime.strptime(str(options[key]), '%Y%m%d')
                except ValueError as exc:
                    raise ValueError(f'Invalid worklist {key}: {options[key]}') from exc
        if 'mode' in value and value['mode'] not in ('auto', 'quick', 'sparse', 'full'):
            raise ValueError('worklist mode must be auto, quick, sparse or full')
        lifecycle = ModelLifecycle.training_options(options=value.get('lifecycle'))
        if 'mode' in value:
            if 'mode' in lifecycle and lifecycle['mode'] != value['mode']:
                raise ValueError('Conflicting worklist mode and lifecycle.mode')
            lifecycle['mode'] = value['mode']
        if lifecycle:
            options['lifecycle'] = lifecycle
        return options

    @classmethod
    def entries(cls, worklist: Mapping) -> dict[str, dict[str, Any]]:
        """Normalize string/dictionary entries once for manual and automatic runners.

        Item settings override global defaults. Identical repeats are deduplicated;
        conflicting duplicate schedules are rejected rather than run ambiguously.
        """
        if not isinstance(worklist, Mapping) or not isinstance(worklist.get('fit'), list):
            raise ValueError('worklist.fit must be a list of schedule names or mappings')
        defaults: dict[str, Any] = {'resume': False, 'force': False} | cls._options(worklist)
        entries: dict[str, dict[str, Any]] = {}
        allowed = {'name', 'resume', 'force', 'short_test', 'start', 'end', 'mode', 'lifecycle'}
        for item in worklist['fit']:
            if isinstance(item, str):
                name, local = item, {}
            elif isinstance(item, Mapping):
                unknown = set(item) - allowed
                if unknown:
                    raise ValueError(f'Unknown worklist entry options: {sorted(unknown)}')
                name, local = item.get('name'), cls._options(item)
            else:
                raise ValueError('Each worklist entry must be a schedule string or mapping')
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError('A worklist entry requires a nonempty schedule name')
            resolved = defaults | local
            lifecycle = dict(defaults.get('lifecycle', {})) | local.get('lifecycle', {})
            if lifecycle:
                resolved['lifecycle'] = lifecycle
            if resolved.get('start', 0) > resolved.get('end', 99991231):
                raise ValueError(f'worklist start exceeds end: {name}')
            if name in entries and entries[name] != resolved:
                raise ValueError(f'Conflicting worklist entries for schedule: {name}')
            entries[name] = resolved
        return entries

    @classmethod
    def options(cls, worklist: Mapping, name: str) -> dict[str, Any]:
        """Standalone requests inherit global options without requiring a list entry."""
        return cls.entries(worklist).get(name, {'resume': False, 'force': False} | cls._options(worklist))

    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return portalocker.Lock(str(self.path.with_suffix('.lock')), mode='a', timeout=0)

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else None

    @property
    def request_path(self) -> Path:
        return self.root / 'requests' / self.path.name

    def request_training(self, request_id: str, *, allow_unlisted: bool = False):
        """Request this schedule after maintenance; the idle worker owns GPU execution.

        A stable ID is idempotent and changes only this schedule's revision.
        Keep the marker after completion so removing it cannot trigger another run.
        allow_unlisted enables an explicit standalone experiment without editing the worklist.
        """
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('A nonempty training request ID is required')
        worklist = yaml.safe_load(PATH.sched_worklist.read_text())
        if not allow_unlisted and self.schedule_name not in self.entries(worklist):
            raise ValueError(f'Schedule [{self.schedule_name}] is not in the training worklist')
        self.revisions(self.schedule_name)  # Verify the schedule config before writing.
        request = {'id': request_id}
        if allow_unlisted:
            request['schedule_name'] = self.schedule_name
        with self.lock():
            if not self.request_path.exists() or json.loads(self.request_path.read_text()) != request:
                write_json(self.request_path, request)
        return request

    @classmethod
    def requested_schedules(cls) -> list[str]:
        """Pending standalone requests, excluding successful matching request IDs.

        Completion uses saved state only, before any schedule YAML lookup. Callers
        merge this list with the current worklist, whose validation stays independent.
        Request markers remain on disk for idempotency and history.
        """
        root = PATH.lc_machine / 'schedule_worklist' / 'requests'
        names = []
        for path in sorted(root.glob('*.json')):
            request = json.loads(path.read_text())
            name = request.get('schedule_name')
            if isinstance(name, str) and name and cls(name).request_path == path and request.get('id'):
                previous = cls(name).read() or {}
                completed_request = previous.get('revisions', {}).get('training_request', {})
                if previous.get('status') == 'success' and completed_request.get('id') == request['id']:
                    continue
                names.append(name)
        return names

    @staticmethod
    def revisions(schedule_name: str):
        from src.res.model.util.config.config import ScheduleConfig
        path = ScheduleConfig.find_path(name=schedule_name)
        if path is None:
            raise FileNotFoundError(f'Schedule config missing: {schedule_name}')
        revisions = {'worklist': file_revision(PATH.sched_worklist), 'schedule': file_revision(path)}
        reset_path = PATH.lc_machine / 'schedule_worklist' / 'reset.json'
        if reset_path.exists():
            revisions['reset_generation'] = {'generation': json.loads(reset_path.read_text())['generation']}
        request_path = WorklistState(schedule_name).request_path
        if request_path.exists():
            revisions['training_request'] = json.loads(request_path.read_text())
        return revisions

    @staticmethod
    def rerun_mark(worklist: dict | None) -> str:
        """Symbolic token for mail; missing or null is displayed as ``none``."""
        if not worklist or worklist.get('rerun_mark') is None:
            return 'none'
        value = worklist['rerun_mark']
        if isinstance(value, (dict, list)):
            raise ValueError('worklist rerun_mark must be a YAML scalar')
        return str(value)

    @staticmethod
    def decision(previous, revisions, *, force: bool, resume: bool):
        """Return run/skip reason and effective resume; no time-based policy."""
        if force:
            return 'forced', resume
        if not previous or previous['status'] != 'success':
            return 'no successful completion', resume
        if previous['revisions'] != revisions:
            return 'worklist or schedule changed', resume
        return 'completed', resume

    def save(self, **data):
        write_json(self.path, {'schema_version': 1, 'schedule_name': self.schedule_name,
                               'updated_at': now(), **data})
