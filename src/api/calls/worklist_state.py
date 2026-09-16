"""Worklist completion policy; general training history lives separately."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import portalocker

from src.proj import PATH
from src.res.model.util.training_history import file_revision, now, write_json

__all__ = ['WorklistState']


class WorklistState:
    def __init__(self, schedule_name: str):
        self.root = PATH.lc_machine / 'schedule_worklist'
        key = hashlib.sha256(schedule_name.encode()).hexdigest()
        self.path = self.root / f'{key}.json'
        self.schedule_name = schedule_name

    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return portalocker.Lock(str(self.path.with_suffix('.lock')), mode='a', timeout=0)

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else None

    @staticmethod
    def revisions(schedule_name: str):
        from src.res.model.util.config.config import ScheduleConfig
        path = ScheduleConfig.find_path(name=schedule_name)
        if path is None:
            raise FileNotFoundError(f'Schedule config missing: {schedule_name}')
        return {'worklist': file_revision(PATH.sched_worklist), 'schedule': file_revision(path)}

    @staticmethod
    def decision(previous, revisions, *, force: bool, resume: bool):
        """Return run/skip reason and effective resume; no time-based policy."""
        if force:
            return 'forced', resume
        if previous and previous.get('model_path') and not Path(previous['model_path']).is_dir():
            return 'training directory missing', False
        if not previous or previous['status'] != 'success':
            return 'no successful completion', resume
        if previous['revisions'] != revisions:
            return 'worklist or schedule changed', resume
        if not previous.get('model_path'):
            return 'training directory unknown', resume
        return 'completed', resume

    def save(self, **data):
        write_json(self.path, {'schema_version': 1, 'schedule_name': self.schedule_name,
                               'updated_at': now(), **data})
