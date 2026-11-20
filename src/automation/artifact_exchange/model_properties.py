"""Bounded summaries of actual training and testing, stored with the source model."""
from __future__ import annotations

from contextlib import contextmanager, suppress
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .core import fsync_dir, real_tree
from .model_logs import now

KINDS = ('initial_training', 'resume_training', 'last_test')
PROPERTY_FILE = 'logs/property.jsonl'


def read_properties(path):
    with suppress(FileNotFoundError):
        return {row['kind']: row for line in real_tree(path).read_text().splitlines() if line.strip()
                for row in [json.loads(line)] if row['kind'] in KINDS}
    return {}


def write_properties(path, records):
    path = real_tree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            for kind in KINDS:
                if kind in records:
                    stream.write(json.dumps(records[kind], ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            temp.replace(path)
            fsync_dir(path.parent)
        finally:
            temp.unlink(missing_ok=True)


def coverage(*ranges):
    ranges = [span for span in ranges if span]
    return [min(span[0] for span in ranges), max(span[-1] for span in ranges)] if ranges else None


class Phase:
    def __init__(self, trainer, kind):
        history = trainer.training_run
        self.path = trainer.base_path.base / PROPERTY_FILE
        self.kind = kind
        self.resume = trainer.config.is_resuming
        self.record = {
            'kind': kind, 'run_id': history.run_id, 'git_head': history.data['git']['head'],
            'started_at': None, 'finished_at': None, 'status': 'running', 'date_range': None,
        }

    def begin(self):
        if self.record['started_at'] is not None:
            return
        self.record['started_at'] = now()
        if self.kind == 'initial_training':
            records = read_properties(self.path)
            if self.kind not in records:
                records[self.kind] = self.record
                write_properties(self.path, records)

    def worked(self, first, last):
        self.begin()
        self.record['date_range'] = coverage(self.record['date_range'], [int(first), int(last)])

    def finish(self, status):
        if self.record['started_at'] is None:
            return
        records = read_properties(self.path)
        self.record.update(status=status, finished_at=now())
        old = records.get(self.kind)
        if self.kind == 'initial_training':
            if old is None or old.get('run_id') == self.record['run_id']:
                records[self.kind] = self.record
            elif status == 'success' and self.record['date_range']:
                if self.resume:
                    previous = records.get('resume_training', {})
                    if previous.get('run_id') == self.record['run_id']:
                        return
                    records['resume_training'] = self.record | {
                        'kind': 'resume_training', 'count': previous.get('count', 0) + 1,
                        'first_started_at': previous.get('first_started_at', self.record['started_at']),
                        'cumulative_date_range': coverage(previous.get('cumulative_date_range'),
                                                          self.record['date_range']),
                    }
                else:
                    records[self.kind] = old | {'latest_full_training': self.record}
            else:
                return
        elif status == 'success' and self.record['date_range']:
            records[self.kind] = self.record
        else:
            return
        write_properties(self.path, records)


@contextmanager
def track_phase(trainer, kind):
    """Keep provenance failures from changing the outcome of training or testing."""
    if trainer.training_run is None:
        yield
        return
    phase = Phase(trainer, kind)
    trainer.property_phase = phase
    status = 'failed'
    try:
        yield
        if phase.record['date_range']:
            from src.proj import Save
            Save.async_wait_all(caller_name='model_properties')
        status = 'success'
    finally:
        trainer.property_phase = None
        try:
            phase.finish(status)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            trainer.logger.warning(f'Model properties write failed: {exc}', vb_level=0)
