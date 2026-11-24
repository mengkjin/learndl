"""Immutable log ranges with journaled publication and two-stage compaction."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

CHUNK_BYTES = 256 * 1024
ROUND_BYTES = 4 * 1024 * 1024
RETENTION = 86400


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save(path, value):
    from .remote import atomic_json
    atomic_json(path, value)


class Publisher:
    def __init__(self, shared: Path, private: Path, host: str):
        self.shared = shared / 'runtime' / key(host)
        self.private = private / 'remote_monitor'
        self.host = host

    def _finish(self, journal):
        for operation in journal['writes']:
            target = self.shared / operation['target']
            source = self.private / operation['source']
            target.parent.mkdir(parents=True, exist_ok=True)
            if operation['immutable'] and target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise ValueError('Immutable log range has conflicting contents')
                continue
            temporary = target.with_suffix('.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        save(self.private / 'state.json', journal['state'])
        for name in journal['delete']:
            path = self.shared / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        for name in journal['pending_delete']:
            (self.private / 'pending' / name).unlink(missing_ok=True)
        (self.private / 'journal.json').unlink()
        shutil.rmtree(self.private / 'staging', ignore_errors=True)

    def tick(self, tasks, *, now=None, limits=None, hidden=()):
        now = time.time() if now is None else now
        limits = {} if limits is None else limits
        self.private.mkdir(parents=True, exist_ok=True)
        journal_path = self.private / 'journal.json'
        if journal_path.exists():
            self._finish(json.loads(journal_path.read_text()))
        staging = self.private / 'staging'
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        state_path = self.private / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'tasks': {}, 'cursor': 0}
        writes, deleted, pending_delete = [], [], []
        self._staged = {}

        def stage(target, data, immutable=False):
            source = Path('staging') / str(len(writes))
            (self.private / source).write_bytes(data)
            self._staged[str(target)] = self.private / source
            writes.append(dict(target=str(target), source=str(source), immutable=immutable))

        def metadata(target, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            digest = hashlib.sha256(data).hexdigest()
            published = state.setdefault('metadata', {})
            if published.get(str(target)) != digest:
                stage(target, data)
                published[str(target)] = digest

        for task in tasks:
            record = state['tasks'].setdefault(key(task['task_id']), {'offset': 0, 'generation': 0})
            record['task'] = task
            record.pop('hidden_since', None)
        for task_id in hidden:
            if (record := state['tasks'].get(key(task_id))) is not None:
                record.setdefault('hidden_since', now)
        summaries = []
        keys = sorted(state['tasks'])
        if keys:
            pivot = state['cursor'] % len(keys)
            keys = keys[pivot:] + keys[:pivot]
        budget = ROUND_BYTES
        stamp = datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        for identity in keys:
            record = state['tasks'][identity]
            task = record['task']
            directory = Path('tasks') / identity
            hidden_since = record.get('hidden_since')
            if (record.get('expires_at', float('inf')) <= now
                    or hidden_since is not None and hidden_since + RETENTION <= now):
                deleted.append(str(directory))
                pending_delete.extend([identity + '.md', identity + '.json'])
                del state['tasks'][identity]
                state.get('metadata', {}).pop(str(directory / 'manifest.json'), None)
                continue
            if hidden_since is not None:
                # Retain its output briefly in case a distinct scheduler phase returns.
                continue
            # Adopt v1 volumes as immutable historical ranges, including its partial last volume.
            if 'segments' not in record:
                record['segments'] = []
                histories = record.get('previous', []) + [record]
                for history in histories:
                    for start in range(0, history['offset'], CHUNK_BYTES):
                        record['segments'].append(dict(generation=history['generation'], start=start,
                            end=min(start + CHUNK_BYTES, history['offset']), kind='sealed',
                            file=f"{history['generation']}-{start // CHUNK_BYTES:08d}.bin"))
                record.pop('preview', None)
                record.pop('preview_mtime', None)
            pending = self.private / 'pending' / (identity + '.md')
            source = pending if pending.exists() else Path(task['log']) if task.get('log') else None
            active = task['status'] in ('starting', 'running')
            if source and not record.get('expires_at') and source.is_file():
                with source.open('rb') as stream:
                    stat = os.fstat(stream.fileno())
                    source_identity = [stat.st_dev, stat.st_ino]
                    if record.get('identity') not in (None, source_identity) or stat.st_size < record['offset']:
                        record.setdefault('previous', []).append(dict(generation=record['generation'], offset=record['offset']))
                        record.update(generation=record['generation'] + 1, offset=0, gap=True)
                    record['identity'] = source_identity
                    stream.seek(record['offset'])
                    target = min(stat.st_size, limits.setdefault((identity, record['generation']), stat.st_size))
                    available = min(target - record['offset'], budget,
                                    max(CHUNK_BYTES, ROUND_BYTES // max(1, len(keys))))
                    while available:
                        data = stream.read(min(available, CHUNK_BYTES))
                        if not data:
                            break
                        start = record['offset']
                        end = start + len(data)
                        filename = f"{record['generation']}-{start:016d}-{end:016d}-{stamp}.bin"
                        stage(directory / filename, data, True)
                        record['segments'].append(dict(generation=record['generation'], start=start,
                            end=end, kind='sealed' if len(data) == CHUNK_BYTES else 'raw', file=filename))
                        record['offset'] = end
                        record['output_at'] = stat.st_mtime
                        budget -= len(data)
                        available -= len(data)
                    record.update(size=stat.st_size, complete=record['offset'] == stat.st_size)
            elif not active and not record.get('expires_at'):
                record.update(complete=True, gap=True)
            if not active and record.get('complete') and 'expires_at' not in record:
                record['expires_at'] = now + RETENTION
            if not record.get('expires_at') or record['expires_at'] == now + RETENTION:
                self._compact(record, directory, stage, deleted)
            summary = dict(task, key=identity, generation=record['generation'], offset=record['offset'],
                           size=record.get('size', 0), complete=record.get('complete', False),
                           output_at=record.get('output_at'), gap=record.get('gap', False),
                           previous=record.get('previous', []), expires_at=record.get('expires_at'),
                           segments=record['segments'], version=2)
            summary.pop('log', None)
            metadata(directory / 'manifest.json', summary)
            summaries.append(summary)
        summaries.sort(key=lambda task: task['key'])
        index_tasks = [{name: value for name, value in task.items() if name in
                       {'task_id', 'script', 'cmd', 'status', 'start_time', 'end_time', 'key', 'phase', 'pid_alive'}}
                       for task in summaries]
        metadata(Path('index.json'), dict(version=2, host=self.host, tasks=index_tasks))
        metadata(Path('heartbeat.json'), dict(sampled_at=now))
        state['cursor'] += 1
        state['collected_at'] = now
        journal = dict(state=state, writes=writes, delete=deleted, pending_delete=pending_delete)
        save(journal_path, journal)
        self._finish(journal)
        return dict(version=2, host=self.host, sampled_at=now, tasks=summaries)

    def _compact(self, record, directory, stage, deleted):
        segments = record['segments']
        raw = [s for s in segments if s['kind'] == 'raw']
        protected = {s['file'] for s in raw[-3:]}
        def merge(candidates, kind):
            data = b''.join(self._staged_path(directory / part['file']).read_bytes() for part in candidates)
            first, last = candidates[0], candidates[-1]
            filename = f"{first['generation']}-{first['start']:016d}-{last['end']:016d}-merged.bin"
            stage(directory / filename, data, True)
            merged = dict(generation=first['generation'], start=first['start'], end=last['end'],
                          kind=kind, file=filename)
            position = segments.index(first)
            for candidate in candidates:
                segments.remove(candidate)
                deleted.append(str(directory / candidate['file']))
            segments.insert(position, merged)

        # Pack raw fragments first. Intermediate fragments then participate in size compaction.
        for raw_only in (True, False):
            candidates = []
            size = 0
            for segment in list(segments):
                eligible = segment['kind'] == 'raw' if raw_only else segment['kind'] != 'sealed'
                if not eligible or segment['file'] in protected:
                    candidates, size = [], 0
                    continue
                if candidates and (candidates[-1]['end'] != segment['start'] or
                                   candidates[-1]['generation'] != segment['generation']):
                    candidates, size = [], 0
                candidates.append(segment)
                size += segment['end'] - segment['start']
                if size >= CHUNK_BYTES or (raw_only and len(candidates) >= 64):
                    merge(candidates, 'sealed' if size >= CHUNK_BYTES else 'intermediate')
                    candidates, size = [], 0

    def _staged_path(self, target):
        # Resolve an in-flight immutable range without opening the source log again.
        return self._staged.get(str(target), self.shared / target)
