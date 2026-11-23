"""Session-local range cache independent of transport file compaction."""
from __future__ import annotations

import codecs
import json
from collections import OrderedDict
from pathlib import Path

from .remote import CHUNK_BYTES, task_key


class RemoteCache:
    def __init__(self, max_bytes=64 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.ranges = OrderedDict()
        self.total = 0
        self.documents = {}
        self.rendered = OrderedDict()

    def document(self, path):
        try:
            stat = path.stat()
            token = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            previous = self.documents.get(str(path))
            if previous is None or previous[0] != token:
                value = json.loads(path.read_text())
                self.documents[str(path)] = (token, value)
            return self.documents[str(path)][1]
        except (OSError, ValueError):
            previous = self.documents.get(str(path))
            return previous[1] if previous else None

    def _trim(self):
        while self.total > self.max_bytes and (self.rendered or self.ranges):
            if self.rendered:
                _, text = self.rendered.popitem(last=False)
                self.total -= len(text.encode('utf-8'))
            else:
                _, data = self.ranges.popitem(last=False)
                self.total -= len(data)

    def render(self, root, task, start, end):
        identity = (str(root), task['task_id'], task['generation'], start, end)
        if identity in self.rendered:
            self.rendered.move_to_end(identity)
            return self.rendered[identity]
        text = self.text(root, task, start, end)
        if text is None:
            return None
        from .output import sanitize_inline_html
        text = sanitize_inline_html(text, 'text')
        self.rendered[identity] = text
        self.total += len(text.encode('utf-8'))
        self._trim()
        return text

    def clear_task(self, root, task_id):
        prefix = (str(root), task_id)
        for identity in list(self.ranges):
            if identity[:2] == prefix:
                self.total -= len(self.ranges.pop(identity))
        for identity in list(self.rendered):
            if identity[:2] == prefix:
                self.total -= len(self.rendered.pop(identity).encode('utf-8'))
        self.documents.pop(str(root / 'tasks' / task_key(task_id) / 'manifest.json'), None)

    def prune(self, root, task_ids):
        for identity in list(self.ranges):
            if identity[0] == str(root) and identity[1] not in task_ids:
                self.total -= len(self.ranges.pop(identity))
        for identity in list(self.rendered):
            if identity[0] == str(root) and identity[1] not in task_ids:
                self.total -= len(self.rendered.pop(identity).encode('utf-8'))
        live = {task_key(task_id) for task_id in task_ids}
        for name in list(self.documents):
            path = Path(name)
            if path.parent.parent == root / 'tasks' and path.parent.name not in live:
                del self.documents[name]

    def read(self, root, task, start, end):
        identity = (str(root), task['task_id'], task['generation'])
        position = start
        parts = []
        segments = task.get('segments')
        if segments is None:
            segments = [dict(generation=task['generation'], start=offset,
                             end=min(offset + CHUNK_BYTES, task['offset']),
                             file=f"{task['generation']}-{offset // CHUNK_BYTES:08d}.bin")
                        for offset in range(0, task['offset'], CHUNK_BYTES)]
        while position < end:
            cached = next((key for key in self.ranges if key[:3] == identity and key[3] <= position < key[4]), None)
            if cached is not None:
                data = self.ranges[cached]
                stop = min(end, cached[4])
                parts.append(data[position - cached[3]:stop - cached[3]])
                self.ranges.move_to_end(cached)
            else:
                segment = next((s for s in segments if s['generation'] == task['generation'] and s['start'] <= position < s['end']), None)
                if segment is None:
                    return None
                stop = min(end, segment['end'])
                starts = [key[3] for key in self.ranges if key[:3] == identity and position < key[3] < stop]
                if starts:
                    stop = min(starts)
                filename = segment['file']
                if Path(filename).name != filename:
                    raise ValueError('Invalid remote log filename')
                path = root / 'tasks' / task_key(task['task_id']) / filename
                try:
                    with path.open('rb') as stream:
                        stream.seek(position - segment['start'])
                        data = stream.read(stop - position)
                except OSError:
                    return None
                if len(data) != stop - position:
                    return None
                cache_key = (*identity, position, stop)
                self.ranges[cache_key] = data
                self.total += len(data)
                self._trim()
                parts.append(data)
            position = stop
        return b''.join(parts)

    def text(self, root, task, start, end):
        prefix = max(0, start - 3)
        raw = self.read(root, task, prefix, end)
        if raw is None:
            return None
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        decoder.decode(raw[:start - prefix], final=False)
        return decoder.decode(raw[start - prefix:], final=False)
