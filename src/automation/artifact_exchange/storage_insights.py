"""Publish one static daily storage summary and discover synced reports."""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
from datetime import datetime
from threading import Lock
from html.parser import HTMLParser
from pathlib import Path

import portalocker

from .core import Config, component, sandboxed

__all__ = ['metadata', 'embed_metadata', 'publish_daily', 'publish_daily_safe', 'ReportCatalog']


class _Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        self.capture = tag == 'script' and dict(attrs).get('id') == 'storage-metadata'

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.capture = False


def metadata(content):
    if '</html>' not in content.lower():
        raise ValueError('Incomplete storage report')
    parser = _Metadata()
    parser.feed(content)
    value = json.loads(''.join(parser.parts))
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('Unsupported storage report')
    for field in ('root', 'started', 'finished', 'report_id'):
        if not isinstance(value.get(field), str):
            raise ValueError('Invalid storage report metadata')
    datetime.fromisoformat(value['started'])
    datetime.fromisoformat(value['finished'])
    return value


def embed_metadata(content, values):
    content = re.sub(r'<script id="storage-metadata" type="application/json">.*?</script>', '', content, flags=re.S)
    encoded = json.dumps(values, ensure_ascii=False).replace('<', '\\u003c')
    tag = f'<script id="storage-metadata" type="application/json">{encoded}</script>'
    if '</head>' in content:
        return content.replace('</head>', tag + '</head>', 1)
    return content.replace('<style>', tag + '<style>', 1)


def publish_daily(root, updated_at, *, config=None, now=None, force=False, report=None):
    from src.proj.env.path import storage_reports
    from src.proj.util.filesys.storage_report import scan_storage, render_html
    config = config or Config.load()
    if not config.enabled or config.role != 'publisher' or sandboxed():
        return 'disabled'
    root = Path(root).resolve(strict=True)
    now = now or datetime.now().astimezone()
    today = now.date().isoformat()
    target = config.shared_root / 'persistent' / 'storage_insights' / component(config.source_host) / 'latest.html'
    private = config.state / 'storage_insights'
    private.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(private / 'publish.lock'), timeout=0):
            try:
                current = metadata(target.read_text(encoding='utf-8'))
                if not force and current.get('day') == today and current['root'] == str(root) and current.get('source_host') == config.source_host:
                    return 'already published'
            except (OSError, ValueError):
                pass
            content = None
            if report is not None:
                content = Path(report).read_text(encoding='utf-8')
                if metadata(content)['root'] != str(root):
                    raise ValueError('Storage report root does not match publication root')
            candidates = [] if content is not None else sorted(
                storage_reports(root).glob('*.html'), key=lambda p: p.stat().st_mtime, reverse=True)
            for candidate_path in candidates:
                try:
                    candidate = candidate_path.read_text(encoding='utf-8')
                    info = metadata(candidate)
                    if info['root'] == str(root) and datetime.fromisoformat(info['started']).timestamp() >= updated_at:
                        content = candidate
                        break
                except (OSError, ValueError):
                    continue
            if content is None:
                content = render_html(scan_storage(root))
            info = dict(metadata(content), source_host=config.source_host, day=today)
            content = embed_metadata(content, info)
            content = content.replace('<h1>Project Storage Report</h1>',
                                      '<h1>Project Storage Report</h1><p>Source: ' + html.escape(config.source_host) + '</p>', 1)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                                 prefix='.storage-', suffix='.tmp', delete=False) as output:
                    temporary = Path(output.name)
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return 'published'
    except portalocker.LockException:
        return 'publication in progress'


def publish_daily_safe(root, updated_at, *, force=False, report=None):
    from src.proj import Logger
    try:
        result = publish_daily(root, updated_at, force=force, report=report)
        Logger.stdout(f'Storage insights: {result}')
    except Exception as exc:
        Logger.error(f'Storage insights publication failed: {exc}')


class ReportCatalog:
    """Cache metadata and HTML by local replacement identity, retaining the last good report."""
    def __init__(self, shared_root):
        self.root = Path(shared_root) / 'persistent' / 'storage_insights'
        self.entries = {}
        self.lock = Lock()

    def refresh(self):
        with self.lock:
            return self._refresh()

    def _refresh(self):
        found = set()
        for path in self.root.glob('*/latest.html'):
            identity = path.parent.name
            if identity.startswith('.') or not identity:
                continue
            found.add(identity)
            try:
                stat = path.stat()
                revision = f'{stat.st_ino}-{stat.st_mtime_ns}-{stat.st_size}'
                if self.entries.get(identity, {}).get('revision') == revision:
                    continue
                content = path.read_text(encoding='utf-8')
                info = metadata(content)
                if not isinstance(info.get('source_host'), str) or component(info['source_host']) != identity:
                    continue
                self.entries[identity] = dict(info, id=identity, revision=revision, content=content)
            except (OSError, ValueError):
                continue
        for identity in set(self.entries) - found:
            del self.entries[identity]
        return [{name: value for name, value in report.items() if name != 'content'}
                for report in self.entries.values()]
