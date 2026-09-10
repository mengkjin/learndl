"""Safe, bounded parsing and disk caching for task-monitor output."""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, TypeAlias

from src.proj import PATH

OutputStyle: TypeAlias = Literal['text', 'colored']
PARSER_VERSION = 1
MAX_RECORDS_PER_CHUNK = 200
MAX_CHUNK_BYTES = 256 * 1024
MAX_RECORD_CHARS = 64 * 1024
DEFAULT_CACHE_BYTES = 512 * 1024 * 1024
DEFAULT_CACHE_AGE_DAYS = 30

_LOG_PREFIX = re.compile(r'^- \d{2}:\d{2}:\d{2}\.\d{3}:\s*')
_CONTROL_ONLY = re.compile(r'^\^+$')
_SAFE_CSS_PROPERTIES = frozenset((
    'color', 'background-color', 'font-weight', 'font-style', 'text-decoration',
))
_SAFE_CSS_VALUE = re.compile(r'^[#(),.%\w\s-]+$')

__all__ = [
    'CachedOutputPage', 'OutputCache', 'OutputStyle', 'parse_live_output',
    'sanitize_inline_html',
]


@dataclass(frozen=True)
class CachedOutputPage:
    """One bounded output page loaded from the on-disk cache."""

    content: str
    page: int
    total_pages: int
    record_count: int
    cache_hit: bool


def _safe_style(raw_style: str) -> str:
    styles: list[str] = []
    for declaration in raw_style.split(';'):
        if ':' not in declaration:
            continue
        name, value = (part.strip().lower() for part in declaration.split(':', 1))
        if name not in _SAFE_CSS_PROPERTIES or not value or not _SAFE_CSS_VALUE.fullmatch(value):
            continue
        if 'url' in value or 'expression' in value:
            continue
        styles.append(f'{name}: {value}')
    return '; '.join(styles)


class _InlineSanitizer(HTMLParser):
    def __init__(self, style: OutputStyle) -> None:
        super().__init__(convert_charrefs=True)
        self.style = style
        self.parts: list[str] = []
        self.open_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.style == 'text':
            if tag == 'br':
                self.parts.append('\n')
            return
        if tag == 'br':
            self.parts.append('<br>')
        elif tag == 'u':
            self.parts.append('<u>')
            self.open_tags.append('u')
        elif tag == 'span':
            safe = _safe_style(dict(attrs).get('style') or '')
            if safe:
                self.parts.append(f'<span style="{html.escape(safe, quote=True)}">')
                self.open_tags.append('span')

    def handle_endtag(self, tag: str) -> None:
        if self.style == 'colored' and tag in ('span', 'u') and tag in self.open_tags:
            while self.open_tags:
                opened = self.open_tags.pop()
                self.parts.append(f'</{opened}>')
                if opened == tag:
                    break

    def handle_data(self, data: str) -> None:
        self.parts.append(data if self.style == 'text' else html.escape(data))

    def result(self) -> str:
        if self.style == 'colored':
            while self.open_tags:
                self.parts.append(f'</{self.open_tags.pop()}>')
        return ''.join(self.parts)


def sanitize_inline_html(fragment: str, style: OutputStyle) -> str:
    """Retain only catcher-generated inline color markup."""
    parser = _InlineSanitizer(style)
    parser.feed(fragment)
    parser.close()
    return parser.result()


def _plain_payload(line: str) -> str:
    return _LOG_PREFIX.sub('', sanitize_inline_html(line, 'text')).strip()


def _compact_records(records: Iterator[str], style: OutputStyle) -> Iterator[str]:
    suppressed = 0
    for record in records:
        payload = _plain_payload(record)
        if not payload or _CONTROL_ONLY.fullmatch(payload):
            suppressed += 1
            continue
        if suppressed:
            message = f'… {suppressed} blank/control-output lines suppressed …'
            yield html.escape(message) if style == 'colored' else message
            suppressed = 0
        sanitized = sanitize_inline_html(record.rstrip(), style)
        if len(sanitized) > MAX_RECORD_CHARS:
            sanitized = sanitized[:MAX_RECORD_CHARS] + ' … [record truncated]'
        yield sanitized
    if suppressed:
        message = f'… {suppressed} blank/control-output lines suppressed …'
        yield html.escape(message) if style == 'colored' else message


def _iter_markdown(path: Path, style: OutputStyle) -> Iterator[str]:
    with path.open(encoding='utf-8', errors='replace') as source:
        yield from _compact_records(iter(source), style)


class _HtmlOutputParser(HTMLParser):
    """Stream HtmlCatcher rows without retaining the source document."""

    def __init__(self, style: OutputStyle, emit: Callable[[str], None]) -> None:
        super().__init__(convert_charrefs=True)
        self.style = style
        self.emit = emit
        self.in_row = False
        self.cell: str | None = None
        self.fields: dict[str, list[str]] = {}
        self.inline_tags: list[str] = []
        self.plain_content: list[str] = []
        self.suppressed = 0
        self.content_chars = 0
        self.content_truncated = False

    def _append_content(self, rendered: str, plain: str = '') -> None:
        if self.content_truncated:
            return
        remaining = MAX_RECORD_CHARS - self.content_chars
        if remaining <= 0:
            self.fields['content'].append(' … [record truncated]')
            self.content_truncated = True
            return
        if len(rendered) > remaining:
            self.fields['content'].append(rendered[:remaining] + ' … [record truncated]')
            self.plain_content.append(plain[:remaining])
            self.content_chars = MAX_RECORD_CHARS
            self.content_truncated = True
            return
        self.fields['content'].append(rendered)
        self.plain_content.append(plain)
        self.content_chars += len(rendered)

    def _emit_suppressed(self) -> None:
        if not self.suppressed:
            return
        message = f'… {self.suppressed} blank/control-output lines suppressed …'
        if self.style == 'colored':
            self.emit(f'<div class="output-line muted">{html.escape(message)}</div>')
        else:
            self.emit(message)
        self.suppressed = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get('class') or '').split())
        if tag == 'tr' and 'output-row' in classes:
            self.in_row = True
            self.fields = {'type': [], 'time': [], 'content': []}
            self.plain_content = []
            self.content_chars = 0
            self.content_truncated = False
            return
        if not self.in_row:
            return
        if tag == 'td':
            if 'type-cell' in classes:
                self.cell = 'type'
            elif 'time-cell' in classes:
                self.cell = 'time'
            elif 'content-cell' in classes:
                self.cell = 'content'
            return
        if self.cell != 'content' or self.style != 'colored':
            return
        if tag == 'br':
            self._append_content('<br>', '\n')
        elif tag == 'u':
            if not self.content_truncated:
                self._append_content('<u>')
                self.inline_tags.append('u')
        elif tag == 'span':
            safe = _safe_style(attributes.get('style') or '')
            if safe and not self.content_truncated:
                self._append_content(f'<span style="{html.escape(safe, quote=True)}">')
                self.inline_tags.append('span')

    def handle_endtag(self, tag: str) -> None:
        if not self.in_row:
            return
        if tag == 'td':
            self.cell = None
        elif tag in ('span', 'u') and self.style == 'colored' and tag in self.inline_tags:
            while self.inline_tags:
                opened = self.inline_tags.pop()
                self.fields['content'].append(f'</{opened}>')
                if opened == tag:
                    break
        elif tag == 'tr':
            while self.inline_tags:
                self.fields['content'].append(f'</{self.inline_tags.pop()}>')
            kind = ''.join(self.fields['type']).strip()
            when = ''.join(self.fields['time']).strip()
            content = ''.join(self.fields['content']).strip()
            plain_content = ''.join(self.plain_content).strip()
            if not plain_content or _CONTROL_ONLY.fullmatch(plain_content):
                self.suppressed += 1
            else:
                self._emit_suppressed()
                prefix = f'{when} [{kind}] '.strip() + ' '
                if self.style == 'colored':
                    kind_class = ' stderr' if kind.casefold() == 'stderr' else ''
                    self.emit(
                        f'<div class="output-line{kind_class}">'
                        f'<span class="meta">{html.escape(prefix)}</span>{content}</div>'
                    )
                else:
                    self.emit(prefix + content)
            self.in_row = False
            self.cell = None

    def handle_data(self, data: str) -> None:
        if not self.in_row or self.cell is None:
            return
        if self.cell == 'content':
            self._append_content(data if self.style == 'text' else html.escape(data), data)
        else:
            self.fields[self.cell].append(data if self.style == 'text' else html.escape(data))

    def close(self) -> None:
        super().close()
        self._emit_suppressed()


def _parse_html(path: Path, style: OutputStyle, emit: Callable[[str], None]) -> None:
    parser = _HtmlOutputParser(style, emit)
    with path.open(encoding='utf-8', errors='replace') as source:
        while chunk := source.read(256 * 1024):
            parser.feed(chunk)
    parser.close()


def parse_live_output(
    path: Path | None, style: OutputStyle, *, max_bytes: int = 512 * 1024, max_lines: int = 500,
) -> str:
    """Parse a bounded tail for a selected running task without disk caching."""
    if path is None or not path.is_file():
        return ''
    try:
        with path.open('rb') as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - max_bytes))
            raw = source.read()
    except OSError:
        return ''
    text = raw.decode('utf-8', errors='replace')
    if size > max_bytes:
        text = text.split('\n', 1)[-1]
    records = list(_compact_records(iter(text.splitlines()), style))[-max_lines:]
    return '\n'.join(records) if style == 'text' else ''.join(f'<div>{record}</div>' for record in records)


class OutputCache:
    """Chunked cache for immutable, completed task output."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else PATH.cache / 'task_monitor_outputs' / f'v{PARSER_VERSION}'

    @staticmethod
    def _base_key(task_id: str, source: Path) -> str:
        return hashlib.sha256(f'{task_id}\0{source}'.encode()).hexdigest()[:24]

    @staticmethod
    def _fingerprint(source: Path) -> dict[str, int] | None:
        try:
            stat = source.stat()
        except OSError:
            return None
        return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

    def _base_dir(self, task_id: str, source: Path, style: OutputStyle) -> Path:
        return self.root / style / self._base_key(task_id, source)

    def _find_cache(self, task_id: str, source: Path, style: OutputStyle) -> tuple[Path | None, bool]:
        base_dir = self._base_dir(task_id, source, style)
        fingerprint = self._fingerprint(source)
        if fingerprint is not None:
            fingerprint_key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
            candidate = base_dir / fingerprint_key
            return (candidate if (candidate / 'manifest.json').is_file() else None), False
        manifests_with_time: list[tuple[float, Path]] = []
        for manifest in base_dir.glob('*/manifest.json'):
            try:
                manifests_with_time.append((manifest.stat().st_mtime, manifest))
            except OSError:
                continue
        manifests = [item for _, item in sorted(manifests_with_time, reverse=True)]
        return (manifests[0].parent if manifests else None), True

    def ensure(self, task_id: str, source: Path, style: OutputStyle) -> Path | None:
        """Return a valid cache directory, building it once when necessary."""
        cached, source_missing = self._find_cache(task_id, source, style)
        if cached is not None:
            try:
                os.utime(cached / 'manifest.json')
                return cached
            except OSError:
                pass
        if source_missing or not source.is_file():
            return None
        fingerprint = self._fingerprint(source)
        assert fingerprint is not None
        base_dir = self._base_dir(task_id, source, style)
        fingerprint_key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        target = base_dir / fingerprint_key
        base_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix='.building-', dir=base_dir))
        try:
            chunk: list[str] = []
            chunk_bytes = 0
            chunk_count = 0
            record_count = 0

            def flush() -> None:
                nonlocal chunk, chunk_bytes, chunk_count
                if not chunk:
                    return
                with gzip.open(temp_dir / f'chunk-{chunk_count:06d}.json.gz', 'wt', encoding='utf-8') as output:
                    json.dump(chunk, output, ensure_ascii=False)
                chunk_count += 1
                chunk = []
                chunk_bytes = 0

            def add(record: str) -> None:
                nonlocal chunk_bytes, record_count
                size = len(record.encode('utf-8'))
                if chunk and (len(chunk) >= MAX_RECORDS_PER_CHUNK or chunk_bytes + size > MAX_CHUNK_BYTES):
                    flush()
                chunk.append(record)
                chunk_bytes += size
                record_count += 1

            if source.suffix.lower() in ('.html', '.htm'):
                _parse_html(source, style, add)
            else:
                for record in _iter_markdown(source, style):
                    add(record)
            flush()
            manifest = {
                'parser_version': PARSER_VERSION,
                'task_id': task_id,
                'source': str(source),
                'fingerprint': fingerprint,
                'style': style,
                'record_count': record_count,
                'chunk_count': chunk_count,
                'created_at': time.time(),
            }
            (temp_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
            try:
                os.replace(temp_dir, target)
            except OSError:
                if not target.is_dir():
                    raise
                shutil.rmtree(temp_dir, ignore_errors=True)
            return target
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def load_page(
        self, task_id: str, source: Path, style: OutputStyle, page: int | None = None,
    ) -> CachedOutputPage | None:
        cached_before, _ = self._find_cache(task_id, source, style)
        cache_dir = self.ensure(task_id, source, style)
        if cache_dir is None:
            return None
        try:
            manifest = json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8'))
            total_pages = int(manifest['chunk_count'])
            if total_pages == 0:
                return CachedOutputPage('', 0, 0, int(manifest['record_count']), cached_before is not None)
            selected_page = total_pages - 1 if page is None else min(max(page, 0), total_pages - 1)
            with gzip.open(cache_dir / f'chunk-{selected_page:06d}.json.gz', 'rt', encoding='utf-8') as cached:
                records: list[str] = json.load(cached)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        separator = '\n' if style == 'text' else ''
        return CachedOutputPage(
            separator.join(records), selected_page, total_pages,
            int(manifest['record_count']), cached_before is not None,
        )

    def ensure_both(self, task_id: str, source: Path) -> bool:
        """Prewarm both display variants before a recovered raw log is removed."""
        return all(self.ensure(task_id, source, style) is not None for style in ('text', 'colored'))

    def cleanup(
        self, *, max_bytes: int = DEFAULT_CACHE_BYTES, max_age_days: int = DEFAULT_CACHE_AGE_DAYS,
    ) -> dict[str, int]:
        """Apply age and total-size limits using manifest mtime as last access."""
        if not self.root.is_dir():
            return {'removed_entries': 0, 'removed_bytes': 0, 'remaining_bytes': 0}
        now = time.time()
        removed_entries = 0
        removed_bytes = 0
        for building_dir in self.root.glob('*/*/.building-*'):
            try:
                if building_dir.stat().st_mtime >= now - 3600:
                    continue
                size = sum(path.stat().st_size for path in building_dir.rglob('*') if path.is_file())
            except OSError:
                continue
            shutil.rmtree(building_dir, ignore_errors=True)
            removed_entries += 1
            removed_bytes += size
        entries: list[tuple[float, int, Path]] = []
        for manifest in self.root.glob('*/*/*/manifest.json'):
            cache_dir = manifest.parent
            try:
                size = sum(path.stat().st_size for path in cache_dir.iterdir() if path.is_file())
                entries.append((manifest.stat().st_mtime, size, cache_dir))
            except OSError:
                continue
        keep: list[tuple[float, int, Path]] = []
        cutoff = now - max_age_days * 86400
        for accessed, size, cache_dir in entries:
            if accessed < cutoff:
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed_entries += 1
                removed_bytes += size
            else:
                keep.append((accessed, size, cache_dir))
        remaining = sum(size for _, size, _ in keep)
        for _, size, cache_dir in sorted(keep):
            if remaining <= max_bytes:
                break
            shutil.rmtree(cache_dir, ignore_errors=True)
            remaining -= size
            removed_entries += 1
            removed_bytes += size
        return {
            'removed_entries': removed_entries,
            'removed_bytes': removed_bytes,
            'remaining_bytes': remaining,
        }
