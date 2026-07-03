"""Automatic file reader that picks loaders by path type."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.proj import Load
from src.proj.db.basic import TAR_SUFFIXES

__all__ = ['AutomaticReader', 'ReadResult']

ReadMode = Literal['text', 'object']


@dataclass(frozen=True)
class ReadResult:
    """Outcome of :meth:`AutomaticReader.read`."""

    mode: ReadMode
    path: Path
    payload: Any | None
    summary: str


class AutomaticReader:
    """Select a preview strategy from file suffix, tar archive, or mmap directory."""

    TEXT_SUFFIXES: frozenset[str] = frozenset({
        '.py', '.md', '.txt', '.yaml', '.yml', '.json', '.toml', '.log', '.sh',
        '.bat', '.sql', '.ini', '.cfg', '.html', '.css', '.js', '.ts', '.tsx',
        '.jsx', '.xml', '.csv', '.rst', '.lua', '.env',
    })
    TEXT_LINE_LIMIT = 30
    DIRECTORY_LIST_LIMIT = 40

    def read(self, path: Path) -> ReadResult:
        """Load or summarize *path* for CLI preview."""
        if not path.exists():
            raise FileNotFoundError(f'Path does not exist: {path}')

        if path.is_dir():
            if path.name.endswith('.mmap'):
                return self._read_mmap(path)
            return self._read_directory_listing(path)

        suffix = path.suffix.lower()
        name = path.name.lower()

        if suffix == '.feather':
            return self._read_dataframe(path, loader='feather')
        if suffix == '.parquet':
            return self._read_dataframe(path, loader='parquet')
        if suffix in {'.pt', '.pth'}:
            return self._read_torch(path)
        if any(name.endswith(tar_suffix) for tar_suffix in TAR_SUFFIXES):
            return self._read_tar(path)
        if suffix in self.TEXT_SUFFIXES or self._looks_like_text(path):
            return self._read_text(path)
        return self._read_text(path, unknown=True)

    @classmethod
    def _looks_like_text(cls, path: Path) -> bool:
        try:
            sample = path.read_bytes()[:4096]
        except OSError:
            return False
        if b'\x00' in sample:
            return False
        try:
            sample.decode('utf-8')
        except UnicodeDecodeError:
            return False
        return True

    def _read_text(self, path: Path, *, unknown: bool = False) -> ReadResult:
        content = path.read_text(encoding='utf-8', errors='replace')
        lines = content.splitlines()
        preview_lines = lines[: self.TEXT_LINE_LIMIT]
        header = 'Unknown file type; showing as text.' if unknown else 'Text preview'
        body = '\n'.join(preview_lines)
        if len(lines) > self.TEXT_LINE_LIMIT:
            body += f'\n... ({len(lines) - self.TEXT_LINE_LIMIT} more lines)'
        summary = f'{header} ({len(lines)} lines total):\n{body}'
        return ReadResult(mode='text', path=path, payload=None, summary=summary)

    def _read_directory_listing(self, path: Path) -> ReadResult:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        shown = entries[: self.DIRECTORY_LIST_LIMIT]
        lines = [
            f'{"[dir] " if entry.is_dir() else "[file]"} {entry.name}'
            for entry in shown
        ]
        if len(entries) > self.DIRECTORY_LIST_LIMIT:
            lines.append(f'... ({len(entries) - self.DIRECTORY_LIST_LIMIT} more entries)')
        summary = f'Directory listing ({len(entries)} entries):\n' + '\n'.join(lines)
        return ReadResult(mode='text', path=path, payload=None, summary=summary)

    def _read_dataframe(self, path: Path, *, loader: str) -> ReadResult:
        import pandas as pd

        if loader == 'feather':
            obj = Load.df(path)
        else:
            obj = pd.read_parquet(path, engine='fastparquet')
        summary = self._object_summary(obj, path)
        return ReadResult(mode='object', path=path, payload=obj, summary=summary)

    def _read_torch(self, path: Path) -> ReadResult:
        try:
            obj = Load.torch(path, weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f'Failed to load torch checkpoint with weights_only=True: {exc}. '
                'Refusing weights_only=False for safety.',
            ) from exc
        summary = self._object_summary(obj, path)
        return ReadResult(mode='object', path=path, payload=obj, summary=summary)

    def _read_tar(self, path: Path) -> ReadResult:
        obj = Load.dfs(path)
        keys = list(obj.keys()) if isinstance(obj, dict) else []
        summary = self._object_summary(obj, path, extra=f'keys={keys[:10]}')
        return ReadResult(mode='object', path=path, payload=obj, summary=summary)

    def _read_mmap(self, path: Path) -> ReadResult:
        obj = Load.mmap(path, values=True, index=True)
        extra_parts: list[str] = []
        if isinstance(obj, dict):
            extra_parts.append(f'keys={list(obj.keys())}')
            values = obj.get('values')
            shape = getattr(values, 'shape', None)
            if shape is not None:
                extra_parts.append(f'values.shape={shape}')
        extra = ', '.join(extra_parts) if extra_parts else None
        summary = self._object_summary(obj, path, extra=extra)
        return ReadResult(mode='object', path=path, payload=obj, summary=summary)

    @staticmethod
    def _object_summary(obj: Any, path: Path, *, extra: str | None = None) -> str:
        lines = [f'Loaded object from {path.name}:', repr(obj)]
        if hasattr(obj, 'shape'):
            lines.append(f'shape={obj.shape}')
        if hasattr(obj, 'dtypes'):
            lines.append(f'dtypes={obj.dtypes}')
        if extra:
            lines.append(str(extra))
        return '\n'.join(lines)
