"""Validated ZIP imports cached separately from downloaded experiment attachments."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

__all__ = ['unpack_experiment']


def unpack_experiment(kind: str, archive: Path, cache: Path) -> Path:
    """Reuse immutable imports; reject unsafe paths, links and oversized archives."""
    stamp = archive.stat()
    key = hashlib.sha256(f'{archive}:{stamp.st_size}:{stamp.st_mtime_ns}:{kind}'.encode()).hexdigest()
    target = cache / key
    marker = target / 'source.json'
    if marker.exists():
        return target / json.loads(marker.read_text())['source']
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache, prefix='.import-') as temporary:
        work = Path(temporary)
        name = re.sub(r'-\d{8}-\d{6}$', '', archive.stem)
        logs = work / name / 'snapshot' / 'tensorboard'
        logs.mkdir(parents=True)
        with zipfile.ZipFile(archive) as package:
            entries = package.infolist()
            if len(entries) > 50000 or sum(item.file_size for item in entries) > 4 * 1024**3:
                raise ValueError('Experiment ZIP exceeds import limits (50000 files / 4 GiB)')
            for item in entries:
                path = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                        or ':' in item.filename or stat.S_ISLNK(mode)):
                    raise ValueError('Unsafe path or symbolic link in experiment ZIP')
                if item.is_dir():
                    continue
                destination = logs.joinpath(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(item) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output)
        if kind == 'tensorboard':
            if not any(p.is_file() for p in logs.rglob('*tfevents*')):
                raise ValueError('No TensorBoard event files in ZIP')
            selected = logs
        elif kind == 'optuna':
            databases = []
            for file in logs.rglob('*'):
                if file.is_file():
                    with file.open('rb') as stream:
                        if stream.read(16) == b'SQLite format 3\x00':
                            databases.append(file)
            if len(databases) != 1:
                raise ValueError('Optuna ZIP must contain exactly one SQLite database')
            selected = databases[0]
        else:
            raise ValueError('Unknown dashboard type')
        relative = selected.relative_to(work)
        (work / 'source.json').write_text(json.dumps({'source': str(relative)}))
        # The caller serializes imports with the instance operations lock.
        work.rename(target)
    return target / relative
