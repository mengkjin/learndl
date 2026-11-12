"""Package one dashboard's experiment data and send via the project mail outbox."""
from __future__ import annotations

import sqlite3
import tempfile
import time
import zipfile
import re
from datetime import datetime
from contextlib import closing
from pathlib import Path

__all__ = ['send_experiment', 'experiment_name']


def experiment_name(kind: str, source: Path) -> str:
    """Recover the full model name without importing the training stack."""
    if kind == 'optuna':
        return re.sub(r'@\d{14}$', '', source.stem)
    model = source.parent.parent if source.parent.name == 'snapshot' else source
    if model.parent.name in ('nn', 'boost', 'linear', 'factor', 'st'):
        return f'{model.parent.name}@{model.name}'
    return re.sub(r'\.\d{14}$', '', model.name)


def send_experiment(kind: str, source: Path) -> bool:
    """Send only the log directory or a consistent SQLite snapshot, never model weights."""
    from src.proj import MACHINE
    from src.proj.util.web.emailer import Email

    if not MACHINE.emailable:
        raise ValueError('Email is not configured on this machine')
    source = source.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='learndl-experiment-') as directory:
        work = Path(directory)
        name = experiment_name(kind, source)
        archive = work / f'{name}-{datetime.now().astimezone():%Y%m%d-%H%M%S}.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
            if kind == 'tensorboard':
                if not source.is_dir():
                    raise ValueError('TensorBoard log directory is unavailable')
                # Do not follow links into unrelated model or project data.
                for path in sorted(source.rglob('*')):
                    relative = path.relative_to(source)
                    if any((source / parent).is_symlink() for parent in [relative, *relative.parents]):
                        continue
                    if path.is_file():
                        package.write(path, str(Path(source.name) / relative))
                if not package.namelist():
                    raise ValueError('TensorBoard log directory is empty')
            elif kind == 'optuna':
                if not source.is_file():
                    raise ValueError('Optuna SQLite database is unavailable')
                snapshot = work / 'snapshot.sqlite3'
                deadline = time.monotonic() + 60

                def progress(status, remaining, total):
                    if time.monotonic() > deadline:
                        raise RuntimeError('SQLite snapshot timed out; please retry')

                with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as database:
                    with closing(sqlite3.connect(snapshot)) as destination:
                        database.backup(destination, pages=256, progress=progress)
                package.write(snapshot, source.name)
            else:
                raise ValueError('Unknown dashboard type')
        return Email.send(
            f'{"TensorBoard" if kind == "tensorboard" else "Optuna"} experiment: {name}',
            f'Experiment data: {source.name}\nThe ZIP contains only the TensorBoard logs or Optuna SQLite snapshot.',
            attachments=[archive], project_attachments=False,
        )
