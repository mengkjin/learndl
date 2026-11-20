# coding: utf-8
# description: Rename the saved model status callback to StatusLogger.
# email: False
# mode: os
import os
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from src.shared.file_ops import fsync_dir, real_tree
from src.res.model.util.config.status_logger import OLD, rename_status_callback


def changes(root):
    for directory in sorted(root.glob('*/*/configs')):
        for path in sorted(real_tree(directory).rglob('*')):
            if path.suffix.lower() not in ('.yaml', '.yml') or not real_tree(path).is_file():
                continue
            original = path.read_bytes()
            if OLD.encode() not in original:
                continue
            loaded = yaml.safe_load(original)
            try:
                migrated = rename_status_callback(loaded)
            except ValueError as exc:
                raise ValueError(f'{path}: {exc}') from exc
            if migrated != loaded:
                yield path, original, migrated


def apply(ctx):
    from src.proj import PATH, Logger

    pending = list(changes(PATH.model))
    for path, original, migrated in pending:
        backup = ctx.backup_dir / 'models' / path.relative_to(PATH.model)
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open('xb') as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_dir(backup.parent)
        stream = NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False)
        temporary = Path(stream.name)
        try:
            with stream:
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                yaml.safe_dump(migrated, stream, allow_unicode=True, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            fsync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    Logger.stdout(f'Migrated {len(pending)} model YAML files to StatusLogger')


def verify(ctx):
    from src.proj import PATH

    return next(changes(PATH.model), None) is None
