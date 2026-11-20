"""Stable identity for one incarnation of a source model directory."""
from datetime import datetime, timedelta, timezone
import json
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

import portalocker

from src.file_ops import atomic_json, fsync_dir, real_tree
from src.model_provenance import PROPERTY_FILE, read_properties, validate_build
from .model_properties import write_properties

def build_identity(base, *, fresh=False, reset=False):
    from src.proj import PATH

    path = real_tree(base / PROPERTY_FILE)
    state = real_tree(PATH.lc_machine / 'model_builds')
    state.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(state / 'identity.lock'), timeout=10):
        props = read_properties(path)
        if 'build' in props:
            return validate_build(props['build'])

        stat = base.stat()
        source = 'directory_birthtime'
        created = getattr(stat, 'st_birthtime', None)
        if created is None and not reset and sys.platform.startswith('linux'):
            try:
                result = subprocess.run(['stat', '--format=%W', '--', str(base)], capture_output=True,
                                        text=True, check=True, timeout=3)
                created = int(result.stdout) or None
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if reset:
            created = datetime.now(timezone.utc).timestamp()
            source = 'reset_observed'
        elif created is None:
            source = 'creation_observed' if fresh else 'estimated_file_mtime'
            created = (datetime.now(timezone.utc).timestamp() if fresh else
                       min((p.stat().st_mtime for p in (base / 'configs').glob('*.yaml')), default=stat.st_mtime))
        when = datetime.fromtimestamp(created, timezone.utc)
        prefix = when.strftime('%Y%m%d.%H%M%S')
        ledger = state / 'names.json'
        names = json.loads(ledger.read_text()) if ledger.exists() else {}
        counters = names.setdefault(base.parent.name + '/' + base.name, {})
        number = counters.get(prefix, -1) + 1
        name = f'build{prefix}.{number}'
        counters[prefix] = number
        atomic_json(ledger, names)
        props['build'] = {'kind': 'build', 'name': name, 'fingerprint': uuid4().hex,
                          'created_at': when.isoformat(), 'time_source': source}
        write_properties(path, props)
        return props['build']


def preserve_build(model):
    """Retain visible artifacts before an explicit from-scratch reset."""
    from src.proj import PATH

    base = model.base
    if not any((base / folder).is_dir() and any((base / folder).iterdir())
               for folder in ('configs', 'logs', 'run_logs', 'results')):
        return
    build = build_identity(base)
    archive = real_tree(PATH.model_archive)
    archive.mkdir(parents=True, exist_ok=True)
    for item in archive.glob(model.full_name + '.*'):
        original = read_properties(item / PROPERTY_FILE).get('build', {})
        if original.get('fingerprint') == build['fingerprint']:
            return

    when = datetime.now()
    while (target := archive / f'{model.full_name}.{when:%Y%m%d%H%M%S}').exists():
        when += timedelta(seconds=1)
    with TemporaryDirectory(dir=archive, prefix='.preserve-') as temporary:
        from pathlib import Path

        staging = Path(temporary) / 'model'
        staging.mkdir()
        for folder in ('configs', 'logs', 'run_logs', 'results'):
            source = real_tree(base / folder)
            if source.is_dir():
                shutil.copytree(source, staging / folder, symlinks=True)
        (staging / '.build-preview-only').touch()
        staging.rename(target)
        fsync_dir(archive)
