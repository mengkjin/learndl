"""Stable identity for one incarnation of a source model directory."""
from datetime import datetime, timedelta, timezone
import json
import re
import subprocess
import sys
from uuid import uuid4

import portalocker

from .core import atomic_json, real_tree
from .model_properties import PROPERTY_FILE, read_properties, write_properties

BUILD_NAME = re.compile(r'build[0-9]{8}\.[0-9]{6}\.[0-9]+')
PHASES = ('train', 'test')


def validate_build(value):
    if (not isinstance(value, dict) or not BUILD_NAME.fullmatch(value.get('name', ''))
            or not re.fullmatch(r'[0-9a-f]{32}', value.get('fingerprint', ''))):
        raise ValueError('Invalid model build identity')
    return value


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
    build_identity(base)
    when = datetime.now()
    while (target := PATH.model_archive / f'{model.full_name}.{when:%Y%m%d%H%M%S}').exists():
        when += timedelta(seconds=1)
    target.mkdir(parents=True)
    for folder in ('configs', 'logs', 'run_logs', 'results'):
        source = real_tree(base / folder)
        if source.is_dir():
            source.rename(target / folder)
