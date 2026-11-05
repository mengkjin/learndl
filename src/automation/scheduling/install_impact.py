"""Conservative installation classification; compatibility is explicitly reviewed."""
from __future__ import annotations

__all__ = [
    'PROTOCOL',
    'COMPATIBILITY_FILE',
    'reviewed_transitions',
    'COMPATIBLE_SOURCE_UPGRADES',
    'sources',
    'classify',
    'publish',
]

import hashlib
import json
from pathlib import Path
import re

from src.automation.paths import ROOT

PROTOCOL = 1
# Reviewed transitions only: relative source path -> {(old SHA256, new SHA256)}.
# Changing a Python file is never implicitly considered a compatible upgrade.
COMPATIBILITY_FILE = ROOT / 'configs/automation/framework_compatibility.json'


def reviewed_transitions():
    document = json.loads(COMPATIBILITY_FILE.read_text())
    if not isinstance(document, dict):
        raise ValueError('Framework compatibility registry must be an object')
    transitions = {}
    for name, pairs in document.items():
        if not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2 or
                any(not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value) for value in pair) for pair in pairs):
            raise ValueError('Invalid reviewed source transition: ' + name)
        transitions[name] = {tuple(pair) for pair in pairs}
    return transitions


COMPATIBLE_SOURCE_UPGRADES = reviewed_transitions()


def sources():
    directory = Path(__file__).parent
    files = list(directory.glob('*.py')) + list(directory.parent.glob('*.py'))
    root = directory.parents[2]
    files += [root / 'src/proj/util/script/code_update_lock.py', root / 'src/proj/util/script/script_lock.py']
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.exists()}


def classify(plan, directory):
    old_path, manifest_path = directory / 'installed.json', directory / 'manifest.json'
    current_sources = sources()
    result = dict(mode='exclusive', groups=None, reasons=[], protocol=PROTOCOL, sources=current_sources)
    def exclusive(reason):
        return dict(result, reasons=[reason])
    if not old_path.exists() or not manifest_path.exists():
        return exclusive('First installation requires exclusive admission')
    old, new = json.loads(old_path.read_text()), plan.get('config')
    manifest = json.loads(manifest_path.read_text())
    if not new or manifest.get('install_protocol') != PROTOCOL:
        return exclusive('Unknown installation protocol; migrate in an idle window')
    previous_sources = manifest.get('sources', {})
    changed = {key for key in previous_sources.keys() | current_sources.keys()
               if previous_sources.get(key) != current_sources.get(key)}
    compatible = all((previous_sources.get(key), current_sources.get(key)) in
                     COMPATIBLE_SOURCE_UPGRADES.get(key, set()) for key in changed)
    if changed and not compatible:
        return exclusive('Unreviewed framework source changes: ' + ', '.join(sorted(changed)))
    for name, before in manifest.get('units', {}).items():
        after = plan['units'].get(name)
        if name.endswith('.service') and after is not None:
            # Timer cadence may change the watchdog's own bounded service timeout.
            def stable(text):
                return [line for line in text.splitlines() if not line.startswith('TimeoutStartSec=')]
            if stable(before) != stable(after):
                return exclusive('Service execution environment changed: ' + name)
    ignored = {'autofix_initialized', 'tasks', 'schedules', 'watchdog'}
    if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
        return exclusive('Host, backend, timezone or scheduler protocol changed')
    if old['tasks'].keys() != new['tasks'].keys():
        return exclusive('Task registry changed')
    if old['watchdog'].get('units') != new['watchdog'].get('units'):
        return exclusive('Persistent service supervision changed')
    old_jobs, new_jobs = old['watchdog']['jobs'], new['watchdog']['jobs']
    for key in ('task_lifecycle', 'task_timeouts', 'autofix'):
        if old_jobs.get(key, {}).get('enabled') and not new_jobs.get(key, {}).get('enabled'):
            return exclusive('Disabling lifecycle supervision requires draining active tasks')
    for key in old_jobs.keys() | new_jobs.keys():
        a, b = old_jobs.get(key, {}), new_jobs.get(key, {})
        if {k: v for k, v in a.items() if k not in {'enabled', 'interval_seconds', 'priority'}} != \
           {k: v for k, v in b.items() if k not in {'enabled', 'interval_seconds', 'priority'}}:
            return exclusive('Maintenance execution policy changed: ' + key)
    groups = set()
    for key, before in old['tasks'].items():
        after = new['tasks'][key]
        if {k: v for k, v in before.items() if k not in {'enabled', 'timeout_seconds', 'priority'}} != \
           {k: v for k, v in after.items() if k not in {'enabled', 'timeout_seconds', 'priority'}}:
            groups.update((before.get('channel', 'data'), after.get('channel', 'data'), 'manual'))
    if groups:
        return dict(result, groups=sorted(groups), reasons=['Task execution/lock policy changed'])
    return dict(result, mode='compatible' if changed else 'config', groups=[],
                reasons=['Reviewed compatible source transitions' if changed else 'Only future scheduling policy changes'])


def publish(directory, config):
    """An immutable snapshot plus one atomic pointer, also readable by old workers."""
    from .installer import atomic_json
    payload = json.dumps(config, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    snapshots = directory / 'generations'
    snapshots.mkdir(exist_ok=True)
    target = snapshots / (digest + '.json')
    if not target.exists():
        atomic_json(target, config)
    pointer = directory / 'installed.next'
    pointer.unlink(missing_ok=True)
    pointer.symlink_to(Path('generations') / target.name)
    pointer.replace(directory / 'installed.json')
    return digest
