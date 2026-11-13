"""Receiver service installation and conservative, journaled local report migration."""
from __future__ import annotations
import json
import re
import shutil
import os
import plistlib
import subprocess
import sys
import time
from typing import Any
from pathlib import Path
from .core import ROOT, atomic_json, digest, locked, real_tree, fsync_dir

LABEL = 'com.learndl.artifact-exchange'
MIGRATIONS = {'model': 'model_training', 'gru_day_rtn': 'model_training',
              'model_compare': 'model_compare', 'storage_reports': 'storage', 'coverage_diag': 'data_quality'}


def service(action, config_path, config) -> dict[str, Any]:
    if config.role != 'receiver':
        raise ValueError('Receiver role required')
    plist = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')
    definition = {'Label': LABEL, 'ProgramArguments': [sys.executable, '-m', 'src.automation.artifact_exchange',
                  '--config', str(Path(config_path).resolve()), 'receive', '--once'],
                  'WorkingDirectory': str(ROOT), 'RunAtLoad': True, 'StartInterval': 300,
                  'ProcessType': 'Background'}
    if action == 'plan':
        return {'path': str(plist), 'definition': definition}
    if sys.platform != 'darwin':
        raise ValueError('launchd installation requires macOS')
    domain = f'gui/{os.getuid()}'
    if action == 'status':
        result = subprocess.run(['launchctl', 'print', domain + '/' + LABEL], capture_output=True, text=True)
        return {'installed': plist.exists(), 'returncode': result.returncode, 'output': result.stdout or result.stderr}
    if action == 'uninstall':
        subprocess.run(['launchctl', 'bootout', domain + '/' + LABEL], capture_output=True)
        plist.unlink(missing_ok=True)
        return {'status': 'uninstalled', 'data_retained': True}
    if not config.enabled:
        raise ValueError('Enable receiver before installation')
    data = plistlib.dumps(definition)
    if plist.exists() and plist.read_bytes() == data:
        result = service('status', config_path, config)
        if result['returncode'] == 0:
            return {'status': 'already installed'}
    plist.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['launchctl', 'bootout', domain + '/' + LABEL], capture_output=True)
    tmp = plist.with_suffix('.tmp')
    tmp.write_bytes(data)
    tmp.replace(plist)
    subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], check=True)
    return {'status': 'installed', 'path': str(plist)}


def inventory(path):
    real_tree(path)
    result = {}
    for p in sorted(path.rglob('*')):
        real_tree(p)
        if p.is_file():
            before = p.stat()
            h = digest(p)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'File changing: {p}')
            result[p.relative_to(path).as_posix()] = {'size': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': h}
    return result


def migration(action, config):
    if config.role != 'receiver':
        raise ValueError('Migration is local receiver-only')
    output = []
    with locked(config, 'receive'):
        for name, category in MIGRATIONS.items():
            src = config.results / name
            dst = config.results / 'reports' / category / 'legacy' / name
            ledger = config.state / 'migrations' / (name + '.json')
            row: dict[str, Any] = {'source': str(src), 'target': str(dst)}
            try:
                old = json.loads(ledger.read_text()) if ledger.exists() else None
                if action == 'rollback':
                    if old is None:
                        continue
                    if src.is_dir() and not src.is_symlink() and not dst.exists():
                        row['status'] = 'already rolled back'
                    elif src.is_symlink() and src.resolve() == dst and inventory(dst) == old['files']:
                        src.unlink()
                        dst.rename(src)
                        atomic_json(ledger, old | {'status': 'rolled_back'})
                        row['status'] = 'rolled_back'
                    elif not src.exists() and dst.exists() and inventory(dst) == old['files']:
                        dst.rename(src)
                        atomic_json(ledger, old | {'status': 'rolled_back'})
                        row['status'] = 'rolled_back'
                    else:
                        raise ValueError('Paths/content changed; rollback refused')
                elif src.is_symlink() and src.resolve() == dst:
                    row['status'] = 'already migrated'
                elif not src.exists() and old and old['status'] == 'prepared' and dst.exists():
                    if inventory(dst) != old['files']:
                        raise ValueError('Interrupted migration target changed')
                    if action == 'apply':
                        src.symlink_to(dst.relative_to(src.parent), target_is_directory=True)
                        atomic_json(ledger, old | {'status': 'migrated'})
                    row['status'] = 'recoverable migration'
                elif not src.exists():
                    row['status'] = 'absent'
                elif dst.exists() or dst.is_symlink():
                    raise ValueError('Destination exists; no merge')
                else:
                    if any(json.loads(p.read_text()).get('schema_version') == 2 for p in src.glob('*/*/source.json')):
                        raise ValueError('Contains received experiment batches; not a legacy report directory')
                    files = inventory(src)
                    if any(time.time_ns() - v['mtime_ns'] < 300 * 10**9 for v in files.values()):
                        raise ValueError('Recently written files; retry after idle window')
                    row.update(status='ready', files=len(files), bytes=sum(v['size'] for v in files.values()))
                    if action == 'apply':
                        if not shutil.which('lsof'):
                            raise ValueError('lsof required to check open writers before migration')
                        probe = subprocess.run(['lsof', '-Ff', '+D', str(src)], capture_output=True, text=True, timeout=60)
                        if probe.returncode not in (0, 1) or probe.stderr or any(re.fullmatch(r'f[0-9]+[wu]', line) for line in probe.stdout.splitlines()):
                            raise ValueError('Open writers or uncertain lsof result; retry when idle')
                        time.sleep(1)
                        if inventory(src) != files:
                            raise ValueError('Directory changing')
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        real_tree(dst.parent)
                        atomic_json(ledger, row | {'files': files, 'status': 'prepared'})
                        src.rename(dst)
                        fsync_dir(dst.parent)
                        fsync_dir(src.parent)
                        if inventory(dst) != files:
                            dst.rename(src)
                            raise ValueError('Post-move verification failed')
                        src.symlink_to(dst.relative_to(src.parent), target_is_directory=True)
                        fsync_dir(src.parent)
                        atomic_json(ledger, row | {'files': files, 'status': 'migrated'})
                        row['status'] = 'migrated'
            except Exception as exc:
                row.update(status='skipped', error=str(exc))
            output.append(row)
    return output
