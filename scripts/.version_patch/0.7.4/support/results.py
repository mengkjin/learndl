"""Read-only preflight and resumable project results migration for 0.7.4."""
from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import NotRequired, TypedDict

from src.automation.artifact_exchange.core import (
    Config, _target, archive_target, atomic_json, component, digest, fsync_dir, locked, real_tree,
)
from src.automation.artifact_exchange.management import MIGRATIONS
from src.automation.artifact_exchange.model_logs import log_root

KEEP = {'test', 'trade', 'factor', 'gp', 'external', 'custom'}
LOCAL = {'rl_experimental', 'factor_sequence', 'factor_sequence_fixes', 'factor_sequence_diagnostics',
         'data_repairs', 'microcap_repair', 'market_risk_rebuild', 'model_compare', 'storage_reports',
         'coverage_diag', 'nn_training_review'}
CATEGORIES = ('model', 'rl_experiment', 'data_repair', 'factor_sequence')


class Move(TypedDict):
    source: str
    target: str
    sha256: NotRequired[str]
    done: NotRequired[bool]


class Plan(TypedDict):
    moves: list[Move]
    redirects: dict[str, str]
    aliases: dict[str, str]
    containers: list[str]
    missing_receipts: list[str]
    complete: NotRequired[bool]


def read(path):
    return json.loads(real_tree(path).read_text())


def fingerprint(path):
    """Include empty directories and all ordinary files, including dotfiles."""
    paths = sorted(path.rglob('*')) if path.is_dir() else []
    checksum = sha256()
    for member in [path, *paths]:
        real_tree(member)
        if not (member.is_dir() or member.is_file()):
            raise ValueError(f'Unexpected file type: {member}')
        entry = [str(member.relative_to(path)), None if member.is_dir() else digest(member)]
        checksum.update(json.dumps(entry).encode() + b'\n')
    return checksum.hexdigest()


def discover(config, monitor) -> Plan:
    root = config.results
    moves, aliases, containers, missing = {}, {}, set(), []
    redirects = {}

    def add(src, dst):
        real_tree(src)
        real_tree(dst)
        if src == dst or dst.is_relative_to(src):
            raise ValueError(f'Invalid move {src} -> {dst}')
        moves[src] = dst
        containers.update(p for p in src.parents if p != root and p.is_relative_to(root))

    def batch(src, manifest):
        if src != _target(manifest, config):
            raise ValueError(f'Unexpected archive location: {src}')
        target = archive_target(manifest, config)
        # Keep verified loopback acceptance material alongside local tests.
        if manifest.get('host') == config.source_host:
            target = root / 'custom/legacy' / src.relative_to(root)
        redirects[src] = target
        if src.exists():
            add(src, target)

    # Only inspect the old protocol depths. In particular custom/ is also an old category.
    for pattern in ('*/*/*/source.json', 'reports/*/*/*/source.json',
                    'model_experiments/*/*/*/source.json', 'reports/*/legacy/*/*/*/source.json'):
        for source in sorted(root.glob(pattern)):
            first = root / source.relative_to(root).parts[0]
            if first.is_symlink() or first.name == 'external':
                continue
            manifest = read(source)
            if manifest.get('schema_version') in (1, 2) and 'batch' in manifest and 'category' in manifest:
                batch(real_tree(source).parent, manifest)

    for ident in (root / 'model_run_logs').glob('*/*/model.json'):
        info = read(ident)
        if ident.parent != root / 'model_run_logs' / component(info['host']) / component(info['model']):
            raise ValueError(f'Unexpected model log location: {ident}')
        add(ident.parent, log_root(config, info['host'], info['model']))

    for item in monitor.glob('*/trash/*/item.json'):
        original = Path(read(item)['original'])
        payload = item.parent / 'payload/source.json'
        if original.is_relative_to(root) and not original.is_relative_to(root / 'external') and payload.is_file():
            batch(original, read(payload))

    for name, category in MIGRATIONS.items():
        src = root / name
        old = root / 'reports' / category / 'legacy' / name
        if src.is_symlink():
            if src.resolve() != old or not old.is_dir():
                raise ValueError(f'Unexpected migration alias: {src}')
            aliases[src] = old
        if old.is_dir() and name in LOCAL and old not in containers:
            add(old, root / 'custom' / name)

    def remainder(path):
        if path in aliases or path in moves:
            return
        if path in containers:
            for child in sorted(real_tree(path).iterdir()):
                remainder(child)
            return
        rel = path.relative_to(root)
        if rel.parts[0] == 'custom':
            return
        if rel.parts[0] in LOCAL:
            dest = root / 'custom' / rel
        else:
            dest = root / 'custom/legacy' / rel
        add(path, dest)

    for child in sorted(root.iterdir()) if root.exists() else []:
        if child.name not in KEEP and not child.name.startswith('.') and child.name != 'README.md':
            remainder(child)

    redirects.update(moves)
    # Receipts may outlive deliberately removed log-only batches or trashed files.
    for folder, dest in list(redirects.items()):
        if dest.is_relative_to(root / 'external') and len(dest.relative_to(root).parts) == 5:
            redirects.setdefault(folder.parent, dest.parent)
    for ident in (root / 'model_run_logs').glob('*/*/model.json'):
        host, model = ident.parent.parent.name, ident.parent.name
        redirects[root / 'model_training' / (host + '__' + model)] = root / 'external/model' / host / model
    for receipt in (config.state / 'received').glob('*.json'):
        target = Path(read(receipt)['target'])
        if not target.exists() and remap(target, redirects) == target:
            missing.append(str(target))

    # If two intact trees need the same destination, merge only disjoint leaves.
    expanded = {}
    def expand(src, dst):
        real_tree(src)
        real_tree(dst)
        if src.is_dir() and (src / 'source.json').is_file() and (dst.exists() or dst in expanded):
            raise FileExistsError(f'Existing archive destination: {src} -> {dst}')
        if dst in expanded:
            other = expanded.pop(dst)
            if not other.is_dir() or not src.is_dir():
                raise FileExistsError(f'Conflicting sources {other} and {src} -> {dst}')
            containers.update((src, other))
            for parent in (other, src):
                for child in sorted(parent.iterdir()):
                    expand(child, dst / child.name)
        elif dst.is_dir() and src.is_dir():
            containers.add(src)
            for child in sorted(src.iterdir()):
                expand(child, dst / child.name)
        elif dst.exists() or dst.is_symlink():
            raise FileExistsError(f'Existing destination: {src} -> {dst}')
        else:
            expanded[dst] = src
    for src, dst in moves.items():
        expand(src, dst)
    return Plan(moves=[Move(source=str(s), target=str(d)) for d, s in expanded.items()],
                redirects={str(s): str(d) for s, d in redirects.items()},
                aliases={str(s): str(d) for s, d in aliases.items()},
                containers=[str(p) for p in sorted(containers, key=lambda p: (-len(p.parts), str(p)))
                            if p != root / 'custom'], missing_receipts=sorted(missing))


def remap(path, mapping):
    for parent in (path, *path.parents):
        if parent in mapping:
            return mapping[parent] / path.relative_to(parent)
    return path


def update_state(config, monitor, mapping):
    def update_field(path, keys):
        record = read(path)
        updated = dict(record)
        for key in keys:
            if record.get(key):
                updated[key] = str(remap(Path(record[key]), mapping))
        if updated != record:
            atomic_json(path, updated)

    for receipt in (config.state / 'received').glob('*.json'):
        update_field(receipt, ('target',))
    for meta in monitor.glob('*/trash/*/item.json'):
        update_field(meta, ('original',))
    for record in (monitor / 'experiments').glob('[0-9]*.json'):
        update_field(record, ('archive_source', 'source'))
    marks = list(monitor.glob('*/archived.json'))
    if any(read(path) for path in marks):
        ids = {}
        for src, dest in mapping.items():
            if dest.exists():
                for actual in (dest, *dest.rglob('*')):
                    old = src / actual.relative_to(dest)
                    ids[sha256(str(old).encode()).hexdigest()] = sha256(str(actual).encode()).hexdigest()
        for path in marks:
            old = read(path)
            new = sorted({ids.get(key, key) for key in old})
            if old != new:
                atomic_json(path, new)


def apply(config, monitor, report):
    import portalocker
    journal = config.state / 'migrations/results-0.7.4.json'
    with ExitStack() as stack:
        # Match the order in Library.manage before acquiring the receiver lock.
        for kind in ('mail_archive', 'model_preview'):
            if (monitor / kind).is_dir():
                stack.enter_context(portalocker.Lock(monitor / kind / 'operations.lock', timeout=0))
        stack.enter_context(locked(config, 'receive'))
        if (monitor / 'experiments').is_dir():
            stack.enter_context(portalocker.Lock(monitor / 'experiments/operations.lock', timeout=0))
        previous: Plan | None = read(journal) if journal.exists() else None
        if previous is None or previous.get('complete'):
            plan = discover(config, monitor)
            if previous and not plan['moves'] and not plan['aliases']:
                atomic_json(report, previous)
                return
            for row in plan['moves']:
                row['sha256'] = fingerprint(Path(row['source']))
            atomic_json(journal, plan)
        else:
            plan = previous
        atomic_json(report, plan)
        for row in plan['moves']:
            src, dst = real_tree(row['source']), real_tree(row['target'])
            if row.get('done'):
                continue
            expected = row.get('sha256')
            if not expected:
                raise ValueError(f'Missing fingerprint for {src}')
            if src.exists():
                if dst.exists() or fingerprint(src) != expected:
                    raise ValueError(f'Source changed or destination occupied: {src} -> {dst}')
                if src.is_dir():
                    probe = subprocess.run(['lsof', '-Ff', '+D', str(src)], capture_output=True, text=True, timeout=120)
                    if probe.returncode not in (0, 1) or probe.stderr or probe.stdout:
                        raise ValueError(f'Directory in use or lsof failed: {src} {probe.stderr}')
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                fsync_dir(src.parent)
                fsync_dir(dst.parent)
            if not dst.exists() or fingerprint(dst) != expected:
                raise ValueError(f'Missing or changed migration target: {dst}')
            row['done'] = True
            atomic_json(journal, plan)
        for name, old in plan['aliases'].items():
            path = Path(name)
            if path.is_symlink() and path.resolve() == Path(old):
                path.unlink()
        for name in plan['containers']:
            path = real_tree(name)
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        update_state(config, monitor, {Path(s): Path(d) for s, d in plan['redirects'].items()})
        for name in KEEP:
            real_tree(config.results / name).mkdir(parents=True, exist_ok=True)
        for name in CATEGORIES:
            real_tree(config.results / 'external' / name).mkdir(exist_ok=True)
        pending = discover(config, monitor)
        plan['complete'] = True
        atomic_json(journal, plan)
        atomic_json(report, plan)
        if pending['moves'] or pending['aliases']:
            raise ValueError('Additional results appeared during migration; retry to migrate them')


if __name__ == '__main__':
    from src.automation.console import logger
    from src.automation.paths import ROOT

    logger.info(json.dumps(discover(Config.load(), ROOT / '.local_resources/learndl_monitor'), ensure_ascii=False, indent=2))
