"""Preflight and migrate model experiment files for 0.7.2."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing, nullcontext
from dataclasses import asdict, dataclass
from itertools import zip_longest
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory

from .core import (
    Config, _target, atomic_json, category_root, digest, experiment_members,
    fsync_dir, locked, real_tree, received_member,
)


@dataclass(frozen=True)
class Move:
    source: Path
    target: Path
    kind: str
    expected_sha: str = ''


def inventory(config, model_root, archive_root, optuna_root):
    """Use full model identities, preserving ambiguous standalone studies."""
    from src.res.model.util import ModelPath

    models = defaultdict(list)
    moves = []
    unmatched = []
    roots = [*archive_root.glob('*'), *(p for p in model_root.glob('*/*') if not p.is_relative_to(archive_root))]
    for base in sorted(roots):
        if base.is_symlink() or not base.is_dir() or base.name.startswith('.'):
            continue
        old = base / 'snapshot/tensorboard'
        if old.exists():
            moves.append(Move(old, base / 'results/tensorboard', 'tensorboard'))
        try:
            if base.parent == archive_root:
                name, _, stamp = base.name.rpartition('.')
                if not (len(stamp) == 14 and stamp.isdigit()):
                    continue
                identity = ModelPath(name).full_name
            else:
                identity = ModelPath(base).full_name
        except (AssertionError, ValueError, KeyError):
            continue
        if identity:
            models[identity].append(base)

    for old in sorted(optuna_root.glob('*/*.sqlite3')):
        identity, _, stamp = old.stem.rpartition('@')
        candidates = models.get(identity, [])
        if len(stamp) == 14 and stamp.isdigit() and len(candidates) == 1:
            moves.append(Move(old, candidates[0] / 'results/optuna' / old.name, 'sqlite'))
        else:
            unmatched.append(str(old))

    category = category_root(config.results, 'model_training')
    for source in sorted(category.glob('*/*/source.json')):
        manifest = json.loads(real_tree(source).read_text())
        if manifest.get('schema_version') != 2 or manifest.get('category') != 'model_training':
            continue
        batch = _target(manifest, config)
        if source.parent != batch:
            raise ValueError(f'Invalid experiment manifest location: {source}')
        if not (config.state / 'received' / (manifest['id'] + '.json')).is_file():
            continue  # Receiver will finish any interrupted receipt.
        kinds = experiment_members(manifest)
        for item in manifest['files']:
            name = item['path']
            canonical = received_member(manifest, name, kinds)
            old = real_tree(batch / name)
            if name != canonical.as_posix() and old.exists():
                moves.append(Move(old, batch / canonical, 'zip', item['sha256']))
    return moves, unmatched


def _tree_hashes(path):
    return {p.relative_to(path).as_posix(): digest(real_tree(p))
            for p in sorted(path.rglob('*')) if not real_tree(p).is_dir()}


def _db(path):
    return closing(sqlite3.connect(real_tree(path).as_uri() + '?mode=ro', uri=True))


def _same_db(left, right):
    with _db(left) as a, _db(right) as b:
        return all(x == y for x, y in zip_longest(a.iterdump(), b.iterdump()))


def _idle(path):
    if not path.exists():
        return
    args = ['+D', str(path)] if path.is_dir() else ['--', str(path)]
    probe = subprocess.run(['lsof', '-Fn', *args], capture_output=True, timeout=60, check=False)
    if probe.stdout or probe.stderr or probe.returncode != 1:
        raise RuntimeError(f'Open files or uncertain lsof result: {path}')


def _check(move, *, check_open=True):
    source, target = real_tree(move.source), real_tree(move.target)
    if check_open:
        for path in (source, target):
            _idle(path)
            if move.kind == 'sqlite':
                for suffix in ('-wal', '-shm'):
                    _idle(real_tree(path.with_name(path.name + suffix)))
    if move.kind == 'sqlite':
        same = target.is_file() and _same_db(source, target)
        evidence = None
    elif move.kind == 'tensorboard':
        evidence = _tree_hashes(source)
        same = target.is_dir() and evidence == _tree_hashes(target)
    else:
        evidence = digest(source)
        if evidence != move.expected_sha:
            raise ValueError(f'Archive differs from its receipt: {source}')
        same = target.is_file() and evidence == digest(target)
    if target.exists() and not same:
        raise FileExistsError(f'Different files at {source} and {target}')
    return same, evidence


def _apply(move, same, evidence):
    source, target = move.source, move.target
    target.parent.mkdir(parents=True, exist_ok=True)
    if move.kind == 'sqlite':
        if not same:
            with TemporaryDirectory(dir=target.parent) as scratch:
                copied = Path(scratch) / 'study.sqlite3'
                with _db(source) as original, closing(sqlite3.connect(copied)) as backup:
                    original.backup(backup)
                    if backup.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                        raise ValueError(f'SQLite backup failed integrity check: {source}')
                if not _same_db(source, copied):
                    raise ValueError(f'SQLite source changed during backup: {source}')
                with copied.open('rb') as stream:
                    os.fsync(stream.fileno())
                copied.rename(target)
        # Original and sidecars stay in place as the recovery copy.
        verified = _same_db(source, target)
    elif same:
        # An identical existing copy is safe; preserve the redundant original.
        verified = True
    else:
        source.rename(target)
        fsync_dir(source.parent)
        verified = (_tree_hashes(target) == evidence if move.kind == 'tensorboard' else digest(target) == evidence)
    fsync_dir(target.parent)
    if not verified:
        raise ValueError(f'Migration verification failed: {target}')


def run(config, model_root, archive_root, optuna_root, *, report=None):
    """Without a report path this is a read-only preflight."""
    guard = locked(config, 'receive') if report is not None and config.role == 'receiver' else nullcontext()
    with guard:
        moves, unmatched = inventory(config, model_root, archive_root, optuna_root)
        rows = []
        ready = []
        for move in moves:
            row = {k: str(v) for k, v in asdict(move).items()}
            rows.append(row)
            try:
                same, evidence = _check(move)
                row['state'] = 'already-migrated' if same else 'ready'
                ready.append((row, move, same, evidence))
            except (OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as exc:
                row.update(state='blocked', error=str(exc))
        summary = {'moves': rows, 'unmatched_optuna': unmatched}
        if report is not None:
            atomic_json(report, summary)
            if len(ready) != len(moves):
                raise ValueError(f'Experiment migration is blocked; see {report}')
            for row, move, same, evidence in ready:
                if not same:
                    try:
                        _apply(move, same, evidence)
                        row.update(state='migrated', sha256=evidence if isinstance(evidence, str) else None)
                    except Exception as exc:
                        row.update(state='blocked', error=str(exc))
                        atomic_json(report, summary)
                        raise
                    atomic_json(report, summary)
        return summary


def main():
    from src.proj import Logger, PATH

    Logger.stdout(json.dumps(run(Config.load(), PATH.model, PATH.model_archive, PATH.optuna),
                             indent=2, ensure_ascii=False), vb_level=0)


if __name__ == '__main__':
    main()
