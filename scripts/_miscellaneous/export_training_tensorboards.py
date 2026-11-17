"""Copy the five November training TensorBoards to Syncthing/persistent.

Run after the relevant training jobs have stopped and closed their writers:
    .venv/bin/python scripts/_miscellaneous/export_training_tensorboards.py

This is a persistent manual export, not an artifact_exchange transfer packet.
Original logs are retained. The receiver will not consume/delete these ZIPs.
Use --dry-run to inspect exact source directories before exporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from datetime import datetime, timezone
from uuid import uuid4
import zipfile

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = (
    'resnet_gru_new@resnet_gru_new_15m@4',
    'resnet_gru_new@resnet_gru_new_15m_att@4',
    'resnet_gru_new@resnet_gru_new_15m_conv2@4',
    'resnet_gru_new@resnet_gru_new_15m_transition_soft_topk',
    'resnet_gru_new@resnet_gru_new_15m_transition_capped_soft_topk',
)


def inventory(source: Path) -> dict[str, tuple[int, int, int]]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f'Missing directory or symbolic link: {source}')
    result = {}
    for path in sorted(source.rglob('*')):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f'Not an ordinary file: {path}')
        result[path.relative_to(source).as_posix()] = (info.st_ino, info.st_size, info.st_mtime_ns)
    if not any('tfevents' in Path(name).name for name in result):
        raise ValueError(f'No TensorBoard event files: {source}')
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_one(source: Path, destination: Path, identity: str) -> dict:
    """Build privately on the same filesystem, verify, then atomically publish."""
    before = inventory(source)
    members = []
    # destination is .../Syncthing/persistent/<batch>/<file>.zip; stage outside share.
    shared_root = destination.parents[2]
    with tempfile.TemporaryDirectory(prefix='.tensorboard-export-', dir=shared_root.parent) as temporary:
        archive = Path(temporary) / destination.name
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
            for name, (_, size, _) in before.items():
                digest = hashlib.sha256()
                count = 0
                with (source / name).open('rb') as original, package.open('tensorboard/' + name, 'w', force_zip64=True) as output:
                    for block in iter(lambda: original.read(1024 * 1024), b''):
                        output.write(block)
                        digest.update(block)
                        count += len(block)
                if count != size:
                    raise RuntimeError(f'Source changed during export: {source / name}')
                members.append({'path': 'tensorboard/' + name, 'bytes': count, 'sha256': digest.hexdigest()})
            manifest = {'model': identity, 'source': str(source), 'created_utc': datetime.now(timezone.utc).isoformat(),
                        'kind': 'tensorboard', 'members': members}
            package.writestr('export_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        if inventory(source) != before:
            raise RuntimeError(f'Logs changed during export; stop training and retry: {source}')
        with zipfile.ZipFile(archive) as package:
            if package.testzip() is not None:
                raise RuntimeError(f'ZIP integrity check failed: {archive}')
        checksum = sha256(archive)
        size = archive.stat().st_size
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(archive, destination)
    return {'model': identity, 'source': str(source), 'zip': destination.name,
            'bytes': size, 'sha256': checksum, 'files': len(members)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models-root', type=Path, default=REPO / 'models' / 'nn')
    parser.add_argument('--shared-root', type=Path, default=Path.home() / 'workspace' / 'Syncthing')
    parser.add_argument('--model', action='append', help='Exact model directory name; repeat to override the five defaults')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    from src.automation.console import logger

    root = args.models_root.expanduser().resolve()
    shared = args.shared_root.expanduser().resolve()
    names = args.model or DEFAULT_MODELS
    sources = []
    # Check all inputs before creating any shared payload.
    for name in names:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError(f'Expected a model directory name, not a path: {name}')
        source = root / name / 'snapshot' / 'tensorboard'
        if (root / name).is_symlink() or source.parent.is_symlink():
            raise ValueError(f'Symlinked model/snapshot directory: {source}')
        files = inventory(source)
        logger.info(f'{name}: {len(files)} files, {sum(v[1] for v in files.values()) / 1024**2:.1f} MiB; source={source}')
        sources.append((name, source))
    if args.dry_run:
        logger.info(f'Dry run only; destination root: {shared / "persistent"}')
        return 0
    if not (shared / '.stfolder').exists():
        raise ValueError(f'Syncthing marker missing: {shared / ".stfolder"}; check --shared-root')
    batch_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '__' + uuid4().hex[:12]
    destination = shared / 'persistent' / f'tensorboard_recovery_{batch_id}'
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for name, source in sources:
        target = destination / f'nn@{name}.zip'
        record = export_one(source, target, f'nn@{name}')
        records.append(record)
        logger.info(f'Exported {target} ({record["bytes"] / 1024**2:.1f} MiB); SHA256={record["sha256"]}')
    # The completion manifest appears only after every requested ZIP succeeds.
    with tempfile.TemporaryDirectory(prefix='.tensorboard-manifest-', dir=shared.parent) as temporary:
        manifest = Path(temporary) / 'manifest.json'
        manifest.write_text(json.dumps({'batch_id': batch_id, 'status': 'complete', 'archives': records},
                                       ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(manifest, destination / 'manifest.json')
    logger.info(f'Complete: {destination}; originals retained. Syncthing delivery is asynchronous.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
