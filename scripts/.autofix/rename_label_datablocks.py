# description: Rename legacy X_y_* DataBlocks to Y_* without recalculation
# content: Rename fit/predict mmap directories and pt/feather files; stop on conflicts.
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: Preview renames without changing data or writing a report

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.proj import PATH, Logger
from src.proj.util.script import ScriptTool

SUFFIXES = ('.mmap', '.pt', '.feather')
FRAMES = ('fit', 'predict')


def plan_renames(root: Path) -> list[tuple[Path, Path]]:
    """Preflight all targets before moving any file; never select raw data."""
    moves = []
    for frame in FRAMES:
        folder = root / frame
        if folder.is_symlink():
            raise ValueError(f'Refusing symlink frame directory: {folder}')
        if not folder.exists():
            continue
        for source in sorted(folder.glob('X_y_*')):
            if source.suffix not in SUFFIXES or not source.stem.removeprefix('X_y_'):
                continue
            if source.is_symlink():
                raise ValueError(f'Refusing symlink DataBlock: {source}')
            if (source.suffix == '.mmap' and not source.is_dir()) or (source.suffix != '.mmap' and not source.is_file()):
                raise ValueError(f'Invalid DataBlock path type: {source}')
            target = source.with_name('Y_' + source.name.removeprefix('X_y_'))
            for suffix in SUFFIXES:
                candidate = target.with_suffix(suffix)
                if candidate.exists() or candidate.is_symlink():
                    raise FileExistsError(f'Both legacy and canonical DataBlocks exist: {source} -> {candidate}')
            moves.append((source, target))
    return moves


def migrate(root: Path, report_dir: Path, dry_run: bool = False):
    moves = plan_renames(root)
    for source, target in moves:
        Logger.note(f'Rename only: {source} -> {target}')
    if dry_run or not moves:
        Logger.note(f'{len(moves)} DataBlock renames; dry_run={dry_run}')
        return moves
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        'rename_label_datablocks_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8] + '.json')
    report = {'status': 'running', 'moves': [
        {'source': str(source), 'target': str(target), 'status': 'pending'} for source, target in moves]}

    def save_report():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(report_path)

    save_report()
    renamed_targets = set()
    try:
        for (source, target), record in zip(moves, report['moves']):
            # Recheck immediately before rename, including other canonical formats.
            candidates = [target.with_suffix(suffix) for suffix in SUFFIXES]
            if any(candidate not in renamed_targets and (candidate.exists() or candidate.is_symlink()) for candidate in candidates):
                raise FileExistsError(f'Canonical target appeared during migration: {target}')
            record['status'] = 'renaming'
            save_report()
            source.rename(target)
            renamed_targets.add(target)
            record['status'] = 'renamed'
            save_report()
        report['status'] = 'success'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_report()
    Logger.note(f'Renamed {len(moves)} DataBlocks without reading/recomputing values. Journal: {report_path}')
    return moves


@ScriptTool('autofix_rename_label_datablocks')
def main(dry_run: bool = False, **kwargs):
    migrate(PATH.datablock, PATH.interim / 'autofix_reports', dry_run=dry_run)


if __name__ == '__main__':
    main()
