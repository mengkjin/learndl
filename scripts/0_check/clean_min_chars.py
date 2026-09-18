"""Remove non-stock rows from historical min_chars files, with backups.

Default: dry run. Use --apply to replace files after backing up each original.
Membership uses DB-mapped secids; retained rows keep original values and schema.
Run without concurrent min_chars writers. No feature recomputation or row filling.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KEYS = ('min_chars', 'min_chars_roll', 'min_chars_tag')


def fingerprint(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def clean_file(path, known, mapper, *, apply=False, backup=None):
    """Backup + atomic replacement; rerunning an already cleaned file is a no-op."""
    if not len(known):
        raise ValueError('Empty stock description; refusing to remove data')
    before = fingerprint(path)
    reader = feather.read_table if path.suffix == '.feather' else parquet.read_table
    writer = feather.write_feather if path.suffix == '.feather' else parquet.write_table
    # Arrow preserves schema metadata, including pandas index metadata.
    table = reader(path)
    keys = pl.from_arrow(table.select(['secid']))
    if not keys.schema['secid'].is_integer():
        raise ValueError(f'{path}: secid must be integer')
    mapped = mapper(keys)
    if mapped.height != table.num_rows:
        raise ValueError('Mapping changed row count')
    keep = mapped['secid'].is_in(pl.Series(sorted(known)).implode()).fill_null(False)
    removed = table.num_rows - int(keep.sum())
    result = {'file': str(path), 'rows': table.num_rows, 'removed_rows': removed,
              'retained_rows': table.num_rows - removed,
              'removed_ids_sample': mapped.filter(~keep)['secid'].unique().head(30).to_list(),
              'status': 'would_clean' if removed else 'unchanged'}
    if not apply or not removed:
        return result
    if backup is None:
        raise ValueError('Backup path is required')
    cleaned = table.filter(pa.array(keep.to_list()))
    backup.parent.mkdir(parents=True, exist_ok=True)
    if fingerprint(path) != before:
        raise RuntimeError('Source changed during read; stop concurrent writers')
    # Exclusive backup creation: never overwrite a previous backup.
    with path.open('rb') as source, backup.open('xb') as dest:
        shutil.copyfileobj(source, dest)
        dest.flush()
        os.fsync(dest.fileno())
    result['backup'] = str(backup)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        writer(cleaned, temp)
        verified = reader(temp)
        if not verified.schema.equals(table.schema, check_metadata=True) or not pl.from_arrow(verified).equals(pl.from_arrow(cleaned)):
            raise RuntimeError('Replacement validation failed')
        with temp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.chmod(temp, path.stat().st_mode & 0o777)
        if fingerprint(path) != before:
            raise RuntimeError('Source changed before replacement; stop concurrent writers')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    result['status'] = 'cleaned'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=20100101)
    parser.add_argument('--end', type=int, default=20991231)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', type=Path, help='New directory for journal and original-file backups')
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error('start must not exceed end')
    from src.proj import DB, PATH
    from src.proj.db.basic.df_handler import dfHandler
    from src.data.util.minchars_stock import historical_stock_ids

    known = set(map(int, historical_stock_ids()))
    output = args.output or PATH.runtime / 'min_chars_cleanup' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    summary = {'apply': args.apply, 'stock_ids': len(known), 'files': 0, 'changed_files': 0,
               'removed_rows': 0, 'errors': 0, 'output': str(output.resolve())}
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {output.resolve()}", flush=True)
    with (output / 'files.jsonl').open('w') as journal:
        for key in KEYS:
            for path in sorted(DB.paths('min_chars', key, start=args.start, end=args.end)):
                try:
                    result = clean_file(path, known, dfHandler.default_mapper, apply=args.apply,
                                        backup=output / 'backups' / key / path.parent.name / path.name)
                    summary['changed_files'] += bool(result['removed_rows'])
                    summary['removed_rows'] += result['removed_rows']
                except Exception as exc:
                    result = {'file': str(path), 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
                    summary['errors'] += 1
                summary['files'] += 1
                journal.write(json.dumps(result, ensure_ascii=False) + '\n')
                journal.flush()
                if summary['files'] % 250 == 0 or result['status'] == 'error':
                    print(f"files={summary['files']}, removed_rows={summary['removed_rows']}, errors={summary['errors']}", flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 2 if summary['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
