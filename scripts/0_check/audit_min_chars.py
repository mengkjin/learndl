"""Read-only, key-column-only audit of daily min_chars files.

Run from the project root: uv run python scripts/0_check/audit_min_chars.py
Exit codes: 0 clean; 1 anomalies; 2 incomplete scan/read errors.
Reports are flushed after every file so terminal loss does not erase evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KEYS = ('min_chars', 'min_chars_roll', 'min_chars_tag')


def inspect_keys(df, expected_date, known, mapper, max_rows=7000):
    """Keep raw evidence even if normalization fails; never discard unknown IDs."""
    issues = []
    report = {'rows': df.height, 'schema': {k: str(v) for k, v in df.schema.items()}}
    for col in ('secid', 'date'):
        if col not in df.columns:
            issues.append(f'missing_{col}')
        elif not df.schema[col].is_integer() or df[col].null_count():
            issues.append(f'invalid_{col}')
    if df.height == 0:
        issues.append('empty_file')
    if df.height > max_rows:
        issues.append('row_count_above_threshold')
    if any(x.startswith(('missing_', 'invalid_')) for x in issues):
        report.update(issues=issues)
        return report, set(), set(), []
    raw = set(df['secid'].to_list())
    dates = df['date'].unique().sort().to_list()
    report.update(raw_unique=len(raw), dates=dates,
                  duplicate_rows=df.height - df.unique(['secid', 'date']).height)
    if report['duplicate_rows']:
        issues.append('duplicate_keys')
    if dates != [expected_date]:
        issues.append('date_mismatch')
    raw_unknown = raw - known
    report.update(raw_unknown=len(raw_unknown), raw_unknown_sample=sorted(raw_unknown)[:30])
    try:
        mapped_df = mapper(df.clone())
        mapped_values = mapped_df['secid'].to_list()
        if len(mapped_values) != df.height or any(v is None or not isinstance(v, int) for v in mapped_values):
            raise ValueError('mapper changed row count or produced invalid secids')
        mapped = set(mapped_values)
        changes = sorted(set(zip(df['secid'].to_list(), mapped_values)) - {(x, x) for x in raw})
        unknown = mapped - known
        report.update(mapped_unique=len(mapped), mapped_unknown=len(unknown),
                      mapped_unknown_sample=sorted(unknown)[:30], mapping_changes=len(changes),
                      mapping_sample=changes[:30])
        if unknown:
            issues.append('unknown_mapped_secids')
        if mapped_df.unique(['secid', 'date']).height < df.unique(['secid', 'date']).height:
            issues.append('mapping_key_collision')
    except Exception as exc:
        mapped, changes = set(), []
        report['mapping_error'] = f'{type(exc).__name__}: {exc}'
        issues.append('mapping_error')
    report['issues'] = issues
    return report, raw, mapped, changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=20100101)
    parser.add_argument('--end', type=int, default=20991231)
    parser.add_argument('--max-rows', type=int, default=7000,
                        help='Diagnostic threshold only; historical files need not have exactly 5000 rows')
    parser.add_argument('--output', type=Path, help='New report directory; existing directory is never overwritten')
    args = parser.parse_args(argv)
    if args.start > args.end or args.max_rows < 1:
        parser.error('invalid date range or row threshold')
    # Use the same machine mapping and historical stock reference as reconstruction.
    from src.proj import DB, PATH
    from src.proj.db.basic.df_handler import dfHandler
    from src.data.util.stock_info import INFO

    known = set(map(int, INFO.get_secid()))
    if not known:
        raise RuntimeError('Historical stock reference is empty')
    output = args.output or PATH.runtime / 'min_chars_audit' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    summary = {'start': args.start, 'end': args.end, 'known_historical_secids': len(known),
               'max_rows': args.max_rows, 'files': 0, 'anomalous_files': 0, 'errors': 0,
               'tables': {}, 'output': str(output.resolve())}
    print(f'Reports: {output.resolve()}', flush=True)
    # Only cumulative ID sets and per-year date sets persist between files, never feature frames.
    with (output / 'files.jsonl').open('w') as details, (output / 'first_seen.csv').open('w', newline='') as ids_file, (output / 'mapping.csv').open('w', newline='') as mapping_file:
        ids_writer = csv.writer(ids_file)
        ids_writer.writerow(['table', 'stage', 'secid', 'known', 'first_file'])
        mapping_writer = csv.writer(mapping_file)
        mapping_writer.writerow(['table', 'raw_secid', 'mapped_secid', 'first_file'])
        for key in KEYS:
            paths = sorted(DB.paths('min_chars', key, start=args.start, end=args.end))
            seen = {'raw': set(), 'mapped': set()}
            years, mapping_seen, daily_seen = {}, set(), {}
            table = {'file_count': len(paths), 'first_anomaly': None, 'years': {}}
            summary['tables'][key] = table
            if not paths:
                summary['errors'] += 1
                table['error'] = 'no_files_in_requested_range'
            for path in paths:
                row = {'table': key, 'file': str(path)}
                try:
                    match = re.search(r'\.(\d{8})\.(?:feather|parquet)$', path.name)
                    if not match:
                        raise ValueError('unrecognized daily filename')
                    date = int(match[1])
                    reader = pl.read_ipc if path.suffix == '.feather' else pl.read_parquet
                    kwargs = {'memory_map': False} if path.suffix == '.feather' else {}
                    df = reader(path, columns=['secid', 'date'], **kwargs)
                    result, raw, mapped, changes = inspect_keys(df, date, known, dfHandler.default_mapper, args.max_rows)
                    del df
                    row.update(result)
                    if 'mapping_error' in result:
                        summary['errors'] += 1
                    if date in daily_seen:
                        row['issues'].append('multiple_files_for_date')
                        row['other_file'] = daily_seen[date]
                    daily_seen[date] = str(path)
                    year = years.setdefault(date // 10000, {'raw': set(), 'mapped': set(), 'dates': set()})
                    year['dates'].add(date)
                    for stage, values in [('raw', raw), ('mapped', mapped)]:
                        new = values - seen[stage]
                        row[f'{stage}_new_ids'] = len(new)
                        for secid in sorted(new):
                            ids_writer.writerow([key, stage, secid, secid in known, str(path)])
                        seen[stage].update(values)
                        year[stage].update(values)
                        row[f'{stage}_cumulative_ids'] = len(seen[stage])
                    row['mapped_ids_sha256'] = hashlib.sha256(','.join(map(str, sorted(mapped))).encode()).hexdigest()
                    for pair in changes:
                        if pair not in mapping_seen:
                            mapping_writer.writerow([key, *pair, str(path)])
                            mapping_seen.add(pair)
                except Exception as exc:
                    row.update(error=f'{type(exc).__name__}: {exc}', issues=['read_error'])
                    summary['errors'] += 1
                summary['files'] += 1
                if row['issues']:
                    summary['anomalous_files'] += 1
                    if table['first_anomaly'] is None:
                        table['first_anomaly'] = row
                    if summary['anomalous_files'] <= 10:
                        print(f"ANOMALY {path}: {row['issues']}", flush=True)
                details.write(json.dumps(row, ensure_ascii=False) + '\n')
                details.flush()
                ids_file.flush()
                mapping_file.flush()
                if summary['files'] % 250 == 0:
                    print(f"Scanned {summary['files']} files, anomalies={summary['anomalous_files']}, last={path}", flush=True)
            table.update(raw_union=len(seen['raw']), mapped_union=len(seen['mapped']),
                         unknown_mapped_union=len(seen['mapped'] - known))
            for year, values in sorted(years.items()):
                table['years'][str(year)] = {'raw_union': len(values['raw']), 'mapped_union': len(values['mapped']),
                                            'dates': len(values['dates'])}
            print(f"{key}: files={len(paths)}, raw union={len(seen['raw'])}, mapped union={len(seen['mapped'])}", flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(f"Done: files={summary['files']}, anomalies={summary['anomalous_files']}, errors={summary['errors']}. See {output / 'summary.json'}", flush=True)
    return 2 if summary['errors'] else 1 if summary['anomalous_files'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
