#!/usr/bin/env python3
# author: jinmeng
# date: 2026-10-10
# description: Repair Day Limit Duplicates
# content: 修复 trade_ts/day_limit 完全相同的重复记录；原文件先备份，冲突记录不改写并报告。
# email: True
# mode: shell
# parameters:
#   start:
#       type: int
#       default: 20170101
#       desc: 开始日期（含）
#   end:
#       type: int
#       default: 20251231
#       desc: 结束日期（含），应覆盖 RL 最后决策日的下一交易日
#   dry_run:
#       type: [True, False]
#       default: True
#       desc: True 仅检查；False 备份并修复

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.proj import DB, PATH, Proj
from src.proj.db.io.dataframe import dfIOHandler
from src.proj.util.script import ScriptTool


def clean_day_limit(frame: pd.DataFrame, date: int) -> pd.DataFrame:
    """Only remove identical rows; never choose between conflicting prices."""
    required = {'secid', 'up_limit', 'down_limit'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f'{date}: empty table or missing columns {required}')
    if frame['secid'].isna().any():
        raise ValueError(f'{date}: null secid')
    if 'date' in frame and not (frame['date'] == date).all():
        raise ValueError(f'{date}: date column does not match file date')
    cleaned = frame.drop_duplicates().reset_index(drop=True)
    conflicts = cleaned.loc[cleaned.duplicated('secid', keep=False)]
    if not conflicts.empty:
        raise ValueError(f'{date}: conflicting duplicate secid rows: {conflicts.head(12).to_dict("records")}')
    return cleaned


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def repair_file(path: Path, backup: Path, date: int, dry_run: bool) -> dict:
    original_hash = digest(path)
    original = dfIOHandler.load_pandas(path, missing_ok=False)
    cleaned = clean_day_limit(original, date)
    removed = len(original) - len(cleaned)
    record = {'date': date, 'path': str(path), 'before_sha256': original_hash,
              'rows_before': len(original), 'rows_after': len(cleaned), 'removed': removed,
              'status': 'would_repair' if removed else 'unchanged'}
    if not removed or dry_run:
        return record
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise FileExistsError(f'backup already exists: {backup}')
    shutil.copy2(path, backup)
    if digest(backup) != original_hash or digest(path) != original_hash:
        raise RuntimeError(f'source changed during backup: {path}')
    # Serialize and verify before replacing the database file. The project IO
    # writer itself is atomic; this extra staging allows round-trip validation.
    staging = path.with_name(f'.repair-{backup.parent.parent.name}-{path.name}')
    try:
        dfIOHandler.save_df(cleaned, staging)
        pd.testing.assert_frame_equal(dfIOHandler.load_pandas(staging, missing_ok=False), cleaned)
        if digest(path) != original_hash:
            raise RuntimeError(f'source changed during repair: {path}')
        shutil.copymode(path, staging)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
    record.update(status='repaired', backup=str(backup), after_sha256=digest(path))
    return record


@ScriptTool('repair_day_limit', markdown_catcher=True)
def main(start: int = 20170101, end: int = 20251231, dry_run: bool = True, **kwargs):
    if start > end:
        raise ValueError('start must not exceed end')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = PATH.main / 'results' / 'data_repairs' / 'day_limit' / stamp
    output.mkdir(parents=True, exist_ok=False)
    report = output / 'report.json'
    Proj.email_attachments.append(report)
    dates = DB.dates('trade_ts', 'day_limit', start=start, end=end)
    records = []
    payload = {'start': start, 'end': end, 'dry_run': dry_run, 'records': records}
    try:
        if not len(dates):
            raise FileNotFoundError('no day_limit files in requested interval')
        for date in dates:
            date = int(date)
            path = DB.path('trade_ts', 'day_limit', date)
            try:
                record = repair_file(path, output / 'backups' / path.name, date, dry_run)
            except Exception as exc:
                record = {'date': date, 'path': str(path), 'status': 'failed', 'error': str(exc)}
            records.append(record)
            if record['status'] != 'unchanged':
                print(json.dumps(record, ensure_ascii=False))
        failures = sum(row['status'] == 'failed' for row in records)
        payload['summary'] = {
            'checked_files': len(records), 'failed_files': failures,
            'changed_files': sum(row['status'] in ('repaired', 'would_repair') for row in records),
            'removed_rows': sum(row.get('removed', 0) for row in records),
        }
        print(json.dumps(payload['summary'], ensure_ascii=False))
        if failures:
            raise RuntimeError(f'{failures} files unresolved; inspect {report}. Conflicting files were not changed.')
    finally:
        report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Audit report and original-file backups: {output}')
    return report


if __name__ == '__main__':
    main()
