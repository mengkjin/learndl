# coding: utf-8
# author: jinmeng
# date: 2026-09-24
# description: Build sellside valid-date index
# content: |
#   扫描已落盘的 sellside 日文件，生成有效日索引。
#   有限值个数大于 0 的日期写入索引；全空日保留原文件，不进入索引。
#   默认 dry_run 只打印，不覆盖已有索引。
# email: False
# mode: shell
# parameters:
#   keys:
#       type: str
#       desc: comma-separated sellside db keys, or all
#       required: False
#       default: huayuan.scores_v5
#   dry_run:
#       type: [True, False]
#       desc: print the split without writing the index
#       required: False
#       default: True

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.proj import DB, Logger
from src.proj.util.script import ScriptTool
from src.data.download.sellside.from_sql import factor_settings
from src.data.download.sellside.valid_dates import finite_value_count, write_index, index_path


def _as_bool(value: bool | str | None) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def _selected_keys(keys: str) -> list[str]:
    text = (keys or '').strip()
    if text.lower() == 'all':
        return list(factor_settings)
    selected = [part.strip() for part in text.split(',') if part.strip()]
    unknown = [key for key in selected if key not in factor_settings]
    if unknown or not selected:
        raise ValueError(f'keys must be "all" or a subset of {tuple(factor_settings)}, got {keys!r}')
    return selected


def _split_dates(db_key: str) -> tuple[np.ndarray, np.ndarray]:
    stored = DB.dates('sellside', db_key)
    if len(stored) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    valid: list[int] = []
    unusable: list[int] = []
    for day in stored.dates:
        frame = DB.load('sellside', db_key, int(day), vb_level='never')
        if finite_value_count(frame) > 0:
            valid.append(int(day))
        else:
            unusable.append(int(day))
    return np.array(valid, dtype=np.int64), np.array(unusable, dtype=np.int64)


@ScriptTool('build_sellside_valid_dates')
def main(keys: str = 'huayuan.scores_v5', dry_run: bool = True, **kwargs):
    """Scan stored sellside days and write the valid-date index."""
    del kwargs
    dry_run_i = _as_bool(dry_run)
    for db_key in _selected_keys(keys):
        valid, unusable = _split_dates(db_key)
        Logger.stdout(
            f'{db_key}: stored={len(valid) + len(unusable)}, '
            f'valid={len(valid)}, all_null_or_empty={len(unusable)}'
        )
        if len(unusable):
            sample = ', '.join(str(int(day)) for day in unusable[:20])
            Logger.stdout(f'  excluded dates: {sample}')
            if len(unusable) > 20:
                Logger.stdout(f'  ... {len(unusable) - 20} more')
        if dry_run_i:
            Logger.stdout(f'  dry_run, index not written ({index_path(db_key)})')
            continue
        write_index(index_path(db_key), valid)
        Logger.success(f'  wrote {len(valid)} valid dates to {index_path(db_key)}')
    if dry_run_i:
        Logger.stdout('Dry run complete. Pass --dry_run False to write indexes.')


if __name__ == '__main__':
    task = main()
    if getattr(task, 'status', 'Success') == 'Error':
        raise SystemExit(1)
