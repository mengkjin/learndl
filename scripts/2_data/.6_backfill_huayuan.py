# coding: utf-8
# author: jinmeng
# date: 2026-09-23
# description: Backfill huayuan sellside history
# content: |
#   按远端表补全 sellside/huayuan.cmm 与 sellside/huayuan.scores_v5。
#   cmm 从配置起点 20260101 起，scores_v5 从 20171229 起，结束日默认今天。
#   默认只写本地缺失的日期；overwrite=True 才覆盖已有日期文件。
#   中断后直接再跑即可续补，已落盘日期会跳过。
#   与日常 sellside 更新分开跑，避免同时写同一库。
# email: True
# mode: shell
# parameters:
#   start:
#       type: int
#       desc: inclusive start yyyyMMdd; 0 = each factor's configured start_date
#       required: False
#       default: 0
#   end:
#       type: int
#       desc: inclusive end yyyyMMdd; 0 = today, clamped to factor end_date
#       required: False
#       default: 0
#   batch_dates:
#       type: int
#       desc: remote query batch size in dates
#       required: False
#       default: 20
#   keys:
#       type: str
#       desc: comma-separated subset of huayuan.cmm,huayuan.scores_v5
#       required: False
#       default: huayuan.cmm,huayuan.scores_v5
#   overwrite:
#       type: [True, False]
#       desc: replace existing date files; False only fills missing dates
#       required: False
#       default: False
#   dry_run:
#       type: [True, False]
#       desc: list remote and pending dates without writing
#       required: False
#       default: False

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.proj.util.script import ScriptTool

KEYS = ('huayuan.cmm', 'huayuan.scores_v5')
DEFAULT_KEYS = ','.join(KEYS)


def _optional_date(value: int | str | None) -> int | None:
    """Return yyyyMMdd, or None when *value* is omitted (0 / empty)."""
    if value is None or value == '':
        return None
    text = str(value).strip()
    if text in {'0', 'None'}:
        return None
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f'Expected YYYYMMDD, got {value!r}')
    datetime.strptime(text, '%Y%m%d')
    return int(text)


def _as_bool(value: bool | str | None) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def _normalize_keys(keys: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if keys is None or keys == '':
        selected = list(KEYS)
    elif isinstance(keys, str):
        selected = [part.strip() for part in keys.split(',') if part.strip()]
    else:
        selected = [str(part).strip() for part in keys if str(part).strip()]
    unknown = [key for key in selected if key not in KEYS]
    if unknown or not selected:
        raise ValueError(f'keys must be a non-empty subset of {KEYS}, got {keys!r}')
    return [str(key) for key in selected]


def _normalize_argv(argv: list[str]) -> list[str]:
    """Accept the original hyphen flags alongside header parameter names."""
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == '--dry-run':
            out.extend(['--dry_run', 'True'])
        elif token.startswith('--dry-run='):
            out.append('--dry_run=' + token.split('=', 1)[1])
        elif token == '--batch-dates':
            if index + 1 >= len(argv):
                raise SystemExit('--batch-dates requires a value')
            out.extend(['--batch_dates', argv[index + 1]])
            index += 1
        elif token.startswith('--batch-dates='):
            out.append('--batch_dates=' + token.split('=', 1)[1])
        elif token == '--overwrite':
            nxt = argv[index + 1] if index + 1 < len(argv) else None
            if nxt is None or nxt.startswith('--'):
                out.extend(['--overwrite', 'True'])
            else:
                out.extend(['--overwrite', nxt])
                index += 1
        else:
            out.append(token)
        index += 1
    return out


def backfill(downloader, db, start=None, end=None, batch_dates=20,
             overwrite=False, dry_run=False):
    from sqlalchemy import text

    start = max(start or downloader.start_date, downloader.start_date)
    end = min(end or int(datetime.now().strftime('%Y%m%d')), downloader.end_date)
    if start > end or batch_dates < 1:
        raise ValueError('Require start <= end and batch_dates >= 1')
    table, column = downloader.factor_set, downloader.date_col
    if not all(name.replace('_', '').isalnum() for name in (table, column)):
        raise ValueError('Unexpected SQL identifier in configuration')
    connection = downloader.connection
    try:
        with connection.get_connection() as conn:
            rows = conn.execute(text(
                f'SELECT DISTINCT `{column}` FROM `{table}` '
                f'WHERE `{column}` BETWEEN :start AND :end ORDER BY `{column}`'
            ), {
                'start': datetime.strptime(str(start), '%Y%m%d').date(),
                'end': datetime.strptime(str(end), '%Y%m%d').date(),
            }).scalars().all()
        dates = [int(d.strftime('%Y%m%d')) for d in rows]
        if not dates:
            raise RuntimeError(f'{table}: no remote data in [{start}, {end}]')
        stored = {int(d) for d in db.dates('sellside', downloader.db_key)}
        pending = dates if overwrite else [d for d in dates if d not in stored]
        print(f'{downloader.db_key} <- {table}: {len(dates)} remote dates '
              f'({dates[0]}..{dates[-1]}), {len(pending)} pending, '
              f'overwrite={overwrite}', flush=True)
        if dry_run:
            return
        for offset in range(0, len(pending), batch_dates):
            batch = pending[offset:offset + batch_dates]
            data = downloader.query_factor_values(batch[0], batch[-1])
            if data is None or data.empty:
                raise RuntimeError(f'{table}: failed/empty query for {batch[0]}..{batch[-1]}')
            required = {'date', 'secid', downloader.local_name}
            if not required.issubset(data.columns):
                raise RuntimeError(f'{table}: unexpected columns {list(data.columns)}')
            # A range query can include already stored dates between missing ones.
            data = data.loc[data['date'].isin(batch)]
            if set(data['date'].unique()) != set(batch):
                raise RuntimeError(f'{table}: incomplete date coverage for {batch[0]}..{batch[-1]}')
            if data.duplicated(['date', 'secid']).any():
                raise RuntimeError(f'{table}: duplicate (date, secid) rows')
            if not data.groupby('date')[downloader.local_name].count().gt(0).all():
                raise RuntimeError(f'{table}: a date has no non-null factor values')
            for day, frame in data.groupby('date', sort=True):
                if not db.save(frame.drop(columns='date').sort_values('secid'),
                               'sellside', downloader.db_key, int(day), overwrite=overwrite):
                    raise RuntimeError(f'{table}: failed to save {day}; rerun to resume')
            print(f'{downloader.db_key}: saved {offset + len(batch)}/{len(pending)} '
                  f'dates, through {batch[-1]}', flush=True)
    finally:
        connection.close()


@ScriptTool('backfill_huayuan')
def main(
    start: int = 0,
    end: int = 0,
    batch_dates: int = 20,
    keys: str = DEFAULT_KEYS,
    overwrite: bool = False,
    dry_run: bool = False,
    **kwargs,
):
    """Fill missing huayuan sellside dates; overwrite only when explicitly requested."""
    del kwargs
    start_i = _optional_date(start)
    end_i = _optional_date(end)
    batch = int(batch_dates)
    overwrite_i = _as_bool(overwrite)
    dry_run_i = _as_bool(dry_run)
    if batch < 1 or (start_i and end_i and start_i > end_i):
        raise ValueError('Require start <= end and batch_dates >= 1')
    from src.data.download.sellside.from_sql import SellsideSQLDownloader
    from src.proj import DB

    for key, downloader in SellsideSQLDownloader.default_factors(_normalize_keys(keys)).items():
        if downloader.db_key != key:
            raise RuntimeError(f'Local key mismatch: {key} -> {downloader.db_key}')
        backfill(downloader, DB, start_i, end_i, batch, overwrite_i, dry_run_i)
    print('Dry run complete.' if dry_run_i else 'Backfill complete.', flush=True)


if __name__ == '__main__':
    sys.argv = [sys.argv[0], *_normalize_argv(sys.argv[1:])]
    task = main()
    if getattr(task, 'status', 'Success') == 'Error':
        raise SystemExit(1)
