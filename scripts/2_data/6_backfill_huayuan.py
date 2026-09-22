"""Backfill huayuan.cmm and huayuan.scores_v5 using configured remote tables.

From the server repository root:
    .venv/bin/python scripts/2_data/6_backfill_huayuan.py --dry-run
    .venv/bin/python scripts/2_data/6_backfill_huayuan.py

Default: fill missing dates from configured start through today. --overwrite
also replaces existing dates (useful for previously misconfigured scores_v5).
Run separately from the regular sellside update to avoid concurrent writes.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

KEYS = ('huayuan.cmm', 'huayuan.scores_v5')


def date_arg(value):
    try:
        if len(value) != 8 or not value.isdigit():
            raise ValueError('Expected YYYYMMDD')
        datetime.strptime(value, '%Y%m%d')
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return int(value)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', type=date_arg)
    parser.add_argument('--end', type=date_arg)
    parser.add_argument('--batch-dates', type=int, default=20)
    parser.add_argument('--keys', nargs='+', choices=KEYS, default=list(KEYS))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='List remote/pending dates without writing')
    args = parser.parse_args()
    if args.batch_dates < 1 or (args.start and args.end and args.start > args.end):
        parser.error('Require start <= end and batch-dates >= 1')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.download.sellside.from_sql import SellsideSQLDownloader
    from src.proj import DB

    for key, downloader in SellsideSQLDownloader.default_factors(args.keys).items():
        if downloader.db_key != key:
            raise RuntimeError(f'Local key mismatch: {key} -> {downloader.db_key}')
        backfill(downloader, DB, args.start, args.end, args.batch_dates,
                 args.overwrite, args.dry_run)
    print('Dry run complete.' if args.dry_run else 'Backfill complete.', flush=True)


if __name__ == '__main__':
    main()
