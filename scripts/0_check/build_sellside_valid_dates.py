# coding: utf-8
# author: jinmeng
# date: 2026-09-24
# description: Build sellside valid-value stats
# content: |
#   扫描全部 sellside key 的已落盘日文件，写入
#   DB_sellside/.data_stats/valid_values/{db_key}.feather。
#   每行记录 date、secid_count、nan_count、is_valid。
# email: False
# mode: shell

from __future__ import annotations

from src.proj import Logger
from src.proj.util.script import ScriptTool
from src.data.download.sellside.from_sql import factor_settings
from src.data.download.sellside.valid_dates import (
    backfill_valid_values , read_valid_values , valid_values_path ,
)


@ScriptTool('build_sellside_valid_dates')
def main(**kwargs):
    """Rebuild valid-value stats for every sellside key."""
    del kwargs
    for db_key in factor_settings:
        written = backfill_valid_values(db_key , full = True)
        frame = read_valid_values(valid_values_path(db_key))
        if frame is None or frame.empty:
            Logger.stdout(f'{db_key}: stored=0, wrote {valid_values_path(db_key)}')
            continue
        valid_n = int(frame['is_valid'].sum())
        unusable = frame.loc[~frame['is_valid'] , 'date']
        Logger.stdout(
            f'{db_key}: stored={len(frame)}, valid={valid_n}, '
            f'all_null_or_empty={len(frame) - valid_n}, scanned={written}'
        )
        if len(unusable):
            sample = ', '.join(str(int(day)) for day in unusable.iloc[:20])
            Logger.stdout(f'  is_valid=False dates: {sample}')
            if len(unusable) > 20:
                Logger.stdout(f'  ... {len(unusable) - 20} more')
        Logger.success(f'  wrote {len(frame)} rows to {valid_values_path(db_key)}')


if __name__ == '__main__':
    task = main()
    if getattr(task, 'status', 'Success') == 'Error':
        raise SystemExit(1)
