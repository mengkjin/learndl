"""Publish recent tracking portfolios and their internal predictions for inspection."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from src.shared.file_ops import fsync_dir, real_tree

from .core import Config, component, locked, sandboxed

PORTS = ('use_daily', 'master_daily', 'gru_daily')
PREDICTION = 'gru_day_V1'
LOOKBACK = 10


def publish(*, config: Config | None = None) -> str:
    config = config or Config.load()
    if not config.enabled or config.role != 'publisher' or sandboxed():
        return 'disabled'
    if not (config.shared_root / '.stfolder').is_dir():
        raise ValueError('Syncthing folder marker missing')

    from src.proj import CALENDAR, PATH
    from src.proj.db.interface.db_path import DBPath

    def port_path(name, date):
        return PATH.trade_port / name / f'{name}.{date}.feather'

    end_date = int(CALENDAR.updated())
    if not port_path(PORTS[0], end_date).is_file():
        return f'no use_daily portfolio for {end_date}'
    start = CALENDAR.td(end_date, 1 - LOOKBACK).td
    dates = [int(day) for day in CALENDAR.range(start, end_date, 'td', until_today=False)]
    target = real_tree(config.shared_root / 'persistent' / 'trading_audit' /
                       component(config.source_host) / 'latest.zip')
    with locked(config, 'trading-audit'):
        predictions = DBPath('pred', PREDICTION).get_paths(dates, closest=True)
        portfolios = {
            name: {day: path for day in dates if (path := port_path(name, day)).is_file()}
            for name in PORTS
        }
        manifest = {
            'version': 1,
            'source_host': config.source_host,
            'dates': dates,
            'predictions': {PREDICTION: sorted(int(day) for day in predictions)},
            'portfolios': {name: sorted(paths) for name, paths in portfolios.items()},
        }
        with TemporaryDirectory(dir=config.shared_root.parent, prefix='.trading-audit-') as directory:
            staging = Path(directory) / target.name
            with ZipFile(staging, 'w', ZIP_DEFLATED) as archive:
                for path in predictions.values():
                    archive.write(path, f'pred/{PREDICTION}/{path.name}')
                for name, paths in portfolios.items():
                    for path in paths.values():
                        archive.write(path, f'portfolio/{name}/{path.name}')
                archive.writestr('manifest.json', json.dumps(manifest, indent=2))
            with staging.open('rb') as stream:
                os.fsync(stream.fileno())
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)
            fsync_dir(target.parent)
    return f'published through {end_date} ({len(dates)} trade dates)'


def publish_daily_safe() -> None:
    from src.proj import Logger

    try:
        Logger.stdout(f'Trading audit snapshot: {publish()}')
    except Exception as exc:
        Logger.warning(f'Trading audit snapshot publication failed: {exc}')


if __name__ == '__main__':
    from src.proj import Logger

    Logger.stdout(publish(), vb_level=0)
