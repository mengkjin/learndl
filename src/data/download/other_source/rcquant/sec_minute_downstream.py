"""Replay minute consumers in dependency order after the source revision is published."""
from itertools import batched
from hashlib import sha256

import pandas as pd

from src.data.preprocess.nightly_rebuild import rebuild_block, retire_60m
from src.proj import CALENDAR, DB, Logger
from src.proj.db.basic import dfHandler
from src.proj.db.interface.db_path import DBPath
from src.proj.db.io.output_lock import output_lock

from .sec_none_migration import MINUTE_DATABLOCKS, write_frame
from .sec_native_revision import REVISION

HF_KEYS = ('hf_correlation', 'hf_liquidity', 'hf_momentum', 'hf_volatility')


def _save_frame(frame, src, key, date=None):
    if frame.empty:
        raise ValueError(f'{src}/{key} produced no rows for {date}')
    descriptor = DBPath(src, key)
    write_frame(dfHandler.reset_index_pandas(frame), descriptor.path_exact(date))
    descriptor.clear_date_cache()


def _save_daily(frame, src, key, date):
    with output_lock(DB.path(src, key, date)):
        _save_frame(frame, src, key, date)


def _tables(job, revision, start, end):
    from src.data.update.custom.daily_risk import DailyRiskUpdater, calc_daily_risk
    from src.data.update.custom.market_daily_risk import MarketDailyRiskUpdater as MarketRisk
    from src.data.update.custom.min_chars._common import trailing_aligned_dates
    from src.data.update.custom.min_chars.daily import MinCharsDailyUpdater, calc_min_chars
    from src.data.update.custom.min_chars.rolling import ROLL_WINDOW, MinCharsRollUpdater, calc_min_chars_roll
    from src.data.update.custom.min_chars.tagged import MinCharsTaggedUpdater, calc_min_chars_tag

    stage = f'{revision}_tables'
    for cls, key, calculate in (
        (MinCharsDailyUpdater, 'min', calc_min_chars),
        (MinCharsRollUpdater, 'min', calc_min_chars_roll),
        (MinCharsTaggedUpdater, 'min', calc_min_chars_tag),
        (DailyRiskUpdater, '5min', calc_daily_risk),
    ):
        target = DB.dates('trade_ts', key, use_alt=True).slice(start=max(start, cls.START_DATE), end=end)
        if cls is MinCharsTaggedUpdater:
            target = target.intersect(DB.dates('min_chars', 'min_chars_roll'))
        for date in map(int, target):
            job.guard()
            item = f'{cls.DB_SRC}/{cls.DB_KEY}/{date}'
            if job.read(stage, item) and DB.path(cls.DB_SRC, cls.DB_KEY, date).is_file():
                continue
            if cls is MinCharsRollUpdater and len(trailing_aligned_dates(date, ROLL_WINDOW, 'min_chars')) < ROLL_WINDOW:
                continue
            _save_daily(calculate(date), cls.DB_SRC, cls.DB_KEY, date)
            job.mark(stage, item)
        Logger.stdout(f'Nightly replay {cls.DB_SRC}/{cls.DB_KEY}: checked through {end}')

    target = DB.dates('trade_ts', '5min', use_alt=True).slice(start=max(start, MarketRisk.START_DATE), end=end)
    pending = [int(date) for date in target if not job.read(stage, f'market_daily/risk/{date}')]
    for dates in batched(pending, 10):
        job.guard()
        data = pd.concat([MarketRisk.calculate(day, backend='polars') for day in dates])
        if set(data['date']) != set(dates):
            raise ValueError(f'Market risk did not cover all required dates {dates}')
        MarketRisk.append_result(data, save=_save_frame)
        for day in dates:
            job.mark(stage, f'market_daily/risk/{day}')
        Logger.stdout(f'Nightly market risk replay through {dates[-1]}')


def _factors(job, revision, start, end):
    from src.res.factor.calculator.factor_calc import (
        FactorCalculator, HfCorrelationFactor, HfLiquidityFactor, HfMomentumFactor, HfVolatilityFactor,
    )
    stage = f'{revision}_factors'
    updated = []
    for factor in FactorCalculator.iter(meta_type='stock', updatable=True):
        if not (isinstance(factor, (HfCorrelationFactor, HfLiquidityFactor, HfMomentumFactor, HfVolatilityFactor))
                or factor.file_name.rsplit('/', 1)[-1] in ('momentum_umr_new', 'momentum_umr_raw')):
            continue
        name = factor.factor_name
        updated.append(name)
        for day in map(int, factor.target_dates(start=start, end=end, overwrite=True)):
            job.guard()
            item = f'{name}/{day}'
            if job.read(stage, item) and DB.path(factor.db_src, factor.db_key, day).is_file():
                continue
            data = factor.validate_value(factor(day), day, strict=True)
            _save_daily(data, factor.db_src, factor.db_key, day)
            job.mark(stage, item)
        for period, dates in factor.stats_target_dates(start, end, overwrite=True).items():
            for offset in range(0, len(dates), 40):
                job.guard()
                batch = list(map(int, dates[offset:offset + 40]))
                item = f'{name}/{period}/{batch[0]}-{batch[-1]}'
                if not job.read(stage, item):
                    factor.update_periodic_stats(period, batch, overwrite=True, save=_save_frame)
                    job.mark(stage, item)
        Logger.stdout(f'Nightly factor replay completed {name}')
    if not updated:
        raise RuntimeError('Nightly factor registry contains no minute-derived factors')
    return updated


def _review_production(factors):
    """Resolve the declared production consumers of the repaired inputs."""
    from src.proj import PATH
    import yaml

    strategy = PATH.main / 'configs/strategy'
    configured = yaml.safe_load((strategy / 'model.yaml').read_text())
    models = {entry['name'] for key in ('prediction', 'hidden_extraction')
              for entry in configured.get(key, {}).values()}
    verified_daily = set()
    for model in models:
        path = PATH.main / 'models/nn' / model / 'configs/production.yaml'
        if path.is_file():
            config = yaml.safe_load(path.read_text())['model']
            if config.get('input.type') == 'data' and config.get('input.data.types') in ('day', ['day']):
                verified_daily.add(model)
    ports = yaml.safe_load((strategy / 'trading_port.yaml').read_text()).get('trading_ports', {})
    alphas = set()
    for port in ports.values():
        names = port['alpha']
        alphas.update(names if isinstance(names, list) else names.split(','))
    affected = alphas.intersection({f'factor@{name}' for name in factors})
    # The current production portfolios use these external sell-side inputs and
    # a daily-only GRU. Require an explicit replay mapping if production changes.
    known = {'factor@ht_master_combined', 'factor@hy_scores_v5', 'factor@df_scores_v3',
             *configured.get('prediction', {})}
    if models - verified_daily or affected or alphas - known:
        raise RuntimeError(f'Nightly needs production replay mapping for {sorted(models - verified_daily | affected | (alphas - known))}')
    Logger.stdout('Nightly production dependency check: daily-only models and externally supplied alphas')
    return {'models': sorted(models), 'portfolios': sorted(ports), 'replayed': []}


def rebuild_minute_consumers(job, dates, source_receipt):
    source = ','.join(map(str, dates)) + REVISION + ''.join(x['digest'] for x in source_receipt['outputs'])
    revision = 'minute_consumers_v1_' + sha256(source.encode()).hexdigest()[:16]
    if job.read(revision, 'complete_v2'):
        return True
    bounds = job.read(revision, 'bounds')
    tail = int(CALENDAR.updated())
    if bounds is None:
        bounds = [min(dates), tail]
        job.mark(revision, 'bounds', bounds)
    elif tail > bounds[1]:
        # Daily jobs can append new dates while earlier replay stages span nights.
        bounds[1] = tail
        job.mark(revision, 'bounds', bounds)
    retire_60m(job, revision)
    _tables(job, revision, *bounds)
    factors = _factors(job, revision, *bounds)
    all_published = True
    for frame in ('fit', 'predict'):
        for key in (*MINUTE_DATABLOCKS, 'market', 'minc', 'mincr', *HF_KEYS, 'momentum'):
            job.guard()
            if not rebuild_block(job, revision, frame, key):
                all_published = False
    if not all_published:
        return False
    job.mark(revision, 'complete_v2', _review_production(factors))
    return True
