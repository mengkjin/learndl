"""Replay minute consumers in dependency order after the source revision is published."""
from __future__ import annotations

__all__ = [
    'rebuild_minute_consumers',
]

from contextlib import contextmanager
from functools import partial
from itertools import batched, groupby
from time import perf_counter
from hashlib import sha256

import pandas as pd

from src.data.preprocess.nightly_rebuild import rebuild_block, retire_60m
from src.proj import CALENDAR, DB, Logger
from src.proj.db.basic import dfHandler
from src.proj.db.interface.db_path import DBPath
from src.proj.db.io.output_lock import output_lock

from .sec_none_migration import MINUTE_DATABLOCKS
from .sec_native_revision import REVISION

HF_KEYS = ('hf_correlation', 'hf_liquidity', 'hf_momentum', 'hf_volatility')


def _save_frame(job, frame, src, key, date=None):
    if frame.empty:
        raise ValueError(f'{src}/{key} produced no rows for {date}')
    descriptor = DBPath(src, key)
    job.write_frame(dfHandler.reset_index_pandas(frame), descriptor.path_exact(date))
    descriptor.clear_date_cache()


def _save_daily(job, frame, src, key, date):
    with output_lock(DB.path(src, key, date)):
        _save_frame(job, frame, src, key, date)


@contextmanager
def _month_progress(label, month):
    start = perf_counter()
    counts = {'written': 0, 'reused': 0}
    try:
        yield counts
    finally:
        if counts['written']:
            Logger.stdout(f'Nightly {label} {month}: {counts["written"]} written, '
                          f'{counts["reused"]} reused, {perf_counter()-start:.1f}s')


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
        label = f'{cls.DB_SRC}/{cls.DB_KEY}'
        for month, dates in groupby(map(int, target), key=lambda day: day // 100):
            with _month_progress(label, month) as counts:
                for date in dates:
                    job.guard()
                    item = f'{label}/{date}'
                    if job.read(stage, item) and DB.path(cls.DB_SRC, cls.DB_KEY, date).is_file():
                        counts['reused'] += 1
                        continue
                    if cls is MinCharsRollUpdater and len(trailing_aligned_dates(date, ROLL_WINDOW, 'min_chars')) < ROLL_WINDOW:
                        continue
                    _save_daily(job, calculate(date), cls.DB_SRC, cls.DB_KEY, date)
                    job.mark(stage, item)
                    counts['written'] += 1
        Logger.stdout(f'Nightly replay {cls.DB_SRC}/{cls.DB_KEY}: checked through {end}')

    target = DB.dates('trade_ts', '5min', use_alt=True).slice(start=max(start, MarketRisk.START_DATE), end=end)
    for month, days in groupby(map(int, target), key=lambda day: day // 100):
        with _month_progress('market_daily/risk', month) as counts:
            pending = []
            for day in days:
                if job.read(stage, f'market_daily/risk/{day}'):
                    counts['reused'] += 1
                else:
                    pending.append(day)
            for dates in batched(pending, 10):
                job.guard()
                data = pd.concat([MarketRisk.calculate(day, backend='polars') for day in dates])
                if set(data['date']) != set(dates):
                    raise ValueError(f'Market risk did not cover all required dates {dates}')
                MarketRisk.append_result(data, save=partial(_save_frame, job))
                for day in dates:
                    job.mark(stage, f'market_daily/risk/{day}')
                counts['written'] += len(dates)


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
        days = factor.target_dates(start=start, end=end, overwrite=True)
        for month, dates in groupby(map(int, days), key=lambda day: day // 100):
            with _month_progress(name, month) as counts:
                for day in dates:
                    job.guard()
                    item = f'{name}/{day}'
                    if job.read(stage, item) and DB.path(factor.db_src, factor.db_key, day).is_file():
                        counts['reused'] += 1
                        continue
                    data = factor.validate_value(factor(day), day, strict=True)
                    _save_daily(job, data, factor.db_src, factor.db_key, day)
                    job.mark(stage, item)
                    counts['written'] += 1
        for period, dates in factor.stats_target_dates(start, end, overwrite=True).items():
            # Preserve established batch keys on resume.
            batches = batched(map(int, dates), 40)
            for month, month_batches in groupby(batches, key=lambda batch: batch[-1] // 100):
                with _month_progress(f'{name} {period} stats', month) as counts:
                    for batch in month_batches:
                        job.guard()
                        item = f'{name}/{period}/{batch[0]}-{batch[-1]}'
                        if job.read(stage, item):
                            counts['reused'] += len(batch)
                            continue
                        factor.update_periodic_stats(period, list(batch), overwrite=True,
                                                     save=partial(_save_frame, job), log_success=False)
                        job.mark(stage, item)
                        counts['written'] += len(batch)
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


def rebuild_minute_consumers(job, dates, source_receipt, *, refreshed=None):
    source = ','.join(map(str, dates)) + REVISION
    if source_receipt is not None:
        source += ''.join(x['digest'] for x in source_receipt['outputs'])
    if refreshed is not None:
        # A fully published refresh starts replay from the historical lower bound.
        source += f'|native-refresh:{refreshed}'
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
