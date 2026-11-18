"""Manual, serial, missing-only factor history repair; no daily-update TTL cache."""
from __future__ import annotations

from src.proj import CALENDAR, Const, Logger
from src.proj.util.functional.parallel import serial_execution
from .factor_calc import FactorCalculator

__all__ = ['fill_calculator', 'run_backfill']


def fill_calculator(calc, start, end, batch_size=32, dry_run=False):
    """Finish values then statistics before allowing dependent calculators to run."""
    name = calc.factor_name
    dates = [int(d) for d in calc.target_dates(start, end, overwrite=False)]
    Logger.stdout(f"[SCAN] {name}: {len(dates)} missing value dates", flush=True)
    for index, date in enumerate(dates, 1):
        Logger.stdout(f"[VALUE {index}/{len(dates)}] {name} date={date}", flush=True)
        if not dry_run:
            # Keep calculator-specific preparation, including pooling weights.
            calc.update_all_factors(start=date, end=date, overwrite=False)
            if len(calc.target_dates(date, date, overwrite=False)):
                raise RuntimeError(f"Factor remains missing: {name}@{date}")
            Logger.stdout(f"[DONE VALUE] {name} date={date}", flush=True)
    if calc.meta_type == 'market':
        return
    targets = calc.stats_target_dates(start, end, overwrite=False)
    for period, missing in targets.items():
        dates = [int(d) for d in missing]
        Logger.stdout(f"[SCAN STATS] {name}/{period}: {len(dates)} missing dates", flush=True)
        for offset in range(0, len(dates), batch_size):
            batch = dates[offset:offset + batch_size]
            Logger.stdout(f"[STATS] {name}/{period} dates={batch}", flush=True)
            if not dry_run:
                calc.update_periodic_stats(period, batch, overwrite=False)
                remaining = calc.stats_target_dates(batch[0], batch[-1], overwrite=False)[period]
                if set(batch).intersection(int(d) for d in remaining):
                    raise RuntimeError(f"Stats remain missing: {name}/{period} dates={batch}")
                Logger.stdout(f"[DONE STATS] {name}/{period} dates={batch}", flush=True)


def run_backfill(start=None, end=None, batch_size=32, dry_run=False):
    """Respect each factor's configured frequency and all historical missing dates."""
    start = Const.Factor.UPDATE.start if start is None else int(start)
    end = min(CALENDAR.updated(), Const.Factor.UPDATE.end) if end is None else int(end)
    if start > end or end > CALENDAR.updated() or batch_size < 1:
        raise ValueError('Require start <= end <= latest confirmed date and batch_size >= 1')
    Logger.stdout(f"[BACKFILL] range={start}..{end} batch_size={batch_size} dry_run={dry_run}", flush=True)
    with serial_execution():
        calculators = sorted(
            FactorCalculator.iter(all=True, updatable=True),
            key=lambda c: (c.meta_type != 'market', c.level, c.meta_type == 'pooling', c.factor_name),
        )
        for index, calc in enumerate(calculators, 1):
            Logger.stdout(f"[FACTOR {index}/{len(calculators)}] {calc.factor_name} level={calc.level}", flush=True)
            try:
                fill_calculator(calc, start, end, batch_size, dry_run)
            except Exception as exc:
                Logger.stdout(f"[FAILED] {calc.factor_name}: {type(exc).__name__}: {exc}", flush=True)
                raise
    Logger.stdout('[COMPLETE] Historical missing-date scan finished', flush=True)
