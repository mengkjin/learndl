"""
Smoke test parallel stock factor update
"""
from __future__ import annotations

import time
from typing import Any , Literal , TypeAlias

from src.proj import CALENDAR, Logger, Proj
from src.res.factor.api import FactorUpdaterAPI
from src.res.factor.calculator.update_jobs import UpdateJobDate

CompareMode : TypeAlias = Literal['forloop', 'process', 'both']

def _resolve_date(date: int | None) -> int:
    if date is None or int(date) < 0:
        return int(CALENDAR.updated())
    return int(date)

def _pick_factors_for_date(date: int, n_factors: int , pick_pead : bool = False) -> list[str]:
    """
    First ``n_factors`` updatable stock calculators with ``init_date <= date``.

    Does not require ``target_dates`` to be non-empty (smoke test may force jobs).
    """
    calcs = FactorUpdaterAPI.Stock.calculators(all=True, updatable=True)
    names: list[str] = []
    for calc in calcs:
        if int(date) < int(calc.init_date):
            continue
        if pick_pead and not calc.factor_name.lower().startswith('pead_'):
            continue
        names.append(calc.factor_name)
        if len(names) >= n_factors and n_factors > 0:
            break
    return names


def _resolve_factors_and_date(
    requested_date: int,
    n_factors: int,
    lookback_td: int,
) -> tuple[int, list[str]]:
    """
    Pick a trading day and factor list.

    Walks back at most ``lookback_td`` trading days until at least one factor applies.
    """
    for back in range(lookback_td + 1):
        d = int(CALENDAR.td(requested_date, -back)) if back else requested_date
        names = _pick_factors_for_date(d, n_factors)
        if names:
            if back:
                Logger.alert1(
                    f'Using date {d} ({back} td back from {requested_date}); '
                    f'{len(names)} factors with init_date <= {d}',
                )
            return d, names
    raise RuntimeError(
        f'No updatable stock factor with init_date <= {requested_date} '
        f'(looked back {lookback_td} trading days)',
    )


def _collect_jobs_force(date: int, factors: list[str], overwrite: bool) -> int:
    """
    Append one job per factor for ``date`` (ignores ``target_dates``).

    Use only when calendar says the day is effective but the updater would skip jobs.
    """
    stock = FactorUpdaterAPI.Stock
    stock.jobs.clear()
    vb = Proj.vb(3)
    registry = {c.factor_name: c for c in stock.calculators(all=True, updatable=True)}
    for name in factors:
        calc = registry.get(name)
        if calc is None:
            Logger.alert1(f'Skip unknown factor: {name}')
            continue
        if int(date) < calc.init_date:
            Logger.alert1(f'Skip {name}: date {date} < init_date {calc.init_date}')
            continue
        stock.jobs.append(UpdateJobDate(calc, date, overwrite, vb_level = vb))
    stock.jobs.sort_jobs()
    return len(stock.jobs)


def _run_once(
    date: int,
    factors: list[str],
    *,
    use_process: bool,
    overwrite: bool,
    force_jobs: bool,
) -> dict[str, Any]:
    """Run ``StockFactorUpdater`` for one day; restore ``multi_thread`` afterward."""
    stock = FactorUpdaterAPI.Stock
    stock.jobs.clear()
    prev_mp = stock.groups_multiprocessing
    stock.groups_multiprocessing = use_process
    mode = 'process' if use_process else 'forloop'
    Logger.note(f'Parallel smoke test: mode={mode}, date={date}, n_factors={len(factors)}, overwrite={overwrite}')
    Logger.stdout(f'Factors: {factors}', indent=1)
    t0 = time.perf_counter()
    try:
        if force_jobs:
            n_jobs = _collect_jobs_force(date, factors, overwrite)
            if n_jobs == 0:
                raise RuntimeError('force_jobs: no jobs collected')
            stock.before_process_jobs(
                start=date, end=date, all=False, selected_factors=factors, overwrite=overwrite,
            )
            for level, level_jobs in stock.leveled_jobs():
                stock.process_level_jobs(level, level_jobs, timeout=-1)
            stock.after_process_jobs(indent=1, vb_level=1)
        else:
            stock.process_jobs(
                start=date,
                end=date,
                all=False,
                selected_factors=factors,
                overwrite=overwrite,
                indent=0,
                vb_level=1,
                timeout=-1,
            )
    finally:
        stock.groups_multiprocessing = prev_mp
    elapsed = time.perf_counter() - t0
    failed = [j for j in stock.jobs if not j.done]
    ok = len(stock.jobs) - len(failed)
    return {
        'mode': mode,
        'elapsed_s': elapsed,
        'jobs': len(stock.jobs),
        'ok': ok,
        'failed': [repr(j) for j in failed],
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    Logger.note('--- summary ---')
    for r in results:
        line = (
            f"{r['mode']:8s}  elapsed={r['elapsed_s']:.2f}s  "
            f"jobs={r['jobs']}  ok={r['ok']}/{r['jobs']}"
        )
        if r['failed']:
            line += f"  failed={r['failed'][:5]}{'...' if len(r['failed']) > 5 else ''}"
        Logger.stdout(line, indent=1)
    if len(results) == 2 and results[0]['ok'] == results[1]['ok'] == results[0]['jobs']:
        t0, t1 = results[0]['elapsed_s'], results[1]['elapsed_s']
        if t0 > 0 and t1 > 0:
            faster = results[0]['mode'] if t0 < t1 else results[1]['mode']
            ratio = max(t0, t1) / min(t0, t1)
            Logger.stdout(f'Both modes finished same job count; faster={faster} (~{ratio:.2f}x)', indent=1)


def test_parallel_factor_calculation(
    date: int | None = None,
    n_factors: int | None = 60,
    compare: CompareMode = 'both',
    overwrite: bool | None = True,
    lookback_td: int | None = 10,
    **kwargs,
) -> None:
    requested = _resolve_date(date)
    n_factors = int(n_factors or 20)
    overwrite = True if overwrite is None else bool(overwrite)
    lookback_td = int(lookback_td if lookback_td is not None else 10)

    if not CALENDAR.is_trade_date(requested):
        raise ValueError(f'{requested} is not a trading day in CALENDAR')

    date, factors = _resolve_factors_and_date(requested, n_factors, lookback_td)
    if len(factors) < n_factors and n_factors > 0:
        Logger.alert1(f'Only {len(factors)} factors applicable on {date} (requested {n_factors})')

    # Bypass ``target_dates`` filtering so we always exercise parallel calc on the chosen day.
    force_jobs = True

    results: list[dict[str, Any]] = []
    if compare in ('forloop', 'both'):
        results.append(
            _run_once(date, factors, use_process=False, overwrite=overwrite, force_jobs=force_jobs),
        )
    if compare in ('process', 'both'):
        results.append(
            _run_once(date, factors, use_process=True, overwrite=overwrite, force_jobs=force_jobs),
        )

    _print_summary(results)
    if any(r['failed'] for r in results):
        raise RuntimeError('Some factor jobs failed; see log above')