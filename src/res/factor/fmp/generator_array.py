"""Shared NumPy selection, evolution and accounting for portfolio generators.

Pandas is confined to input/output adapters. Historical top/Yale semantics,
including missing quote rows and rounding, are preserved. Positions and input
caches belong to each run. The original builder supplies the reference backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.proj import CALENDAR, Const
from src.res.factor.util import Portfolio, Port, PortCreateResult, PortfolioAccount
from src.res.factor.util.agency.portfolio_accountant import AccountConfig

T50Backend = Literal['legacy', 'array', 'auto']
BLOCK_DAYS = 128


class UnsupportedT50(ValueError):
    """A known unsupported contract; only this exception permits auto fallback."""


@dataclass(frozen=True)
class Position:
    ids: np.ndarray
    weights: np.ndarray
    date: int
    value: float = 1.
    exists: bool = True

    @classmethod
    def empty(cls, date):
        return cls(np.empty(0, dtype=np.int64), np.empty(0), int(date), exists=False)


@dataclass(frozen=True)
class Quotes:
    ids: np.ndarray  # unique, ascending
    returns: np.ndarray


@dataclass(frozen=True)
class Selection:
    ids: np.ndarray
    alpha: np.ndarray
    industry: np.ndarray
    rankpct: np.ndarray
    industry_rank: np.ndarray
    descending: np.ndarray


def descending_order(values):
    """pandas nargsort's default descending quicksort, including tie ordering."""
    valid = np.flatnonzero(~np.isnan(values))
    reverse = valid[::-1]
    order = reverse[np.argsort(values[reverse], kind='quicksort')][::-1]
    return np.concatenate((order, np.flatnonzero(np.isnan(values))))


def prepare_selection(ids, alpha, industry):
    n = len(ids)
    rankpct = np.full(n, np.nan)
    valid = np.flatnonzero(~np.isnan(alpha))
    ascending = valid[np.argsort(alpha[valid], kind='stable')]
    rankpct[ascending] = np.arange(1, len(valid) + 1) / max(1, len(valid))
    ranks = np.full(n, np.nan)
    # rank(method='first') breaks ties by original candidate row order.
    for code in np.unique(industry):
        members = np.flatnonzero((industry == code) & ~np.isnan(alpha))
        order = members[np.argsort(-alpha[members], kind='stable')]
        ranks[order] = np.arange(1, len(order) + 1)
    return Selection(ids, alpha, industry, rankpct, ranks, descending_order(alpha))


def select(day: Selection, old: Position, conf, date: int) -> Position:
    held = np.isin(day.ids, old.ids[old.weights > 0])
    cumulative = np.empty(len(held), dtype=np.int64)
    cumulative[day.descending] = np.cumsum(held[day.descending])
    buffered = ((day.rankpct >= conf.buffer_zone)
                | ((cumulative <= conf.stay_num) & (day.rankpct >= conf.no_zone)))
    stay = held & buffered
    counts = np.bincount(day.industry[stay], minlength=int(day.industry.max()) + 1 if len(day.ids) else 0)
    eligible = (~stay) & (day.industry_rank < conf.indus_slots - counts[day.industry])
    # The reference sorts the already descending, filtered frame AGAIN. This
    # matters for ties; reusing the first ordering changes the chosen stocks.
    remaining = day.descending[eligible[day.descending]]
    entry = remaining[descending_order(day.alpha[remaining])]
    count = conf.n_best - int(stay.sum())
    entry = entry[:count]  # also mirrors DataFrame.head for negative counts
    chosen = np.union1d(day.ids[stay], day.ids[entry])
    return Position(chosen, np.full(len(chosen), 1. / len(chosen)) if len(chosen) else np.empty(0),
                    int(date), old.value)


def matching(position: Position, quotes: Quotes):
    index = np.searchsorted(quotes.ids, position.ids)
    valid = index < len(quotes.ids)
    valid[valid] &= quotes.ids[index[valid]] == position.ids[valid]
    return valid, index[valid]


def weighted_return(position: Position, quotes: Quotes):
    valid, index = matching(position, quotes)
    return (position.weights[valid] * quotes.returns[index]).sum()


def evolve(position: Position, date: int, quotes: Quotes | None) -> Position:
    if quotes is None:
        return Position(position.ids, position.weights, int(date), position.value, position.exists)
    valid, index = matching(position, quotes)
    weights = position.weights[valid]
    new = weights * (1. + quotes.returns[index])
    value = position.value * (1. + new.sum() - weights.sum())
    if not np.isfinite(value) or value == 0:
        raise UnsupportedT50('nonfinite or zero portfolio value during evolution')
    weights = new * position.value / value
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise UnsupportedT50('evolution left the finite long-only domain')
    order = descending_order(weights)
    return Position(position.ids[valid][order], weights[order], int(date), value, position.exists)


def turnover(new: Position, old: Position, decimals: int):
    if not new.exists or new is old:
        return 0.
    ids = np.union1d(new.ids, old.ids)
    weights = np.zeros(len(ids))
    weights[np.searchsorted(ids, new.ids)] += new.weights
    # Port.__sub__ uses ``another * -1``; __mul__ calls rescale, so the
    # reference normalizes the OLD position before subtracting it. Missing
    # quote rows can leave its weight sum below one. Do not "fix" that here.
    with np.errstate(divide='ignore', invalid='ignore'):
        previous = old.weights / old.weights.sum()
    weights[np.searchsorted(ids, old.ids)] -= np.nan_to_num(previous, nan=0.)
    return np.round(np.abs(weights).sum(), decimals)


def account_step(old, new, bench, close, bench_close, old_open, new_open, old_vwap, new_vwap,
                 ret_decimals, turn_decimals):
    pf_close = weighted_return(new, close)
    bm = weighted_return(bench, bench_close)
    overnight_open = (weighted_return(old, old_open) + 1.) - 1.
    shadow_open = (weighted_return(new, new_open) + 1.) - 1.
    overnight_vwap = (weighted_return(old, old_vwap) + 1.) - 1.
    shadow_vwap = (weighted_return(new, new_vwap) + 1.) - 1.
    pf_open = (pf_close + 1.) * (overnight_open + 1.) / (shadow_open + 1.) - 1.
    pf_vwap = (pf_close + 1.) * (overnight_vwap + 1.) / (shadow_vwap + 1.) - 1.
    return np.array([np.round(pf_open, ret_decimals), np.round(bm, ret_decimals),
                     turnover(new, old, turn_decimals), np.round(overnight_open, ret_decimals),
                     np.round(pf_close, ret_decimals), np.round(pf_open, ret_decimals),
                     np.round(pf_vwap, ret_decimals)])


def from_port(port):
    ids, weights = port.secid.copy(), port.weight.copy()
    if len(np.unique(ids)) != len(ids) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise UnsupportedT50('positions must have unique IDs and finite long-only weights')
    if not np.isfinite(port.value) or port.value <= 0:
        raise UnsupportedT50('positions must have positive finite value')
    return Position(ids, weights, int(port.date), float(port.value), bool(port))


class Inputs:
    """Bounded, block-local adapters; quote requests shared across submodels."""
    def __init__(self, prepared=None, generic=False):
        self.prepared = prepared
        self.generic = generic
        self.quotes = {}
        self.industries = {}
        self.benchmarks = {}
        self.quote_requests = 0
        self.seconds = 0.

    def clear(self):
        self.quotes.clear()
        self.industries.clear()
        self.benchmarks.clear()

    def quote(self, date0, date1=None, price: Literal['close', 'open', 'vwap'] = 'close'):
        date0 = int(DATAVENDOR.td(date0))
        date1 = int(DATAVENDOR.td(date0, 1)) if date1 is None else int(date1)
        key = (date0, date1, price)
        if key not in self.quotes:
            started = perf_counter()
            prepared = self.prepared.quote(date0, date1, price) if self.prepared else None
            if prepared is not None:
                ids, raw = prepared
                values = np.where(np.isnan(raw), 0., raw)
                if not np.isfinite(values).all():
                    raise UnsupportedT50('nonfinite prepared quote returns')
                self.quotes[key] = Quotes(ids, values)
                self.quote_requests += 1
                self.seconds += perf_counter() - started
                return self.quotes[key]
            q = DATAVENDOR.get_quote_ret(date0, date1, 'close', price)
            if q is None:
                raise ValueError(f'Ret Quote (at {date1}) does not exist')
            ids = q['secid'].to_numpy(dtype=np.int64) if 'secid' in q else q.index.to_numpy(dtype=np.int64)
            if len(np.unique(ids)) != len(ids):
                raise UnsupportedT50('duplicate security IDs in returns')
            values = q['ret'].fillna(0).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise UnsupportedT50('nonfinite quote returns')
            order = np.argsort(ids)
            self.quotes[key] = Quotes(ids[order], values[order])
            self.quote_requests += 1
            self.seconds += perf_counter() - started
        return self.quotes[key]

    def drift(self, state, date):
        old, new = int(DATAVENDOR.td(state.date)), int(DATAVENDOR.td(date))
        q = None if old == new else self.quote(old, new)
        return evolve(state, date, q)

    def benchmark(self, portfolio, date):
        key = (id(portfolio), int(date))
        if key not in self.benchmarks:
            # T50 receives a plain Portfolio produced by Universe, not a lazy
            # Benchmark that might load/write projected weights.
            use_date = portfolio.closest_avail_date(date)
            port = portfolio.ports.get(use_date)
            state = Position.empty(date) if port is None else self.drift(from_port(port), date)
            self.benchmarks[key] = state
        return self.benchmarks[key]

    def selection(self, builder, date):
        started = perf_counter()
        nested_before = self.seconds
        model = builder.alpha.get(date)
        if len(np.unique(model.secid)) != len(model.secid):
            raise UnsupportedT50('duplicate alpha security IDs')
        bench = self.benchmark(builder.benchmark, date)
        candidates = (np.intersect1d(model.secid, bench.ids) if bench.exists and len(bench.ids)
                      else np.unique(model.secid))
        if self.generic:
            # Keep screening semantics, including universe-relative ranks and
            # external alpha normalization, in the existing configuration layer.
            benchmark = builder.benchmark.get(date, closest=True)
            candidates = builder.creator.conf.get_candidate_pool(date, model, benchmark)
        if not len(candidates):
            # The reference cannot add an industry column to an empty frame
            # and raises in groupby. Do not silently invent an empty portfolio.
            raise UnsupportedT50('empty candidate pool is unsupported by the legacy creator')
        # Preserve the self-composite normalization, including constant/NaN alpha.
        sorted_model = builder.creator.conf.get_sorting_alpha(date, model).align(candidates)
        alpha = sorted_model.alpha
        if np.isinf(alpha).any():
            raise UnsupportedT50('infinite normalized alpha')
        # A missing prediction date uses the closest older Amodel; its
        # to_dataframe(indus=True) uses that model's date in the legacy path.
        industry_date = sorted_model.date
        if industry_date not in self.industries:
            industry = self.prepared.industry(industry_date) if self.prepared else None
            self.industries[industry_date] = (industry if industry is not None
                                              else DATAVENDOR.INFO.get_indus(industry_date))
        labels = self.industries[industry_date].reindex(candidates)['indus'].fillna('unknown')
        industry = pd.factorize(labels, sort=False)[0]
        result = prepare_selection(candidates, alpha, industry)
        self.seconds += perf_counter() - started - (self.seconds - nested_before)
        return result


def validate_group(group, generic=False):
    if group.category not in (('top', 'screen', 'revscreen', 'reinforce') if generic else ('top',)) or group.lags != [0]:
        raise UnsupportedT50('only unlagged top generation is supported')
    if group.acc_kwargs != dict(analytic=False, attribution=False, trade_engine='yale', daily=False):
        raise UnsupportedT50('requires yale accounting without daily mode or risk attribution')
    for builder in group.builders:
        conf = builder.creator.conf
        if not generic and (conf.sorter != ['self'] or conf.screener):
            raise UnsupportedT50('only self alpha without an additional screener is supported')
        if type(builder.benchmark) is not Portfolio or builder.benchmark.is_default:
            raise UnsupportedT50('requires an explicit, preloaded universe Portfolio')
        values = [conf.n_best, conf.turn_control, conf.buffer_zone, conf.no_zone, conf.indus_control]
        if (not np.isfinite(values).all() or conf.n_best <= 0 or not isinstance(conf.n_best, (int, np.integer))
                or not all(0 <= v <= 1 for v in values[1:])):
            raise UnsupportedT50('invalid long-only top selection parameters')


def generate_arrays(group, inputs, block_days):
    records = [{int(d): from_port(p) for d, p in b.portfolio.ports.items()} for b in group.builders]
    states = [r[max(r)] if r else None for r in records]
    for start in range(0, len(group.relevant_dates), block_days):
        inputs.clear()
        for date in group.relevant_dates[start:start + block_days]:
            for i, builder in enumerate(group.builders):
                if not builder.alpha.has(date):
                    continue
                old = states[i]
                old = Position.empty(date) if old is None else inputs.drift(old, date)
                day = inputs.selection(builder, date)
                state = select(day, old, builder.creator.conf, date)
                records[i][int(date)] = states[i] = state
    return records


def account_arrays(group, records, inputs, block_days):
    plans = []
    for builder, positions in zip(group.builders, records, strict=True):
        config = AccountConfig(builder.portfolio.name, builder.benchmark_name, group.start, group.end,
                               False, False, 'yale', False)
        resumed = (PortfolioAccount.load(builder.resume_path_account) if Const.Model.resume_fmp_account
                   else PortfolioAccount())
        if Const.Model.resume_fmp_account:
            resumed.filter_dates(end=int(builder.resumed_portfolio_end_date))
        dates = np.array(sorted(positions), dtype=np.int64)
        if len(dates):
            begin = max(int(dates.min()), group.start, resumed.max_model_date + 1)
            end = min(int(DATAVENDOR.td(int(dates.max()), 5)), group.end,
                      int(DATAVENDOR.td(CALENDAR.updated(), -1)))
            dates = np.intersect1d(DATAVENDOR.td_within(begin, end), dates)
        ends = (np.concatenate((dates[1:], [DATAVENDOR.td(end, 1)])) if len(dates) else dates.copy())
        starts = CALENDAR.offset(dates, 1, 'td') if len(dates) else dates.copy()
        old = Position.empty(dates[0]) if len(dates) else None
        plans.append(dict(builder=builder, positions=positions, config=config, resumed=resumed,
                          dates=dates, ends=ends, starts=starts, old=old,
                          rows=np.zeros((len(dates), 7)), indices={int(d): i for i, d in enumerate(dates)}))
    all_dates = np.unique(np.concatenate([p['dates'] for p in plans])) if plans else np.empty(0, dtype=int)
    for start in range(0, len(all_dates), block_days):
        inputs.clear()
        for date in all_dates[start:start + block_days]:
            for plan in plans:
                i = plan['indices'].get(int(date))
                if i is None:
                    continue
                old, new = plan['old'], plan['positions'][int(date)]
                end = int(plan['ends'][i])
                bench = inputs.benchmark(plan['builder'].benchmark, date)
                close = inputs.quote(new.date, end)
                row = account_step(old, new, bench, close, inputs.quote(bench.date, end),
                                   inputs.quote(old.date, price='open'), inputs.quote(new.date, price='open'),
                                   inputs.quote(old.date, price='vwap'), inputs.quote(new.date, price='vwap'),
                                   Const.Factor.ROUNDING.ret, Const.Factor.ROUNDING.turnover)
                if not np.isfinite(row).all():
                    raise UnsupportedT50('nonfinite account result')
                plan['rows'][i] = row
                plan['old'] = inputs.drift(new, end)
    return plans


def publish(group, records, plans):
    """Only after successful computation, materialize legacy objects and archives."""
    for builder, positions, plan in zip(group.builders, records, plans, strict=True):
        new_dates = set(positions) - set(builder.portfolio.ports)
        for date in sorted(new_dates):
            state = positions[date]
            port = Port(pd.DataFrame({'secid': state.ids, 'weight': state.weights}), date,
                        builder.portfolio.name, state.value)
            builder.portfolio.append(port)
            builder.creations.append(PortCreateResult(port, len(port) == builder.creator.conf.n_best))
            builder.port = port
        builder.portfolio.activate_accountant()
        if len(plan['dates']):
            frame = pd.DataFrame(plan['rows'], columns=['pf', 'bm', 'turn', 'overnight', 'pf_close', 'pf_open', 'pf_vwap'])
            frame['model_date'], frame['start'], frame['end'] = plan['dates'], plan['starts'], plan['ends']
            frame['pf'] -= frame['turn'] * plan['config'].trade_cost
            frame['excess'] = frame['pf'] - frame['bm']
            frame['analytic'], frame['attribution'] = None, None
            fresh = PortfolioAccount(frame[PortfolioAccount.columns_all], plan['config'])
            account = PortfolioAccount.Concat(plan['resumed'], fresh)
        else:
            account = plan['resumed']
        account.with_index(builder.port_index)
        builder.portfolio.accountant.account = account
    group.accounted = True
    group.save_portfolios()
    for builder in group.builders:
        builder.account.save(builder.resume_path_account, indent=builder.indent + 1, vb_level=builder.vb_level)


def run_t50(group, backend: T50Backend = 'legacy', *, block_days=BLOCK_DAYS,
            prepared_root=None, generic=False):
    """Return the normal total-account frame; diagnostics live on the group."""
    if backend not in ('legacy', 'array', 'auto'):
        raise ValueError(f'unknown t50_backend: {backend}')
    if block_days <= 0:
        raise ValueError('block_days must be positive')
    started = perf_counter()
    diagnostics: dict[str, Any] = dict(requested=backend, selected='legacy', fallback_reason=None, block_days=block_days)
    group.t50_diagnostics = diagnostics
    if backend != 'legacy':
        try:
            group.builders_setup()
            validate_group(group, generic)
            if generic or prepared_root is not None:
                for builder in group.builders:
                    conf = builder.creator.conf
                    default = type(conf)()
                    for key in ('n_best', 'turn_control', 'buffer_zone', 'no_zone',
                                'indus_control', 'screen_ratio', 'sorter', 'screener'):
                        if getattr(conf, key) != getattr(default, key):
                            raise UnsupportedT50(f'nondefault {key}: use legacy input and calculation path')
            from src.res.factor.training_analysis_snapshot import CURRENT
            prepared = CURRENT.get() if generic else None
            if prepared_root is not None:
                from .prepared_market import PreparedMarket
                try:
                    prepared = PreparedMarket(prepared_root)
                except (OSError, ValueError, KeyError) as exc:
                    raise UnsupportedT50(str(exc)) from exc
            group.builders_resume()
            if generic:
                for builder in group.builders:
                    builder.creator.conf.alpha_sorter.get_alpha_models(group.relevant_dates)
                    builder.creator.conf.alpha_screener.get_alpha_models(group.relevant_dates)
            inputs = Inputs(prepared, generic)
            setup_seconds = perf_counter() - started
            at = perf_counter()
            records = generate_arrays(group, inputs, block_days)
            generation_elapsed = perf_counter() - at
            generation_preparation = inputs.seconds
            at = perf_counter()
            plans = account_arrays(group, records, inputs, block_days)
            accounting_elapsed = perf_counter() - at
            accounting_preparation = inputs.seconds - generation_preparation
        except UnsupportedT50 as exc:
            if backend == 'array':
                raise
            diagnostics['fallback_reason'] = str(exc)
            group.logger.warning(f'T50 array fallback: {exc}')
        else:
            at = perf_counter()
            publish(group, records, plans)
            result = group.total_account()
            diagnostics.update(selected='array', setup_seconds=setup_seconds, prepared_hits=prepared.hits if prepared else 0,
                               prepared_misses=prepared.misses if prepared else 0, preparation_seconds=inputs.seconds,
                               generation_seconds=generation_elapsed - generation_preparation,
                               accounting_seconds=accounting_elapsed - accounting_preparation,
                               output_seconds=perf_counter() - at, quote_requests=inputs.quote_requests,
                               total_seconds=perf_counter() - started)
            group.logger.stdout(f'T50 backend: {diagnostics}')
            return result
    at = perf_counter()
    group.build()
    diagnostics['build_seconds'] = perf_counter()-at
    at = perf_counter()
    result = group.total_account()
    diagnostics['accounting_seconds'] = perf_counter()-at
    diagnostics['total_seconds'] = perf_counter() - started
    group.logger.stdout(f'T50 backend: {diagnostics}')
    return result


def run_generator(group, backend='auto', *, prepared_root=None, block_days=128):
    result = run_t50(group, backend, prepared_root=prepared_root,
                     block_days=block_days, generic=True)
    group.generator_diagnostics = group.t50_diagnostics
    return result
