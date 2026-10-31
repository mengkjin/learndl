"""Causal NumPy/pandas transforms. Missing values stay missing until packing."""
import numpy as np
import pandas as pd
from .schema import BuildConfig, EVENTS, STOCK_CHANNELS


def clean(x):
    out = np.asarray(x, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def age_scale(age, cfg):
    return np.log1p(np.minimum(age, cfg.age_cap)) / np.log1p(cfg.age_cap)


def pack(columns):
    """Values followed by explicit valid channels, with zero only as an input placeholder."""
    names, arrays = [], []
    for name, array in columns.items():
        array = clean(array)
        names.extend((name, name + '__valid'))
        arrays.extend((np.nan_to_num(array), np.isfinite(array).astype(float)))
    return np.stack(arrays, axis=-1).astype('float32'), names


def cross_section(raw, eligible, cfg):
    x = clean(raw)
    x[~eligible] = np.nan
    out = np.full_like(x, np.nan)
    degenerate = np.zeros_like(x)
    for t in range(x.shape[1]):
        for k in range(x.shape[2]):
            ok = np.isfinite(x[:, t, k])
            if ok.sum() < cfg.min_cross_section:
                continue
            v = x[ok, t, k]
            lo, hi = np.quantile(v, [cfg.lower_quantile, cfg.upper_quantile])
            v = np.clip(v, lo, hi)
            std = v.std(ddof=0)
            if std <= 1e-12:
                out[ok, t, k] = 0
                degenerate[ok, t, k] = 1
            else:
                out[ok, t, k] = (v-v.mean()) / std
    return out, degenerate


def stock_block(raw, eligible, cfg):
    x = clean(raw)
    directions = np.array([-1 if f.startswith('-') else 1 for f in cfg.factors])
    x *= directions
    x[~eligible] = np.nan
    z, degenerate = cross_section(x, eligible, cfg)
    observed = np.isfinite(x)
    last = np.full((len(x), x.shape[2]), -1, dtype=int)
    previous = np.full_like(last, np.nan, dtype=float)
    gap, never, delta = np.zeros_like(x), np.zeros_like(x), np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        ok = observed[:, t]
        # Gap is time since previous valid daily observation, including at reappearance.
        seen = last >= 0
        gap[:, t] = np.where(seen, age_scale(t-last, cfg), 0)
        never[:, t] = ~seen & ~ok
        delta[:, t] = np.where(ok & seen, x[:, t]-previous, np.nan)
        previous = np.where(ok, x[:, t], previous)
        last = np.where(ok, t, last)
    dz, _ = cross_section(delta, eligible, cfg)
    channels = (np.nan_to_num(z), np.isfinite(z), observed, gap, never,
                np.nan_to_num(dz), np.isfinite(dz), np.zeros_like(x),
                np.zeros_like(x), degenerate)
    # Channel-major layout permits deterministic slices without a changing factor count.
    values = np.concatenate(channels, axis=-1).astype('float32')
    names = [f'{channel}__{factor}' for channel in STOCK_CHANNELS for factor in cfg.factors]
    return values[:, :, None, :], names, x


def group_metrics(factor, returns, minimum):
    ok = np.isfinite(factor) & np.isfinite(returns)
    if ok.sum() < minimum:
        return np.full(4, np.nan)
    f, r = factor[ok], returns[ok]
    if np.ptp(f) <= 1e-12:
        return np.full(4, np.nan)
    rank = pd.Series(f).rank(method='average').to_numpy()
    # Ties are never split by stock identifier; an absent extreme group is invalid.
    group = np.minimum(9, np.floor((rank-.5) * 10 / len(rank))).astype(int)
    top, bot = r[group == 9], r[group == 0]
    ic = pd.Series(f).corr(pd.Series(r), method='spearman') if np.ptp(r) > 1e-12 else np.nan
    if not len(top) or not len(bot):
        return np.array([ic, np.nan, np.nan, np.nan])
    return np.array([ic, top.mean()-r.mean(), bot.mean()-r.mean(), top.mean()-bot.mean()])


def performance_block(signed, returns, dates, events, cfg):
    n, t_count, k_count = signed.shape
    daily = np.full((t_count, k_count, 4), np.nan)
    for t in range(1, t_count):
        for k in range(k_count):
            daily[t, k] = group_metrics(signed[:, t-1, k], returns[:, t], cfg.min_performance)
    columns = {}
    for k, factor in enumerate(cfg.factors):
        for j, metric in enumerate(('ic', 'top', 'bottom', 'spread')):
            series = pd.Series(daily[:, k, j])
            scale = 1 if j == 0 else 100
            columns[f'{factor}__{metric}'] = series.to_numpy()*scale
            for w in (20, 60):
                roll = series.rolling(w, min_periods=w//2)
                mean, std = roll.mean(), roll.std(ddof=0)
                columns[f'{factor}__{metric}_mean{w}'] = mean.to_numpy()*scale
                columns[f'{factor}__{metric}_std{w}'] = std.to_numpy()*scale
                if j == 0:
                    columns[f'{factor}__icir{w}'] = (mean / std.where(std > 1e-12)).clip(-5, 5).to_numpy()
            if j > 0:
                rms = series.shift(1).pow(2).rolling(cfg.history_window, min_periods=cfg.history_min).mean().pow(.5)
                columns[f'{factor}__{metric}_risk_scaled'] = (series/rms.clip(lower=1e-4)).clip(-5, 5).to_numpy()
    date_index = {int(d): i for i, d in enumerate(dates)}
    if events.duplicated(['event', 'event_date']).any():
        raise ValueError('Duplicate event type/confirmation date')
    for event in EVENTS:
        records = events.loc[events.event == event].sort_values('event_date')
        evalue = np.full((t_count, k_count, 4), np.nan)
        age, duration = np.full(t_count, np.nan), np.full(t_count, np.nan)
        for row in records.itertuples():
            if row.start > row.end or row.end > row.event_date:
                raise ValueError('Event interval must end no later than confirmation')
            if row.event_date not in date_index:
                continue
            confirm = date_index[row.event_date]
            evalue[confirm:] = np.nan
            age[confirm:] = age_scale(np.arange(t_count-confirm), cfg)
            duration[confirm:] = np.nan
            if row.start not in date_index or row.end not in date_index:
                continue
            start, end = date_index[row.start], date_index[row.end]
            # Invalidate previous event even when a new interval cannot be evaluated.
            evalue[confirm:] = np.nan
            age[confirm:] = age_scale(np.arange(t_count-confirm), cfg)
            duration[confirm:] = age_scale(end-start+1, cfg)
            if start == 0:
                continue
            window = returns[:, start:end+1]
            complete = np.isfinite(window).all(axis=1)
            r = np.where(complete, np.prod(1+window, axis=1)-1, np.nan)
            metrics = np.stack([group_metrics(signed[:, start-1, k], r, cfg.min_performance) for k in range(k_count)])
            evalue[confirm:] = metrics
        columns[f'{event}__age'] = age
        columns[f'{event}__duration'] = duration
        for k, factor in enumerate(cfg.factors):
            for j, metric in enumerate(('ic', 'top', 'bottom', 'spread')):
                columns[f'{event}__{factor}__{metric}'] = evalue[:, k, j] * (1 if j == 0 else 100)
    values, names = pack(columns)
    return values[None, :, None, :], names


def market_block(market, cfg):
    columns = {}
    for name in market.columns:
        s = pd.Series(clean(market[name].to_numpy()))
        if name in EVENTS:
            if not s.dropna().isin([0, 1]).all():
                raise ValueError(f'Nonbinary event label: {name}')
            columns[name] = s.to_numpy()
            last = None
            age = np.full(len(s), np.nan)
            for t, value in enumerate(s):
                if value == 1:
                    last = t
                if last is not None:
                    age[t] = age_scale(t-last, cfg)
            columns[name+'__age'] = age
        elif name.endswith('_ratio'):
            if not s.dropna().between(0, 1).all():
                raise ValueError(f'Invalid proportion {name}')
            columns[name] = s.to_numpy()
        elif name == 'market_return':
            columns[name] = s.to_numpy()*100
        else:
            history = s.shift(1).rolling(cfg.history_window, min_periods=cfg.history_min)
            std = history.std(ddof=0)
            columns[name] = ((s-history.mean())/std.where(std > 1e-12)).clip(-5, 5).to_numpy()
    values, names = pack(columns)
    return values[None, :, None, :], names
