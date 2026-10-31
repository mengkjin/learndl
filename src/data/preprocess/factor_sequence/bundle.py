"""Immutable content-addressed bundles, audit evidence and strict prefix replay."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .schema import BuildConfig, KEYS, EVENTS, STOCK_CHANNELS
from .numeric import clean, stock_block, performance_block, market_block


def digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def build(root, raw, returns, dates, secids, market, events, eligible=None,
          cfg=None, source=None, parent=None, allow_revision=False, chunk_size=None):
    cfg = cfg or BuildConfig()
    raw, returns = clean(raw), clean(returns)
    dates, secids = np.asarray(dates, dtype='int64'), np.asarray(secids, dtype='int64')
    if raw.shape != (len(secids), len(dates), len(cfg.factors)) or returns.shape != raw.shape[:2]:
        raise ValueError('Raw input shape mismatch')
    if len(dates) < 2 or np.any(np.diff(dates) <= 0) or len(set(secids)) != len(secids):
        raise ValueError('Dates must be ordered/unique; securities unique')
    if not np.array_equal(market.index.to_numpy(), dates) or not set(EVENTS) <= set(market.columns):
        raise ValueError('Market calendar or event columns mismatch')
    if not {'event', 'event_date', 'start', 'end'} <= set(events.columns):
        raise ValueError('Event schema missing')
    if not events.event.isin(EVENTS).all():
        raise ValueError('Unknown event type')
    if eligible is None:
        eligible = np.ones(raw.shape[:2], dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != raw.shape[:2]:
        raise ValueError('Eligibility shape mismatch')
    for k, name in enumerate(cfg.factors):
        if not np.isfinite(raw[:, :, k][eligible]).any():
            raise ValueError(f'Required factor entirely missing: {name}')
    snapshots = dict(raw=raw, returns=returns, dates=dates, secids=secids, eligible=eligible,
                     market=clean(market.to_numpy()), market_names=np.array(market.columns, dtype=str))
    # Events beyond the cutoff do not belong to this snapshot.
    events = events.loc[events.event_date <= dates[-1], ['event', 'event_date', 'start', 'end']].sort_values(['event_date','event']).reset_index(drop=True)
    sources = {k: digest(v) for k, v in snapshots.items()}
    sources['events'] = hashlib.sha256(events.to_json(orient='records').encode()).hexdigest()
    transform_hashes = {name:file_digest(Path(__file__).with_name(name))
                        for name in ('schema.py','numeric.py','bundle.py','sources.py')}
    identity = {'config': asdict(cfg), 'inputs': sources, 'source': source or {},
                'transform_hashes': transform_hashes}
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    destination = Path(root)/version
    if destination.exists():
        load(destination, expected_config=cfg)
        return destination
    revision = False
    if parent:
        previous = load(parent)
        old = np.load(Path(parent)/'inputs.npz', allow_pickle=False)
        t = len(old['dates'])
        compatible = (previous['config_hash'] == cfg.fingerprint and
                      previous.get('transform_hashes') == transform_hashes)
        for name in snapshots:
            a = snapshots[name]
            if name in ('raw','returns','eligible'):
                a = a[:, :t]
            elif name in ('dates','market'):
                a = a[:t]
            compatible &= digest(a) == digest(old[name])
        old_events = pd.read_json(Path(parent)/'events.json').reindex(columns=['event','event_date','start','end'])
        prefix = events.loc[events.event_date <= old['dates'][-1]].reset_index(drop=True)
        compatible &= prefix.to_json(orient='records') == old_events.to_json(orient='records')
        revision = not compatible
        if revision and not allow_revision:
            raise ValueError('Historical source/config revision; explicitly allow a new version')
    # Prefix replay is the reference incremental algorithm: never truncate state by
    # guessing a finite lookback for last-observation or last-event age.
    x, xnames, signed = stock_block(raw, eligible, cfg)
    if chunk_size is None:
        p, pnames = performance_block(signed, returns, dates, events, cfg)
        m, mnames = market_block(market, cfg)
    else:
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        ps, ms = [], []
        for start in range(0, len(dates), chunk_size):
            end = min(start+chunk_size, len(dates))
            part_events = events.loc[events.event_date <= dates[end-1]]
            pp, pnames = performance_block(signed[:, :end], returns[:, :end], dates[:end], part_events, cfg)
            mm, mnames = market_block(market.iloc[:end], cfg)
            ps.append(pp[:, start:end]); ms.append(mm[:, start:end])
        p, m = np.concatenate(ps, axis=1), np.concatenate(ms, axis=1)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(destination/'inputs.npz', **snapshots)
        events.to_json(destination/'events.json', orient='records', indent=2)
        blocks = {}
        for key, values, names, ids in (
            ('factorseq', x, xnames, secids), ('factorperf', p, pnames, [0]), ('factormarket', m, mnames, [0])):
            np.savez_compressed(destination/f'{key}.npz', values=values, dates=dates,
                                secids=np.asarray(ids), features=np.array(names))
            blocks[key] = dict(shape=list(values.shape), features=names, shared=key != 'factorseq',
                               sha256=file_digest(destination/f'{key}.npz'))
        manifest = dict(version=version, schema_version=cfg.version, config=asdict(cfg),
                        config_hash=cfg.fingerprint, source_fingerprints=sources, source=source or {},
                        transform_hashes=transform_hashes,
                        dates=[int(dates[0]),int(dates[-1])], blocks=blocks,
                        parent=str(parent) if parent else None, historical_revision=revision,
                        snapshot_sha256=file_digest(destination/'inputs.npz'),
                        events_sha256=file_digest(destination/'events.json'),
                        limitations=['Source update time unknown', 'Daily factor snapshots may contain upstream imputation or revisions'])
        write_json(destination/'manifest.json', manifest)  # completion marker, last
    except BaseException:
        write_json(destination/'incomplete.json', {'status':'failed'})
        raise
    return destination


def load(path, expected_config=None, verify=True):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    cfg = BuildConfig(**(manifest['config'] | {'factors':tuple(manifest['config']['factors'])}))
    if cfg.fingerprint != manifest['config_hash'] or (expected_config and expected_config.fingerprint != cfg.fingerprint):
        raise ValueError('Config fingerprint mismatch')
    if verify:
        for key in KEYS:
            if file_digest(path/f'{key}.npz') != manifest['blocks'][key]['sha256']:
                raise ValueError(f'Corrupt data block: {key}')
        for filename, field in (('inputs.npz','snapshot_sha256'),('events.json','events_sha256')):
            if file_digest(path/filename) != manifest[field]:
                raise ValueError(f'Corrupt source snapshot: {filename}')
    return manifest


def audit(path):
    manifest = load(path)
    raw = np.load(Path(path)/'inputs.npz', allow_pickle=False)
    cfg = BuildConfig(**(manifest['config'] | {'factors':tuple(manifest['config']['factors'])}))
    x = np.load(Path(path)/'factorseq.npz', allow_pickle=False)['values'][:, :, 0]
    k = len(cfg.factors)
    valid = x[..., k:2*k] > 0
    observed = x[..., 2*k:3*k] > 0
    denominator=raw['eligible'].sum(axis=0)
    coverage = observed.sum(axis=0)/np.maximum(denominator[:,None],1)
    coverage[denominator==0]=np.nan
    # Long form coverage is intentionally a file, not a giant JSON list.
    pd.DataFrame(coverage, index=raw['dates'], columns=cfg.factors).rename_axis('date').to_csv(Path(path)/'coverage.csv')
    records = {}
    for j, f in enumerate(cfg.factors):
        z = x[..., j][valid[..., j]]
        r = raw['raw'][..., j][observed[..., j]]
        tails=0
        for t in range(len(raw['dates'])):
            daily=raw['raw'][:,t,j][observed[:,t,j]]
            if len(daily):
                lo,hi=np.quantile(daily,[cfg.lower_quantile,cfg.upper_quantile])
                tails+=int(((daily<lo)|(daily>hi)).sum())
        records[f] = dict(observed_fraction=float(observed[..., j].mean()),
                          eligible_missing_fraction=float(1-observed[...,j].sum()/max(raw['eligible'].sum(),1)),
                          extreme_fraction=float(tails/max(len(r),1)),
                          normalized_fraction=float(valid[..., j].mean()),
                          degenerate_count=int(x[..., 9*k+j].sum()),
                          raw_quantiles=np.quantile(r,[0,.01,.5,.99,1]).tolist() if len(r) else [],
                          normalized_quantiles=np.quantile(z,[0,.01,.5,.99,1]).tolist() if len(z) else [])
    blocks = {}
    for key in KEYS:
        b = np.load(Path(path)/f'{key}.npz', allow_pickle=False)
        if not np.isfinite(b['values']).all():
            raise ValueError(f'Nonfinite packed block: {key}')
        names = list(b['features'])
        mask = [i for i,n in enumerate(names) if str(n).endswith('__valid')]
        if key != 'factorseq' and not b['values'][...,mask].any():
            raise ValueError(f'Entire public block unavailable: {key}')
        blocks[key] = dict(shape=list(b['values'].shape), bytes=int(b['values'].nbytes),
                           maturity={str(names[i]):float(b['values'][...,i].mean()) for i in mask})
    if not valid.any():
        raise ValueError('No usable normalized factor observations')
    # Independently check fixed dates with pandas quantiles/ranks, not the production helper.
    checks, event_checks = [], []
    p = np.load(Path(path)/'factorperf.npz', allow_pickle=False)
    pnames = list(p['features'])
    def verify_metrics(factor, v, r, t, prefix=''):
        frame = pd.DataFrame({'v':v,'r':r}).replace([np.inf,-np.inf],np.nan).dropna()
        if len(frame) < cfg.min_performance or frame.v.nunique() < 2:
            return None
        group = np.floor((frame.v.rank(method='average')-.5)*10/len(frame)).clip(upper=9)
        top, bottom = frame.r[group == 9], frame.r[group == 0]
        expected = {'ic':frame.v.rank().corr(frame.r.rank()) if frame.r.nunique()>1 else np.nan,
                    'top':top.mean()-frame.r.mean() if len(top) and len(bottom) else np.nan,
                    'bottom':bottom.mean()-frame.r.mean() if len(top) and len(bottom) else np.nan,
                    'spread':top.mean()-bottom.mean() if len(top) and len(bottom) else np.nan}
        for metric,value in expected.items():
            name=f'{prefix}{factor}__{metric}'
            if bool(p['values'][0,t,0,pnames.index(name+'__valid')]) != bool(np.isfinite(value)):
                raise ValueError(f'Independent audit validity mismatch: {name}')
            if np.isfinite(value) and not np.isclose(p['values'][0,t,0,pnames.index(name)],
                                                    value*(1 if metric=='ic' else 100),atol=1e-6):
                raise ValueError(f'Independent audit metric mismatch: {name}')
        return {'date':int(raw['dates'][t]),'factor':factor,'metrics_checked':list(expected)}
    for t in sorted(set([1,len(raw['dates'])//2,len(raw['dates'])-1])):
        for j, factor in enumerate(cfg.factors):
            v = pd.Series(raw['raw'][:, t-1, j]) * (-1 if factor.startswith('-') else 1)
            r = pd.Series(raw['returns'][:, t])
            ok = v.notna() & r.notna() & raw['eligible'][:,t-1]
            if ok.sum() < cfg.min_performance or v[ok].nunique() < 2 or r[ok].nunique() < 2:
                continue
            checks.append(verify_metrics(factor,v[ok].to_numpy(),r[ok].to_numpy(),t))
    events = pd.read_json(Path(path)/'events.json').reindex(columns=['event','event_date','start','end'])
    indices = {int(d):i for i,d in enumerate(raw['dates'])}
    for row in events.groupby('event').head(3).itertuples():
        if any(d not in indices for d in (row.start,row.end,row.event_date)) or indices[row.start]==0:
            continue
        a,b,t = indices[row.start],indices[row.end],indices[row.event_date]
        r = pd.DataFrame(raw['returns'][:,a:b+1]).add(1).prod(axis=1,skipna=False)-1
        for j,factor in enumerate(cfg.factors):
            v = raw['raw'][:,a-1,j]*(-1 if factor.startswith('-') else 1)
            v = np.where(raw['eligible'][:,a-1],v,np.nan)
            check = verify_metrics(factor,v,r.to_numpy(),t,row.event+'__')
            if check:
                event_checks.append(dict(check,event=row.event,start=int(row.start),end=int(row.end)))
    report = dict(status='passed', version=manifest['version'], factors=records, blocks=blocks,
                  source_update_time_unknown_fraction=1., daily_checks=checks,event_checks=event_checks,
                  event_counts=events.event.value_counts().to_dict(),event_records=events.to_dict('records'))
    write_json(Path(path)/'audit.json',report)
    return report
