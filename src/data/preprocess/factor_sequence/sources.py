"""Read-only project source adapter. Missing required sources are errors, not synthetic fallbacks."""
import numpy as np
import pandas as pd
from .schema import BuildConfig, EVENTS


def read_daily_factor(db, source, key, dates, fingerprints=None):
    """Normalize each file before merging; refuse to infer identifiers from row order."""
    pieces = []
    for date in dates:
        path = db.path(source,key,int(date))
        if not path.exists():
            continue
        frame = db.load(source,key,int(date),key_column='date')
        if 'secid' not in frame or frame.secid.isna().any():
            raise ValueError(f'Missing stock identifiers in {path}; row order is not a stock mapping')
        if fingerprints is not None:
            from .bundle import file_digest
            fingerprints[str(path)] = file_digest(path)
        pieces.append(frame)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()


def preflight(cfg=None):
    from src.res.factor.calculator import StockFactorHierarchy
    from src.proj import DB
    cfg = cfg or BuildConfig()
    missing, factors = [], {}
    for signed in cfg.factors:
        name = signed.lstrip('+-')
        calc = StockFactorHierarchy.get_factor(name)
        dates = calc.stored_dates()
        factors[signed] = {'count':len(dates), 'start':int(dates.min) if len(dates) else None,
                           'end':int(dates.max) if len(dates) else None}
        if not len(dates):
            missing.append('factor:'+name)
    for event in EVENTS:
        frame = StockFactorHierarchy.get_factor(event).Load(None)
        if frame.empty:
            missing.append('event:'+event)
    if DB.load('index_daily_ts','000985.CSI').empty:
        missing.append('index:000985.CSI')
    from src.data.util import DataBlock
    y = DataBlock.load_preprocess('y',frame='fit')
    if not y.initiated or 'std_lag1_10' not in y.feature:
        missing.append('label:std_lag1_10')
    return {'status':'failed' if missing else 'passed','missing':missing,'factors':factors}


def read_project(start, end, cfg=None):
    from src.proj import CALENDAR, DB
    from src.data import DATAVENDOR
    from src.data.util import DataBlock
    from src.res.factor.calculator import StockFactorHierarchy
    from src.res.factor.calculator.factor_calc import AffiliateFactorCalculator
    cfg = cfg or BuildConfig()
    dates = np.asarray(CALENDAR.range(start,end,'td'),dtype=int)
    frames, fingerprints = [], {}
    for signed in cfg.factors:
        name = signed.lstrip('+-')
        calc = StockFactorHierarchy.get_factor(name)
        try:
            if calc.db_src == 'stock_factor' and not issubclass(calc, AffiliateFactorCalculator):
                # Normalize identifiers BEFORE concatenation: old feather files
                # use a named secid index, newer files use a secid column.
                df = read_daily_factor(DB,calc.db_src,calc.db_key,dates,fingerprints)
            else:
                df = calc.Loads(dates, normalize=False)
                from .bundle import file_digest
                for db in calc.full_dbs():
                    for date in dates:
                        path = DB.path(db['src'],db['key'],int(date))
                        if path.exists() and str(path) not in fingerprints:
                            fingerprints[str(path)] = file_digest(path)
        except Exception as exc:
            raise ValueError(f'Required source {name} failed for {start}..{end}: {exc}') from exc
        if df.empty or name not in df or df[['secid','date']].isna().any().any():
            raise ValueError(f'Empty or invalid identifiers in source {name}')
        if df.duplicated(['secid','date']).any():
            raise ValueError(f'Duplicate stock/date in source {name}; refusing implicit aggregation')
        frames.append(df.set_index(['secid','date'])[[name]])
    factor = DataBlock.from_pandas(pd.concat(frames,axis=1))
    if not factor.initiated:
        raise ValueError('Required factor files unavailable')
    factor = factor.align_date(dates).align_feature([f.lstrip('+-') for f in cfg.factors])
    secids = factor.secid
    raw = factor.values[:,:,0].cpu().numpy()
    price = DATAVENDOR.TRADE.get_specific_data(
        int(CALENDAR.td(start,-1)),end,'adj','close',prev=False,pivot=True)
    price = price.reindex(columns=secids).sort_index()
    returns = price.pct_change(fill_method=None).reindex(dates).to_numpy().T
    if not np.isfinite(returns).any():
        raise ValueError('Adjusted close returns unavailable')
    desc = DATAVENDOR.INFO.get_list_dt(offset=91,keep_columns=['list_dt','delist_dt']).reindex(secids)
    eligible = ((dates[None,:] >= desc.list_dt.to_numpy()[:,None]) &
                (dates[None,:] < desc.delist_dt.to_numpy()[:,None]))
    market = pd.DataFrame(index=dates)
    records = []
    for event in EVENTS:
        frame = StockFactorHierarchy.get_factor(event).Load(None)
        if frame.empty or frame.date.duplicated().any():
            raise ValueError(f'Empty or duplicate event history: {event}')
        frame = frame.set_index('date').sort_index()
        market[event] = frame[event].reindex(dates).astype(float)
        for date,row in frame.loc[(frame.index <= end) & (frame[event] == 1)].iterrows():
            event_start = int(row.rebound_start) if event == 'selloff_rebound' else int(date)
            records.append(dict(event=event,event_date=int(date),start=event_start,end=int(date)))
        if event == 'selloff_rebound':
            for name in ('selloff','rebound','atr'):
                market[name] = frame[name].reindex(dates)
        if event == 'high_level_switch':
            # Denominator uses the same date's available industry quote universe.
            codes = DB.load('information_ts','zx_industry')['l1_code'].unique()
            counts = pd.Series(0.,index=dates)
            for code in codes:
                quote = DB.load('index_daily_ts',code)
                if not quote.empty:
                    counts += pd.Series(quote.close.notna().to_numpy().astype(float),
                                        index=quote.trade_date.astype(int)).reindex(dates).fillna(0)
            for name in ('peaked','peaking'):
                market[name+'_ratio'] = frame[name].reindex(dates)/counts.where(counts > 0)
    quote = DB.load('index_daily_ts','000985.CSI').sort_values('trade_date').set_index('trade_date')
    market['market_return'] = (quote.pct_chg/100).reindex(dates)
    # Relative range avoids measuring an arbitrary index-point level.
    market['bandwidth'] = ((quote.high.rolling(5).max()-quote.low.rolling(5).min())/quote.close).reindex(dates)
    events = pd.DataFrame(records,columns=['event','event_date','start','end'])
    return dict(raw=raw,returns=returns,dates=dates,secids=secids,market=market,events=events,
                eligible=eligible,source={'kind':'project_daily_snapshots','start':start,'end':end,
                                          'listing_offset':91,'revision_history':'not_available',
                                          'factor_file_sha256':fingerprints})
