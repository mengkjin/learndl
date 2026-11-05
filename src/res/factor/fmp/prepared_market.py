"""Default-generator market inputs; predictions and portfolios are never persisted."""
from __future__ import annotations

__all__ = [
    'default_contract',
    'fingerprints',
    'prepare_market',
    'PreparedMarket',
]
import hashlib
import json
import os
import uuid
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.proj import DB, Const


def default_contract():
    return {'creator': Const.Factor.FMP.creator, 'price_semantics': 'get_quote_ret-v1',
            'rounding': [Const.Factor.ROUNDING.ret, Const.Factor.ROUNDING.turnover],
            'version': 1}


def fingerprints(dates):
    from src.res.factor.util.classes.prepared_universe import PreparedUniverse
    paths = {DB.path('information_ts', 'industry')}
    for date in dates:
        for d in (int(date), int(DATAVENDOR.td(date, 1))):
            paths.update((DB.path('trade_ts', 'adjprice', d), DB.path('trade_ts', 'day', d)))
    return {str(path): PreparedUniverse.fingerprint(path) for path in sorted(paths)}


def prepare_market(root, dates):
    """Compute standard next-session returns and industry panels, then publish."""
    root = Path(root)
    dates = sorted(set(map(int, dates)))
    if not dates:
        raise ValueError('empty preparation interval')
    started = perf_counter()
    before = fingerprints(dates)
    DATAVENDOR.TRADE.clear_all()
    DATAVENDOR.INFO.clear_all()
    packs = {}
    for date in dates:
        arrays = packs.setdefault(date//100, {})
        ind = DATAVENDOR.INFO.get_indus(date)
        arrays[f'{date}_industry_ids'] = ind.index.to_numpy(dtype=np.int64)
        arrays[f'{date}_industry'] = ind.indus.fillna('unknown').to_numpy(dtype=str)
        for price in ('close', 'open', 'vwap'):
            q = DATAVENDOR.get_quote_ret(date, int(DATAVENDOR.td(date, 1)), 'close', price)
            if q is None:
                raise ValueError(f'missing quote {date}/{price}')
            ids = q['secid'].to_numpy(dtype=np.int64) if 'secid' in q else q.index.to_numpy(dtype=np.int64)
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f'duplicate quote IDs: {date}')
            order = np.argsort(ids)
            arrays[f'{date}_{price}_ids'] = ids[order]
            arrays[f'{date}_{price}'] = q.ret.to_numpy(dtype=float)[order]
    compute_seconds = perf_counter()-started
    started = perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    folder = root/generation
    folder.mkdir()
    hashes = {}
    for month, arrays in packs.items():
        path = folder/f'{month}.npz'
        np.savez(path, **arrays)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = default_contract()
    # JSON serialization detaches mutable configuration objects.
    meta = dict(dates=dates, next_dates={str(d):int(DATAVENDOR.td(d,1)) for d in dates},
                contract=contract, sources=before, hashes=hashes)
    (folder/'manifest.json').write_text(json.dumps(meta))
    if fingerprints(dates) != before:
        raise ValueError('market sources changed during preparation')
    pointer = root/f'.{generation}.tmp'
    pointer.write_text(generation)
    os.replace(pointer, root/'current')
    return dict(compute_seconds=compute_seconds, save_seconds=perf_counter()-started,
                dates=len(dates), bytes=sum(p.stat().st_size for p in folder.iterdir()))


class PreparedMarket:
    def __init__(self, root):
        self.folder = Path(root)/(Path(root)/'current').read_text().strip()
        self.meta = json.loads((self.folder/'manifest.json').read_text())
        if self.meta['contract'] != json.loads(json.dumps(default_contract())):
            raise ValueError('prepared market default configuration changed')
        if fingerprints(self.meta['dates']) != self.meta['sources']:
            raise ValueError('prepared market source revision')
        self.month = None
        self.arrays = {}

    def _month(self, date):
        month = int(date)//100
        if month != self.month:
            path = self.folder/f'{month}.npz'
            if not path.exists():
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.meta['hashes'][path.name]:
                raise ValueError('prepared market partition corrupted')
            with np.load(path, allow_pickle=False) as data:
                self.arrays = {key:data[key] for key in data.files}
            self.month = month
        return True

    def quote(self, date, end, price):
        key = f'{date}_{price}'
        if self.meta['next_dates'].get(str(date)) != end or not self._month(date) or key not in self.arrays:
            return None
        return self.arrays[key+'_ids'], self.arrays[key]

    def industry(self, date):
        key = f'{date}_industry'
        if not self._month(date) or key not in self.arrays:
            return None
        return pd.DataFrame({'indus':self.arrays[key]}, index=self.arrays[key+'_ids'])
