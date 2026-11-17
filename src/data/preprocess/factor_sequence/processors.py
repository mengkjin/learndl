"""Explicit immutable-bundle loading through the existing processor registry."""
from __future__ import annotations

__all__ = [
    'load_block',
    'PrePro_factorseq',
    'PrePro_factorperf',
    'PrePro_factormarket',
]
from pathlib import Path
import numpy as np
import torch
from ..core import MicellaneousPreProcessor
from .schema import KEYS, SHARED_KEYS, STOCK_CHANNELS
from .bundle import load as load_manifest


def load_block(dataset, key, dates=None, secid=None, view="full"):
    from src.data.util import DataBlock
    if not dataset:
        raise ValueError('input.factor_sequence.dataset must identify an immutable bundle')
    manifest = load_manifest(dataset)
    repaired = manifest.get('format') == 'factor-sequence-repair-2'
    if key not in KEYS:
        raise KeyError(key)
    if repaired:
        b = dict(values=np.load(Path(dataset)/f'{key}.npy', mmap_mode='r'),
                 dates=np.load(Path(dataset)/'dates.npy'),
                 secids=np.load(Path(dataset)/'secids.npy') if key == 'factorseq' else np.array([0]),
                 features=np.array(manifest['blocks'][key]['features']))
    else:
        b = np.load(Path(dataset)/f'{key}.npz',allow_pickle=False)
    if list(b['features']) != manifest['blocks'][key]['features']:
        raise ValueError('Feature order does not match manifest')
    names=list(b['features'])
    selected=names
    if view == 'compact' or (repaired and view == 'slim'):
        factors=manifest['config']['factors']
        if key == 'factorseq':
            selected=[f'{c}__{f}' for c in ('value','valid','gap','never') for f in factors]
        elif key == 'factorperf' and not repaired:
            selected=[f'{f}__{m}{suffix}' for f in factors for m in ('ic','spread','ic_mean20','ic_mean60','spread_mean20','spread_mean60') for suffix in ('','__valid')]
    elif view != 'full':
        raise ValueError('Unknown factor input view')
    indices=[names.index(n) for n in selected]  # missing features fail, never silently fill
    values=b['values']
    if repaired:
        day_indices = np.arange(len(b['dates'])) if dates is None else np.flatnonzero(np.isin(b['dates'], dates))
        stock_indices = np.arange(len(b['secids']))
        if key not in SHARED_KEYS and secid is not None:
            stock_indices = np.flatnonzero(np.isin(b['secids'], secid))
        # Select all axes together: do not materialize 332 channels before
        # reducing to 132, or copy the full historical memory map first.
        values = values[np.ix_(stock_indices, day_indices, np.arange(values.shape[2]), np.asarray(indices, dtype=np.intp))]
        b['dates'], b['secids'] = b['dates'][day_indices], b['secids'][stock_indices]
    elif selected != names:
        values=values[...,indices].copy()
    block = DataBlock(values=torch.from_numpy(values),date=b['dates'],
                      secid=b['secids'],feature=np.asarray(selected))
    if not isinstance(b, dict):
        b.close()
    if dates is not None:
        block = block.align_date(np.asarray(dates))
    if key not in SHARED_KEYS and secid is not None:
        block = block.align_secid(secid)
    holes = ~torch.isfinite(block.values).any(dim=-1)
    block.values.nan_to_num_(nan=0.,posinf=0.,neginf=0.)
    if key == 'factorseq':
        k = len(manifest['config']['factors'])
        never = 3 if view in ('compact','slim') else STOCK_CHANNELS.index('never')
        block.values[...,never*k:(never+1)*k] = torch.where(
            holes[...,None],1.,block.values[...,never*k:(never+1)*k])
    return block


class _BundleProcessor(MicellaneousPreProcessor):
    ENABLED = False  # opt-in experiment; never included in daily bulk updates
    AllowInactive = True

    def pre_process(self, *args, **kwargs):
        raise ValueError('Build all three blocks together with the factor_sequence build stage')

    def load(self, dates=None, secid=None, *, dataset=None, view="full", **kwargs):
        return load_block(dataset,self.key,dates,secid,view=view)

class PrePro_factorseq(_BundleProcessor):
    pass

class PrePro_factorperf(_BundleProcessor):
    pass

class PrePro_factormarket(_BundleProcessor):
    pass
