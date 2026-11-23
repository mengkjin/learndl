# coding: utf-8
# description: Remove out-of-interval stocks from archived all/top universes.
# email: False
# mode: os
"""Clean daily snapshots without reranking stocks or recalculating exclusions."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd

from src.automation.console import logger
from src.proj import DB
from src.proj.db.basic.code_mapper import secid2secid


def _scan(ctx, *, repair):
    root = ctx.root / 'data/Export/universe'
    paths = sorted(path for path in root.glob('*/*/*.feather')
                   if path.parent.parent.name == 'all' or path.parent.parent.name.startswith('top'))
    if not paths:
        return True
    desc = DB.load('information_ts', 'description', vb_level='never')
    if desc.empty or not desc.list_dt.gt(0).any():
        raise ValueError('Cannot verify archived universes without listing dates')
    known = desc.loc[desc.list_dt.gt(0)]
    checked = changed = removed = 0
    for path in paths:
        name, date_text = path.stem.rsplit('.', 1)
        if name != path.parent.parent.name or date_text[:4] != path.parent.name or len(date_text) != 8:
            raise ValueError(f'Unexpected universe snapshot path: {path}')
        date = int(date_text)
        frame = pd.read_feather(path)
        if 'date' in frame and not frame.date.eq(date).all():
            raise ValueError(f'Conflicting snapshot dates in {path}')
        ids = secid2secid(frame.secid)
        # Preserve undocumented IDs. One valid interval is enough for relistings.
        active = known.loc[known.list_dt.le(date) &
                           (known.delist_dt.isna() | known.delist_dt.le(0) |
                            known.delist_dt.gt(date)), 'secid']
        drop = ids.isin(known.secid) & ~ids.isin(active)
        checked += 1
        if not drop.any():
            continue
        if not repair:
            raise ValueError(f'Inactive securities remain in {path}')
        backup = ctx.backup_dir / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            frame.loc[~drop].reset_index(drop=True).to_feather(temporary)
            shutil.copymode(path, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        changed += 1
        removed += int(drop.sum())
    logger.info('Universe snapshots checked=%d changed=%d rows_removed=%d', checked, changed, removed)
    return True


def apply(ctx):
    _scan(ctx, repair=True)


def verify(ctx):
    return _scan(ctx, repair=False)
