"""Immutable monthly universe packs with incremental, dependency-aware publication."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.proj import DB, Load


class StaleUniverse(ValueError):
    """An upstream change requires explicit daily-snapshot regeneration."""


class PreparedUniverse:
    VERSION = 2

    def __init__(self, root, name):
        self.root = Path(root) / hashlib.sha256(name.encode()).hexdigest()[:20]
        self.name = name

    @staticmethod
    def fingerprint(path):
        path = Path(path)
        try:
            stat = path.stat()
            return [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        except FileNotFoundError:
            return [str(path), None]

    def sources(self, dates):
        return {str(d): self.fingerprint(DB.path('universe', self.name, d))
                for d in sorted(set(map(int, dates)))}

    def dependencies(self, dates):
        """Track absent files too, so late-arriving records invalidate old packs.

        Conservative for information tables; precise 90-calendar-day windows for
        announcements. Benchmark-derived universes are not supported yet.
        """
        if self.name != 'all' and not self.name.startswith('top'):
            raise ValueError('prepared universes currently support all/top only')
        values = sorted(map(int, DB.dates('trade_ts', 'day_val')))
        memo = {}
        paths_cache = {}
        def dbpath(src, key, date=None):
            address = (src, key, date)
            if address not in paths_cache:
                paths_cache[address] = DB.path(src, key, date)
            return paths_cache[address]
        def token(path):
            key = str(path)
            if key not in memo:
                memo[key] = self.fingerprint(path)
            return memo[key]
        shared = [dbpath('information_ts', key) for key in ('description', 'change_name')]
        result = {}
        for date in sorted(set(map(int, dates))):
            previous = [d for d in values if d <= date]
            val_date = previous[-1] if previous else date
            paths = shared + [dbpath('trade_ts', 'day_val', val_date),
                              dbpath('trade_ts', 'day_val', date),
                              dbpath('exposure', 'week_rank_loser', date),
                              dbpath('exposure', 'warns_st_and_delist_90', date)]
            end = datetime.strptime(str(date), '%Y%m%d')
            for offset in range(90):
                current = end-timedelta(days=offset)
                day = current.year*10000 + current.month*100 + current.day
                paths.extend(dbpath('crawler', f'announcement_{ex}', day) for ex in ('sse','szse','bse'))
            result[str(date)] = hashlib.sha256(json.dumps([token(p) for p in paths]).encode()).hexdigest()
        return result

    def current(self):
        try:
            folder = self.root / (self.root/'current').read_text().strip()
            meta = json.loads((folder/'manifest.json').read_text())
        except FileNotFoundError:
            return None, None
        if meta['version'] != self.VERSION or meta['name'] != self.name:
            raise ValueError('prepared universe schema/name mismatch; rebuild in a new root')
        return folder, meta

    def prepare_from_sources(self, dates, *, verified_warning_history=False):
        """Regenerate only uncovered/revised days, then incrementally publish.

        Recomputing warning flags needs an explicit assertion of source coverage;
        sparse historical announcement files cannot certify a negative result.
        """
        from .universe import Universe, UniverseExclusions
        from src.data import DATAVENDOR
        from src.proj import Save
        dates = sorted(set(map(int, dates)))
        _, old = self.current()
        deps = self.dependencies(dates)
        sources = self.sources(dates)
        changed = [d for d in dates if old is None or
                   old['dependencies'].get(str(d)) != deps[str(d)] or
                   old['sources'].get(str(d)) != sources[str(d)]]
        started = time.perf_counter()
        if changed:
            if not verified_warning_history:
                raise ValueError('warning-history coverage must be verified before regenerating daily universes')
            DATAVENDOR.INFO.clear_all()
            DATAVENDOR.TRADE.clear_all()
            for cache in UniverseExclusions._cache_exclusions.values():
                for date in changed:
                    cache.pop(date, None)
            UniverseExclusions.prepare_warnst(changed, refresh=True)
            universe = Universe(self.name)
            for date in changed:
                universe.get_universe_df(date, rebuild=True)
                Universe._cache_universes.get(self.name, {}).pop(date, None)
            Save.async_wait_all(caller_name='prepared_universe')
        source_seconds = time.perf_counter()-started
        report = self.prepare(dates, _regenerated=set(changed))
        report['source_finalize_seconds'] = source_seconds
        report['total_seconds'] += source_seconds
        return report

    def prepare(self, dates, *, _regenerated=frozenset()):
        import fcntl
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root/'writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._prepare(dates, _regenerated=_regenerated)

    def _prepare(self, dates, *, _regenerated=frozenset()):
        """Import finalized snapshots; unchanged months are hard-linked.

        Initial import trusts existing daily snapshots. Subsequent upstream
        changes cannot be blessed by repacking an unchanged stale daily file.
        """
        started = time.perf_counter()
        dates = sorted(set(map(int, dates)))
        if not dates:
            raise ValueError('no universe dates requested')
        old_folder, old = self.current()
        sources, deps = self.sources(dates), self.dependencies(dates)
        for date in dates:
            d = str(date)
            if old and d in old['dependencies'] and old['dependencies'][d] != deps[d] and date not in _regenerated:
                raise StaleUniverse(f'upstream revision at {date}: use prepare_from_sources')
        changed = [d for d in dates if not old or old['sources'].get(str(d)) != sources[str(d)]
                   or old['dependencies'].get(str(d)) != deps[str(d)]]
        frames = {d: Load.df(Path(sources[str(d)][0])).assign(date=d) for d in changed}
        if any(frame.empty for frame in frames.values()):
            raise ValueError('empty source snapshot requires explicit coverage handling')
        read_seconds = time.perf_counter()-started
        if old and not changed:
            return dict(read_seconds=read_seconds, compute_seconds=0., save_seconds=0.,
                        total_seconds=time.perf_counter()-started, changed_dates=0, rewritten_months=0,
                        generation=old['generation'])
        begin = time.perf_counter()
        coverage = dict(old['coverage']) if old else {}
        coverage.update({str(d): list(df.columns) for d, df in frames.items()})
        all_sources = dict(old['sources']) if old else {}
        all_deps = dict(old['dependencies']) if old else {}
        all_sources.update(sources); all_deps.update(deps)
        months = sorted({d//100 for d in changed})
        packs = {}
        for month in months:
            parts = []
            if old_folder and (old_folder/f'{month}.feather').exists():
                prior = pd.read_feather(old_folder/f'{month}.feather')
                parts.append(prior[~prior.date.isin(changed)])
            parts.extend(frames[d] for d in changed if d//100 == month)
            packs[month] = pd.concat(parts, ignore_index=True).sort_values('date', kind='stable').reset_index(drop=True)
        compute_seconds = time.perf_counter()-begin
        begin = time.perf_counter()
        self.root.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        folder = self.root/generation
        folder.mkdir()
        if old_folder:
            for path in old_folder.glob('*.feather'):
                if int(path.stem) not in months:
                    os.link(path, folder/path.name)
        for month, frame in packs.items():
            frame.to_feather(folder/f'{month}.feather')
        manifest = dict(version=self.VERSION, name=self.name, generation=generation,
                        sources=all_sources, dependencies=all_deps, coverage=coverage)
        (folder/'manifest.json').write_text(json.dumps(manifest))
        if self.sources(dates) != sources or self.dependencies(dates) != deps:
            raise RuntimeError('sources changed during preparation; generation not published')
        # Prevent simultaneous writers from silently dropping another writer's dates.
        _, latest = self.current()
        if (latest or {}).get('generation') != (old or {}).get('generation'):
            raise RuntimeError('another writer published a generation; retry preparation')
        pointer = self.root/f'.{generation}.tmp'
        pointer.write_text(generation)
        os.replace(pointer, self.root/'current')
        return dict(read_seconds=read_seconds, compute_seconds=compute_seconds,
                    save_seconds=time.perf_counter()-begin, total_seconds=time.perf_counter()-started,
                    bytes=sum(p.stat().st_size for p in folder.iterdir()), dates=len(all_sources),
                    changed_dates=len(changed), rewritten_months=len(months), generation=generation)

    def load(self, dates, exclusions):
        """Validate requested dependencies, then read a single pinned generation."""
        dates = sorted(set(map(int, dates)))
        folder, meta = self.current()
        if meta is None:
            raise FileNotFoundError('no prepared universe generation')
        sources, deps = self.sources(dates), self.dependencies(dates)
        for date, source in sources.items():
            if meta['sources'].get(date) != source or meta['dependencies'].get(date) != deps[date]:
                raise StaleUniverse(f'stale or uncovered universe date: {date}')
            required = set(exclusions.split('_'))-{''}
            if not required.issubset(meta['coverage'][date]):
                raise ValueError(f'missing exclusion coverage at {date}: {required-set(meta["coverage"][date])}')
        frames = []
        for month in sorted({d//100 for d in dates}):
            frame = pd.read_feather(folder/f'{month}.feather')
            frames.append(frame[frame.date.isin(dates)])
        if self.sources(dates) != sources or self.dependencies(dates) != deps:
            raise StaleUniverse('sources changed during load')
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
