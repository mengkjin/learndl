"""Resumable conversion of existing RCQuant stock bars to native price and volume."""
from __future__ import annotations

from contextlib import closing
from datetime import date as calendar_date, datetime, timedelta
from hashlib import file_digest, sha256
from itertools import batched
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3

import numpy as np
import pandas as pd
import portalocker
import rqdatac

from src.data.util import trade_min_reform
from src.proj import CALENDAR, DB, Dates, Logger, MACHINE, PATH
from src.proj.db.interface.db_path import DBPath

from . import bar_min
from .initializer import MinDataType, QuotaExceeded, RQInitializer
from .migration_factors import (
    EMPTY_EVENTS, PRICE, event_series, normalize_events, restore_stock_day, vwap_outside,
)
from .storage import atomic_frame, date_lock

START = 20070101
END = 20261124
BEGIN_NONE = bar_min.SEC_NONE_START
SEC = MinDataType.SEC
KEYS = ['code', 'datetime']
CHECK_COLUMNS = [*PRICE, 'volume', 'amount', 'num_trades']
METHODS = ('get_ex_factor', 'get_split')
BATCH_SIZE = 10
DAILY_FACTOR_BATCH_SIZE = 1000
MINUTE_BATCH_SIZE = 100


class EveningEnded(Exception):
    """Save progress and resume in the next evening window."""


def fingerprint(path: Path) -> str:
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def file_matches(path: Path, digest: str) -> bool:
    return path.is_file() and fingerprint(path) == digest


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_frame(frame, path, *, empty_ok=False):
    atomic_frame(frame.reset_index(drop=True), path, vb_level='never', empty_ok=empty_ok)
    sync_directory(path.parent)


def to_order_book_ids(codes: list[str], instruments: pd.DataFrame | None) -> dict[str, str]:
    """Recover exchange suffixes, using the dated instrument list for naked codes."""
    mapping = {}
    ambiguous = set()
    provider_codes = set()
    if instruments is not None:
        provider_codes = set(instruments['code'].dropna().unique())
        for instrument in provider_codes:
            ticker = instrument.split('.')[0]
            if ticker in mapping and mapping[ticker] != instrument:
                ambiguous.add(ticker)
            mapping[ticker] = instrument
    ids = {}
    for code in codes:
        if re.fullmatch(r'\d{6}\.(XSHG|XSHE)', code):
            ids[code] = code
        elif re.fullmatch(r'\d{6}\.(SH|SZ)', code):
            ticker, exchange = code.split('.')
            ids[code] = ticker + {'SH': '.XSHG', 'SZ': '.XSHE'}[exchange]
        elif code in provider_codes:
            ids[code] = code
        elif code in mapping and code not in ambiguous:
            ids[code] = mapping[code]
        else:
            raise ValueError(f'Unmapped historical stock code {code!r}')
    if len(ids) != len(set(ids.values())):
        raise ValueError('Multiple raw codes map to the same order_book_id')
    return ids


def valid_raw_day(frame: pd.DataFrame, date: int) -> None:
    times = pd.to_datetime(frame['datetime'])
    minute = times.dt.hour * 60 + times.dt.minute
    session = minute.between(571, 690) | minute.between(781, 900)
    if (frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any()
            or not times.dt.strftime('%Y%m%d').eq(str(date)).all()
            or not times.eq(times.dt.floor('min')).all() or not session.all()):
        raise ValueError(f'{date}: invalid or duplicate RCQuant stock-minute keys')
    if not times.dt.strftime('%H%M').eq(frame['time'].astype(str)).all():
        raise ValueError(f'{date}: datetime and time disagree')


def validate_values(frame: pd.DataFrame, date: int) -> None:
    traded = frame['volume'].gt(0)
    if (not np.isfinite(frame[CHECK_COLUMNS[-3:]]).all().all()
            or frame[CHECK_COLUMNS[-3:]].lt(0).any().any()
            or not np.isfinite(frame.loc[traded, PRICE]).all().all()
            or not np.isclose(frame['volume'], np.rint(frame['volume']), rtol=0, atol=1e-5).all()
            or not np.isclose(frame['num_trades'], np.rint(frame['num_trades']), rtol=0, atol=1e-5).all()
            or frame.loc[traded, PRICE].le(0).any().any()
            or frame.loc[traded, 'low'].gt(frame.loc[traded, ['open', 'close']].min(axis=1)).any()
            or frame.loc[traded, 'high'].lt(frame.loc[traded, ['open', 'close']].max(axis=1)).any()):
        raise ValueError(f'{date}: invalid unadjusted quote, share or trade-count values')


class Migration:
    def __init__(self, root: Path, factor_dir: Path, db: sqlite3.Connection, started_at: datetime | None):
        self.root = root
        self.state = root / '.sec_none_migration'
        self.factor_dir = factor_dir
        self.db = db
        self.started_at = started_at
        self.db.execute('CREATE TABLE IF NOT EXISTS checkpoints (stage TEXT, item TEXT, value TEXT,'
                        ' PRIMARY KEY (stage, item))')
        self.updater = bar_min.RcquantMinBarDownloader()

    def read(self, stage: str, item: int | str):
        row = self.db.execute('SELECT value FROM checkpoints WHERE stage=? AND item=?',
                              (stage, str(item))).fetchone()
        return json.loads(row[0]) if row else None

    def mark(self, stage: str, item: int | str, value=True) -> None:
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO checkpoints VALUES (?,?,?)',
                            (stage, str(item), json.dumps(value)))

    def guard(self):
        if self.started_at is not None and bar_min._backfill_deadline_hit(self.started_at):
            raise EveningEnded

    def remote(self, func, *args, **kwargs):
        self.guard()
        if not RQInitializer.init(raise_on_quota=True):
            raise RuntimeError('RCQuant initialization failed')
        return bar_min.rq_call(func, *args, raise_on_quota=True, **kwargs)

    def minute_path(self, date: int) -> Path:
        return self.root / 'secmin' / f'{date}.feather'

    def scan(self, expected: list[int]) -> dict:
        previous = self.read('manifest', 'history')
        if previous is not None and not previous['missing'] and not previous['mapping_errors']:
            return previous
        dates, missing, codes, early = [], [], set(), set()
        errors = {}
        if previous is not None:
            dates = previous['dates']
            codes.update(previous['ids'])
            early.update(previous['early'])
            expected = sorted({*previous['missing'], *map(int, previous['mapping_errors'])})
        for date in expected:
            self.guard()
            raw = self.minute_path(date)
            if not raw.is_file():
                missing.append(date)
                continue
            try:
                source = self.read('scan', date)
                if source is None:
                    with date_lock(self.root, SEC, date):
                        raw_codes = pd.read_feather(raw, columns=['code'])['code']
                        if raw_codes.isna().any() or raw_codes.empty:
                            raise ValueError('empty raw archive or missing stock code')
                        instruments = bar_min.load_list(date, SEC)
                        mapping = to_order_book_ids(sorted(raw_codes.unique()), instruments)
                        # The pre-2007 event history is only relevant to stocks
                        # that could already have had an action at the start.
                        listed = {}
                        if instruments is not None and 'listed_date' in instruments:
                            listed = instruments.set_index('code')['listed_date'].to_dict()
                        source = {
                            'mapping': mapping, 'raw': fingerprint(raw),
                            'early': sorted(code for code in mapping.values()
                                            if str(listed.get(code, '0000-00-00')) < '2007-01-01'),
                        }
                        self.mark('scan', date, source)
            except ValueError as exc:
                errors[date] = str(exc)
                continue
            dates.append(date)
            codes.update(source['mapping'].values())
            early.update(source['early'])
        # Keep existing batch boundaries if a previously missing file introduces
        # new stock IDs. Appended IDs form new batches.
        manifest = {'dates': sorted(dates), 'missing': missing, 'mapping_errors': errors}
        for name, values in [('ids', codes), ('early', early)]:
            existing = [] if previous is None else previous[name]
            manifest[name] = [*existing, *sorted(values - set(existing))]
            batches = [] if previous is None else previous[f'{name}_batches']
            manifest[f'{name}_batches'] = [
                *batches, *map(list, batched(manifest[name][len(existing):], BATCH_SIZE)),
            ]
        self.mark('manifest', 'history', manifest)
        return manifest

    def factor_batch(self, method: str, ids: list[str], start, end: int) -> pd.DataFrame:
        key = f'{method}-{start}-{end}-' + sha256(','.join(ids).encode()).hexdigest()
        path = self.state / 'events' / f'{key}.feather'
        if path.is_file():
            return pd.read_feather(path)
        if self.read('empty_event_batch', key):
            return normalize_events(method, ids, None)
        result = self.remote(getattr(rqdatac, method), ids,
                             start_date=start, end_date=end, market='cn')
        frame = normalize_events(method, ids, result)
        if frame.empty:
            self.mark('empty_event_batch', key)
        else:
            write_frame(frame, path)
        return frame

    def get_events(self, manifest: dict):
        events = {}
        key = sha256(json.dumps([manifest['ids_batches'], manifest['early_batches']]).encode()).hexdigest()
        for method in METHODS:
            path = self.state / 'events' / f'{method}-merged-{key}.feather'
            if path.is_file():
                events[method] = event_series(method, pd.read_feather(path))
                continue
            frames = [self.factor_batch(method, ids, START, END)
                      for ids in manifest['ids_batches']]
            frames.extend(self.factor_batch(method, ids, None, 20061231)
                          for ids in manifest['early_batches'])
            frames = [part for part in frames if not part.empty]
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['order_book_id'])
            events[method] = event_series(method, frame)
            if frames:
                write_frame(frame, path)
        return events

    def collect_daily_actions(self, date: int) -> bool:
        """Freeze the day's provider universe and archive both event responses."""
        output = self.factor_dir / 'daily' / str(date)
        universe_path = output / 'order_book_ids.feather'
        event_paths = {name: output / f'{name}.feather' for name in METHODS}
        if (self.read('daily_actions', date) is not None and universe_path.is_file()
                and all(path.is_file() for path in event_paths.values())):
            return False
        self.guard()
        if universe_path.is_file():
            ids = pd.read_feather(universe_path)['order_book_id'].tolist()
        else:
            instruments = bar_min.rcquant_instrument_list(date, SEC, raise_on_quota=True)
            if instruments.empty or 'code' not in instruments or instruments['code'].isna().any():
                raise ValueError(f'{date}: RCQuant returned an invalid stock universe')
            ids = instruments['code'].unique().tolist()
            if not all(isinstance(code, str) and code.strip() for code in ids):
                raise ValueError(f'{date}: RCQuant returned an invalid order_book_id')
            ids.sort()
            write_frame(pd.DataFrame({'order_book_id': ids}), universe_path)

        counts = {'order_book_ids': len(ids)}
        for method, destination in event_paths.items():
            if destination.is_file():
                counts[method] = len(pd.read_feather(destination))
                continue
            frames = []
            for batch in batched(ids, DAILY_FACTOR_BATCH_SIZE):
                self.guard()
                frame = self.factor_batch(method, list(batch), date, date)
                if not frame.empty:
                    frames.append(frame)
            events = pd.concat(frames, ignore_index=True) if frames else normalize_events(method, ids, None)
            write_frame(events, destination, empty_ok=True)
            counts[method] = len(events)
        self.mark('daily_actions', date, counts)
        Logger.stdout(f'RCQuant {date}: archived stock IDs and corporate actions, {counts}')
        return True

    def collect_recent_actions(self, through: int) -> bool:
        """Give the daily update target priority, then resume missed dates."""
        latest = calendar_date.fromisoformat(str(through))
        first = calendar_date.fromisoformat(str(BEGIN_NONE))
        if latest < first:
            return False
        updated = self.collect_daily_actions(through)
        for offset in range((latest - first).days):
            day = first + timedelta(days=offset)
            updated = self.collect_daily_actions(int(day.strftime('%Y%m%d'))) or updated
        return updated

    def daily_factors(self, date, mapping, events):
        path = self.factor_dir / f'{date}.feather'
        digest = self.read('factors', date)
        if digest and path.is_file():
            return
        codes = list(mapping)
        ids = list(mapping.values())
        frame = pd.DataFrame({
            'date': date, 'order_book_id': ids, 'secid': DB.code2secid(pd.Series(codes)),
            'ex_cum_factor': [events['get_ex_factor'].get(code, EMPTY_EVENTS).on(date) for code in ids],
            'split_cum_factor': [events['get_split'].get(code, EMPTY_EVENTS).on(date) for code in ids],
        })
        write_frame(frame, path)
        self.mark('factors', date, fingerprint(path))

    def direct_none(self, date: int, ids: list[str]) -> pd.DataFrame:
        self.guard()
        if not RQInitializer.init(raise_on_quota=True):
            raise RuntimeError('RCQuant initialization failed')
        data = bar_min.rq_get_price(ids, date, adjust_type='none', raise_on_quota=True)
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame()
        # Keep canonical IDs until joining to the exact legacy stock-minute keys.
        frame = data.reset_index().rename(columns={'total_turnover': 'amount'})
        frame['datetime'] = pd.to_datetime(frame['datetime'])
        if not frame['order_book_id'].isin(ids).all():
            raise ValueError(f'{date}: unexpected IDs in none response')
        return frame

    def prepare_raw(self, date, raw, mapping, events):
        restored = raw.copy()
        if bar_min.is_none_min(raw):
            return restored, {}, []
        daily = DB.load('trade_ts', 'day', date, vb_level='never')
        if not daily.empty:
            daily = daily.set_index('secid', verify_integrity=True)
        reference = {} if daily.empty else daily[[*PRICE, 'volume']].T.to_dict('list')
        secid = dict(zip(mapping, DB.code2secid(pd.Series(list(mapping)))))
        needs_none = {}
        counts = {'price_and_original_volume': 0, 'price_and_split_volume': 0,
                  'direct_none': 0, 'revised_amount_or_trades': 0}

        restored[[*PRICE, 'volume']] = restored[[*PRICE, 'volume']].astype(float)
        groups = raw.groupby('code', sort=False).indices
        for code, indexes in groups.items():
            provider = mapping[code]
            values = reference.get(secid[code])
            if values is None:
                needs_none[provider] = indexes
                continue
            ref = np.asarray(values, dtype=float)
            ref[-1] *= 1000  # Tushare day stores thousands of shares.
            part = raw.iloc[indexes]
            recovered = restore_stock_day(
                part[PRICE].to_numpy(float), part['volume'].to_numpy(float), ref, date,
                events['get_ex_factor'].get(provider, EMPTY_EVENTS),
                events['get_split'].get(provider, EMPTY_EVENTS),
            )
            if recovered is None:
                needs_none[provider] = indexes
                continue
            prices, volume, scaled = recovered
            candidate = part.assign(**dict(zip(PRICE, prices.T)), volume=volume)
            if vwap_outside(candidate).any():
                needs_none[provider] = indexes
                continue
            restored.loc[indexes, PRICE] = prices
            restored.loc[indexes, 'volume'] = volume
            counts['price_and_split_volume' if scaled else 'price_and_original_volume'] += 1

        missing = set(needs_none)

        def apply_direct(direct):
            verified = []
            for provider, part in direct.groupby('order_book_id', sort=False):
                if provider not in missing or part['datetime'].duplicated().any():
                    continue
                indexes = needs_none[provider]
                legacy = raw.iloc[indexes]
                native = part.set_index('datetime')
                if not legacy['datetime'].isin(native.index).all() or len(native) != len(legacy):
                    continue
                native = native.loc[legacy['datetime'], CHECK_COLUMNS].reset_index()
                outside = vwap_outside(native)
                if outside.any():
                    audit = native.loc[outside].assign(order_book_id=provider)
                    audit['vwap'] = audit['amount'] / audit['volume']
                    write_frame(audit, self.state / 'vwap_failures' / f'{date}.feather')
                    raise ValueError(f'{date}: {provider} has {outside.sum()} VWAP violations in direct none data')
                changes = ~np.isclose(native[['amount', 'num_trades']].to_numpy(),
                                      legacy[['amount', 'num_trades']].to_numpy(), rtol=0, atol=1e-8)
                counts['revised_amount_or_trades'] += int(changes.any(axis=1).sum())
                restored.loc[indexes, CHECK_COLUMNS] = native[CHECK_COLUMNS].to_numpy()
                counts['direct_none'] += 1
                missing.remove(provider)
                verified.append(part)
            return verified

        if missing:
            directory = self.state / 'none' / str(date)
            for path in sorted(directory.glob('*.feather')):
                apply_direct(pd.read_feather(path))
            for batch in batched(sorted(missing), MINUTE_BATCH_SIZE):
                direct = self.direct_none(date, list(batch))
                verified = [] if direct.empty else apply_direct(direct)
                if verified:
                    key = sha256(','.join(batch).encode()).hexdigest()
                    write_frame(pd.concat(verified, ignore_index=True), directory / f'{key}.feather')
        restored[bar_min.ADJUST_COLUMN] = 'none'
        return restored, counts, sorted(missing)

    def stage(self, date, restored, counts):
        valid_raw_day(restored, date)
        validate_values(restored, date)
        outside = vwap_outside(restored)
        if outside.any():
            audit = restored.loc[outside, [*KEYS, *CHECK_COLUMNS]].copy()
            audit['vwap'] = audit['amount'] / audit['volume']
            write_frame(audit, self.state / 'vwap_failures' / f'{date}.feather')
            raise ValueError(f'{date}: {outside.sum()} native one-minute VWAP violations')
        minute = self.updater.rcquant_min_to_normal_min(restored, SEC)
        if minute.duplicated(['secid', 'minute']).any():
            raise ValueError(f'{date}: overlapping normalized stock-minute keys')
        if minute['minute'].max() != 239:
            raise ValueError(f'{date}: normalized stock bars are missing the last session minute')
        outputs = [(self.minute_path(date), restored)]
        for period in bar_min.SEC_PERIODS:
            frame = minute if period == 1 else trade_min_reform(minute.copy(), period)
            if frame.empty or vwap_outside(frame).any():
                raise ValueError(f'{date}: invalid {period}-minute VWAP')
            if period != 1 and not np.allclose(frame[['volume', 'amount', 'num_trades']].sum(),
                                              minute[['volume', 'amount', 'num_trades']].sum(),
                                              rtol=1e-12, atol=1e-6):
                raise ValueError(f'{date}: inconsistent {period}-minute totals')
            outputs.append((DBPath('trade_ts', bar_min.src_key(SEC, period)).path_exact(date), frame))

        prepared = []
        required = {}
        for target, frame in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            device = target.parent.stat().st_dev
            size = target.stat().st_size if target.exists() else frame.memory_usage(deep=False).sum()
            required[device] = (target.parent, required.get(device, (None, 0))[1] + int(size))
        for directory, size in required.values():
            if shutil.disk_usage(directory).free < size:
                raise OSError(f'{date}: insufficient space to stage RCQuant bars on {directory}')
        for target, frame in outputs:
            pending = target.parent / '.rcquant_none_stage' / target.name
            write_frame(frame, pending)
            prepared.append({'target': str(target), 'staged': str(pending), 'digest': fingerprint(pending)})
        self.mark('prepared', date, {'outputs': prepared, 'counts': counts})

    def publish(self, date: int, prepared: dict):
        raw = self.minute_path(date)
        backup = raw.parent / '.pre_backup' / raw.name
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            expected = self.read('scan', date)['raw']
            if not file_matches(raw, expected):
                raise RuntimeError(f'{date}: original minute changed since inventory')
            os.link(raw, backup)
            sync_directory(backup.parent)
            sync_directory(backup.parent.parent)
        if not file_matches(backup, self.read('scan', date)['raw']):
            raise RuntimeError(f'{date}: invalid pre-migration backup')

        for item in prepared['outputs']:
            target, pending = Path(item['target']), Path(item['staged'])
            digest = item['digest']
            if pending.exists():
                if not file_matches(pending, digest):
                    raise RuntimeError(f'{date}: staged file changed: {pending}')
                os.replace(pending, target)
                sync_directory(target.parent)
                sync_directory(pending.parent)
            elif not file_matches(target, digest):
                raise RuntimeError(f'{date}: cannot reconcile interrupted publication: {target}')
            self.mark('published', str(date) + '/' + target.name, digest)
        for period in bar_min.SEC_PERIODS:
            DBPath('trade_ts', bar_min.src_key(SEC, period)).clear_date_cache()
        self.mark('done', date, prepared['counts'])
        Logger.stdout(f"RCQuant {date}: published unadjusted raw and six periods, {prepared['counts']}")

    def migrate_date(self, date: int, events):
        if self.read('done', date) is not None or self.read('blocked', date) is not None:
            return
        raw_path = self.minute_path(date)
        with date_lock(self.root, SEC, date):
            prepared = self.read('prepared', date)
            if prepared is None:
                source = self.read('scan', date)
                if not file_matches(raw_path, source['raw']):
                    raise RuntimeError(f'{date}: raw minute changed since inventory')
                raw = pd.read_feather(raw_path).sort_values(KEYS).reset_index(drop=True)
                try:
                    valid_raw_day(raw, date)
                    restored, counts, missing = self.prepare_raw(date, raw, source['mapping'], events)
                    if missing:
                        self.mark('pending_none', date, missing)
                        return
                    self.stage(date, restored, counts)
                except ValueError as exc:
                    self.mark('blocked', date, str(exc))
                    Logger.warning(f'RCQuant migration pending review: {exc}')
                    return
                prepared = self.read('prepared', date)
            self.publish(date, prepared)

    def run(self):
        outstanding = self.db.execute(
            "SELECT item FROM checkpoints WHERE stage='prepared' EXCEPT "
            "SELECT item FROM checkpoints WHERE stage='done'"
        ).fetchall()
        for (date,) in outstanding:
            self.guard()
            self.migrate_date(int(date), None)

        manifest = self.scan(list(map(int, Dates(START, END))))
        Logger.stdout(f"RCQuant source inventory: {len(manifest['dates'])} dates, {len(manifest['ids'])} stocks, "
                      f"{len(manifest['missing'])} missing dates, {len(manifest['mapping_errors'])} mapping errors")
        events = self.get_events(manifest)
        for date in manifest['dates']:
            self.guard()
            self.daily_factors(date, self.read('scan', date)['mapping'], events)
        for date in reversed(manifest['dates']):
            self.guard()
            self.migrate_date(date, events)

        summary = {'done': 0, 'blocked': 0, 'awaiting_none': 0, 'price_and_original_volume': 0,
                   'price_and_split_volume': 0, 'direct_none': 0, 'revised_amount_or_trades': 0}
        for date in manifest['dates']:
            counts = self.read('done', date)
            if counts is not None:
                summary['done'] += 1
                for key, val in counts.items():
                    summary[key] += val
            elif self.read('blocked', date):
                summary['blocked'] += 1
            else:
                summary['awaiting_none'] += 1
        Logger.stdout(f'RCQuant historical migration: {summary}')
        if (summary['done'] and summary['done'] == len(manifest['dates']) and not manifest['missing']
                and not manifest['mapping_errors']):
            self.mark('complete', 'history', summary)


def collect_daily_actions_through(through: int) -> bool:
    _, through = CALENDAR.update_schedule(end=through, key='rcquant_min')
    if through < BEGIN_NONE:
        return False
    root = bar_min.RQ_PATH
    state = root / '.sec_none_migration'
    state.mkdir(parents=True, exist_ok=True)
    # The daily collector and historical task share durable checkpoints, but
    # their request dates are disjoint. Only daily collectors serialize here.
    with date_lock(root, 'stock_actions', BEGIN_NONE), closing(sqlite3.connect(state / 'progress.sqlite')) as db:
        return Migration(root, PATH.miscel / 'exfactor', db, started_at=None).collect_recent_actions(through)


def run():
    if MACHINE.name != 'mengkjin-server':
        raise RuntimeError('RCQuant production minute migration is configured for mengkjin-server')
    started = CALENDAR.now(bj_tz=True)
    if started.hour < 21 or bar_min._backfill_deadline_hit(started):
        Logger.stdout('RCQuant migration is outside its Beijing evening window')
        return
    root = bar_min.RQ_PATH
    state = root / '.sec_none_migration'
    state.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(state / 'migration.lock', timeout=0), closing(sqlite3.connect(state / 'progress.sqlite')) as db:
        migration = Migration(root, PATH.miscel / 'exfactor', db, started)
        if migration.read('complete', 'history') is not None:
            Logger.stdout('RCQuant minute migration already complete')
            return
        today = started.strftime('%Y%m%d')
        if migration.read('quota_exhausted', today):
            Logger.stdout('RCQuant migration already reached its quota this evening')
            return
        try:
            migration.run()
        except QuotaExceeded:
            migration.mark('quota_exhausted', today)
            Logger.warning('RCQuant quota reached; migration will resume next evening')
        except EveningEnded:
            Logger.stdout('RCQuant evening window ended; migration will resume next evening')
