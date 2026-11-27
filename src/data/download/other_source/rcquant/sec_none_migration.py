"""Resumable conversion of existing RCQuant stock bars to native price and volume."""
from __future__ import annotations

from contextlib import closing
from datetime import date as calendar_date, datetime, timedelta
from hashlib import file_digest, sha256
from itertools import batched, groupby
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
from time import perf_counter
from uuid import uuid4

import pandas as pd
import polars as pl
import rqdatac

from src.proj import CALENDAR, DB, Dates, Logger, PATH, Save
from src.proj.db.interface.db_path import DBPath

from . import bar_min
from .initializer import MinDataType, RQInitializer
from .migration_factors import normalize_events
from .sec_history_polars import (
    FactorHistory, ID, PRICE, QUOTE, STOCK_TIME, TOTALS, aggregate_bars, daily_reference,
    daily_volume_mismatches, normalize_bars, restore_factors, vwap_audit,
)
from .storage import (
    BAR_IPC, RAW_IPC, SpaceDeferred, atomic_frame, check_space, date_lock, disk_full, frame_bytes,
)

__all__ = ['collect_daily_actions_through', 'run']

START = 20070101
END = 20261124
BEGIN_NONE = bar_min.SEC_NONE_START
SEC = MinDataType.SEC
KEYS = ['code', 'datetime']
METHODS = ('get_ex_factor', 'get_split')
BATCH_SIZE = 10
DAILY_FACTOR_BATCH_SIZE = 1000
MINUTE_BATCH_SIZE = 100
REPAIR = 'polars_roundtrip_v2'
MINUTE_DATABLOCKS = ('15m', '15mraw', '30m', '30mcont', '15mt', '30mt')
PRICE_CHECK_ID = '000001.XSHE'


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


def write_frame(frame, path, *, empty_ok=False, ipc_compression=None):
    if isinstance(frame, pd.DataFrame):
        frame = frame.reset_index(drop=True)
    elif ipc_compression is None:
        ipc_compression = BAR_IPC
    atomic_frame(frame, path, vb_level='never', empty_ok=empty_ok, ipc_compression=ipc_compression)
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


def valid_raw_day(frame: pl.DataFrame, date: int) -> None:
    at = pl.col('datetime')
    minute = at.dt.hour().cast(pl.Int64) * 60 + at.dt.minute()
    session = minute.is_between(571, 690) | minute.is_between(781, 900)
    session_date = pl.date(date // 10000, date // 100 % 100, date % 100)
    keys_valid, codes_present, clock_matches, last_minute = frame.select(
        (at.is_not_null() & at.dt.date().eq(session_date) & at.eq(at.dt.truncate('1m'))
         & session).all(),
        pl.col('code').is_not_null().all(),
        (at.dt.hour().cast(pl.Int64) * 100 + at.dt.minute())
        .eq(pl.col('time').cast(pl.Int64, strict=False)).fill_null(False).all().alias('time_valid'),
        minute.max().alias('last_minute'),
    ).row(0)
    if not (len(frame) and keys_valid and codes_present and frame.select(KEYS).n_unique() == len(frame)):
        raise ValueError(f'{date}: invalid or duplicate RCQuant stock-minute keys')
    if not clock_matches:
        raise ValueError(f'{date}: datetime and time disagree')
    if last_minute != 900:
        raise ValueError(f'{date}: stock bars are missing the last session minute')


def valid_quote() -> pl.Expr:
    valid_prices = (
        pl.all_horizontal(pl.col(PRICE).is_finite() & pl.col(PRICE).gt(0))
        & (pl.col('low') <= pl.min_horizontal('open', 'close'))
        & (pl.col('high') >= pl.max_horizontal('open', 'close'))
    )
    total = pl.col(TOTALS)
    counts = pl.col('num_trades', 'volume')
    return (
        pl.all_horizontal(total.is_finite() & total.ge(0))
        & pl.all_horizontal((counts - counts.round(0)).abs().le(1e-5))
        & (pl.col('volume').eq(0) | valid_prices)
    ).fill_null(False)


def validate_values(frame: pl.DataFrame, date: int) -> None:
    if not frame.select(valid_quote().all()).item():
        raise ValueError(f'{date}: invalid unadjusted quote, share or trade-count values')


class Migration:
    def __init__(self, root: Path, factor_dir: Path, db: sqlite3.Connection, started_at: datetime | None):
        self.root = root
        self.state = root / '.sec_none_migration'
        self.factor_dir = factor_dir
        self.db = db
        self.started_at = started_at
        self.reserve_bytes = 0
        self.db.execute('CREATE TABLE IF NOT EXISTS checkpoints (stage TEXT, item TEXT, value TEXT,'
                        ' PRIMARY KEY (stage, item))')

    def read(self, stage: str, item: int | str):
        row = self.db.execute('SELECT value FROM checkpoints WHERE stage=? AND item=?',
                              (stage, str(item))).fetchone()
        return json.loads(row[0]) if row else None

    def mark(self, stage: str, item: int | str, value: bool | str | list | dict = True) -> None:
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO checkpoints VALUES (?,?,?)',
                            (stage, str(item), json.dumps(value)))

    def require_record(self, stage: str, item: int | str) -> dict:
        record = self.read(stage, item)
        if not isinstance(record, dict):
            raise RuntimeError(f'{item}: missing or invalid {stage} checkpoint')
        return record

    def repair_stage(self, stage: str) -> str:
        return f'{REPAIR}_{stage}'

    def guard(self):
        if self.started_at is not None and bar_min._backfill_deadline_hit(self.started_at):
            raise EveningEnded

    def check_space(self, path, write_bytes, *, operation):
        check_space(path, write_bytes, self.reserve_bytes, operation=operation)

    def write_frame(self, frame, path, **kwargs):
        size = frame_bytes(frame)
        self.check_space(path.parent, size, operation=f'write {path.name}')
        try:
            write_frame(frame, path, **kwargs)
        except Exception as exc:
            if disk_full(exc):
                raise SpaceDeferred(f'writing {path}: {exc}') from exc
            raise

    def remote(self, func, *args, **kwargs):
        self.guard()
        self.check_space(self.state, 0, operation=getattr(func, '__name__', 'RCQuant query'))
        if not RQInitializer.init(raise_on_quota=True):
            raise RuntimeError('RCQuant initialization failed')
        return bar_min.rq_call(func, *args, raise_on_quota=True, **kwargs)

    def minute_path(self, date: int) -> Path:
        return self.root / 'secmin' / f'{date}.feather'

    def backup_path(self, date: int) -> Path:
        raw = self.minute_path(date)
        return raw.parent / '.pre_backup' / raw.name

    def original_path(self, date: int, *, previously_written: bool) -> Path:
        path = self.backup_path(date) if previously_written else self.minute_path(date)
        if not file_matches(path, self.require_record('scan', date)['raw']):
            raise ValueError(f'{date}: original RCQuant minutes unavailable or changed: {path}')
        return path

    def scan(self, expected: list[int]) -> dict:
        previous = self.read('manifest', 'history')
        if not expected:
            raise ValueError('Cannot inventory RCQuant history with an empty trading calendar')
        dates, missing, codes, early = [], [], set(), set()
        errors = {}
        if previous is not None:
            dates = [date for date in previous['dates'] if self.minute_path(date).is_file()]
            codes.update(previous['ids'])
            early.update(previous['early'])
            expected = sorted(set(expected).difference(dates) | set(previous['missing'])
                              | set(map(int, previous['mapping_errors'])))
            if not expected:
                return previous
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
                        raw_codes = pl.read_ipc(raw, columns=['code'], memory_map=False)['code']
                        if raw_codes.null_count() or raw_codes.is_empty():
                            raise ValueError('empty raw archive or missing stock code')
                        instruments = bar_min.load_list(date, SEC)
                        mapping = to_order_book_ids(raw_codes.unique().sort().to_list(), instruments)
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
            except (ValueError, pl.exceptions.PolarsError) as exc:
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
            self.write_frame(frame, path)
        return frame

    def get_events(self, manifest: dict):
        events = {}
        key = sha256(json.dumps([manifest['ids_batches'], manifest['early_batches']]).encode()).hexdigest()
        for method in METHODS:
            path = self.state / 'events' / f'{method}-merged-{key}.feather'
            if path.is_file():
                events[method] = pl.read_ipc(path, memory_map=False)
                Logger.stdout(f'RCQuant [3/7] {method}: already consolidated, {len(events[method])} rows')
                continue
            Logger.stdout(f'RCQuant [3/7] {method}: collecting or reusing '
                          f'{len(manifest["ids_batches"]) + len(manifest["early_batches"])} batches')
            batches = [
                *((ids, START, END) for ids in manifest['ids_batches']),
                *((ids, None, 20061231) for ids in manifest['early_batches']),
            ]
            frames = []
            for index, (ids, start, end) in enumerate(batches, 1):
                frames.append(self.factor_batch(method, ids, start, end))
                if index % 50 == 0 or index == len(batches):
                    Logger.stdout(f'RCQuant [3/7] {method}: {index}/{len(batches)} batches ready')
            parts = [pl.from_pandas(part) for part in frames if not part.empty]
            frame = pl.concat(parts, how='diagonal_relaxed') if parts else pl.DataFrame()
            events[method] = frame
            self.write_frame(frame if parts else pl.DataFrame(schema={ID: pl.String}), path, empty_ok=True)
            Logger.stdout(f'RCQuant [3/7] {method}: consolidated {len(frame)} rows')
        return FactorHistory.from_events(events['get_ex_factor'], events['get_split'])

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
            self.write_frame(pd.DataFrame({'order_book_id': ids}), universe_path)

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
            self.write_frame(events, destination, empty_ok=True)
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
            return False
        self.write_frame(events.daily(date, self.stock_mapping(mapping)), path)
        self.mark('factors', date, fingerprint(path))
        return True

    @staticmethod
    def stock_mapping(mapping: dict[str, str]) -> pl.DataFrame:
        codes = list(mapping)
        return pl.DataFrame({'code': codes, ID: list(mapping.values()),
                             'secid': DB.code2secid(pd.Series(codes)).to_list()})

    def direct_none(self, date: int, ids: list[str]) -> pd.DataFrame:
        self.guard()
        self.check_space(self.state, 0, operation=f'{date} download native minutes')
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

    def backfill_missing_raw(self, dates: list[int]) -> list[int]:
        """Backfill missing source days before rebuilding their minute tables."""
        filled = []
        for date in reversed(dates):
            self.guard()
            with date_lock(self.root, SEC, date):
                path = self.minute_path(date)
                if path.is_file():
                    filled.append(date)
                    continue
                if self.read('scan', date) is not None or self.backup_path(date).exists():
                    Logger.warning(f'RCQuant {date}: scanned raw source disappeared; original recovery required')
                    continue
                self.check_space(path.parent, 0, operation=f'{date} backfill secmin')
                Logger.stdout(f'RCQuant {date}: attempting missing secmin source backfill (none)')
                try:
                    instruments = bar_min.rcquant_instrument_list(date, SEC, raise_on_quota=True)
                    if instruments.empty:
                        raise ValueError('empty stock universe')
                    instruments = instruments.loc[instruments['is_active']]
                    ids = sorted(instruments['code'].unique())
                    if not ids:
                        raise ValueError('no active stocks')
                    native = self.direct_none(date, ids)
                    bar_min.check_sec_backfill_quality(
                        native.rename(columns={'order_book_id': 'code'}), instruments, date,
                    )
                    absent = set(ids) - set(native['order_book_id'])
                    if absent:
                        raise ValueError(f'none response omitted {len(absent)} requested stocks')
                    frame = pl.from_pandas(
                        bar_min.provider_min_frame(native.set_index([ID, 'datetime']), SEC),
                        nan_to_null=False,
                    ).with_columns(pl.col(QUOTE).cast(pl.Float64))
                    valid_raw_day(frame, date)
                    validate_values(frame, date)
                    self.write_frame(frame.with_columns(pl.lit('none').alias(bar_min.ADJUST_COLUMN)), path,
                                     ipc_compression=RAW_IPC)
                    self.mark('raw_backfill', date, fingerprint(path))
                except ValueError as exc:
                    Logger.warning(f'RCQuant {date}: raw secmin still missing: {exc}')
                    continue
                filled.append(date)
                Logger.stdout(f'RCQuant {date}: raw secmin backfilled for {frame["code"].n_unique()} stocks')
        return filled

    def repair_cohorts(self, dates: list[int]) -> tuple[list[int], list[int]]:
        """Find unchanged source days and dates touched by previous publication."""
        completed = {int(row[0]) for row in self.db.execute("SELECT item FROM checkpoints WHERE stage='done'")}
        published = {int(row[0].split('/', 1)[0]) for row in self.db.execute(
            "SELECT item FROM checkpoints WHERE stage='published'"
        )}
        untouched, previously_written = [], []
        for date in dates:
            # A kill can land after raw replacement but before its receipt.
            # The original hard link is created before the first replacement.
            backup = self.backup_path(date)
            if (date in completed or date in published or (
                backup.is_file() and not file_matches(self.minute_path(date), self.require_record('scan', date)['raw'])
            )):
                previously_written.append(date)
            else:
                untouched.append(date)
        return untouched, previously_written

    def log_price_check(self, date: int, raw: pl.DataFrame, stocks: pl.DataFrame, refs: pl.DataFrame,
                        *, from_backup: bool) -> None:
        codes = stocks.filter(pl.col(ID) == PRICE_CHECK_ID)['code'].to_list()
        if not codes:
            return
        reference = refs.filter(pl.col(ID) == PRICE_CHECK_ID).select(f'{name}_day' for name in PRICE)
        expected = reference.row(0) if not reference.is_empty() else None

        def ohlc(bars: pl.DataFrame):
            if bars.is_empty():
                return None
            return bars.select(pl.col('open').first(), pl.col('high').max(),
                               pl.col('low').min(), pl.col('close').last()).row(0)

        def matches_reference(observed):
            if observed is None or expected is None or None in (*observed, *expected):
                return None
            return all(math.isclose(actual, target, rel_tol=0., abs_tol=1e-5)
                       for actual, target in zip(observed, expected))

        input_ohlc = ohlc(raw.filter(pl.col('code').is_in(codes)))
        record = {'source': 'backup' if from_backup else 'archive', 'input_ohlc': input_ohlc,
                  'tushare_ohlc': expected, 'input_matches_tushare': matches_reference(input_ohlc),
                  'previous_block': self.read('blocked', date)}
        if from_backup and self.minute_path(date).is_file():
            try:
                current = (pl.scan_ipc(self.minute_path(date), memory_map=False)
                           .filter(pl.col('code').is_in(codes)).sort('datetime').select(PRICE).collect())
                record['archive_ohlc'] = ohlc(current)
                record['archive_matches_tushare'] = matches_reference(record['archive_ohlc'])
            except (OSError, pl.exceptions.PolarsError) as exc:
                record['archive_read_error'] = str(exc)
        self.mark(self.repair_stage('price_check'), date, record)

    def prepare_raw(self, date: int, raw: pl.DataFrame, stocks: pl.DataFrame, refs: pl.DataFrame,
                    events: FactorHistory):
        raw_columns = raw.columns
        raw = raw.join(stocks.select('code', ID), on='code', validate='m:1', maintain_order='left')
        if raw[ID].null_count():
            raise ValueError(f'{date}: unmapped raw codes')

        def audit_native_volume(native: pl.DataFrame) -> int:
            # Native bars are the minute source of record; log vendor disagreement.
            ids = daily_volume_mismatches(native, refs)
            if ids:
                Logger.warning(f'RCQuant {date}: native minute volume differs from Tushare for '
                               f'{len(ids)} stocks, first IDs: {ids[:10]}')
            return len(ids)

        if bar_min.ADJUST_COLUMN in raw:
            native = raw[bar_min.ADJUST_COLUMN].eq('none').fill_null(False)
            if native.all():
                return raw.select(raw_columns), {'source_none': stocks.height,
                                                 'direct_none_daily_volume_mismatch': audit_native_volume(raw)}, []
            if native.any():
                raise ValueError(f'{date}: archive mixes native and unverified adjustment states')
        restored, split_count = restore_factors(raw, refs, events, date, stocks)
        factor_count = restored[ID].n_unique()
        missing = set(stocks[ID].to_list()).difference(restored[ID].unique().to_list())
        counts = {'price_and_original_volume': factor_count - split_count,
                  'price_and_split_volume': split_count, 'direct_none': 0,
                  'revised_amount_or_trades': 0, 'direct_none_daily_volume_mismatch': 0}
        records = [restored]
        directory = self.state / 'none' / str(date)
        expected = raw.select(*STOCK_TIME, 'amount', 'num_trades')

        def reconcile(direct: pl.DataFrame) -> pl.DataFrame:
            direct = direct.filter(pl.col(ID).is_in(sorted(missing))).select(
                ID, pl.col('datetime').cast(raw.schema['datetime']), pl.col(QUOTE).cast(pl.Float64),
            )
            coverage = direct.group_by(ID).agg(
                pl.len().alias('returned'), pl.col('datetime').n_unique().alias('unique_returned'),
                valid_quote().all().alias('valid_quotes'),
            )
            intersect = direct.join(expected, on=STOCK_TIME, suffix='_legacy')
            verified = (
                expected.filter(pl.col(ID).is_in(sorted(missing))).group_by(ID).len(name='expected')
                .join(coverage, on=ID).join(intersect.group_by(ID).len(name='matched'), on=ID)
                .filter((pl.col('expected') == pl.col('matched')) & (pl.col('returned') == pl.col('matched'))
                        & (pl.col('returned') == pl.col('unique_returned')) & pl.col('valid_quotes'))
                .select(ID)
            )
            confirmed = intersect.join(verified, on=ID)
            counts['revised_amount_or_trades'] += confirmed.filter(pl.any_horizontal([
                (pl.col(name) - pl.col(f'{name}_legacy')).abs().gt(1e-8) for name in ['amount', 'num_trades']
            ])).height
            counts['direct_none_daily_volume_mismatch'] += audit_native_volume(confirmed)
            missing.difference_update(verified[ID].to_list())
            counts['direct_none'] += verified.height
            return confirmed.select(*STOCK_TIME, *QUOTE)

        if missing:
            for path in sorted(directory.glob('*.feather')):
                if not missing:
                    break
                records.append(reconcile(pl.read_ipc(path, memory_map=False)))
            for batch in batched(sorted(missing), MINUTE_BATCH_SIZE):
                direct = self.direct_none(date, list(batch))
                if direct.empty:
                    continue
                verified = reconcile(pl.from_pandas(direct, nan_to_null=False))
                if not verified.is_empty():
                    key = sha256(','.join(batch).encode()).hexdigest()
                    self.write_frame(verified, directory / f'{key}.feather')
                    records.append(verified)

        combined = pl.concat(records, how='vertical_relaxed')
        rebuilt = (
            raw.drop(QUOTE).join(combined, on=STOCK_TIME, validate='1:1', maintain_order='left')
            .with_columns(pl.lit('none').alias(bar_min.ADJUST_COLUMN))
            .select(list(dict.fromkeys([*raw_columns, bar_min.ADJUST_COLUMN])))
        )
        return rebuilt, counts, sorted(missing)

    def stage(self, date, restored: pl.DataFrame, counts, *, preserve_raw: bool = False,
              revision: str = REPAIR, stocks: pl.DataFrame | None = None):
        valid_raw_day(restored, date)
        validate_values(restored, date)
        if stocks is None:
            stocks = self.stock_mapping(self.require_record('scan', date)['mapping'])
        minute = normalize_bars(restored, stocks)
        if (minute['secid'].null_count() or minute.select('secid', 'minute').n_unique() != len(minute)):
            raise ValueError(f'{date}: invalid or overlapping normalized stock-minute keys')
        outputs = [] if preserve_raw else [(self.minute_path(date), restored)]
        reference = minute.group_by('secid').agg(pl.col(TOTALS).sum())
        evidence = []
        counts = counts.copy()
        counts['vwap_audit_1min'] = 0
        counts['vwap_audit_xmin'] = 0
        for period in bar_min.SEC_PERIODS:
            frame = minute if period == 1 else aggregate_bars(minute, period)
            compared = (frame.group_by('secid').agg(pl.col(TOTALS).sum())
                        .join(reference, on='secid', suffix='_ref', validate='1:1'))
            if compared.height != reference.height or compared.select(pl.any_horizontal(
                (pl.col(name) - pl.col(f'{name}_ref')).abs()
                > 1e-6 + pl.col(f'{name}_ref').abs() * 1e-12 for name in TOTALS
            ).any()).item():
                raise ValueError(f'{date}: inconsistent {period}-minute totals')
            audit = vwap_audit(frame, period)
            if not audit.is_empty():
                evidence.append(audit)
                counts['vwap_audit_1min' if period == 1 else 'vwap_audit_xmin'] += audit.height
            outputs.append((DBPath('trade_ts', bar_min.src_key(SEC, period)).path_exact(date), frame))
        audit_path = (self.state / revision if revision != REPAIR else self.state) / 'vwap_audit' / f'{date}.feather'
        if evidence or audit_path.exists():
            self.write_frame(pl.concat(evidence) if evidence else audit, audit_path, empty_ok=True)

        prepared = []
        required = {}
        for target, frame in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            device = target.parent.stat().st_dev
            size = frame_bytes(frame)
            required[device] = (target.parent, required.get(device, (None, 0))[1] + int(size))
        for directory, size in required.values():
            self.check_space(directory, size, operation=f'{date} stage RCQuant bars')
        for target, frame in outputs:
            pending = target.parent / f'.{revision}_stage' / target.name
            compression = RAW_IPC if target == self.minute_path(date) else BAR_IPC
            self.write_frame(frame, pending, ipc_compression=compression)
            prepared.append({'target': str(target), 'staged': str(pending), 'digest': fingerprint(pending),
                             'ipc_compression': compression})
        record = {'outputs': prepared, 'counts': counts, 'preserve_raw': preserve_raw}
        self.mark(f'{revision}_prepared', date, record)
        return record

    def publish(self, date: int, prepared: dict):
        raw = self.minute_path(date)
        expected = self.require_record('scan', date)['raw']
        if prepared.get('preserve_raw'):
            if not file_matches(raw, expected):
                raise RuntimeError(f'{date}: native minute archive changed since inventory')
        else:
            backup = self.backup_path(date)
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists():
                if not file_matches(raw, expected):
                    raise RuntimeError(f'{date}: original minute changed since inventory')
                os.link(raw, backup)
                sync_directory(backup.parent)
                sync_directory(backup.parent.parent)
            if not file_matches(backup, expected):
                raise RuntimeError(f'{date}: invalid pre-migration backup')

        self.publish_files(date, prepared)
        self.mark(self.repair_stage('done'), date, prepared['counts'])

    def publish_files(self, date: int, prepared: dict, revision: str = REPAIR):
        for item in prepared['outputs']:
            target, pending = Path(item['target']), Path(item['staged'])
            digest = item['digest']
            if file_matches(target, digest):
                if file_matches(pending, digest):
                    pending.unlink()
                    sync_directory(pending.parent)
            elif pending.exists():
                if not file_matches(pending, digest):
                    raise RuntimeError(f'{date}: staged file changed: {pending}')
                os.replace(pending, target)
                sync_directory(target.parent)
                sync_directory(pending.parent)
            else:
                raise RuntimeError(f'{date}: cannot reconcile interrupted publication: {target}')
            self.mark(f'{revision}_published', str(date) + '/' + target.name, digest)
        for period in bar_min.SEC_PERIODS:
            DBPath('trade_ts', bar_min.src_key(SEC, period)).clear_date_cache()

    def migrate_date(self, date: int, events: FactorHistory, *, previously_written: bool = False) -> bool:
        if self.read(self.repair_stage('done'), date) is not None:
            return True
        with date_lock(self.root, SEC, date):
            prepared = self.read(self.repair_stage('prepared'), date)
            if prepared is None:
                self.check_space(self.minute_path(date).parent, 0, operation=f'{date} restore secmin')
                source = self.require_record('scan', date)
                try:
                    raw_path = self.original_path(date, previously_written=previously_written)
                    raw = pl.read_ipc(raw_path, memory_map=False).sort(KEYS).with_columns(pl.col(QUOTE).cast(pl.Float64))
                    valid_raw_day(raw, date)
                    stocks = self.stock_mapping(source['mapping'])
                    refs = daily_reference(stocks, DB.load_pl('trade_ts', 'day', date, vb_level='never'))
                    self.log_price_check(date, raw, stocks, refs, from_backup=previously_written)
                    restored, counts, missing = self.prepare_raw(date, raw, stocks, refs, events)
                    self.mark(self.repair_stage('pending_none'), date, missing)
                    if missing:
                        Logger.warning(f'RCQuant {date}: {len(missing)} stocks still need matching native minute keys')
                        return False
                    self.stage(date, restored, counts,
                               preserve_raw=not previously_written and 'source_none' in counts)
                except (ValueError, pl.exceptions.PolarsError) as exc:
                    if disk_full(exc):
                        raise
                    self.mark(self.repair_stage('blocked'), date, str(exc))
                    Logger.warning(f'RCQuant {date}: repair pending review: {exc}')
                    return False
                prepared = self.require_record(self.repair_stage('prepared'), date)
            self.publish(date, prepared)
            return True

    @staticmethod
    def rebuild_minute_block(frame: str, key: str) -> dict:
        from src.data.preprocess.processors import PrePros
        from src.data.util import DataBlock
        from src.proj.db.io.mmap import ArrayMemoryMap

        with DataBlock.preprocess_lock(key, frame):
            processor = PrePros.get_processor(key, frame, vb_level=1)
            if frame == 'fit':
                processor.DateChunkYears = 1
            canonical = DataBlock.path_preprocess(key, frame, find_if_not_exists=False)
            original = DataBlock.path_preprocess(key, frame)
            block = processor.load_with_extension(reconstruct=True)
            if block.empty or block.last_meaningful_date < block.date[0]:
                raise ValueError(f'RCQuant {frame}/{key}: empty minute DataBlock')

            if original.exists():
                backup = canonical.parent / f'.{REPAIR}_backup' / original.name
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if original.is_dir():
                        staged = backup.with_name(f'.{backup.name}-{uuid4().hex}')
                        shutil.copytree(original, staged, copy_function=os.link)
                        staged.rename(backup)
                    else:
                        os.link(original, backup)
                    sync_directory(backup.parent)
            Save.mmap(block.values, canonical, index={'date': block.date, 'secid': block.secid,
                                                     'feature': block.feature})
            sync_directory(canonical.parent)
            index = ArrayMemoryMap.load(canonical, values=False, index=True)['index']
            if any(index[name].tolist() != getattr(block, name).tolist() for name in ('date', 'secid', 'feature')):
                raise ValueError(f'RCQuant {frame}/{key}: DataBlock index differs after publishing')
            processor.save_norm(block)
            return {'path': str(canonical), 'dates': len(block.date), 'last': int(block.date[-1])}

    def rebuild_datablocks(self, dates: list[int]) -> None:
        version = sha256(','.join(map(str, dates)).encode()).hexdigest()
        stage = self.repair_stage('datablocks')
        remaining = []
        for frame in ('fit', 'predict'):
            for key in MINUTE_DATABLOCKS:
                item = f'{version}/{frame}/{key}'
                record = self.read(stage, item)
                if isinstance(record, dict) and Path(record['path']).is_dir():
                    Logger.stdout(f'RCQuant [7/7] {frame}/{key}: already rebuilt, {record["dates"]} dates')
                else:
                    remaining.append((frame, key, item))
        if not remaining:
            self.mark(self.repair_stage('complete'), 'history', {'dates': len(dates), 'datablocks': version})
            return

        Logger.stdout(f'RCQuant [7/7] Rebuilding {len(remaining)} minute DataBlocks')
        for frame, key, item in remaining:
            self.guard()
            Logger.stdout(f'RCQuant [7/7] Rebuilding {frame}/{key}')
            record = self.rebuild_minute_block(frame, key)
            self.mark(stage, item, record)
            Logger.stdout(f'RCQuant [7/7] {frame}/{key}: saved {record["dates"]} dates through {record["last"]}')
        self.mark(self.repair_stage('complete'), 'history', {'dates': len(dates), 'datablocks': version})
        Logger.stdout('RCQuant [7/7] Historical minute repair and DataBlocks complete')

    def run(self, *, revised_dates=(), rebuild=True):
        from .sec_storage_recovery import transcode_raw

        expected = list(map(int, Dates(START, END)))
        Logger.stdout('RCQuant [1/7] Checking historical inventory and saved stock IDs')
        manifest = self.scan(expected)
        Logger.stdout(f"RCQuant [1/7] Inventory ready: {len(manifest['dates'])} archived dates, "
                      f"{len(manifest['ids'])} stocks, {len(manifest['mapping_errors'])} mapping errors")
        if manifest['mapping_errors']:
            Logger.warning(f"RCQuant unmapped source dates: {manifest['mapping_errors']}")
        if manifest['missing']:
            Logger.warning(f"RCQuant [2/7] Missing raw secmin dates: {manifest['missing']}")
            filled = self.backfill_missing_raw(manifest['missing'])
            if filled:
                Logger.stdout(f'RCQuant [2/7] Raw secmin now present for dates: {sorted(filled)}')
                manifest = self.scan(expected)
            Logger.stdout(f"RCQuant [2/7] Raw secmin dates still missing: {manifest['missing']}")
        else:
            Logger.stdout('RCQuant [2/7] Raw secmin coverage complete')

        Logger.stdout('RCQuant [3/7] Loading equity price and split factors')
        events = self.get_events(manifest)
        Logger.stdout('RCQuant [4/7] Checking daily factor files')
        daily_saved = 0
        for month, month_dates in groupby(sorted(manifest['dates']), key=lambda day: day // 100):
            written = checked = 0
            start = perf_counter()
            try:
                for date in month_dates:
                    self.guard()
                    written += self.daily_factors(date, self.require_record('scan', date)['mapping'], events)
                    checked += 1
            finally:
                daily_saved += written
                if checked:
                    Logger.stdout(f'RCQuant [4/7] {month}: {written} written, {checked-written} reused, '
                                  f'{perf_counter()-start:.1f}s')
        Logger.stdout(f'RCQuant [4/7] Daily factors ready: {daily_saved} written, '
                      f'{len(manifest["dates"])-daily_saved} reused')

        done = {date for date in manifest['dates'] if self.read(self.repair_stage('done'), date) is not None}
        done.update(set(revised_dates) & set(manifest['dates']))
        self.history_progress = len(done), len(expected)
        unfinished = [date for date in manifest['dates'] if date not in done]
        untouched, previously_written = self.repair_cohorts(unfinished)
        Logger.stdout(f'RCQuant [5/7] Polars restoration: {len(done)} already complete, '
                      f'{len(previously_written)} to restart from backups, {len(untouched)} original dates')
        Logger.stdout('RCQuant [6/7] Publishing raw and six periods per validated date, with independent VWAP audit')
        if unfinished:
            Logger.stdout(f'RCQuant [6/7] Resume at {max(unfinished)}')
        previous_dates = set(previously_written)
        for month, month_dates in groupby(sorted(unfinished, reverse=True), key=lambda day: day // 100):
            published = blocked = 0
            start = perf_counter()
            try:
                for date in month_dates:
                    self.guard()
                    previous = date in previous_dates
                    if previous and not self.backup_path(date).is_file():
                        Logger.warning(f'RCQuant {date}: previous publication has no original backup')
                        blocked += 1
                        continue
                    if self.migrate_date(date, events, previously_written=previous):
                        prepared = self.read(self.repair_stage('prepared'), date)
                        if prepared is not None:
                            transcode_raw(self, date, prepared)
                        done.add(date)
                        published += 1
                        self.history_progress = len(done), len(expected)
                    else:
                        blocked += 1
            finally:
                if published or blocked:
                    Logger.stdout(f'RCQuant [6/7] {month}: {published} published, {blocked} pending, '
                                  f'{len(done)}/{len(expected)} total, {perf_counter()-start:.1f}s')

        pending = sorted(set(manifest['dates']) - done)
        Logger.stdout(f'RCQuant [6/7] Published {len(done)} of {len(manifest["dates"])} archived dates, '
                      f'{len(pending)} pending')
        if not done or pending or manifest['missing'] or manifest['mapping_errors']:
            Logger.stdout('RCQuant [7/7] Minute DataBlocks waiting for historical publication to finish')
            return None
        if rebuild:
            self.rebuild_datablocks(sorted(done))
        return sorted(done)


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
    from src.automation.nightly_maintenance import run as run_nightly

    return run_nightly()
