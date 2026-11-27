"""Checkpointed publication of whole-market unadjusted stock minutes."""
from __future__ import annotations

__all__ = [
    'DATE',
    'REVISION',
    'repair_native_day',
    'redownload_flagged_days',
]

from hashlib import sha256
from itertools import groupby
from pathlib import Path
import os
from time import perf_counter

import polars as pl

from src.proj import DB, Logger

from . import bar_min
from .initializer import QuotaExceeded
from .storage import RAW_IPC
from .sec_history_polars import ID, QUOTE, TOTALS, DAILY_VOLUME_TOLERANCE, daily_reference
from .sec_none_migration import (
    KEYS, MINUTE_BATCH_SIZE, SEC, Migration, date_lock, file_matches, fingerprint,
    sync_directory, to_order_book_ids, valid_quote, valid_raw_day, write_frame,
)

DATE = 20260608
REVISION = 'native_20260608_v1'

FLAGGED_REVISION = 'native_volume_warning_days_v1'
# Distinct dates in the completed November 27 history run (task 1795762688).
FLAGGED_DATES = (
    20070122, 20070329, 20070402, 20070413, 20070416, 20070417, 20070418, 20070419,
    20070423, 20070508, 20070523, 20070524, 20070601, 20070613, 20070615, 20071107,
    20080514, 20080519, 20080708, 20080709, 20080718, 20080807, 20081024, 20081027,
    20100317, 20100512, 20100514, 20111219, 20130621, 20130926, 20130927, 20131111,
    20131113, 20131205, 20131225, 20131226, 20140313, 20140530, 20140818, 20150504,
    20150506, 20150507, 20150511, 20150512, 20150513, 20150514, 20150515, 20150518,
    20150519, 20150520, 20150522, 20150525, 20150526, 20150527, 20150528, 20150529,
    20150601, 20150602, 20150603, 20150604, 20150605, 20150608, 20150611, 20150612,
    20150615, 20150616, 20150617, 20150618, 20150619, 20150623, 20150624, 20150625,
    20150626, 20150629, 20150630, 20150701, 20150702, 20150703, 20150706, 20150707,
    20150708, 20150710, 20150713, 20150714, 20150715, 20150716, 20150717, 20150720,
    20150722, 20150723, 20150724, 20150727, 20150728, 20150729, 20150730, 20150731,
    20150803, 20150804, 20150805, 20150806, 20150807, 20150810, 20150811, 20150812,
    20150813, 20150814, 20150817, 20150818, 20150819, 20150820, 20150821, 20150824,
    20150825, 20150826, 20150827, 20150828, 20150831, 20150901, 20150902, 20150907,
    20150908, 20150909, 20150910, 20150911, 20150914, 20150915, 20150916, 20150917,
    20150918, 20150921, 20150922, 20150923, 20150924, 20150925, 20150928, 20150929,
    20150930, 20151008, 20151009, 20151012, 20151013, 20151014, 20151015, 20151016,
    20151019, 20151020, 20151021, 20151022, 20151023, 20151026, 20151027, 20151029,
    20151030, 20151102, 20151103, 20151104, 20151105, 20151106, 20151109, 20151110,
    20151111, 20151112, 20151113, 20151116, 20151117, 20151118, 20151119, 20151120,
    20151123, 20151124, 20151125, 20151126, 20151127, 20151201, 20151202, 20151203,
    20151204, 20151207, 20151208, 20151209, 20151210, 20151211, 20161207,
)

def _usable(frame: pl.DataFrame, expected: pl.DataFrame, date: int) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    at = pl.col('datetime')
    clock = at.dt.hour().cast(pl.Int64) * 60 + at.dt.minute()
    in_session = clock.is_between(571, 690) | clock.is_between(781, 900)
    session_date = pl.date(date // 10000, date // 100 % 100, date % 100)
    keys_ok = (at.dt.date() == session_date) & (at == at.dt.truncate('1m')) & in_session
    coverage = (frame.group_by(ID).agg(
        pl.len().alias('bars'), at.n_unique().alias('unique'),
        (keys_ok & valid_quote()).fill_null(False).all().alias('valid'),
    ).join(expected.group_by(ID).len(name='old_bars'), on=ID, how='left'))
    missing_keys = expected.join(frame.select(ID, 'datetime'), on=[ID, 'datetime'], how='anti')
    good = coverage.filter(
        pl.col('valid') & (pl.col('bars') == pl.col('unique'))
        & (pl.col('bars') >= pl.col('old_bars').fill_null(240))
    ).join(missing_keys.select(ID).unique(), on=ID, how='anti')
    return frame.join(good.select(ID), on=ID, how='semi')


def _download(job: Migration, directory: Path, manifest: dict, expected: pl.DataFrame, *,
              date: int = DATE, revision: str = REVISION) -> pl.DataFrame | None:
    frames = []
    requested = 0
    started = perf_counter()
    for i, batch in enumerate(manifest['batches']):
        job.guard()
        wanted = expected.filter(pl.col(ID).is_in(batch))
        path = directory / 'download' / f'{i:04d}.feather'
        saved = pl.DataFrame(schema={ID: pl.String, 'datetime': expected.schema['datetime'],
                                     **dict.fromkeys(QUOTE, pl.Float64)})
        # Retain the original numeric batch keys for the released June 8 repair.
        batch_key = i if revision == REVISION else f'{date}/{i}'
        receipt = job.read(f'{revision}_batch', batch_key)
        if path.is_file() and (not receipt or not receipt['digest'] or file_matches(path, receipt['digest'])):
            try:
                saved = _usable(pl.read_ipc(path, memory_map=False), wanted, date)
            except pl.exceptions.PolarsError:
                Logger.warning(f'RCQuant {date}: unreadable native download batch {i}')
        missing = sorted(set(batch) - set(saved[ID].to_list()))
        if missing:
            job.check_space(directory, 0, operation=f'{date} download native batch {i + 1}')
            requested += 1
            response = job.direct_none(date, missing)
            if not response.empty:
                incoming = pl.from_pandas(response, nan_to_null=False).select(
                    ID, pl.col('datetime').cast(expected.schema['datetime']), pl.col(QUOTE).cast(pl.Float64),
                )
                incoming = _usable(incoming, wanted.filter(pl.col(ID).is_in(missing)), date)
                saved = pl.concat([saved, incoming]).sort(ID, 'datetime')
                # Store a durable batch before its receipt. A pending receipt lets a
                # subsequent run adopt the exact file written before interruption.
                job.mark(f'{revision}_batch', batch_key, {'digest': None})
                job.write_frame(saved, path, empty_ok=True, ipc_compression=RAW_IPC)
        if path.is_file() and not saved.is_empty():
            job.mark(f'{revision}_batch', batch_key, {'digest': fingerprint(path), 'stocks': saved[ID].n_unique()})
        frames.append(saved)
    if revision == REVISION:
        Logger.stdout(f'RCQuant {date}: checked {len(frames)} native batches, '
                      f'{requested} fetched, {perf_counter()-started:.1f}s')
    full = pl.concat(frames)
    missing = sorted(set(manifest['ids']) - set(full[ID].to_list()))
    if missing:
        job.mark(f'{revision}_missing', date, missing)
        Logger.warning(f'RCQuant {date}: {len(missing)} stocks still lack valid native minute data, '
                       f'first IDs: {missing[:10]}')
        return None
    job.mark(f'{revision}_missing', date, [])
    return full


def _audit(directory, old, native, stocks, old_mapping, *, save=write_frame):
    names = ['open_first', 'high_max', 'low_min', 'close_last', *TOTALS, 'bars']
    def one(bars):
        return bars.group_by(ID).agg(
            pl.col('open').first().alias('open_first'), pl.col('high').max().alias('high_max'),
            pl.col('low').min().alias('low_min'), pl.col('close').last().alias('close_last'),
            pl.col(TOTALS).sum(), pl.len().cast(pl.Int64).alias('bars'),
        )
    original = old.with_columns(pl.col('code').replace_strict(old_mapping).alias(ID)).sort(ID, 'datetime')
    reference = daily_reference(stocks, DB.load_pl('trade_ts', 'day', DATE, vb_level='never')).filter(
        pl.col('volume_day').is_finite()
    )
    delta = (
        one(original).join(one(native), on=ID, suffix='_native', how='full', coalesce=True)
        .join(reference.select(ID, 'volume_day'), on=ID, how='left')
        .with_columns(*[(pl.col(f'{name}_native') - pl.col(name)).alias(f'{name}_delta') for name in names],
                      (pl.col('volume_native') - pl.col('volume_day')).alias('volume_vs_tushare'))
        .with_columns(pl.when(pl.col('volume_day') > 0)
                      .then(pl.col('volume_vs_tushare') / pl.col('volume_day')).alias('volume_diff_fraction'),
                      (pl.col('volume') - pl.col('volume_day')).abs().gt(DAILY_VOLUME_TOLERANCE)
                      .alias('old_volume_mismatch'),
                      pl.col('volume_vs_tushare').abs().gt(DAILY_VOLUME_TOLERANCE).alias('native_volume_mismatch'))
        .with_columns(
            pl.when(pl.col('volume_day').is_null()).then(pl.lit('no_reference'))
            .when(pl.col('old_volume_mismatch').fill_null(False))
            .then(pl.when(pl.col('native_volume_mismatch')).then(pl.lit('persisting')).otherwise(pl.lit('resolved')))
            .when(pl.col('native_volume_mismatch')).then(pl.lit('new'))
            .otherwise(pl.lit('matched')).alias('volume_mismatch_transition')
        )
    ).sort(ID)
    minute_delta = (original.select(ID, 'datetime', *QUOTE).join(
        native.select(ID, 'datetime', *QUOTE), on=[ID, 'datetime'], suffix='_native', how='full', coalesce=True,
    ).filter(pl.any_horizontal(~pl.col(name).eq_missing(pl.col(f'{name}_native')) for name in QUOTE)))
    save(delta, directory / 'daily_comparison.feather')
    save(minute_delta, directory / 'minute_differences.feather', empty_ok=True)
    return {'stocks': native[ID].n_unique(), 'changed_minutes': minute_delta.height,
            'old_volume_mismatches': delta['old_volume_mismatch'].sum(),
            'native_volume_mismatches': delta['native_volume_mismatch'].sum()}


def _backup(path: Path, destination: Path) -> None:
    if path.is_file() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, destination)
        sync_directory(destination.parent)


def repair_native_day(job: Migration, date: int = DATE, *, revision: str = REVISION) -> bool:
    from .sec_storage_recovery import published_matches

    directory = job.state / revision
    if revision != REVISION:
        directory /= str(date)
    with date_lock(job.root, SEC, date):
        done = job.read(f'{revision}_done', date)
        if done and len(done['outputs']) == 7 and all(
            published_matches(job, date, item) for item in done['outputs']
        ):
            if revision == FLAGGED_REVISION:
                job.mark(f'{revision}_done', date, _publication_receipt(done))
            else:
                Logger.stdout(f'RCQuant {date}: native revision verified, all 7 files already published')
            return True
        if revision == REVISION:
            Logger.stdout(f'RCQuant {date}: checking native minute download and publication')
        prepared = job.read(f'{revision}_prepared', date)
        archive = job.minute_path(date)
        previous = archive.parent / f'.{revision}_backup' / archive.name
        if prepared is None or not all(
            file_matches(Path(item['target']), item['digest']) or
            file_matches(Path(item['staged']), item['digest']) for item in prepared['outputs']
        ):
            job.guard()
            old = pl.read_ipc(previous if previous.exists() else archive, memory_map=False)
            manifest = job.read(f'{revision}_manifest', date)
            if manifest is None:
                job.check_space(directory, 0, operation=f'{date} download stock list')
                instruments = bar_min.rcquant_instrument_list(date, SEC, raise_on_quota=True)
                if instruments.empty:
                    raise RuntimeError(f'RCQuant {date}: no dated stock universe')
                mapping = to_order_book_ids(old['code'].unique().to_list(), instruments)
                ids = sorted(set(mapping.values()) | set(instruments.loc[instruments['is_active'], 'code']))
                manifest = {'ids': ids, 'mapping': mapping, 'adjust_type': 'none',
                            'batches': [ids[i:i+MINUTE_BATCH_SIZE] for i in range(0, len(ids), MINUTE_BATCH_SIZE)]}
                job.mark(f'{revision}_manifest', date, manifest)
            expected = old.select(pl.col('code').replace_strict(manifest['mapping']).alias(ID), 'datetime')
            native = _download(job, directory, manifest, expected, date=date, revision=revision)
            if native is None:
                return False
            job.guard()
            start = perf_counter()
            code = pl.col(ID).str.replace(r'\.XSHE$', '.SZ').str.replace(r'\.XSHG$', '.SH')
            restored = native.with_columns(code.alias('code'), pl.col('datetime').dt.strftime('%H%M').alias('time'),
                                           pl.lit(str(date)).alias('date'),
                                           pl.lit('none').alias(bar_min.ADJUST_COLUMN))
            mapping = dict(restored.select('code', ID).unique().iter_rows())
            stocks = job.stock_mapping(mapping)
            counts = {'stocks': native[ID].n_unique()}
            if revision == REVISION:
                counts = _audit(directory, old, native, stocks, manifest['mapping'], save=job.write_frame)
            raw = restored.drop(ID).select(*KEYS, 'time', 'date', *QUOTE, bar_min.ADJUST_COLUMN)
            valid_raw_day(raw, date)
            prepared = job.stage(date, raw, counts, revision=revision, stocks=stocks)
            if revision == REVISION:
                Logger.stdout(f'RCQuant {date}: native raw and six periods staged in {perf_counter()-start:.2f}s')
        job.guard()
        original = job.read('scan', date)
        if original and file_matches(archive, original['raw']):
            _backup(archive, job.backup_path(date))
        for item in prepared['outputs']:
            path = Path(item['target'])
            if not file_matches(path, item['digest']):
                _backup(path, path.parent / f'.{revision}_backup' / path.name)
        # Avoid replacing files whose verified publication survived an interruption.
        job.publish_files(date, prepared, revision=revision)
        if len(prepared['outputs']) != 7 or not all(
            file_matches(Path(item['target']), item['digest']) for item in prepared['outputs']
        ):
            raise RuntimeError(f'RCQuant {date}: native revision publication was not verified')
        receipt = _publication_receipt(prepared) if revision == FLAGGED_REVISION else prepared
        job.mark(f'{revision}_done', date, receipt)
        if revision == REVISION:
            Logger.stdout(f'RCQuant {date}: native revision verified, all 7 files published, {prepared["counts"]}')
        return True


def _publication_receipt(prepared):
    return {**prepared, 'outputs': [
        {**item, 'file_identity': bar_min.minute_identity(Path(item['target']))}
        for item in prepared['outputs']
    ]}


def redownload_flagged_days(job: Migration) -> str | None:
    """Return the identity of the full published set, or defer the downstream replay."""
    identities = []
    stage = f'{FLAGGED_REVISION}_done'
    for month, days in groupby(FLAGGED_DATES, key=lambda day: day // 100):
        updated = reused = 0
        try:
            for date in days:
                job.guard()
                receipt = job.read(stage, date)
                if receipt and len(receipt['outputs']) == 7 and all(
                    Path(item['target']).is_file()
                    and item.get('file_identity') == bar_min.minute_identity(Path(item['target']))
                    for item in receipt['outputs']
                ):
                    reused += 1
                else:
                    try:
                        if not repair_native_day(job, date, revision=FLAGGED_REVISION):
                            return None
                    except QuotaExceeded:
                        Logger.warning(f'Nightly RCQuant native refresh quota exhausted at {date}, '
                                       f'{len(identities)}/{len(FLAGGED_DATES)} dates already published')
                        raise
                    receipt = job.require_record(stage, date)
                    updated += 1
                identities.append(f"{date}:" + ','.join(item['digest'] for item in receipt['outputs']))
        finally:
            if updated:
                Logger.stdout(f'Nightly RCQuant native refresh {month}: {updated} published, {reused} reused, '
                              f'{len(identities)}/{len(FLAGGED_DATES)} total')
    Logger.info(f'Nightly RCQuant native refresh: all {len(identities)} dates verified', vb_level=0)
    return sha256((FLAGGED_REVISION + '|'.join(identities)).encode()).hexdigest()
