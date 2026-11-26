"""One durable whole-market native download for the 2026-06-08 correction."""
from pathlib import Path
import os
from time import perf_counter

import polars as pl

from src.proj import DB, Logger

from . import bar_min
from .sec_history_polars import ID, QUOTE, TOTALS, DAILY_VOLUME_TOLERANCE, daily_reference
from .sec_none_migration import (
    KEYS, MINUTE_BATCH_SIZE, SEC, Migration, date_lock, file_matches, fingerprint,
    sync_directory, to_order_book_ids, valid_quote, valid_raw_day, write_frame,
)

DATE = 20260608
REVISION = 'native_20260608_v1'


def _usable(frame: pl.DataFrame, expected: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    at = pl.col('datetime')
    clock = at.dt.hour().cast(pl.Int64) * 60 + at.dt.minute()
    in_session = clock.is_between(571, 690) | clock.is_between(781, 900)
    keys_ok = (at.dt.date() == pl.date(2026, 6, 8)) & (at == at.dt.truncate('1m')) & in_session
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


def _download(job: Migration, directory: Path, manifest: dict, expected: pl.DataFrame) -> pl.DataFrame | None:
    frames = []
    for i, batch in enumerate(manifest['batches']):
        job.guard()
        wanted = expected.filter(pl.col(ID).is_in(batch))
        path = directory / 'download' / f'{i:04d}.feather'
        saved = pl.DataFrame(schema={ID: pl.String, 'datetime': expected.schema['datetime'],
                                     **dict.fromkeys(QUOTE, pl.Float64)})
        receipt = job.read(f'{REVISION}_batch', i)
        if path.is_file() and (not receipt or not receipt['digest'] or file_matches(path, receipt['digest'])):
            try:
                saved = _usable(pl.read_ipc(path, memory_map=False), wanted)
            except pl.exceptions.PolarsError:
                Logger.warning(f'RCQuant {DATE}: unreadable native download batch {i}')
        missing = sorted(set(batch) - set(saved[ID].to_list()))
        if missing:
            start = perf_counter()
            response = job.direct_none(DATE, missing)
            downloaded = perf_counter()
            if not response.empty:
                incoming = pl.from_pandas(response, nan_to_null=False).select(
                    ID, pl.col('datetime').cast(expected.schema['datetime']), pl.col(QUOTE).cast(pl.Float64),
                )
                incoming = _usable(incoming, wanted.filter(pl.col(ID).is_in(missing)))
                saved = pl.concat([saved, incoming]).sort(ID, 'datetime')
                # Store a durable batch before its receipt. A pending receipt lets a
                # subsequent run adopt the exact file written before interruption.
                job.mark(f'{REVISION}_batch', i, {'digest': None})
                write_frame(saved, path, empty_ok=True)
            Logger.stdout(f'RCQuant {DATE}: batch {i + 1}/{len(manifest["batches"])} '
                          f'download={downloaded-start:.2f}s convert/stage={perf_counter()-downloaded:.2f}s')
        if path.is_file() and not saved.is_empty():
            job.mark(f'{REVISION}_batch', i, {'digest': fingerprint(path), 'stocks': saved[ID].n_unique()})
        frames.append(saved)
    full = pl.concat(frames)
    missing = sorted(set(manifest['ids']) - set(full[ID].to_list()))
    if missing:
        job.mark(f'{REVISION}_missing', DATE, missing)
        Logger.warning(f'RCQuant {DATE}: {len(missing)} stocks still lack valid native minute data')
        return None
    job.mark(f'{REVISION}_missing', DATE, [])
    return full


def _audit(directory, old, native, stocks, old_mapping):
    names = ['open_first', 'high_max', 'low_min', 'close_last', *TOTALS, 'bars']
    def one(bars):
        return bars.group_by(ID).agg(
            pl.col('open').first().alias('open_first'), pl.col('high').max().alias('high_max'),
            pl.col('low').min().alias('low_min'), pl.col('close').last().alias('close_last'),
            pl.col(TOTALS).sum(), pl.len().cast(pl.Int64).alias('bars'),
        )
    original = old.with_columns(pl.col('code').replace_strict(old_mapping).alias(ID)).sort(ID, 'datetime')
    reference = daily_reference(stocks, DB.load_pl('trade_ts', 'day', DATE, vb_level='never'))
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
    write_frame(delta, directory / 'daily_comparison.feather')
    write_frame(minute_delta, directory / 'minute_differences.feather', empty_ok=True)
    return {'stocks': native[ID].n_unique(), 'changed_minutes': minute_delta.height,
            'old_volume_mismatches': delta['old_volume_mismatch'].sum(),
            'native_volume_mismatches': delta['native_volume_mismatch'].sum()}


def _backup(path: Path, destination: Path) -> None:
    if path.is_file() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, destination)
        sync_directory(destination.parent)


def repair_native_day(job: Migration) -> bool:
    directory = job.state / REVISION
    with date_lock(job.root, SEC, DATE):
        done = job.read(f'{REVISION}_done', DATE)
        if done and len(done['outputs']) == 7 and all(
            file_matches(Path(item['target']), item['digest']) for item in done['outputs']
        ):
            Logger.stdout(f'RCQuant {DATE}: native revision verified, all 7 files already published')
            return True
        prepared = job.read(f'{REVISION}_prepared', DATE)
        archive = job.minute_path(DATE)
        previous = archive.parent / f'.{REVISION}_backup' / archive.name
        if prepared is None or not all(
            file_matches(Path(item['target']), item['digest']) or
            file_matches(Path(item['staged']), item['digest']) for item in prepared['outputs']
        ):
            job.guard()
            old = pl.read_ipc(previous if previous.exists() else archive, memory_map=False)
            manifest = job.read(f'{REVISION}_manifest', DATE)
            if manifest is None:
                instruments = bar_min.rcquant_instrument_list(DATE, SEC, raise_on_quota=True)
                if instruments.empty:
                    raise RuntimeError(f'RCQuant {DATE}: no dated stock universe')
                mapping = to_order_book_ids(old['code'].unique().to_list(), instruments)
                ids = sorted(set(mapping.values()) | set(instruments.loc[instruments['is_active'], 'code']))
                manifest = {'ids': ids, 'mapping': mapping, 'adjust_type': 'none',
                            'batches': [ids[i:i+MINUTE_BATCH_SIZE] for i in range(0, len(ids), MINUTE_BATCH_SIZE)]}
                job.mark(f'{REVISION}_manifest', DATE, manifest)
            expected = old.select(pl.col('code').replace_strict(manifest['mapping']).alias(ID), 'datetime')
            native = _download(job, directory, manifest, expected)
            if native is None:
                return False
            job.guard()
            start = perf_counter()
            code = pl.col(ID).str.replace(r'\.XSHE$', '.SZ').str.replace(r'\.XSHG$', '.SH')
            restored = native.with_columns(code.alias('code'), pl.col('datetime').dt.strftime('%H%M').alias('time'),
                                           pl.lit(str(DATE)).alias('date'),
                                           pl.lit('none').alias(bar_min.ADJUST_COLUMN))
            mapping = dict(restored.select('code', ID).unique().iter_rows())
            stocks = job.stock_mapping(mapping)
            counts = _audit(directory, old, native, stocks, manifest['mapping'])
            raw = restored.drop(ID).select(*KEYS, 'time', 'date', *QUOTE, bar_min.ADJUST_COLUMN)
            valid_raw_day(raw, DATE)
            prepared = job.stage(DATE, raw, counts, revision=REVISION, stocks=stocks)
            Logger.stdout(f'RCQuant {DATE}: Polars comparison and six periods staged in {perf_counter()-start:.2f}s')
        job.guard()
        original = job.read('scan', DATE)
        if original and file_matches(archive, original['raw']):
            _backup(archive, job.backup_path(DATE))
        for item in prepared['outputs']:
            path = Path(item['target'])
            if not file_matches(path, item['digest']):
                _backup(path, path.parent / f'.{REVISION}_backup' / path.name)
        # Avoid replacing files whose verified publication survived an interruption.
        job.publish_files(DATE, prepared, revision=REVISION)
        if len(prepared['outputs']) != 7 or not all(
            file_matches(Path(item['target']), item['digest']) for item in prepared['outputs']
        ):
            raise RuntimeError(f'RCQuant {DATE}: native revision publication was not verified')
        job.mark(f'{REVISION}_done', DATE, prepared)
        Logger.stdout(f'RCQuant {DATE}: native revision verified, all 7 files published, {prepared["counts"]}')
        return True
