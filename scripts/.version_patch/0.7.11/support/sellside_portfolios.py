"""Replay native portfolio output after the Dongfang-to-Huayuan transition."""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.automation.artifact_exchange import production
from src.automation.artifact_exchange.core import digest
from src.automation.console import logger
from src.data.quality.valid_dates import load_valid_dates
from src.proj import CALENDAR, DB, MACHINE, PATH, Save
from src.res.factor.util import Portfolio, Universe
from src.res.trading.trading_port import BacktestPort, TrackingPort
from src.shared.file_ops import atomic_json, fsync_dir, real_tree

FIRST_AFFECTED_DATE = 20260101
JOURNAL = 'sellside-portfolio-replay.json'
AFFECTED = {f'factor@df_scores_v{n}' for n in (0, 2, 3)}


def affected_factors(config):
    names = set()
    for field in ('alpha', 'components', 'sorter', 'screener'):
        values = config.get(field) or []
        if isinstance(values, str):
            values = values.split(',')
        names.update(value.strip().removeprefix('-') for value in values)
    return names & AFFECTED


def _check_vendor_history(factors, end):
    for factor in sorted(factors):
        key = factor.replace('factor@df_', 'huayuan.')
        valid = load_valid_dates(key, required=True)
        if valid is None or not any(DB.path('sellside', key, int(date)).is_file()
                                    for date in valid[(valid >= FIRST_AFFECTED_DATE) & (valid <= end)]):
            raise ValueError(f'Missing Huayuan replacement history for {factor} through {end}')


class ReplayUniverse(Universe):
    def get_universe_df(self, date, *, rebuild=False):
        return super().get_universe_df(date, rebuild=rebuild, write_cache=False)


class StagedPaths:
    """Use the normal trading writers with their outputs redirected to staging."""

    def stage_path(self, path):
        return self.stage / path.relative_to(self.root)

    @property
    def result_dir(self):
        return self.stage_path(super().result_dir)

    @property
    def portfolio_dir(self):
        return self.stage_path(super().portfolio_dir)


class StagedTrackingPort(StagedPaths, TrackingPort):
    pass


class StagedBacktestPort(StagedPaths, BacktestPort):
    def end_date(self):
        return max(self.first_replay_date - 1, super().end_date())


def _tracking_replay(port, dates, prior):
    rows = {}
    port.Alpha.get_alpha_models(dates)
    for date in dates:
        frame = port.build_portfolio(date, last_port=prior, export=True)
        if frame.empty:
            raise ValueError(f'{port.name} produced no holdings for recorded date {date}')
        rows[date] = frame
        prior = Portfolio.from_dataframe(frame, name=port.name)
    return rows


def _check_holdings(path):
    if not path.is_file():
        raise ValueError(f'Missing rebuilt portfolio {path}')
    frame = pd.read_feather(path)
    if (frame.empty or frame['secid'].isna().any() or frame['secid'].duplicated().any()
            or not np.isfinite(frame[['weight', 'value']].to_numpy()).all()):
        raise ValueError(f'Invalid rebuilt holdings {path}')


def _replay(port_class, staged_class, name, factors, root, stage):
    source = port_class.load(name)
    dates = list(source.stored_dates(start=FIRST_AFFECTED_DATE))
    if not dates:
        return {}, {}
    if dates[-1] > CALENDAR.updated():
        raise ValueError(f'Core market data does not cover {name} through {dates[-1]}')
    _check_vendor_history(factors, dates[-1])

    staged = staged_class.load(name)
    staged.root, staged.stage = root, stage
    staged.Universe = ReplayUniverse(staged.universe)
    staged.first_replay_date = dates[0]
    staged.portfolio_dir.mkdir(parents=True, exist_ok=True)
    for date in source.stored_dates(end=dates[0] - 1):
        shutil.copy2(real_tree(source.export_path(date)), staged.export_path(date))

    logger.info('Replaying %s from %s through %s (%s existing rebalance dates)',
                name, dates[0], dates[-1], len(dates))
    if source.backtest:
        staged.build_backward(dates[-1])
        tracking = {}
    else:
        tracking = _tracking_replay(staged, dates, source.get_last_port(dates[0]))

    generated_dates = set(staged.stored_dates(start=dates[0]))
    missing = sorted(set(dates) - generated_dates)
    if missing:
        raise ValueError(f'{name} did not rebuild recorded dates {missing}')
    outputs = {
        source.export_path(date): staged.export_path(date)
        for date in sorted(generated_dates)
    }
    for path in outputs.values():
        _check_holdings(path)

    # Rebuild from the staged holdings with a fresh account. Resuming an old
    # account would retain returns calculated from the old rebalance chain.
    staged.analyze(key_fig='')
    Save.async_wait_all(raise_first_error=True, caller_name='sellside_portfolio_replay')
    for destination, generated in (
        (source.result_path_account, staged.result_path_account),
        (source.result_path_data, staged.result_path_data),
        (source.result_path_plot, staged.result_path_plot),
    ):
        if generated.is_file() and generated.stat().st_size and production.finished_report(generated):
            outputs[destination] = generated
        elif destination.exists():
            raise ValueError(f'Failed to regenerate existing portfolio result {destination}')
    return outputs, tracking


def _stage_csv(root, stage, tracking):
    outputs = {}
    for date, replacements in tracking.items():
        destination = real_tree(PATH.rslt_trade / 'trading_ports' / f'trading_ports.{date}.csv')
        if not destination.is_file():
            continue
        frame = pd.read_csv(destination, index_col=0, float_precision='round_trip')
        if not frame['name'].isin(replacements).any():
            continue
        updated = [replacements.get(name, group)
                   for name, group in frame.groupby('name', sort=False, dropna=False)]
        generated = stage / destination.relative_to(root)
        generated.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(updated).to_csv(generated)
        outputs[destination] = generated
    return outputs


def _atomic_copy(source, target, *, backup=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix='.sellside-replay-') as directory:
        pending = Path(directory) / target.name
        if backup:
            shutil.copy2(source, pending)
        else:
            # Give regenerated versions their installation timestamp for Syncthing.
            shutil.copyfile(source, pending)
            shutil.copymode(target if target.exists() else source, pending)
        with pending.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(pending, target)
        fsync_dir(target.parent)


def _install(outputs, ctx, root):
    manifest = {}
    for destination, generated in outputs.items():
        target = real_tree(destination)
        relative = production.relative_path(target.relative_to(root).as_posix()).as_posix()
        manifest[relative] = digest(generated)
        backup = ctx.backup_dir / 'original' / relative
        if target.is_file() and not backup.exists():
            _atomic_copy(target, backup, backup=True)
    journal = ctx.backup_dir / JOURNAL
    atomic_json(journal, {'status': 'installing', 'outputs': manifest})
    for target, generated in outputs.items():
        _atomic_copy(generated, target)
    atomic_json(journal, {'status': 'complete', 'outputs': manifest})


def _publish(outputs):
    if not outputs:
        return
    try:
        active = production.publisher()
        if active is not None:
            for record in production.publish(outputs, *active):
                if record['status'] != 'published':
                    logger.warning('Production delivery pending: %s', record.get('error') or record.get('reason'))
    except Exception as exc:
        logger.warning('Production delivery deferred to daily reconciliation: %s', exc)


def apply(ctx):
    # The server holds the sellside and market history. Receivers get the
    # corrected outputs through the normal production artifact channel.
    if not MACHINE.platform_server:
        atomic_json(ctx.backup_dir / JOURNAL, {'status': 'receiver', 'outputs': {}})
        return
    root = real_tree(PATH.production)
    if real_tree(ctx.root) != root:
        raise ValueError(f'Portfolio replay requires the production checkout {root}')
    if verify(ctx):
        return

    tracking = defaultdict(dict)
    outputs = {}
    with TemporaryDirectory(dir=ctx.backup_dir, prefix='replay-') as directory:
        stage = Path(directory)
        for port_class, staged_class in ((TrackingPort, StagedTrackingPort), (BacktestPort, StagedBacktestPort)):
            for name, config in port_class.candidate_ports.items():
                factors = affected_factors(config)
                if not factors:
                    continue
                generated, rows = _replay(port_class, staged_class, name, factors, root, stage)
                outputs.update(generated)
                for date, frame in rows.items():
                    tracking[date][name] = frame
        outputs.update(_stage_csv(root, stage, tracking))
        _install(outputs, ctx, root)
    if not verify(ctx):
        raise ValueError('Installed portfolio replay files failed verification')
    logger.info('Verified %s rebuilt portfolio files', len(outputs))
    _publish(outputs)


def verify(ctx):
    journal = ctx.backup_dir / JOURNAL
    if not journal.is_file():
        return False
    state = json.loads(journal.read_text())
    if state['status'] == 'receiver':
        return not MACHINE.platform_server
    if state['status'] != 'complete':
        return False
    root = real_tree(ctx.root)
    return all((target := real_tree(root / production.relative_path(relative))).is_file() and digest(target) == expected
               for relative, expected in state['outputs'].items())
