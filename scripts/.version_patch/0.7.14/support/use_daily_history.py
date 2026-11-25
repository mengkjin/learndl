"""Restore production tracking history and resume use_daily from its actual prior portfolio."""
import io
import re
import shutil
import sqlite3
from email import policy
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.automation.version_patch import load_patch_support
from src.proj import CALENDAR, MACHINE, PATH, Save
from src.shared.file_ops import atomic_json, real_tree

NAME = 'use_daily'
START = 20261124
SEED = 20261123
COLUMNS = ['secid', 'weight', 'value']


def old_attempts(ctx):
    ledger = ctx.backup_dir.parents[2] / 'ledger.sqlite'
    with sqlite3.connect(f'{ledger.as_uri()}?mode=ro', uri=True) as db:
        return db.execute("SELECT log_path, started FROM attempts WHERE patch='0.7.11/001' ORDER BY attempt").fetchall()


def mail_rows(needed, cutoff):
    """Read successful pre-replay daily-update attachments, latest run per holdings date."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    chosen = {}
    roots = [PATH.email_outbox, PATH.rslt_external / 'mail_archive' / MACHINE.name / 'daily_update']
    for root in roots:
        for path in root.rglob('message.eml'):
            with path.open('rb') as stream:
                parser = BytesParser(policy=policy.default)
                headers = parser.parse(stream, headersonly=True)
                match = re.search(r'Success - Daily Update - (\d{14})', str(headers.get('Subject', '')), re.I)
                if not match:
                    continue
                stamp = datetime.strptime(match[1], '%Y%m%d%H%M%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
                if stamp.timestamp() >= cutoff:
                    continue
                stream.seek(0)
                message = parser.parse(stream)
            for attachment in message.iter_attachments():
                match = re.fullmatch(r'trading_ports\.(\d{8})\.csv', attachment.get_filename() or '')
                if not match:
                    continue
                date = int(match[1])
                if not 20260101 <= date <= SEED or (needed is not None and date not in needed):
                    continue
                payload = attachment.get_payload(decode=True)
                frame = pd.read_csv(io.BytesIO(payload), float_precision='round_trip')
                rows = portfolio_rows(frame, date)
                if rows.empty:
                    continue
                if date not in chosen or stamp > chosen[date][0]:
                    chosen[date] = stamp, rows
                elif chosen[date][0] == stamp and not rows.reset_index(drop=True).equals(chosen[date][1].reset_index(drop=True)):
                    raise ValueError(f'Conflicting mail for {NAME} on {date}')
    return {date: rows for date, (_, rows) in chosen.items()}


def portfolio_rows(frame, date):
    return frame.loc[(frame.name == NAME) & (frame.date == date)].drop(columns=['Unnamed: 0'], errors='ignore')


def restore(ctx, old, source, staged, root, history):
    attempts = old_attempts(ctx)
    backups = [Path(log).parent / 'backup' / 'original' for log, _ in attempts]
    cutoff = attempts[0][1] if attempts else float('inf')
    tracking, outputs, missing = {}, {}, set()
    for date in history:
        target = source.export_path(date)
        generated = staged.export_path(date)
        if date < 20260101:
            shutil.copy2(target, generated)
            continue
        outputs[target] = generated
        csv = PATH.rslt_trade / 'trading_ports' / f'trading_ports.{date}.csv'
        holdings = next((p for b in backups if (p := b / target.relative_to(root)).is_file()), None)
        if holdings:
            shutil.copy2(holdings, generated)
        original_csv = next((p for b in backups if (p := b / csv.relative_to(root)).is_file()), None)
        if original_csv:
            rows = portfolio_rows(pd.read_csv(original_csv, float_precision='round_trip'), date)
            if not rows.empty:
                tracking[date] = {NAME: rows}
        if not holdings or (csv.is_file() and date not in tracking):
            missing.add(date)
    # If the old patch has no surviving portfolio backups, also recover dates
    # that are represented only in outbox history.
    backup_available = any((b / source.portfolio_dir.relative_to(root)).is_dir() for b in backups)
    mail = mail_rows(missing if backup_available else None, cutoff) if missing or not backup_available else {}
    for date in mail.keys() - set(history):
        outputs[source.export_path(date)] = staged.export_path(date)
        missing.add(date)
    for date in sorted(missing):
        if date in mail:
            tracking.setdefault(date, {NAME: mail[date]})
        if date not in tracking:
            raise ValueError(f'Missing original {NAME} holdings or CSV for {date} in backup and mail outbox')
        generated = staged.export_path(date)
        if not generated.exists():
            tracking[date][NAME][COLUMNS].reset_index(drop=True).to_feather(generated)
    for target, generated in outputs.items():
        old._check_holdings(generated)
        date = int(target.stem.split('.')[-1])
        if date in tracking:
            a = pd.read_feather(generated)[COLUMNS].sort_values('secid').reset_index(drop=True)
            b = tracking[date][NAME][COLUMNS].sort_values('secid').reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-12, atol=1e-12)
    return outputs, tracking


def apply(ctx):
    old = load_patch_support('0.7.11', 'sellside_portfolios', root=ctx.root)
    if not MACHINE.platform_server:
        atomic_json(ctx.backup_dir / old.JOURNAL, {'status': 'receiver', 'outputs': {}})
        return
    root = real_tree(ctx.root)
    if root != real_tree(PATH.production):
        raise ValueError('Expected production checkout')
    if old.verify(ctx):
        return
    source = old.TrackingPort.load(NAME)
    history = list(source.stored_dates(end=SEED))
    dates = list(source.stored_dates(start=START))
    if not dates or dates[0] != START or dates[-1] > CALENDAR.updated():
        raise ValueError('Expected November 23 seed, November 24 holdings and market coverage through latest holdings')
    with TemporaryDirectory(dir=ctx.backup_dir) as directory:
        stage = Path(directory)
        staged = old.StagedTrackingPort.load(NAME)
        staged.root, staged.stage = root, stage
        staged.Universe = old.ReplayUniverse(staged.universe)
        staged.portfolio_dir.mkdir(parents=True)
        history = sorted(set(history) | {SEED})
        outputs, tracking = restore(ctx, old, source, staged, root, history)
        old._check_vendor_history(old.affected_factors(source.candidate_ports[NAME]), dates[-1])
        rows = old._tracking_replay(staged, dates, staged.get_last_port(START))
        for date, frame in rows.items():
            generated = staged.export_path(date)
            old._check_holdings(generated)
            outputs[source.export_path(date)] = generated
            tracking[date] = {NAME: frame}
        staged.analyze(key_fig='')
        Save.async_wait_all(raise_first_error=True, caller_name='use_daily_history')
        for key in ('result_path_account', 'result_path_data', 'result_path_plot'):
            generated = getattr(staged, key)
            if not generated.is_file() or not generated.stat().st_size or not old.production.finished_report(generated):
                raise ValueError(f'Missing rebuilt result {generated}')
            outputs[getattr(source, key)] = generated
        outputs.update(old._stage_csv(root, stage, tracking))
        old._install(outputs, ctx, root)
    if not old.verify(ctx):
        raise ValueError('Installed use_daily history failed verification')
    old._publish(outputs)


def verify(ctx):
    return load_patch_support('0.7.11', 'sellside_portfolios', root=ctx.root).verify(ctx)
