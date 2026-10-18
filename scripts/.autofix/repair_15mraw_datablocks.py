# description: Repair 15mraw DataBlocks, then request gru_15mraw training
# content: Back up and rebuild invalid dumps; request GPU training only after both frames pass.
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: Inspect both frames without writing data, reports or training requests

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import torch

from src.proj import CALENDAR, PATH, Logger
from src.proj.util.script import ScriptTool
from src.data import DataBlock
from src.data.preprocess import PrePros

FRAMES = ('fit', 'predict')
EXPECTED_FEATURES = ['open', 'close', 'high', 'low', 'vwap', 'volume']
TRAINING_REQUEST_ID = '20261017_repair_15mraw_train_02'


def block_issue(block: DataBlock) -> str | None:
    """Detect the legacy turn_fl schema and unusable dumps with bounded scan memory."""
    expected = EXPECTED_FEATURES
    if block.empty:
        return 'missing or empty DataBlock'
    if block.shape[2] != 16 or list(block.feature) != expected:
        return f'invalid schema: shape={block.shape}, features={list(block.feature)}'
    if len(np.unique(block.secid)) != len(block.secid) or not np.all(np.diff(block.date) > 0):
        return 'duplicate secid or unordered/duplicate dates'
    finite_features = torch.zeros(len(expected), dtype=torch.bool)
    for start in range(0, len(block.date), 64):
        values = block.values[:, start:start + 64]
        if torch.isinf(values).any():
            return 'infinite values'
        finite_features |= torch.isfinite(values).any(dim=(0, 1, 2)).cpu()
    if not finite_features.all():
        return f'all-NaN features: {np.asarray(expected)[~finite_features.numpy()].tolist()}'
    return None


def repair_datablocks(dry_run: bool = False, *, prepare_block=None, configure_processor=None):
    cutoff = int(CALENDAR.updated())
    plan = []
    for frame in FRAMES:
        proc = PrePros.get_processor('15mraw', frame, vb_level='never')
        try:
            block = proc.load_dump()
            reason = block_issue(block)
            del block
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, AssertionError) as exc:
            reason = f'unreadable dump: {type(exc).__name__}: {exc}'
        plan.append({'frame': frame, 'reason': reason,
                     'path': str(DataBlock.path_preprocess('15mraw', frame))})
        Logger.note(f'15mraw/{frame}: {reason or "healthy; skip"}')
    targets = [item for item in plan if item['reason'] is not None]
    if dry_run or not targets:
        return plan

    folder = PATH.interim / 'autofix_backups' / (
        '15mraw_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    report_path = folder / 'report.json'
    report = {'cutoff': cutoff, 'status': 'running', 'frames': plan}

    def save_report():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(report_path)

    save_report()
    Logger.note(f'15mraw backup/report: {report_path}')
    try:
        for item in targets:
            frame = item['frame']
            source = DataBlock.path_preprocess('15mraw', frame)
            backup = folder / frame / source.name
            item['backup'] = str(backup) if source.exists() else None
            item['status'] = 'backing_up'
            save_report()
            if source.exists():
                backup.parent.mkdir(parents=True)
                if source.is_dir():
                    shutil.copytree(source, backup)
                else:
                    shutil.copy2(source, backup)
            item['status'] = 'rebuilding'
            save_report()
            proc = PrePros.get_processor('15mraw', frame, vb_level='never')
            proc.load_end = cutoff
            proc.DateChunkYears = 1
            if configure_processor is not None:
                configure_processor(proc, item, folder)
            block = proc.load_with_extension(reconstruct=True)
            if prepare_block is not None:
                prepare_block(block, proc, item, folder)
            if reason := block_issue(block):
                raise ValueError(f'15mraw/{frame} rebuilt block: {reason}')
            expected_dates = CALENDAR.range(proc.load_start, cutoff)
            if not np.array_equal(block.date, expected_dates):
                raise ValueError(f'15mraw/{frame}: rebuilt dates do not cover the requested '
                                 f'{proc.load_start}..{cutoff}; check raw 15min data availability')
            shape, dates, secids = tuple(block.shape), block.date.copy(), block.secid.copy()
            item['status'] = 'saving'
            save_report()
            proc.save_dump(block)
            del block
            saved = proc.load_dump()
            if reason := block_issue(saved):
                raise ValueError(f'15mraw/{frame} saved block: {reason}')
            if tuple(saved.shape) != shape or not np.array_equal(saved.date, dates) or not np.array_equal(saved.secid, secids):
                raise ValueError(f'15mraw/{frame}: saved coordinates/shape differ from rebuilt block')
            del saved
            item.update(status='verified', shape=shape, start=int(dates[0]), end=int(dates[-1]))
            save_report()
        report['status'] = 'success'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_report()
    return plan


def request_training(default_request_id: str = TRAINING_REQUEST_ID):
    # Autofix has a private /dev and an exclusive maintenance lease. Persist the
    # request; the ordinary idle worker starts training after that lease exits.
    from src.api.calls.worklist_state import WorklistState
    from src.api.task_monitor.scheduling.maintenance_gate import sandbox_request
    context = sandbox_request()
    request_id = context['id'] if context is not None else default_request_id
    request = WorklistState('gru_15mraw').request_training(request_id)
    Logger.note(f'gru_15mraw training requested: {request["id"]}; '
                'waiting for maintenance completion and an idle GPU')
    return request


def run_task(dry_run: bool = False):
    plan = repair_datablocks(dry_run=dry_run)
    if not dry_run:
        request_training()
    return plan


@ScriptTool('autofix_repair_15mraw_datablocks')
def main(dry_run: bool = False, **kwargs):
    run_task(dry_run=dry_run)


if __name__ == '__main__':
    main()
