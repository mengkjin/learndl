"""Read-only audit/repack of a finished walk-forward suite; never train/load models.

The snapshot is opened only for its date arrays, not market features or a DB.
All new artifacts live in a fresh recovery directory; original results stay intact.
"""
from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import _sha256, create_bundle
from .task_mail import save_json

ARMS = ('original', 'penalty', 'retention')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_suite(run, *, project_root=None, log_path=None):
    run = Path(run).resolve()
    audit = {'run': str(run), 'complete': False, 'errors': [], 'training_tasks_verified': 0,
             'test_paths_verified': 0, 'scope': 'Artifact/date/numerical consistency; no model replay or DB access.'}
    errors = audit['errors']
    try:
        status, config, plan = [read_json(run/name) for name in ('status.json', 'config.json', 'plan.json')]
        if status.get('status') != 'success' or status.get('stage') not in ('all', 'quarters'):
            raise ValueError(f'Not a completed testing stage: {status}')
        quarters = pd.period_range(config['first_quarter'], config['last_quarter'], freq='Q')
        names = [str(q) for q in quarters]
        if not names or [f['name'] for f in plan['folds']] != names:
            raise ValueError('Quarter plan/config mismatch')
        selected = run/'selected.json'
        if _sha256(selected) != read_json(run/'selection_fingerprint.json')['sha256']:
            raise ValueError('Frozen selection fingerprint mismatch')
        if set(read_json(selected)) != set(ARMS):
            raise ValueError('Missing selected experiment arms')
        screen = [(run/'screen'/f'p{p:g}-b{b:g}-s{s}', config['screen_steps'], s)
                  for p, b, s in itertools.product(config['penalties'], config['bonuses'], config['seeds'])]
        formal = [(run/'quarters'/q/arm/f'seed-{s}', config['quarter_steps'], s)
                  for q, arm, s in itertools.product(names, ARMS, config['seeds'])]
        expected = len(screen)+len(formal)
        audit.update(expected_training_tasks=expected, quarters=names, seeds=config['seeds'])
        if plan['total_training_tasks'] != expected:
            raise ValueError('Training task count disagrees with config')
        for folder, budget, seed in screen + formal:
            try:
                task = read_json(folder/'task_status.json')
                if task['status'] != 'success':
                    raise ValueError('training task not successful')
                for name in ('metrics.json', 'normalizer.npz', 'best_model.zip', 'final_model.zip'):
                    if _sha256(folder/name) != task['artifacts'][name]:
                        raise ValueError(f'artifact hash mismatch: {name}')
                metrics = read_json(folder/'metrics.json')
                if metrics['actual_timesteps'] < budget or metrics['completed_updates'] <= 0:
                    raise ValueError('PPO did not reach configured budget')
                if metrics['config']['seed'] != seed or metrics['config']['total_timesteps'] != budget:
                    raise ValueError('Training config disagrees with planned seed/budget')
                if not np.isfinite(metrics['validation']['final_nav']):
                    raise ValueError('Invalid validation result')
                audit['training_tasks_verified'] += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f'{folder.relative_to(run)}: {exc}')
        keys = [f'{b}/{float(c):g}' for b, c in itertools.product(('robust', 'equal', 'retention_rule'), config['costs_bp'])]
        keys += [f'{a}/{s}/{float(c):g}/{m}' for a, s, c, m in
                 itertools.product(ARMS, config['seeds'], config['costs_bp'], ('rolling', 'fixed'))]
        audit['expected_test_paths'] = len(keys)
        summary = read_json(run/'summary/summary.json')
        if set(summary['metrics']) != set(keys):
            errors.append('Summary metrics do not match all configured strategy/seed/cost paths')
        panel = Path(config['panel']).expanduser()
        if not panel.is_absolute():
            panel = Path(project_root or Path.cwd())/panel
        # Loading two small members of the NPZ does not deserialize any model.
        with np.load(panel, allow_pickle=False) as snapshot:
            ds, es = snapshot['decision_dates'], snapshot['return_end_dates']
        first = int(quarters[0].start_time.strftime('%Y%m%d'))
        end = int((quarters[-1]+1).start_time.strftime('%Y%m%d'))
        mask = (ds >= first) & (ds < end) & (es < end)
        expected_dates = np.column_stack([ds[mask], es[mask]])
        if not len(expected_dates) or np.any(np.diff(expected_dates[:, 0]) <= 0):
            raise ValueError('Snapshot contains empty/invalid test calendar')
        audit['test_dates'] = {'first_decision': int(expected_dates[0, 0]),
                               'last_settlement': int(expected_dates[-1, 1]), 'transitions': len(expected_dates)}
        for key in keys:
            try:
                path = run/'summary'/f'{key.replace("/", "_")}.csv'
                rows = pd.read_csv(path)
                dates = rows[['decision_date', 'return_end_date']].to_numpy()
                if not np.array_equal(dates, expected_dates):
                    raise ValueError('Test date coverage differs from snapshot')
                decision_quarters = set(pd.to_datetime(rows.decision_date.astype(str)).dt.to_period('Q').astype(str))
                if decision_quarters != set(names):
                    raise ValueError('Missing test quarter')
                nav, returns = rows.nav.to_numpy(), rows.period_return.to_numpy()
                if not np.isfinite(rows[['nav', 'period_return', 'turnover', 'fee_estimate']].to_numpy()).all() or np.any(nav < 0):
                    raise ValueError('Non-finite or negative test values')
                if not np.allclose(np.cumprod(1+returns), nav, rtol=1e-6, atol=1e-9):
                    raise ValueError('NAV does not reconcile with period returns')
                checks = dict(final_nav=nav[-1], total_return=nav[-1]-1,
                              mean_turnover=rows.turnover.mean(), total_fees=rows.fee_estimate.sum())
                for field, value in checks.items():
                    if not np.isclose(value, summary['metrics'][key][field], rtol=1e-6, atol=1e-9):
                        raise ValueError(f'Summary disagrees with trajectory: {field}')
                audit['test_paths_verified'] += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f'{key}: {exc}')
        if not (run/'summary/report.html').is_file():
            errors.append('Missing offline test report')
        audit['comparison'] = {a: {k: v for k, v in summary['judgments'][a].items() if k != 'bootstrap'} for a in ARMS}
        audit['primary_benchmark'] = summary['metrics'][f'robust/{float(config["primary_cost_bp"]):g}']
        audit['complete'] = not errors
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'Audit stopped: {type(exc).__name__}: {exc}')
    # Bound log inspection; do not touch the global mail outbox or credentials.
    audit['mail_diagnosis'] = {'cause_confirmed': False,
        'known_code_issue': 'Legacy task attached all ZIP parts plus full HTML in one email; splitting ZIP alone did not bound MIME size.',
        'existing_bundles': [{'name': p.name, 'bytes': p.stat().st_size} for p in (run/'bundles').glob('*') if p.is_file()]}
    if log_path and Path(log_path).is_file():
        with Path(log_path).open('rb') as stream:
            stream.seek(max(0, Path(log_path).stat().st_size-512*1024))
            tail = stream.read().decode('utf-8', errors='replace')
        matches = [line[:1000] for line in tail.splitlines() if any(word in line.lower() for word in
                   ('sending email', 'smtp', 'send_email failed', 'message size', 'too large', 'exceeds'))]
        audit['mail_diagnosis']['log_evidence'] = matches[-20:]
    return audit


def recover_suite(run, destination, *, project_root=None, log_path=None):
    """Always save diagnosis. Package analysis artifacts only after all checks pass."""
    run, destination = Path(run).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    audit = audit_suite(run, project_root=project_root, log_path=log_path)
    save_json(destination/'audit.json', audit)
    if not audit['complete']:
        return audit, None
    payload = destination/'results'
    payload.mkdir()
    # Explicit allowlist: no original logs, database, panel, or model binaries.
    for path in (run/'summary').iterdir():
        if path.is_symlink():
            raise ValueError(f'Refusing result symlink: {path}')
        if path.is_file() and path.suffix in ('.csv', '.json', '.html', '.png') and path.name != 'package_manifest.json':
            shutil.copy2(path, payload/path.name)
    for group in ('screen', 'quarters'):
        for status in (run/group).rglob('task_status.json'):
            target = payload/status.parent.relative_to(run)
            target.mkdir(parents=True, exist_ok=True)
            for name in ('task_status.json', 'metrics.json', 'update_history.csv', 'validation_updates.csv'):
                source = status.parent/name
                if source.is_file():
                    shutil.copy2(source, target/name)
    shutil.copy2(destination/'audit.json', payload/'audit.json')
    # Use authoritative root status even if an old summary copy was stale.
    shutil.copy2(run/'status.json', payload/'status.json')
    save_json(payload/'recovery_scope.json', {'source': str(run), 'trained_again': False,
        'contains_models': False, 'contains_market_panel': False,
        'note': 'Includes test curves/report and training update/validation logs. Full models/TensorBoard stay in the original server run.'})
    metadata = create_bundle(payload, destination/'bundles', max_part_bytes=8*1024**2)
    return audit, metadata
