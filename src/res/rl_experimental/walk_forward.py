"""Sequential, resumable research suite. No database writes or production orders."""
from __future__ import annotations

import contextlib
import sys
import hashlib
import html
import itertools
import json
import shutil
import time
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import replay_baseline
from .bundle import create_bundle
from .data import PanelData, Standardizer
from .env import EpisodeRange, PortfolioEnv
from .progress import progress, quiet_output
from .training import ExperimentConfig, build_candidate_cache, history_metrics, train_experiment, write_history


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def day(value):
    return int(pd.Timestamp(value).strftime('%Y%m%d'))


def intervals(panel, start, valid_start, test_start, end):
    """Training labels must settle strictly before the next segment begins."""
    result = {}
    for name, lo, hi in [('train', start, valid_start), ('valid', valid_start, test_start)]:
        idx = np.flatnonzero((panel.decision_dates >= lo) & (panel.return_end_dates < hi))
        if not len(idx):
            raise ValueError(f'empty {name} interval: {lo}-{hi}')
        result[name] = (int(idx[0]), int(idx[-1]) + 1)
    idx = np.flatnonzero((panel.decision_dates >= test_start) & (panel.decision_dates < end))
    if len(idx):
        result['test'] = (int(idx[0]), int(idx[-1]) + 1)
    return result


def slice_panel(panel, start, end):
    mask = (panel.decision_dates >= start) & (panel.decision_dates < end)
    ids = np.flatnonzero(mask)
    if not len(ids):
        raise ValueError('empty panel slice')
    a, b = int(ids[0]), int(ids[-1]) + 1
    values = {}
    for f in fields(panel):
        v = getattr(panel, f.name)
        if isinstance(v, np.ndarray) and f.name != 'stock_ids':
            values[f.name] = v[a:b]
    return replace(panel, **values)


def load_config(path):
    c = json.loads(Path(path).read_text())
    required = ('panel', 'output', 'screen', 'first_quarter', 'last_quarter')
    if any(k not in c for k in required):
        raise ValueError(f'walk-forward config requires {required}')
    c.setdefault('seeds', [7, 17, 29])
    c.setdefault('penalties', [0., .0003, .001, .003])
    c.setdefault('bonuses', [0., .5])
    c.setdefault('costs_bp', [0., 1., 3., 5., 10.])
    c.setdefault('primary_cost_bp', 3.)
    c.setdefault('screen_steps', 30000)
    c.setdefault('quarter_steps', 100000)
    c.setdefault('training_years', 4)
    c.setdefault('validation_years', 1)
    c.setdefault('training', {})
    c.setdefault('bundle_max_mb', 20)
    c.setdefault('alpha_sample_status', 'unknown')
    if len(set(c['seeds'])) != len(c['seeds']) or not c['seeds']:
        raise ValueError('seeds must be nonempty and unique')
    if not any(x > 0 for x in c['penalties']) or not any(x > 0 for x in c['bonuses']) or 0 not in c['bonuses']:
        raise ValueError('screen needs positive penalty, positive bonus and bonus=0')
    if c['primary_cost_bp'] not in c['costs_bp'] or 5 not in c['costs_bp']:
        raise ValueError('cost scenarios must include primary cost and 5 bp stress')
    if min(c['costs_bp']) < 0 or min(c['penalties']) < 0 or min(c['bonuses']) < 0:
        raise ValueError('costs, penalties and bonuses must be nonnegative')
    return c


def folds(c):
    quarters = pd.period_range(c['first_quarter'], c['last_quarter'], freq='Q')
    if not len(quarters):
        raise ValueError('no quarters')
    result = []
    for q in quarters:
        test = q.start_time
        valid = test - pd.DateOffset(years=c['validation_years'])
        train = valid - pd.DateOffset(years=c['training_years'])
        result.append(dict(name=str(q), start=day(train), valid_start=day(valid),
                           test_start=day(test), end=day((q + 1).start_time)))
    if c['screen']['end'] > result[0]['test_start']:
        raise ValueError('screening must finish before first evaluation decision')
    return result


def task_plan(c):
    fs = folds(c)
    screen = len(c['penalties']) * len(c['bonuses']) * len(c['seeds'])
    formal = len(fs) * 3 * len(c['seeds'])
    profile = c.get('profile_report')
    estimates = None
    if profile and Path(profile).is_file():
        measurement = json.loads(Path(profile).read_text()).get('training', {})
        preferred = measurement.get(c['training'].get('device', 'cuda'), {})
        if preferred.get('steps_per_second', 0) > 0:
            steps = screen*c['screen_steps'] + formal*c['quarter_steps']
            estimates = {'training_hours_only': steps/preferred['steps_per_second']/3600,
                         'note': 'Measured short run; excludes preparation and multi-cost replay, not an SLA.'}
    return {'screen_training_tasks': screen, 'quarter_training_tasks': formal,
            'total_training_tasks': screen + formal, 'folds': fs, 'seeds': c['seeds'],
            'costs_bp': c['costs_bp'], 'estimate': estimates,
            'estimate_status': 'measured' if estimates else 'run profile-env first; no measured ETA available'}


def config_for(c, seed, penalty, bonus, steps):
    base = dict(n_envs=4, rollout_steps=128, batch_size=128, update_epochs=4,
                device='cuda', observation_version=2, lean_training=True)
    base.update(c['training'])
    base.update(seed=seed, total_timesteps=steps, holding_bonus=bonus,
                extra_cost_bp=c['primary_cost_bp'], observation_version=2, lean_training=True,
                selection_metric='nav', reward={'turnover_penalty': penalty, 'downside_penalty': .1})
    base['eval_freq'] = 5 * base['n_envs'] * base['rollout_steps']
    return ExperimentConfig(**base)


def train_task(panel, splits, directory, config, prepared, cache):
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'contract': prepared[3], 'splits': splits, 'config': asdict(config),
                'implementation': code_fingerprint()}
    task_hash = fingerprint(identity)
    status_file = directory / 'task_status.json'
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        if previous['fingerprint'] != task_hash:
            raise ValueError(f'task config/data changed; use a new output directory: {directory}')
        if previous['status'] == 'success':
            for name, digest in previous['artifacts'].items():
                if not (directory/name).is_file() or hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'completed task artifact corrupted: {directory/name}')
            progress('resume', f'Reuse completed task {directory.name}')
            return json.loads((directory/'metrics.json').read_text())
    write_json(status_file, {'status': 'running', 'fingerprint': task_hash})
    try:
        from .experiment import _Tee
        with (directory/'stdout.log').open('a') as stdout, (directory/'stderr.log').open('a') as stderr:
            with contextlib.redirect_stdout(_Tee(sys.stdout, stdout)), contextlib.redirect_stderr(_Tee(sys.stderr, stderr)):
                result = train_experiment(panel, directory, config, splits=splits, evaluate_test=False,
                                          prepared=prepared, baseline_cache=cache)
        names = ('metrics.json', 'normalizer.npz', 'best_model.zip', 'final_model.zip')
        write_json(status_file, {'status': 'success', 'fingerprint': task_hash,
                                'artifacts': {n: hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in names}})
        return result
    except Exception:
        write_json(status_file, {'status': 'failed', 'fingerprint': task_hash, 'traceback': traceback.format_exc()})
        raise


def code_fingerprint():
    return fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path(__file__).parent.glob('*.py'))})


def choose_screen(rows, bonus_required):
    candidates = []
    pairs = sorted({(r['penalty'], r['bonus']) for r in rows
                    if r['penalty'] > 0 and (r['bonus'] > 0) == bonus_required})
    for penalty, bonus in pairs:
        group = [r for r in rows if r['penalty'] == penalty and r['bonus'] == bonus]
        turnover = float(np.median([r['turnover_ratio'] for r in group]))
        relative = float(np.median([r['relative_nav'] for r in group]))
        candidates.append(dict(penalty=penalty, bonus=bonus, turnover_ratio=turnover,
                               relative_nav=relative, turnover_goal_met=turnover <= 1.5))
    qualified = [x for x in candidates if x['turnover_goal_met']]
    if qualified:
        return max(qualified, key=lambda x: (x['relative_nav'], -x['penalty'], -x['bonus']))
    return min(candidates, key=lambda x: (x['turnover_ratio'], -x['relative_nav'], x['penalty']))


def cash_account(panel):
    return dict(stock_ids=panel.stock_ids.copy(), weights=np.zeros(panel.n_stocks, np.float32), cash=1., nav=1.)


def model_replay(panel, interval, directory, cost, state, device):
    from stable_baselines3 import PPO
    meta = json.loads((directory/'metrics.json').read_text())
    config = replace(ExperimentConfig(**meta['config']), extra_cost_bp=cost, device=device)
    with np.load(directory/'normalizer.npz', allow_pickle=False) as a:
        normalizer = Standardizer(a['mean'], a['scale'])
    expected = meta['panel_contract']
    if panel.n_features != expected['n_features'] or list(panel.feature_names) != expected['feature_names'] or not np.array_equal(meta['stock_ids'], panel.stock_ids):
        raise ValueError('checkpoint feature/security contract mismatch')
    if panel.n_industries != expected['n_industries']:
        raise ValueError('checkpoint industry vocabulary mismatch')
    normalized = normalizer.transform(panel)
    env = PortfolioEnv(normalized, EpisodeRange(*interval), config.constraints(),
                       reward_config=config.reward, reward_scale=config.reward_scale, observation_version=2,
                       candidate_cache=build_candidate_cache(normalized, config.candidate_count))
    obs, _ = env.reset(options={'account_state': state})
    model = PPO.load(directory/'best_model.zip', device=device)
    while env._time < interval[1] and env._nav > 0:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    state.update(env.export_account())
    history = env.history
    env.close()
    return history


def retention_replay(panel, interval, constraints, state):
    env = PortfolioEnv(panel, EpisodeRange(*interval), constraints, observation_version=2)
    env.reset(options={'account_state': state})
    while env._time < interval[1] and env._nav > 0:
        _, _, done, _, _ = env.step(np.zeros(env.action_space.shape, np.float32))
        if done:
            break
    state.update(env.export_account())
    history = env.history
    env.close()
    return history


def add_exhausted_rows(panel, interval, history, state):
    """Account remains at zero after bankruptcy; these rows contain no decisions/trades."""
    if state['nav'] != 0:
        return history
    completed = {int(r['decision_date']) for r in history}
    for i in range(*interval):
        if int(panel.decision_dates[i]) not in completed:
            history.append(dict(decision_date=int(panel.decision_dates[i]), return_end_date=int(panel.return_end_dates[i]),
                                nav=0., period_return=0., turnover=0., fee_estimate=0., base_fee=0., extra_cost=0.,
                                cash_weight=0., holdings=0, frozen_holdings=0, rejected_buys=0, rejected_sells=0,
                                assets_exhausted=True, delisting_events=0, delisting_loss=0.))
    return history


def quarter_returns(history):
    frame = pd.DataFrame(history)
    quarter = pd.to_datetime(frame.return_end_date.astype(str)).dt.to_period('Q').astype(str)
    return {str(q): float(np.prod(1 + group.period_return) - 1)
            for q, group in frame.groupby(quarter)}


def bootstrap_relative(a, b, block, seed=20261019, repetitions=2000):
    x = np.log1p(np.asarray(a)) - np.log1p(np.asarray(b))
    if not np.isfinite(x).all():
        return {'status': 'undefined_after_bankruptcy'}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), (repetitions, int(np.ceil(len(x)/block))))
    idx = ((starts[:, :, None] + np.arange(block)) % len(x)).reshape(repetitions, -1)[:, :len(x)]
    return {'block': block, 'repetitions': repetitions, 'seed': seed,
            'relative_nav_ci95': np.quantile(np.expm1(x[idx].sum(axis=1)), [.025, .975]).tolist()}


def summarize(histories, c):
    metrics = {k: history_metrics(h) for k, h in histories.items()}
    for key, h in histories.items():
        m = metrics[key]
        m['annualized_return'] = float(m['final_nav'] ** (252 / len(h)) - 1)
        m['mean_cash_weight'] = float(np.mean([r['cash_weight'] for r in h]))
        m['mean_risk_unsold_holdings'] = float(np.mean([r.get('risk_unsold_holdings', 0) for r in h]))
        parts = key.split('/')
        cost = parts[1] if len(parts) == 2 else parts[2]
        benchmark = histories[f'robust/{cost}']
        active = np.asarray([r['period_return'] for r in h]) - np.asarray([r['period_return'] for r in benchmark])
        te = float(active.std(ddof=1) * np.sqrt(252)) if len(active) > 1 else 0.
        bn = metrics[f'robust/{cost}']['final_nav']
        m['return_difference_vs_robust_pp'] = float(100*(m['final_nav']-bn))
        m['relative_nav_vs_robust'] = float(m['final_nav']/bn-1) if bn > 0 else None
        m['tracking_error_vs_robust'] = te
        m['information_ratio_vs_robust'] = float(active.mean()*252/te) if te > 0 else None
    quarters = {k: quarter_returns(h) for k, h in histories.items()}
    judgments = {}
    primary = f'{float(c["primary_cost_bp"]):g}'
    for arm in ('original', 'penalty', 'retention'):
        base_key = f'robust/{primary}'
        bm = metrics[base_key]
        keys = [f'{arm}/{seed}/{primary}/rolling' for seed in c['seeds']]
        bq = quarters[base_key]
        quarter_excess = {q: float(np.median([quarters[k][q] - bq[q] for k in keys])) for q in bq}
        wins = sum(metrics[k]['final_nav'] > bm['final_nav'] for k in keys)
        turns = float(np.median([metrics[k]['mean_turnover'] for k in keys]))
        drawdown = float(np.median([metrics[k]['max_drawdown'] for k in keys]))
        stress = float(np.median([metrics[f'{arm}/{seed}/5/rolling']['final_nav'] for seed in c['seeds']]))
        criteria = {'seed_wins': wins >= int(np.ceil(2*len(keys)/3)),
                    'quarter_win_rate': np.mean([x > 0 for x in quarter_excess.values()]) >= .6,
                    'turnover': turns <= 1.5*bm['mean_turnover'],
                    'drawdown': drawdown >= bm['max_drawdown'] - .02,
                    'stress_5bp': stress > metrics['robust/5']['final_nav']}
        judgments[arm] = {'criteria': {k: bool(v) for k, v in criteria.items()}, 'research_pass': bool(all(criteria.values())),
                          'production_ready': False, 'winning_seeds': wins,
                          'quarter_win_rate': float(np.mean([x > 0 for x in quarter_excess.values()])),
                          'turnover_ratio': turns/max(bm['mean_turnover'], 1e-12),
                          'drawdown_difference_pp': 100*(drawdown-bm['max_drawdown']), 'quarter_excess_median': quarter_excess,
                          'bootstrap': {str(seed): [bootstrap_relative(
                              [r['period_return'] for r in histories[f'{arm}/{seed}/{primary}/rolling']],
                              [r['period_return'] for r in histories[base_key]], block) for block in (5, 10, 20)]
                                        for seed in c['seeds']}}
    dispersion = {}
    for arm in ('original', 'penalty', 'retention'):
        values = [metrics[f'{arm}/{seed}/{primary}/rolling']['total_return'] for seed in c['seeds']]
        dispersion[arm] = {'mean_return': float(np.mean(values)), 'std_return': float(np.std(values)),
                           'worst_quarter': float(min(v for seed in c['seeds'] for v in quarters[f'{arm}/{seed}/{primary}/rolling'].values()))}
    return {'metrics': metrics, 'quarter_returns': quarters, 'seed_dispersion': dispersion, 'judgments': judgments,
            'research_status': 'historical development validation', 'alpha_sample_status': c['alpha_sample_status'],
            'cost_note': '3.5 bp base plus simulated friction; no capacity or impact calibration'}


def report(summary, output):
    rows = ''.join('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{100*m[k]:.2f}%</td>' for k in
                  ('total_return', 'max_drawdown', 'annualized_volatility', 'mean_turnover', 'total_fees'))+'</tr>'
                   for name, m in summary['metrics'].items())
    # Embed machine-readable data and plot individual continuous paths offline.
    curves = summary.get('curves', {})
    svg = []
    for arm, values in curves.items():
        if not values:
            continue
        scale = max(1., max(values))
        points = ' '.join(f'{i/max(1,len(values)-1)*900:.2f},{180-v/scale*160:.2f}' for i,v in enumerate(values))
        svg.append(f'<h3>{html.escape(arm)}：连续净值</h3><svg viewBox="0 0 930 210"><polyline points="{points}" fill="none" stroke="steelblue"/><text x="0" y="205">交易序号 0—{len(values)-1}；纵轴净值 0—{scale:.2f}（起始=1）</text></svg>')
    content = '<!doctype html><meta charset="utf-8"><title>低换手 RL 滚动实验</title><style>body{font:15px/1.7 system-ui;max-width:1100px;margin:auto;padding:24px}td,th{padding:6px;border-bottom:1px solid #ccc}svg{width:100%}</style>'
    content += '<h1>低换手 RL：季度滚动与成本压力</h1><p>历史开发验证，主基准 robust；未校准实盘容量。收益为累计收益，费用为累计金额/初始净值，换手为日均双边换手。</p>'
    content += '<pre>'+html.escape(json.dumps(summary['judgments'], ensure_ascii=False, indent=2))+'</pre>'
    content += '<h2>季度收益与种子离散度</h2><p>季度以结算日归属；数值为小数收益率。</p><pre>'+html.escape(json.dumps({'quarter_returns': summary['quarter_returns'], 'seed_dispersion': summary['seed_dispersion']}, ensure_ascii=False, indent=2))+'</pre>'
    content += '<table><tr><th>方案/种子/额外bp/模式</th><th>净收益</th><th>最大回撤</th><th>年化波动</th><th>换手</th><th>费用</th></tr>'+rows+'</table>'+''.join(svg)
    (output/'report.html').write_text(content, encoding='utf-8')


def run_walk_forward(config_path, *, dry_run=False, stage='all', bundle_ready=None):
    if dry_run:
        return _run_walk_forward(config_path, dry_run=True, stage=stage, bundle_ready=bundle_ready)
    output = Path(load_config(config_path)['output']).resolve()
    with quiet_output(output/'detail.log'):
        progress('suite', f'Start {stage}; full logs: {output}/detail.log', milestone=True)
        try:
            result = _run_walk_forward(config_path, stage=stage, bundle_ready=bundle_ready)
        except Exception as exc:
            progress('suite', f'Failed: {type(exc).__name__}: {exc}', warning=True, milestone=True)
            raise
        progress('suite', f'Completed {stage}; bundle: {result["bundle_metadata"]}', milestone=True)
        return result


def _run_walk_forward(config_path, *, dry_run=False, stage='all', bundle_ready=None):
    c = load_config(config_path)
    plan = task_plan(c)
    if dry_run:
        snapshot = Path(c['panel'])
        if snapshot.is_file():
            with np.load(snapshot, allow_pickle=False) as arrays:
                decisions = arrays['decision_dates']
                ends = arrays['return_end_dates']
            actual = []
            for fold in plan['folds']:
                entry = {'name': fold['name']}
                ranges = [('train', fold['start'], fold['valid_start']),
                          ('valid', fold['valid_start'], fold['test_start']),
                          ('test', fold['test_start'], fold['end'])]
                for role, lo, hi in ranges:
                    mask = (decisions >= lo) & (decisions < hi)
                    if role != 'test':
                        mask &= ends < hi
                    if role == 'test' and fold == plan['folds'][-1]:
                        mask &= ends < hi
                    idx = np.flatnonzero(mask)
                    entry[role] = {'transitions': len(idx),
                                   'decision_start': int(decisions[idx[0]]) if len(idx) else None,
                                   'settlement_end': int(ends[idx[-1]]) if len(idx) else None}
                actual.append(entry)
            plan['actual_windows'] = actual
            plan['snapshot_status'] = 'dates inspected; full feature/contract checks run before training'
        else:
            plan['snapshot_status'] = 'not found; showing planned dates only; prepare v4 snapshot before running'
        return plan
    if stage not in ('screen', 'quarters', 'all'):
        raise ValueError('invalid stage')
    output = Path(c['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = PanelData.load_npz(c['panel'])
    if panel.schema_version < 4 or panel.delisting_event is None:
        raise ValueError('quarterly experiments require a v4 real snapshot')
    required_start = min(c['screen']['start'], plan['folds'][0]['start'])
    if panel.decision_dates[0] > day(pd.Timestamp(str(required_start)) + pd.Timedelta(days=15)) or panel.return_end_dates[-1] < day(pd.Timestamp(str(plan['folds'][-1]['end'])) - pd.Timedelta(days=10)):
        raise ValueError('snapshot does not cover requested windows')
    identity_config = {k: v for k, v in c.items() if k not in ('stage', 'recipient', 'bundle_max_mb', 'profile_report')}
    identity = fingerprint({'config': identity_config, 'panel': panel.contract(), 'code': code_fingerprint()})
    lock = output/'identity.json'
    if lock.exists() and json.loads(lock.read_text())['fingerprint'] != identity:
        raise ValueError('suite config/data/code changed; choose a new output directory')
    write_json(lock, {'fingerprint': identity, 'panel_contract': panel.contract()})
    write_json(output/'config.json', c)
    write_json(output/'plan.json', plan)
    from .experiment import _system_info
    write_json(output/'system.json', _system_info())
    started = time.perf_counter()
    try:
        selected_path = output/'selected.json'
        if stage in ('screen', 'all'):
            progress('screen', f'Start {plan["screen_training_tasks"]} screening tasks', milestone=True)
            screen = c['screen']
            sub = slice_panel(panel, screen['start'], screen['end'])
            splits = intervals(sub, screen['start'], screen['valid_start'], screen['end'], screen['end'])
            norm = Standardizer.fit(sub, *splits['train'])
            prepared = (norm, norm.transform(sub), build_candidate_cache(sub, c["training"].get("candidate_count", 100)), sub.contract())
            cache, results = {}, []
            for penalty, bonus, seed in itertools.product(c['penalties'], c['bonuses'], c['seeds']):
                config = config_for(c, seed, penalty, bonus, c['screen_steps'])
                folder = output/'screen'/f'p{penalty:g}-b{bonus:g}-s{seed}'
                progress('screen', f'Training {folder.name}')
                metrics = train_task(sub, splits, folder, config, prepared, cache)
                baseline = json.loads((folder/'validation_baseline_metrics.json').read_text())['robust_top50']
                results.append(dict(penalty=penalty, bonus=bonus, seed=seed,
                                    relative_nav=metrics['validation']['final_nav']/max(baseline['final_nav'], 1e-12)-1,
                                    turnover_ratio=metrics['validation']['mean_turnover']/max(baseline['mean_turnover'], 1e-12)))
                write_json(output/'screen_results.json', results)
            selected = {'original': {'penalty': 0., 'bonus': 0.},
                        'penalty': choose_screen(results, False), 'retention': choose_screen(results, True)}
            write_json(selected_path, selected)
            write_json(output/'selection_fingerprint.json', {'sha256': hashlib.sha256(selected_path.read_bytes()).hexdigest()})
            progress('screen', f'Completed screening; frozen choices: {selected}', milestone=True)
            del prepared, cache, sub
        if stage in ('quarters', 'all'):
            if not selected_path.exists():
                raise ValueError('run screening before quarterly training')
            if hashlib.sha256(selected_path.read_bytes()).hexdigest() != json.loads((output/'selection_fingerprint.json').read_text())['sha256']:
                raise ValueError('frozen screening selection was modified')
            selected = json.loads(selected_path.read_text())
            for fold in plan['folds']:
                sub = slice_panel(panel, fold['start'], fold['end'])
                splits = intervals(sub, fold['start'], fold['valid_start'], fold['test_start'], fold['end'])
                norm = Standardizer.fit(sub, *splits['train'])
                prepared = (norm, norm.transform(sub), build_candidate_cache(sub, c["training"].get("candidate_count", 100)), sub.contract())
                cache = {}
                for arm, seed in itertools.product(selected, c['seeds']):
                    params = selected[arm]
                    config = config_for(c, seed, params['penalty'], params['bonus'], c['quarter_steps'])
                    folder = output/'quarters'/fold['name']/arm/f'seed-{seed}'
                    progress('quarter', f'{fold["name"]}/{arm}/seed-{seed}')
                    train_task(sub, splits, folder, config, prepared, cache)
                del prepared, cache, sub
                progress('quarter', f'{fold["name"]}: training/validation complete, {len(selected)*len(c["seeds"])} tasks', milestone=True)
            histories, states = {}, {}
            for fold in plan['folds']:
                # Evaluate only this decision segment; preserve positions at every boundary.
                sub = slice_panel(panel, fold['test_start'], fold['end'])
                if fold == plan['folds'][-1]:
                    stop = int(np.searchsorted(sub.return_end_dates, fold['end'], side='left'))
                    if stop == 0:
                        raise ValueError('last quarter has no settled transitions')
                    if stop < sub.n_steps:
                        sub = slice_panel(sub, fold['test_start'], int(sub.decision_dates[stop]))
                interval = (0, sub.n_steps)
                for cost in c['costs_bp']:
                    cfg = config_for(c, c['seeds'][0], 0, 0, c['quarter_steps'])
                    constraints = replace(cfg.constraints(), extra_cost_bp=cost)
                    for name, kind in [('robust', 'robust_top50'), ('equal', 'alpha_equal_weight'), ('retention_rule', None)]:
                        key = f'{name}/{float(cost):g}'
                        state = states.setdefault(key, cash_account(sub))
                        if kind:
                            h = replay_baseline(sub, interval, constraints, kind, account_state=state)
                        else:
                            h = retention_replay(sub, interval, replace(constraints, holding_bonus=selected['retention']['bonus']), state)
                        histories.setdefault(key, []).extend(add_exhausted_rows(sub, interval, h, state))
                    for arm, seed, mode in itertools.product(selected, c['seeds'], ('rolling', 'fixed')):
                        key = f'{arm}/{seed}/{float(cost):g}/{mode}'
                        state = states.setdefault(key, cash_account(sub))
                        quarter = fold['name'] if mode == 'rolling' else plan['folds'][0]['name']
                        folder = output/'quarters'/quarter/arm/f'seed-{seed}'
                        progress('replay', f'{fold["name"]}/{key}')
                        h = model_replay(sub, interval, folder, cost, state, c['training'].get('device', 'cuda'))
                        histories.setdefault(key, []).extend(add_exhausted_rows(sub, interval, h, state))
                write_json(output/'evaluation_progress.json', {'completed_quarter': fold['name'], 'paths': len(histories)})
                progress('quarter', f'{fold["name"]}: test replay complete, {len(histories)} strategy/cost paths', milestone=True)
            # Every strategy/cost must have exactly the same decision/settlement pairs.
            dates = None
            for key, history in histories.items():
                pairs = [(r['decision_date'], r['return_end_date']) for r in history]
                if dates is not None and pairs != dates:
                    raise ValueError(f'evaluation dates mismatch: {key}')
                dates = pairs
            summary_dir = output/'summary'
            summary_dir.mkdir(exist_ok=True)
            for key, h in histories.items():
                write_history(summary_dir/f'{key.replace("/", "_")}.csv', h)
            summary = summarize(histories, c)
            summary['curves'] = {k: [1.] + [r['nav'] for r in h] for k, h in histories.items()
                                 if k.endswith(f'/{c["seeds"][0]}/{float(c["primary_cost_bp"]):g}/rolling') or k == f'robust/{float(c["primary_cost_bp"]):g}'}
            write_json(summary_dir/'summary.json', summary)
            report(summary, summary_dir)
        write_json(output/'status.json', {'status': 'success', 'stage': stage, 'elapsed_seconds': time.perf_counter()-started})
    except Exception:
        write_json(output/'status.json', {'status': 'failed', 'stage': stage, 'traceback': traceback.format_exc()})
        raise
    finally:
        summary_dir = output/'summary'
        summary_dir.mkdir(exist_ok=True)
        for name in ('status.json', 'config.json', 'plan.json', 'selected.json', 'screen_results.json', 'identity.json', 'system.json'):
            if (output/name).exists():
                shutil.copy2(output/name, summary_dir/name)
        metadata = create_bundle(summary_dir, output/'bundles', int(c['bundle_max_mb']*1024**2))
        if bundle_ready:
            bundle_ready(metadata)
    return {'output': str(output), 'bundle_metadata': str(metadata), 'status': 'success', 'stage': stage}
