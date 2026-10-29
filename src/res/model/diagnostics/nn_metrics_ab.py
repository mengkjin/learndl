"""Independent-process default short tests; portable evidence for later local analysis."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ORDER = [('legacy_1', False), ('fast_1', True), ('fast_2', True), ('legacy_2', False)]
SEED = 20261029


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def probe(args, root):
    try:
        p = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=15)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as exc:
        return {'error': str(exc)}


def single(folder, name, fast):
    started = time.perf_counter()
    result = {'name': name, 'fastpath': fast, 'seed': SEED, 'status': 'failed'}
    try:
        import torch
        from src.proj import Save
        from src.res.model.model_module.application.trainer import ModelTrainer
        result.update(import_seconds=time.perf_counter()-started, torch=torch.__version__,
                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      threads=torch.get_num_threads())
        t = time.perf_counter()
        trainer = ModelTrainer.GO(title=f'NN metrics AB {name}', short_test=True,
            stage=0, resume=0, selection=0,
            override={'model.name': name, 'env.random_seed': SEED,
                      'train.experimental_metrics_fastpath': fast})
        Save.async_wait_all(caller_name='nn_metrics_ab')
        result.update(pipeline_seconds=time.perf_counter()-t,
            model_path=str(trainer.base_path.base.resolve()), device=str(trainer.config.device),
            effective_config=dict(trainer.config.Param), model_params=trainer.config.algo_config.params,
            model_dates=trainer.data.model_date_list.tolist(), status='success',
            stage_seconds={s: (trainer.status.times[f'{s}_end']-trainer.status.times[f'{s}_start']).total_seconds()
                           for s in ('data','fit','test')})
    except BaseException:
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        save(folder/'run.json', result)


def run_suite(folder, root, request_id):
    folder.mkdir(parents=True, exist_ok=False)
    report = {'request_id': request_id, 'seed': SEED, 'order': ORDER,
              'host': socket.gethostname(), 'platform': platform.platform(), 'python': sys.version,
              'git': probe(['git','rev-parse','HEAD'], root),
              'git_status': probe(['git','status','--short'], root), 'runs': []}
    save(folder/'suite.json', report)
    # Capture source/configs used so subsequent code updates cannot erase provenance.
    for src, target in [('configs/model', 'source/configs/model'),
                        ('src/res/model/util/metric', 'source/metric'),
                        ('src/res/model/callback/monitor/summary.py', 'source/summary.py')]:
        source = root/src; dest = folder/target; dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(source, dest)
    env = dict(os.environ, PYTHONHASHSEED=str(SEED), PYTHONUNBUFFERED='1',
               PYTHONPATH=str(root), MPLCONFIGDIR=str(folder/'mplconfig'))
    # Prepare fonts before timing, as in the local test.
    subprocess.run([sys.executable, '-c', 'import matplotlib.font_manager'], env=env, cwd=root, check=True)
    from src.api.task_monitor.scheduling.maintenance_gate import child_env, pass_fds
    env.update(child_env())
    for label, fast in ORDER:
        run = folder/label; run.mkdir()
        name = f'ab_{request_id}_{label}'
        save(run/'gpu_before.json', probe(['nvidia-smi'], root))
        t = time.perf_counter()
        with (run/'console.log').open('w') as log:
            try:
                proc = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--single', str(run),
                    '--name', name, '--fast', str(int(fast))], cwd=root, env=env,
                    pass_fds=pass_fds(), stdout=log, stderr=subprocess.STDOUT, timeout=7200)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write('\nChild exceeded 7200 seconds.\n')
        item = {'label': label, 'name': name, 'fastpath': fast, 'returncode': code,
                'process_seconds': time.perf_counter()-t}
        if (run/'run.json').exists():
            item['result'] = json.loads((run/'run.json').read_text())
            model = Path(item['result'].get('model_path', root/'models/st'/f'nn@gru@{name}'))
        else:
            model = root/'models/st'/f'nn@gru@{name}'
        # Preserve partial artifacts too; no market inputs or training dataset caches.
        for part in ('configs', 'archive', 'snapshot', 'results'):
            if (model/part).is_dir():
                shutil.copytree(model/part, run/'model'/part)
        save(run/'gpu_after.json', probe(['nvidia-smi'], root))
        report['runs'].append(item)
        save(folder/'suite.json', report)
        print(f'NN A/B {label}: exit={code}; evidence={run}', flush=True)
    report['status'] = 'success' if all(r['returncode'] == 0 and r.get('result',{}).get('status') == 'success'
                                         for r in report['runs']) else 'failed'
    report['timing'] = {}
    if report['status'] == 'success':
        for metric in ('fit', 'test', 'pipeline_seconds', 'wall_seconds'):
            def value(r):
                data = r['result']
                return data[metric] if metric.endswith('seconds') else data['stage_seconds'][metric]
            old = [value(r) for r in report['runs'] if not r['fastpath']]
            new = [value(r) for r in report['runs'] if r['fastpath']]
            report['timing'][metric] = {'legacy': old, 'fast': new,
                'time_reduction_pct': 100*(1-sum(new)/sum(old))}
    report['interpretation'] = 'Timing only; compare predictions, metrics and weights locally before accepting. GPU may be nondeterministic.'
    save(folder/'suite.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--fast', type=int, choices=(0,1), required=True)
    args = parser.parse_args()
    single(args.single, args.name, bool(args.fast))
