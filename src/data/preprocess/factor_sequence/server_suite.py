"""Server-owned source discovery and isolated walk-forward verification stages.

The entry point sends the suite summary; each training sends its own result mail.
"""
from __future__ import annotations

__all__ = [
    'discover',
    'execute',
    'aggregate',
    'run_suite',
    'report_package',
    'send_result',
]
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
import pandas as pd
from .bundle import write_json, load, file_digest
from .pipeline import available_splits
from .schema import BuildConfig


def discover(cfg):
    from src.proj import CALENDAR, DB
    from .sources import preflight
    report=preflight(cfg)
    if report['status']!='passed':
        return report
    prices=DB.dates('trade_ts','adjprice')
    if not len(prices):
        return {**report,'status':'failed','missing':report['missing']+['trade_ts/adjprice']}
    # An old/sparse individual factor must not truncate everybody else's OOS period.
    end=min(int(CALENDAR.updated()),int(prices.max))
    start=min(item['start'] for item in report['factors'].values())
    if start>=end:
        raise ValueError('No usable source history')
    return {**report,'history_start':start,'data_end':end,
            'date_policy':'earliest stored factor date through latest available adjusted-price trading day',
            'source_limits':'Stored snapshots may contain revisions; per-factor coverage is audited, not backfilled.'}


def execute(command, log, root):
    env=os.environ.copy()
    env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONUNBUFFERED']='1'
    with Path(log).open('x') as stream:
        with subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                # Real child output advances the supervised worker's progress log too.
                print(line,end='',flush=True)
            returncode=process.wait()
    if returncode:
        raise RuntimeError(f'Child exited with code {returncode}; inspect {log}')


def aggregate(fold_directories, output, expected_dates):
    """Date-equal OOS statistics; folds never overlap and no average of fold ICIRs."""
    groups={}
    expected_keys=None
    for folder in fold_directories:
        training=Path(folder)/'training'
        runs=json.loads((training/'runs.json').read_text())
        if not runs or any(run['status']!='passed' for run in runs):
            raise ValueError('Incomplete fold cannot enter the aggregate')
        keys=set()
        for run in runs:
            variant=json.dumps(run.get('variant',{}),sort_keys=True)
            key=(run['model'],int(run['seed']),variant)
            if key in keys:
                raise ValueError('Duplicate model/seed/variant in a fold')
            keys.add(key)
            groups.setdefault(key,[]).append(pd.read_csv(training/f"{run['name']}_daily.csv"))
        if expected_keys is None:
            expected_keys=keys
        elif keys!=expected_keys:
            raise ValueError('Different model sets across folds')
    summaries=[]
    for index,(key,parts) in enumerate(sorted(groups.items())):
        daily=pd.concat(parts,ignore_index=True).sort_values('date')
        if daily.date.duplicated().any() or daily.date.astype(int).tolist()!=list(expected_dates):
            raise ValueError('OOS dates have gaps/overlap or do not match the frozen plan')
        ic=daily.rank_ic.dropna()
        std=ic.std(ddof=0)
        top=daily.top_decile_10d_excess.dropna()
        target=Path(output)/f'daily_{index:03d}_{key[0]}_{key[1]}.csv'
        daily.to_csv(target,index=False)
        summaries.append(dict(model=key[0],seed=key[1],variant=json.loads(key[2]),
            start=int(daily.date.min()),end=int(daily.date.max()),dates=len(daily),ic_dates=len(ic),
            rank_ic=float(ic.mean()) if len(ic) else None,
            icir=float(ic.mean()/std) if std>1e-12 else None,
            top_decile_10d_excess=float(top.mean()) if len(top) else None,
            stock_date_samples=int(daily.n.sum()),daily_file=target.name,
            cost_adjusted_return=None))
    if not summaries:
        raise ValueError('No completed models')
    write_json(Path(output)/'comparison.json',summaries)
    return summaries


def run_suite(root, results_root, run_id, data_config, test_start=20170101, test_end=None,
              research=False, ablations=False, resume_from=None, training_email=True, recipient=None):
    root,results_root=Path(root).resolve(),Path(results_root).resolve()
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Invalid run ID')
    folder=results_root/run_id
    folder.mkdir(parents=True,exist_ok=False)
    report={'status':'running','run_id':run_id,'started_at':datetime.now(timezone.utc).isoformat(),
            'test_start':int(test_start),'requested_test_end':test_end,'research':research,
            'ablations':ablations,'execution_mode':'schedule_model','stages':[],'fold_directories':[],
            'interpretation':'Research run' if research else 'Pipeline verification; not model-selection evidence'}
    write_json(folder/'suite.json',report)
    def stage(name, action):
        item={'name':name,'status':'running'}
        report['stages'].append(item)
        write_json(folder/'suite.json',report)
        print(f'[factor-sequence] {name}: starting',flush=True)
        before=time.perf_counter()
        try:
            result=action()
            item['status']='passed'
            return result
        except BaseException:
            item.update(status='failed',error=traceback.format_exc())
            raise
        finally:
            item['seconds']=time.perf_counter()-before
            write_json(folder/'suite.json',report)
            print(f'[factor-sequence] {name}: {item["status"]} ({item["seconds"]:.1f}s)',flush=True)
    pipeline=[sys.executable,'-m','src.data.preprocess.factor_sequence.pipeline']
    try:
        config=json.loads(Path(data_config).read_text())
        cfg=BuildConfig(**{**config,'factors':tuple(config['factors'])})
        write_json(folder/'data_config.json',config)
        report['code']={str(p.relative_to(root)):file_digest(p) for directory in (
            'src/data/preprocess/factor_sequence','src/res/algo/nn/model/factor_sequence')
            for p in (root/directory).glob('*.py')}
        for p in [root/'scripts/4_train/6_factor_sequence_verify.py',
                  *sorted((root/'configs/algo/nn').glob('factorseq_*.yaml'))]:
            report['code'][str(p.relative_to(root))]=file_digest(p)
        stage('offline',lambda:execute([sys.executable,'-m','unittest','discover','-s','tests',
                  '-p','test_factor_sequence*.py','-v'],folder/'offline.log',root))
        def probe():
            found=discover(cfg)
            write_json(folder/'source_discovery.json',found)
            if found['status']!='passed':
                raise ValueError('Required sources missing: '+', '.join(found['missing']))
            return found
        if resume_from:
            previous_mode=json.loads((results_root/resume_from/'suite.json').read_text()).get('execution_mode')
            if previous_mode!='schedule_model':
                raise ValueError('Legacy per-fold runs cannot resume as a single model directory. Preserve old artifacts and start a new schedule-model suite.')
            from .resume import inspect_resume, verify_current_labels
            source, previous, plan, dataset = stage('inspect_resume', lambda: inspect_resume(root,results_root,resume_from))
            if config != json.loads((source/'data_config.json').read_text()):
                raise ValueError('Resume data configuration changed')
            if (research,ablations) != (previous['research'],previous['ablations']):
                raise ValueError('Resume training budget changed')
            stage('verify_labels',lambda:verify_current_labels(source,dataset))
            report.update(resume_from=resume_from,test_start=previous.get('test_start',test_start),
                requested_test_end=previous.get('requested_test_end'),history_start=previous['history_start'],
                data_end=previous['data_end'],dataset=str(dataset),data_version=previous['data_version'])
            write_json(folder/'dataset.json',{'path':str(dataset),'version':previous['data_version']})
            write_json(folder/'walk_forward.json',plan)
            stage('audit',lambda:execute(pipeline+['audit','--run-id',run_id],folder/'audit.log',root))
        else:
            sources=stage('discover',probe)
            report.update(history_start=sources['history_start'],data_end=sources['data_end'])
            stage('build',lambda:execute(pipeline+['build','--run-id',run_id,
                '--start',str(sources['history_start']),'--end',str(sources['data_end']),
                '--data-config',str((folder/'data_config.json').resolve())],folder/'build.log',root))
            dataset=Path(json.loads((folder/'dataset.json').read_text())['path'])
            report['dataset']=str(dataset)
            report['data_version']=load(dataset)['version']
            stage('audit',lambda:execute(pipeline+['audit','--run-id',run_id],folder/'audit.log',root))
            def resolve_plan():
                resolved=available_splits(dataset,int(test_start),test_end)
                write_json(folder/'walk_forward.json',resolved)
                return resolved
            plan=stage('plan',resolve_plan)
        report['latest_mature_label']=plan['latest_mature_label']
        report['fold_count']=len(plan['folds'])
        from .schedule_training import variants, export_folds
        candidates=variants(ablations)
        seeds=(17,29,43) if research else (17,)
        report['requested_training_runs']=len(plan['folds'])*len(candidates)*len(seeds)
        report['schedule_tasks']=len(candidates)*len(seeds)
        results=[]
        for model,variant in candidates:
            for seed in seeds:
                suffix='_'.join(variant) if variant else 'full'
                key=f'{model}_{suffix}_{seed}'
                command=[sys.executable,'-m','src.data.preprocess.factor_sequence.schedule_training',
                    '--dataset',str(dataset),'--suite',str(folder),'--model',model,
                    '--seed',str(seed),'--variant',json.dumps(variant)]
                if research:
                    command+=['--research']
                if not training_email:
                    command+=['--no-email']
                if recipient:
                    command+=['--recipient',recipient]
                if resume_from:
                    command+=['--resume-suite',str(results_root/resume_from)]
                stage(f'schedule_{key}',lambda command=command,key=key:execute(command,folder/f'{key}.log',root))
                candidates_found=[json.loads(p.read_text()) for p in (folder/'models').glob('*/run.json')]
                matches=[r for r in candidates_found if (r['model'],r['seed'],r['variant'])==(model,seed,variant)]
                if len(matches)!=1 or matches[0]['status']!='passed':
                    raise ValueError('Missing successful ordinary schedule result')
                results.append(matches[0])
        report['fold_directories']=stage('export_folds',lambda:export_folds(dataset,folder,plan,results))
        dates=[date for fold in plan['folds'] for date in fold['test']]
        summaries=stage('compare_all',lambda:aggregate(report['fold_directories'],folder,dates))
        report.update(status='passed',models=len(summaries),test_dates=len(dates),
                      actual_test_start=dates[0],actual_test_end=dates[-1])
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        mail_states=[]
        for child in report['fold_directories']:
            path=Path(child)/'training/runs.json'
            if path.exists():
                try:
                    for run in json.loads(path.read_text()):
                        if 'mail' in run:
                            mail_states.append({'name':run['name'],**run['mail']})
                except (OSError,ValueError):
                    mail_states.append({'status':'failed','error':traceback.format_exc(),'path':str(path)})
        report['training_mail']=mail_states
        report['training_mail_failed']=sum(s['status']=='failed' for s in mail_states)
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        write_json(folder/'suite.json',report)
        from src.automation.artifact_exchange.research import publish_analysis
        publish_analysis(folder, 'factor_sequence', report['status'])
        print(f'[factor-sequence] results: {folder}',flush=True)
    return folder,report


def report_package(folder):
    """Portable summaries and diagnostics; large checkpoints/stock predictions stay on server."""
    folder=Path(folder)
    report=json.loads((folder/'suite.json').read_text())
    target=folder/'result_report.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in folder.iterdir():
            if path.is_file() and path.suffix in ('.json','.log','.csv'):
                archive.write(path,path.name)
        for index,child in enumerate(report['fold_directories']):
            child=Path(child)
            for name in ('smoke.json','compare.json','training/runs.json','training/model_cards.json',
                         'training/split.json','training/comparison.json'):
                path=child/name
                if path.exists():
                    archive.write(path,f'fold_{index:03d}/{name}')
        for path in (folder/'models').glob('*/*'):
            if path.name in {'run.json','schedule.yaml'}:
                archive.write(path,str(path.relative_to(folder)))
        dataset=report.get('dataset')
        if dataset:
            for name in ('audit.json','coverage.csv','manifest.json'):
                path=Path(dataset)/name
                if path.exists():
                    archive.write(path,f'dataset/{name}')
    return target


def send_result(folder, recipient=None):
    from src.proj import MACHINE
    from src.proj.util.web.task_mail import TaskResultMail, retry_delivery
    folder=Path(folder)
    report=json.loads((folder/'suite.json').read_text())
    from src.automation.artifact_exchange.research import publish_analysis, delivered, notice
    transfers = publish_analysis(folder, 'factor_sequence', report['status'])
    if not MACHINE.emailable:
        state={'status':'skipped','reason':'Email is disabled on this machine'}
        write_json(folder/'mail_status.json',state)
        print('[factor-sequence] mail skipped: machine email disabled',flush=True)
        return state
    try:
        if (folder/'mail/delivery.json').exists():
            ok=retry_delivery(folder/'mail')
            state={'status':'sent' if ok else 'failed','receipt':str(folder/'mail/delivery.json')}
            write_json(folder/'mail_status.json',state)
            return state
        package=folder/'result_report.zip'
        if not package.exists():
            package=report_package(folder)
        failed=next((item['name'] for item in report['stages'] if item['status']=='failed'),None)
        body=(f"Factor-sequence verification: {report['status']}\n"
              f"Run: {report['run_id']}\nRequested OOS start: {report['test_start']}\n"
              f"OOS end: {report.get('actual_test_end',report.get('latest_mature_label','unresolved'))}\n"
              f"History start: {report.get('history_start','unresolved')}\n"
              f"Data version: {report.get('data_version','not built')}\n"
              f"Failed stage: {failed or 'none'}\nResults: {folder}\n"
              f"{report['interpretation']}\nAttached: summaries, actual configs and diagnostics. "
              "Checkpoints and stock-level predictions remain on the server.\n")
        ok=TaskResultMail(folder/'mail')(f"Factor sequence [{report['status']}] {report['run_id']}",
            body + notice(transfers),recipient=recipient,attachments=(list(folder.glob('*.log')) if delivered(transfers) else [folder/'suite.json',package]))
        state={'status':'sent' if ok else 'failed','receipt':str(folder/'mail/delivery.json')}
    except Exception:
        state={'status':'failed','error':traceback.format_exc()}
    write_json(folder/'mail_status.json',state)
    return state
