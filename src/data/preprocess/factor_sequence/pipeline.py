"""Opt-in build/audit/smoke/compare stages; never schedules work or sends mail.

python -m src.data.preprocess.factor_sequence.pipeline --help
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import time
import traceback
import resource
import sys
import numpy as np
import pandas as pd
from .bundle import build, load, audit, write_json, file_digest
from .schema import BuildConfig, KEYS


def purged_split(dates, ends, train=252, valid=63, test=63, warmup=239):
    dates, ends = np.asarray(dates), np.asarray(ends)
    if len(dates) != len(ends) or np.any(np.diff(dates) <= 0) or np.any(ends < dates):
        raise ValueError('Invalid dates/label endpoints')
    if len(dates) < warmup+train+valid+test:
        raise ValueError('Insufficient history; refusing to shorten the experiment')
    block=dates[-(train+valid+test):]
    endpoint=dict(zip(dates,ends))
    tr=block[:train];va=block[train:train+valid];te=block[train+valid:]
    tr=np.array([d for d in tr if endpoint[d] < va[0]])
    va=np.array([d for d in va if endpoint[d] < te[0]])
    if not len(tr) or not len(va):
        raise ValueError('Empty split after purging')
    return {k:v.astype(int).tolist() for k,v in [('train',tr),('valid',va),('test',te)]}


def train_overrides(dataset, name, model, split, model_date, research=False, seed=17):
    return {
        'model.name':name, 'model.module':model, 'model.labels':['std_lag1_10'],
        'model.beg_date':int(model_date), 'model.end_date':split['test'][-1],
        'model.interval':100000, 'model.window':400, 'model.submodels':['best'],
        'input.type':'data', 'input.data.types':'+'.join(KEYS),
        'input.factor_sequence.dataset':str(Path(dataset).resolve()),
        'input.sequence.lens':{'factorseq':60,'factorperf':120,'factormarket':120},
        'input.sequence.steps':dict.fromkeys(KEYS,1),
        'input.verify.x.max_nan_ratio':0.,
        'train.max_epoch':30 if research else 3, 'train.fitting_step':1,
        'train.skip_horizon':0,  # exact purged date lists below supersede blanket gap
        'train.dataloader.sample_method':'sequential',
        'train.dataloader.train_ratio':.8, 'train.dataloader.purge_horizon':11,
        'train.factor_sequence.train_dates':split['train'],
        'train.factor_sequence.valid_dates':split['valid'],
        'train.criterion.loss':{'mse':{}},
        'train.criterion.accuracy':{'spearman':{'which_output':[0],'which_label':[0]}},
        'train.criterion.weight':{'fit':'equal','test':'equal'},
        'model.special.global2top':False, 'train.trainer.optimizer':{'name':'Adam','param':{}},
        'train.trainer.learn_rate':{'base':.001,'transfer_multiplier':{}},
        'train.trainer.scheduler':{'name':'step','param':{'step_size':1000,'gamma':1.}},
        'env.random_seed':seed, 'env.short_test':False,
        'train.callbacks':['StatusDisplay','SummaryWriter','BasicTestResult','EarlyStoppage','CallbackTimer'],
        'callbacks.EarlyStoppage':{'peak_patience':5 if research else 100,'converge_patience':100},
    }


def read_predictions(model_path):
    paths=sorted((Path(model_path)/'snapshot'/'pred_recorder'/'preds').glob('*.feather'))
    if not paths:
        raise ValueError(f'No saved out-of-sample predictions: {model_path}')
    df=pd.concat([pd.read_feather(p) for p in paths],ignore_index=True)
    if 'submodel' in df:
        df=df.loc[df.submodel == 'best']
    required={'date','secid','pred','label'}
    if not required <= set(df):
        raise ValueError('Prediction schema mismatch')
    if df.duplicated(['date','secid']).any():
        raise ValueError('Duplicate OOS predictions')
    return df


def smoke(dataset, folder, research=False, ablations=False):
    from src.proj import CALENDAR, Save
    from src.data.util import DataBlock
    from src.data import DATAVENDOR
    from src.res.model.model_module.application.trainer import ModelTrainer
    from src.res.algo.nn.model.factor_sequence import CATALOG, MODELS
    manifest=load(dataset)
    if not (Path(dataset)/'audit.json').exists():
        raise ValueError('Run audit before smoke')
    if json.loads((Path(dataset)/'audit.json').read_text())['status'] != 'passed':
        raise ValueError('Data audit did not pass')
    stock=np.load(Path(dataset)/'factorseq.npz',allow_pickle=False)
    dates=stock['dates']
    y=DataBlock.load_preprocess('y',frame='fit')
    if not y.initiated or 'std_lag1_10' not in y.feature:
        raise ValueError('Real label block std_lag1_10 unavailable')
    y=y.align_secid(stock['secids']).align_date(dates).align_feature(['std_lag1_10'])
    ends=np.asarray([int(CALENDAR.td(int(d),11)) for d in dates])
    # Raw return coverage, not just a finite imputed label, bounds the endpoint.
    latest_return=int(dates[-1])
    ready=(ends <= latest_return) & np.isfinite(y.values[:,:,0,0].cpu().numpy()).any(axis=0)
    last=np.flatnonzero(ready)
    if not len(last):
        raise ValueError('No mature labels')
    last=int(last[-1])
    split=purged_split(dates[:last+1],ends[:last+1])
    for group,group_dates in split.items():
        indices=np.searchsorted(dates,group_dates)
        if not ready[indices].all():
            raise ValueError(f'{group} contains missing label dates; refusing silent omission')
    model_date=int(CALENDAR.td(split['test'][0],-1))
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'split.json',split)
    write_json(folder/'catalog.json',CATALOG)
    np.savez_compressed(folder/'labels.npz',values=y.values.cpu().numpy(),dates=dates,secids=stock['secids'])
    # A separately named realized forward return, not the normalized training label.
    # std_lag1_10 is close(T+11)/close(T+1)-1: daily-return offsets 2..11.
    future=DATAVENDOR.nday_fut_ret(stock['secids'],np.asarray(split['test']),nday=10,lag=2)
    np.savez_compressed(folder/'realized_returns.npz',values=future.values.cpu().numpy(),
                        dates=future.date,secids=future.secid)
    runs=[]
    variants=[(key,{}) for key in MODELS]
    if ablations:
        variants += [('factorseq_m1',{'use_performance':False}),('factorseq_m1',{'use_market':False}),
                     ('factorseq_m2',{'modulation':False})]
    pending=[{'model':model,'seed':seed,'variant':extra,'status':'not_run'}
             for model,extra in variants for seed in ((17,29,43) if research else (17,))]
    write_json(folder/'model_cards.json',pending)
    from src.res.algo.nn.model import factor_sequence as model_code
    provenance = {'model_sha256':file_digest(model_code.__file__),
                  'runner_sha256':file_digest(__file__),
                  'manifest_sha256':file_digest(Path(dataset)/'manifest.json')}
    from src.proj import PATH
    provenance['training_data_code']={str(path.relative_to(PATH.main)):file_digest(path)
        for path in (PATH.main/'src/res/model/util/data').glob('*.py')}
    for model, extra in variants:
        for seed in ((17,29,43) if research else (17,)):
            suffix='_'.join(extra) if extra else 'full'
            name=f'fs_{folder.parent.name}_{model}_{suffix}_{seed}'
            override=train_overrides(dataset,name,model,split,model_date,research,seed)
            for k,v in extra.items():
                override[k]=[v]
            result={'model':model,'name':name,'seed':seed,'status':'running','overrides':override,
                    'data_version':manifest['version'],'cost_adjusted_return':None,
                    'provenance':provenance,'model_version':CATALOG[model]['version']}
            card=pending[len(runs)]
            card.update(status='running',run_id=folder.parent.name,name=name,results='runs.json')
            runs.append(result);write_json(folder/'runs.json',runs)
            started=time.perf_counter()
            try:
                trainer=ModelTrainer.GO(title=name,module=model,override=override,short_test=False,
                                        stage=0,resume=0,selection=0,html_catcher=False)
                Save.async_wait_all(caller_name='factor_sequence')
                result['model_path']=str(trainer.base_path.base)
                pred=read_predictions(trainer.base_path.base)
                if sorted(pred.date.unique()) != split['test']:
                    raise ValueError('Predicted dates differ from the frozen test split')
                pred['data_version']=manifest['version']
                pred['model_version']=CATALOG[model]['version']
                pred['valid']=np.isfinite(pred.pred)&np.isfinite(pred.label)
                if not pred['valid'].all():
                    raise ValueError('Nonfinite prediction or label in the frozen OOS sample')
                pred.to_csv(folder/f'{name}.csv',index=False)
                result.update(status='passed',seconds=time.perf_counter()-started,
                              effective_config=json.loads(json.dumps(dict(trainer.config.Param),default=str)))
            except BaseException:
                result.update(status='failed',seconds=time.perf_counter()-started,error=traceback.format_exc())
                raise
            finally:
                result['process_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
                card.update(status=result['status'],reason=result.get('error'),model_path=result.get('model_path'))
                write_json(folder/'model_cards.json',pending)
                # Config may contain pathlib objects; serialize a stable textual snapshot.
                (folder/'runs.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2,default=str))
    return {'status':'passed','runs':len(runs),'interpretation':'Pipeline verification only' if not research else 'Research comparison'}


def compare(folder):
    folder=Path(folder)
    runs=json.loads((folder/'runs.json').read_text())
    if not runs or any(r['status'] != 'passed' for r in runs):
        raise ValueError('Every requested run must pass before comparison')
    realized=np.load(folder/'realized_returns.npz',allow_pickle=False)
    values=realized['values'].reshape(len(realized['secids']),len(realized['dates']),-1)[...,0]
    raw=pd.DataFrame(values,index=realized['secids'],columns=realized['dates']).rename_axis(index='secid',columns='date').stack().rename('return').reset_index()
    reports=[]; reference=None
    for run in runs:
        pred=pd.read_csv(folder/f"{run['name']}.csv")
        sample=set(zip(pred.date,pred.secid))
        if reference is None:
            reference=sample
        elif sample != reference:
            raise ValueError('Models predicted different samples; comparison is invalid')
        pred=pred.merge(raw,on=['secid','date'],how='left',validate='one_to_one')
        daily=[]
        for date,day in pred.groupby('date',sort=True):
            good=day.loc[np.isfinite(day.pred)&np.isfinite(day.label)]
            ic=good.pred.corr(good.label,method='spearman') if good.pred.nunique()>1 and good.label.nunique()>1 else np.nan
            available=good.loc[np.isfinite(good['return'])]
            group=np.floor((available.pred.rank(method='average')-.5)*10/max(len(available),1)).clip(upper=9)
            top=available.loc[group == 9]
            excess=float(top['return'].mean()-available['return'].mean()) if len(top) else np.nan
            daily.append({'date':int(date),'rank_ic':ic,'top_decile_10d_excess':excess,'n':len(good)})
        df=pd.DataFrame(daily);df.to_csv(folder/f"{run['name']}_daily.csv",index=False)
        std=df.rank_ic.std(ddof=0)
        reports.append({'model':run['model'],'name':run['name'],'seed':run['seed'],
                        'rank_ic':float(df.rank_ic.mean()) if df.rank_ic.notna().any() else None,
                        'icir':float(df.rank_ic.mean()/std) if std>1e-12 else None,
                        'top_decile_10d_excess':float(df.top_decile_10d_excess.mean()) if df.top_decile_10d_excess.notna().any() else None,
                        'n':int(df.n.sum()),'seconds':run['seconds'],
                        'process_peak_rss_bytes':run.get('process_peak_rss_bytes'),
                        'cost_adjusted_return':None,'cost_note':'No daily holdings/trades were produced; overlapping 10d returns are not a daily NAV'})
    write_json(folder/'comparison.json',reports)
    return {'status':'passed','models':len(reports)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['build','audit','smoke','compare','all'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    parser.add_argument('--research',action='store_true')
    parser.add_argument('--ablations',action='store_true')
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--allow-revision',action='store_true')
    parser.add_argument('--data-config',type=Path,help='JSON BuildConfig; defaults to the frozen v1 schema')
    args=parser.parse_args()
    config = json.loads(args.data_config.read_text()) if args.data_config else {}
    if 'factors' in config:
        config['factors']=tuple(config['factors'])
    cfg=BuildConfig(**config)
    if not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):
        parser.error('Invalid run ID')
    from src.proj import PATH, CALENDAR
    folder=PATH.result/'factor_sequence'/args.run_id
    folder.mkdir(parents=True,exist_ok=True)
    stages=['build','audit','smoke','compare'] if args.stage=='all' else [args.stage]
    dataset=args.dataset
    if dataset is None and (folder/'dataset.json').exists():
        dataset=Path(json.loads((folder/'dataset.json').read_text())['path'])
    for stage in stages:
        evidence=folder/f'{stage}.json'
        if evidence.exists():
            raise FileExistsError(f'Preserve existing attempt; use a new run ID: {evidence}')
        started=time.perf_counter()
        invocation={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        write_json(evidence,{'status':'running','stage':stage,'invocation':invocation})
        try:
            if stage=='build':
                from .sources import preflight, read_project
                pre=preflight(cfg);write_json(folder/'preflight.json',pre)
                if pre['status']!='passed':
                    raise ValueError('Required sources missing: '+', '.join(pre['missing']))
                end=args.end or min(f['end'] for f in pre['factors'].values())
                start=args.start or int(CALENDAR.td(end,-720))
                inputs=read_project(start,end,cfg)
                dataset=build(PATH.datablock/'factor_sequence',**inputs,cfg=cfg,parent=args.parent,allow_revision=args.allow_revision)
                write_json(folder/'dataset.json',{'path':str(dataset)})
                result={'status':'passed','dataset':str(dataset)}
            else:
                if not dataset:
                    raise ValueError('Specify --dataset or run build first')
                manifest=load(dataset)
                write_json(folder/'dataset.json',{'path':str(dataset),'version':manifest['version']})
                if stage=='audit':
                    report=audit(dataset)
                    result={'status':report['status'],'report':str(dataset/'audit.json')}
                elif stage=='smoke':
                    result=smoke(dataset,folder/'training',args.research,args.ablations)
                else:
                    result=compare(folder/'training')
            result.update(stage=stage,seconds=time.perf_counter()-started,invocation=invocation)
            write_json(evidence,result)
            print(json.dumps(result,ensure_ascii=False))
        except BaseException:
            write_json(evidence,{'status':'failed','stage':stage,'seconds':time.perf_counter()-started,'invocation':invocation,'error':traceback.format_exc()})
            raise

if __name__=='__main__':
    main()
