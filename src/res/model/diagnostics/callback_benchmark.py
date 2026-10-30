"""Isolated callback A/B evidence and optional four independent short tests."""
from __future__ import annotations
import argparse, contextlib, csv, json, os, platform, shutil, statistics, subprocess, sys, time, traceback, warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import torch
from src.res.model.callback.experimental import DEFAULTS, FLAGS, latest, basic_group_stats, options
from src.res.model.util.metric.functions import RankICFunction
from src.res.model.util.metric.stage_metrics import AttemptMetrics
from src.res.model.callback.fit.early_stop import EarlyStoppage
from src.res.model.callback.fit.retrain import BadAttemptRetrain
from src.res.model.callback.consolidate import ConsolidateCallBack
from src.res.model.util.core import BatchInput, BatchOutput, BatchData
from src.res.model.diagnostics.nn_metrics_ab import save, probe, SEED, ORDER

ROOT=Path(__file__).resolve().parents[4]

class Quiet:
    def __getattr__(self,key): return lambda *a,**k:None

class DispatchFixture(ConsolidateCallBack):
    def __init__(self, enabled, trace=None):
        self.callback_fastpath=DEFAULTS | {'dispatch':enabled}
        self.all_hooks={'on_train_batch_end','on_fit_model_restart'}
        class Child:
            logger=Quiet()
            def __init__(self,i): self.i=i
            def is_hook_implemented(self,h): return self.i%3==0
            def at_enter(self,h):
                if trace is not None: trace.append((self.i,'enter',h))
            def at_exit(self,h):
                if trace is not None: trace.append((self.i,'exit',h))
        self.callbacks=[Child(i) for i in range(10)]
        self.rebuild_dispatch()
    @property
    def logger(self): return Quiet()
    @property
    def status(self): return NS(stage='fit')
    def call(self): self.at_enter('on_train_batch_end'); self.at_exit('on_train_batch_end')

class DecisionFixture(EarlyStoppage):
    def __init__(self,enabled):
        self.callback_fastpath=DEFAULTS | {'decision_local':enabled}
        self.converge_dataset='valid';self.converge_patience=5;self.peak_patience=20
        self.peak_epoch=None;self.peak_epoch_metrics=None;self.events=[]
        self.attempt=object.__new__(AttemptMetrics)
        self.attempt.tables={'valid_epoch_totals':[pd.DataFrame({'total_accuracy':[float(i)],'total_loss':[float(i)]}) for i in range(100)]}
        self.attempt.latest_epoch=lambda:None
    @property
    def metrics(self):return NS(attempt_metrics=self.attempt,compare_epochs=lambda *a:False)
    def add_stop_event(self,*a):self.events.append(a)


class RetrainFixture(BadAttemptRetrain):
    def __init__(self,enabled):
        from src.res.model.util.metric.aggregator import MetricAggregator
        self.callback_fastpath=DEFAULTS | {'decision_local':enabled}
        self.early_exit=10; self.moderate_early_exit=20; self.moderate_min_ic=.1; self.min_ic=.05
        self.max_attempt=4; self.is_nanloss=False; self.events=[]
        self.attempt=AttemptMetrics(MetricAggregator())
        self.attempt.tables['valid_epoch_accuracies']=[pd.DataFrame({'spearman':[i/500]},index=pd.Index([i],name='epoch')) for i in range(100)]
        self.attempt.epoch_metric_results=[NS(valid_rankic=i/500) for i in range(100)]
    @property
    def metrics(self):return NS(attempt_metrics=self.attempt,model_metrics=NS(best_ic=.2))
    @property
    def status(self):return NS(end_attempt_event=NS(effective_epoch=21),loop_end=True,attempt=0)
    def trigger_retrain(self,*args):self.events.append(args)


def basic_legacy(frame):
    grouped=frame.groupby(['model_num','model_date','submodel','date'])
    def ret(df,kind):
        rtn=df.rtn_lag1_10; z=(rtn-rtn.mean())/(rtn.std()+1e-6);r=df.pred.rank(pct=True)
        mask=r>=.95 if kind=='top5pct' else r<.05 if kind=='bot5pct' else (r>=.4)&(r<.6)
        return z[mask].mean()
    result={}
    for col,label in [('rankic_rtn','rtn_lag1_10'),('rankic_std','std_lag1_10')]:
        result[col]=grouped.apply(lambda df:df.pred.corr(df[label],method='spearman'),include_groups=False)
    for col in ('top5pct','mid20pct','bot5pct'):
        result[col]=grouped.apply(lambda df:ret(df,col),include_groups=False)
    return pd.DataFrame(result)


def basic_fast(frame):
    return frame.groupby(['model_num','model_date','submodel','date']).apply(basic_group_stats,include_groups=False)


def compare(a,b,rtol=0,atol=0):
    if isinstance(a,pd.DataFrame):
        pd.testing.assert_frame_equal(a,b,rtol=rtol,atol=atol)
        a=a.to_numpy();b=b.to_numpy()
    elif isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=rtol,atol=atol,equal_nan=True)
        a=a.detach().cpu().numpy();b=b.detach().cpu().numpy()
    elif a is None:
        assert b is None
        return 0.
    a=np.asarray(a);b=np.asarray(b)
    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol,equal_nan=True)
    finite=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.


def measure(old,new,device,repeats):
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
        elif device.type=='mps':torch.mps.synchronize()
    for _ in range(5):old();new()
    sync();start=time.perf_counter();old();sync()
    loops=max(1,min(1000,int(.02/max(time.perf_counter()-start,1e-6))))
    samples=[[],[]]
    for block in range(repeats):
        for i in ((0,1) if block%2==0 else (1,0)):
            sync();start=time.perf_counter()
            for _ in range(loops):(old,new)[i]()
            sync();samples[i].append((time.perf_counter()-start)*1000/loops)
    a,b=map(statistics.median,samples);wins=sum(y<x for x,y in zip(*samples))
    return dict(legacy_ms=a,fast_ms=b,speedup=a/b,time_reduction_pct=100*(1-b/a),
                saved_ms_per_call=a-b,calls_per_block=loops,measured_calls=loops*repeats,
                saved_ms_over_measured_calls=(a-b)*loops*repeats,samples_ms=samples,
                faster_blocks=wins,eligible=wins>=int(np.ceil(7*repeats/9)) and b<a)


def run_micro(args,report):
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    def case(flag,name,old,new,**metadata):
        error=compare(old(),new(),rtol=1e-10 if flag=='basic_test' else 1e-6,atol=1e-12 if flag=='basic_test' else 1e-7)
        row=dict(flag=flag,name=name,max_abs_error=error,**metadata,**measure(old,new,device,args.repeats))
        report['rows'].append(row);persist(args.output,report)
        print(f'{name}: {row["legacy_ms"]:.4f} -> {row["fast_ms"]:.4f} ms; {row["faster_blocks"]}/{args.repeats} faster; eligible={row["eligible"]}',flush=True)
    if 'summary_latest' in args.only:
        for count in (1,50,200):
            attempt=object.__new__(AttemptMetrics)
            attempt.tables={f'{ds}_epoch_totals':[pd.DataFrame({'total_loss':[float(i)],'total_accuracy':[float(i)],'rankic':[float(i)]}) for i in range(count)] for ds in ('train','valid')}
            case('summary_latest',f'latest/{count}',lambda:[attempt.latest(ds,m) for ds in ('train','valid') for m in ('loss','accuracy','rankic')],lambda:[latest(attempt,ds,m) for ds in ('train','valid') for m in ('loss','accuracy','rankic')])
    if 'summary_hidden_rank' in args.only:
        for n in args.sizes:
            for width in (32,64):
                x=torch.randn(n,width,device=device);y=torch.randn(n,1,device=device)
                data=BatchData(BatchInput(x,y,None,torch.zeros(n,2,dtype=torch.long,device=device),torch.ones(n,1,dtype=torch.bool,device=device),np.array([1]),np.arange(n)),BatchOutput((y,{'hidden':x})))
                calc=RankICFunction()
                for threshold in (4096,5000):
                    case('summary_hidden_rank',f'hidden/{n}x{width}/{threshold}',lambda:calc.hidden_rankic(data),lambda:calc.hidden_rankic(data,hybrid_threshold=threshold),threshold=threshold)
    if 'decision_local' in args.only:
        old,new=DecisionFixture(False),DecisionFixture(True)
        case('decision_local','early_stop/valid',old.check_early_stoppage,new.check_early_stoppage)
        assert old.events==new.events
        old,new=RetrainFixture(False),RetrainFixture(True)
        case('decision_local','retrain/best_lookup',old.on_fit_epoch_end_before,new.on_fit_epoch_end_before)
        assert old.events==new.events
    if 'basic_test' in args.only:
        for n in (200,5000):
            frame=pd.DataFrame({'model_num':0,'model_date':20200101,'submodel':'best','date':np.repeat(np.arange(5),n),
                'pred':rng.integers(-30,31,n*5),'rtn_lag1_10':rng.normal(size=n*5),'std_lag1_10':rng.normal(size=n*5)})
            case('basic_test',f'basic/{n}x5',lambda:basic_legacy(frame),lambda:basic_fast(frame))
    if 'dispatch' in args.only:
        a=[];b=[];DispatchFixture(False,a).call();DispatchFixture(True,b).call();assert a==b
        old,new=DispatchFixture(False),DispatchFixture(True)
        case('dispatch','dispatch/10_callbacks',old.call,new.call)
    report['eligible_flags']=[flag for flag in args.only if (rows:=[r for r in report['rows'] if r['flag']==flag and r.get('threshold',args.threshold)==args.threshold]) and all(r['eligible'] for r in rows)]


def persist(folder,report):
    save(folder/'report.json',report)
    fields=['flag','name','threshold','max_abs_error','legacy_ms','fast_ms','speedup','time_reduction_pct','saved_ms_per_call','calls_per_block','measured_calls','saved_ms_over_measured_calls','faster_blocks','eligible']
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report.get('rows',[]))


def single(folder,name,flags,threshold):
    from src.res.model.diagnostics.callback_profile import CallbackProfile
    from src.proj import Save
    from src.res.model.model_module.application.trainer import ModelTrainer
    result=dict(name=name,flags=flags,status='failed');start=time.perf_counter()
    profile=CallbackProfile()
    try:
        with profile:
            trainer=ModelTrainer.GO(title=f'Callback AB {name}',short_test=True,stage=0,resume=0,selection=0,
                override={'model.name':name,'env.random_seed':SEED,'train.experimental_metrics_fastpath':False,
                    'train.experimental_callback_fastpath':DEFAULTS | dict.fromkeys(flags,True) | {'hidden_rank_threshold':threshold}})
            Save.async_wait_all(caller_name='callback_ab',raise_first_error=True)
        effective_flags=options(trainer.config)
        assert all(effective_flags[k] == (k in flags) for k in FLAGS), 'Requested callback switches did not take effect'
        assert trainer.config['train.experimental_metrics_fastpath'] is False
        result.update(effective_callback_options=effective_flags,git=probe(['git','rev-parse','HEAD'],ROOT),status='success',model_path=str(trainer.base_path.base.resolve()),device=str(trainer.config.device),
            effective_config=dict(trainer.config.Param),model_params=trainer.config.algo_config.params,
            model_dates=trainer.data.model_date_list.tolist(),
            stage_seconds={s:(trainer.status.times[f'{s}_end']-trainer.status.times[f'{s}_start']).total_seconds() for s in ('data','fit','test')})
    except BaseException:
        result['traceback']=traceback.format_exc();raise
    finally:
        result.update(wall_seconds=time.perf_counter()-start,profile=profile.report());save(folder/'run.json',result)


def compare_runs(runs):
    """Normalize only generated run names; require decisions and weights exact."""
    base=runs[0];checks=[]
    def norm(frame,name):
        return frame.map(lambda v:v.replace(name,'<run>') if isinstance(v,str) else v).rename(columns=lambda v:v.replace(name,'<run>') if isinstance(v,str) else v)
    for other in runs[1:]:
        a,b=base['result'],other['result'];left,right=Path(base['evidence']),Path(other['evidence'])
        excluded={'model.name','train.experimental_callback_fastpath'}
        assert {k:v for k,v in a['effective_config'].items() if k not in excluded and not k.startswith('train.experimental_callback_fastpath.')}=={k:v for k,v in b['effective_config'].items() if k not in excluded and not k.startswith('train.experimental_callback_fastpath.')}
        assert a['model_params']==b['model_params'] and a['model_dates']==b['model_dates']
        assert json.dumps(a['profile']['events']).replace(a['name'],'<run>')==json.dumps(b['profile']['events']).replace(b['name'],'<run>')
        da,db=a['profile']['diagnostics'],b['profile']['diagnostics'];assert len(da)==len(db)
        max_error=0.
        for x,y in zip(da,db):
            assert {k:v for k,v in x.items() if k!='values'}=={k:v for k,v in y.items() if k!='values'}
            max_error=max(max_error,compare(np.array(x['values']),np.array(y['values']),1e-6,1e-7))
        counts=dict(weights=0,feathers=0,sheets=0)
        for pattern in ('archive/**/*.pt','snapshot/**/*.feather','snapshot/metrics/**/*.xlsx','results/*.xlsx'):
            files=sorted(p.relative_to(left) for p in left.glob(pattern));assert files==sorted(p.relative_to(right) for p in right.glob(pattern))
            for rel in files:
                if rel.suffix=='.pt':
                    x=torch.load(left/rel,map_location='cpu',weights_only=True);y=torch.load(right/rel,map_location='cpu',weights_only=True)
                    assert x.keys()==y.keys()
                    for k in x:torch.testing.assert_close(x[k],y[k],rtol=0,atol=0,equal_nan=True)
                    counts['weights']+=len(x)
                elif rel.suffix=='.feather':
                    x=norm(pd.read_feather(left/rel),a['name']);y=norm(pd.read_feather(right/rel),b['name'])
                    is_basic='basic_test' in str(rel)
                    pd.testing.assert_frame_equal(x,y,check_exact=not is_basic,rtol=1e-10,atol=1e-12);counts['feathers']+=1
                else:
                    x=pd.read_excel(left/rel,sheet_name=None);y=pd.read_excel(right/rel,sheet_name=None);assert x.keys()==y.keys()
                    for sheet in x:
                        pd.testing.assert_frame_equal(norm(x[sheet],a['name']),norm(y[sheet],b['name']),check_exact='basic_test' not in str(rel),rtol=1e-10,atol=1e-12);counts['sheets']+=1
        assert counts['weights']>0 and counts['feathers']>0 and counts['sheets']>0,'Missing training evidence'
        checks.append(dict(name=b['name'],diagnostic_max_abs_error=max_error,**counts))
    return checks


def run_short(args,report):
    flags=report['eligible_flags'];report['short_tests']=[]
    source=args.output/'source'
    for rel in ('configs/model','src/res/model/callback','src/res/model/diagnostics','src/res/model/util/metric','src/res/model/util/trainer/base_callback.py','src/func/metric.py','src/func/tensor.py'):
        dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if (ROOT/rel).is_dir():shutil.copytree(ROOT/rel,dest,ignore=shutil.ignore_patterns('__pycache__'))
        else:shutil.copy2(ROOT/rel,dest)
    # Empty selection still runs four baseline controls, explicitly labelled.
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONHASHSEED=str(SEED),PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(args.output/'mplconfig'))
    subprocess.run([sys.executable,'-c','import matplotlib.font_manager'],env=env,cwd=ROOT,check=True)
    for label,fast in ORDER:
        folder=args.output/label;folder.mkdir()
        name=f'cb_{args.output.name}_{label}'
        command=[sys.executable,'-m','src.res.model.diagnostics.callback_benchmark','--single',str(folder),'--name',name,'--threshold',str(args.threshold),'--flags',','.join(flags if fast else [])]
        launched=time.perf_counter()
        with (folder/'console.log').open('w') as f:
            try:
                p=subprocess.run(command,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=7200);code=p.returncode
            except subprocess.TimeoutExpired:code=124
        process_seconds=time.perf_counter()-launched
        result=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else {}
        model=Path(result.get('model_path',ROOT/'models/st'/f'nn@gru@{name}'))
        evidence=folder/'model'
        for part in ('configs','archive','snapshot','results'):
            if (model/part).exists():shutil.copytree(model/part,evidence/part)
        report['short_tests'].append(dict(label=label,fast=fast,returncode=code,process_seconds=process_seconds,result=result,evidence=str(evidence)))
        persist(args.output,report);print(f'{label}: exit={code}',flush=True)
    assert all(r['returncode']==0 for r in report['short_tests']),'One or more short tests failed; see console logs'
    report['comparison']=compare_runs(report['short_tests'])
    report['training_timing']={}
    for key in ('data','fit','test','wall_seconds','process_seconds'):
        def value(r):return r[key] if key=='process_seconds' else r['result'][key] if key=='wall_seconds' else r['result']['stage_seconds'][key]
        old=[value(r) for r in report['short_tests'] if not r['fast']];new=[value(r) for r in report['short_tests'] if r['fast']]
        report['training_timing'][key]=dict(legacy=old,fast=new,time_reduction_pct=100*(1-sum(new)/sum(old)),
            conclusion='observed improvement' if max(new)<min(old) and flags else '未证实提速')


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for stream in self.streams:stream.write(s);stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams:stream.flush()


def send_results(args, report):
    """Prepare immutable mail only after the log has been closed."""
    from src.proj.util.web.task_mail import TaskResultMail
    archive=Path(shutil.make_archive(str(args.output)+'_evidence','zip',args.output))
    body=json.dumps({k:report.get(k) for k in ('status','eligible_flags','training_timing')},ensure_ascii=False)
    if not TaskResultMail(args.output/'mail')('NN callback benchmark',body,args.recipient,
            attachments=[args.output/'report.json',args.output/'summary.csv',args.output/'console.log',archive]):
        raise RuntimeError(f'Mail incomplete; retry only delivery: python -m src.proj.util.web.task_mail {args.output / "mail"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/nn_training_review'/f'callbacks_{datetime.now():%Y%m%d_%H%M%S_%f}')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--sizes',type=int,nargs='+',default=[2000,4095,4096,4097,5000])
    parser.add_argument('--only',nargs='+',choices=FLAGS,default=list(FLAGS))
    parser.add_argument('--threshold',type=int,choices=(4096,5000),default=5000)
    parser.add_argument('--short-tests',action='store_true',help='After eligibility gate, run old/new/new/old default short tests')
    parser.add_argument('--no-email',action='store_true');parser.add_argument('--recipient')
    parser.add_argument('--single',type=Path,help=argparse.SUPPRESS);parser.add_argument('--name',help=argparse.SUPPRESS);parser.add_argument('--flags',default='',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.single:
        single(args.single,args.name,[f for f in args.flags.split(',') if f],args.threshold);return
    if args.repeats<9 or any(n<1 for n in args.sizes):parser.error('repeats >= 9 and positive sizes required')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',rows=[],config=vars(args)|{'output':str(args.output)},seed=SEED,
        torch=torch.__version__,device=args.device,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        threads=torch.get_num_threads(),platform=platform.platform(),git=probe(['git','rev-parse','HEAD'],ROOT),
        git_status=probe(['git','status','--short'],ROOT),baseline='Current checkout; existing Spearman >=5000 scatter is NOT a new callback gain')
    error=None
    with (args.output/'console.log').open('w') as f,contextlib.redirect_stdout(Tee(sys.stdout,f)),contextlib.redirect_stderr(Tee(sys.stderr,f)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                run_micro(args,report)
            if args.short_tests:run_short(args,report)
            report['status']='success'
        except BaseException as exc:
            error=exc;report['status']='failed';report['traceback']=traceback.format_exc();traceback.print_exc()
        finally:persist(args.output,report)
    if not args.no_email:
        try:send_results(args,report)
        except Exception:
            if error is None:raise
            traceback.print_exc()
    print(f'Results: {args.output}')
    if error is not None:raise error

if __name__=='__main__':main()
