"""Isolated rank experiment; never patches production functions.

From repo root: python scripts/5_test/benchmark_nn_rank_scatter.py
Defaults to CUDA when available; --device cpu or --device cuda:1 overrides it.
Optional: --repeats 15 --output results/rank_server.json.
Writes JSON, CSV and a console log; emails results using project mail settings.
Use --no-email for local testing; --recipient overrides the default recipient.
Compares legacy rank/rank_pct with scatter functions and Spearman with its switch
off/on (fixed threshold 5000). candidate_ms is the tested production implementation.
Full suite adds layouts, distributions, dtypes and weighted Spearman.
"""
import argparse
import contextlib
import csv
import subprocess
import sys
import traceback
from datetime import datetime
import json
import platform
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from src.func.tensor import rank, rank_pct, rank_scatter, rank_pct_scatter
from src.func.metric import spearman


def validate(device):
    count = 0
    dtype_names = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64, torch.bool):
        dtype_names.append(str(dtype))
        for shape in ((), (0,), (1,), (31,), (0,3), (7,0), (17,3), (5,3,4)):
            base = torch.randint(-3,4,shape,device=device).to(dtype)
            values = [('ties',base)]
            if dtype.is_floating_point:
                special=base.clone()
                if special.numel() >= 4:
                    special.flatten()[:4] = torch.tensor([float('nan'),float('inf'),-float('inf'),-0.],device=device,dtype=dtype)
                values += [('special',special),('all_nan',torch.full_like(base,float('nan')))]
            for variant,x in list(values):
                if x.ndim >= 2:
                    values.append((variant+'_transpose',x.transpose(0,-1)))
                if x.ndim and x.shape[0] > 1:
                    values.append((variant+'_slice',x[::2]))
            for variant,x in values:
                dims=[None,*range(x.ndim),*range(-x.ndim,0)] if x.ndim else [None,0,-1]
                for dim in dims:
                    for old,new in ((rank,rank_scatter),(rank_pct,rank_pct_scatter)):
                        a=old(x,dim=dim); b=new(x,dim=dim)
                        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
                        count+=1
    # Downstream weighted/unweighted correlations and broadcasting used by hidden diagnostics.
    correlations=0
    for channels in (1,32):
        for distribution in ('normal','ties','nonfinite'):
            x=torch.randn(257,channels,device=device)
            y=torch.randn(257,1,device=device)
            if distribution != 'normal':
                x=x.round(decimals=1); y=y.round(decimals=1)
            if distribution == 'nonfinite':
                x[0]=torch.nan; y[1]=torch.inf
            for dim in (None,0,-2):
                for w in (None,torch.rand(257,1,device=device)):
                    a=spearman(x,y,w,dim,scatter_enabled=False); b=spearman(x,y,w,dim)
                    torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
                    if distribution != 'nonfinite':
                        assert torch.isfinite(a).all()
                    correlations+=1
    return {'rank_comparisons':count,'spearman_comparisons':correlations,'dtypes':dtype_names,'exact':True}


def scenarios(device,sizes,suite):
    for n in sizes:
        for channels in (1,32):
            x=torch.randn(n,channels,device=device)
            y=torch.randn(n,1,device=device)
            yield dict(name=f'{n}x{channels}/normal',dim=0,distribution='normal'),x,y,None
            x=(x*3).round(); y=(y*3).round()
            x[::20]=torch.nan; y[::25]=torch.nan
            yield dict(name=f'{n}x{channels}/ties_nan',dim=0,distribution='ties_nan'),x,y,None
            mask=torch.isfinite(x).all(dim=1)&torch.isfinite(y).all(dim=1)
            yield dict(name=f'{n}x{channels}/ties_filtered',dim=0,distribution='ties_filtered'),x[mask],y[mask],None
    if suite=='basic':
        return
    # Controlled supplements at two sizes: same cross-section, alternate memory
    # layouts, sorting axes, input precision, ordering and weighting.
    for n in (2000,5000):
        for variant in ('all_equal','sorted','reverse','nan_half','infinity',
                        'transpose','last_contiguous','strided','3d','flatten',
                        'float16','float64','weighted'):
            x=torch.randn(n,32,device=device); y=torch.randn_like(x)
            dim=0; w=None
            if variant=='all_equal': x.zero_(); y.zero_()
            elif variant=='sorted': x=x.sort(dim=0).values
            elif variant=='reverse': x=x.sort(dim=0,descending=True).values
            elif variant=='nan_half': x[::2]=torch.nan
            elif variant=='infinity': x[::3]=torch.inf; x[1::3]=-torch.inf
            elif variant in ('transpose','last_contiguous'):
                x=x.T; y=y.T; dim=-1
                if variant=='last_contiguous': x=x.contiguous(); y=y.contiguous()
            elif variant=='strided':
                x=torch.randn(n*2,64,device=device)[::2,::2]
                y=torch.randn(n*2,64,device=device)[::2,::2]
            elif variant=='3d': x=x.reshape(n,4,8); y=y.reshape(n,4,8)
            elif variant=='flatten': dim=None
            elif variant in ('float16','float64'):
                dtype=getattr(torch,variant); x=x.to(dtype); y=y.to(dtype)
            elif variant=='weighted': w=torch.rand(n,1,device=device)
            yield dict(name=f'{n}x32/{variant}',dim=dim,distribution=variant),x,y,w


def measure(old,new,device,repeats):
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
        elif device.type=='mps': torch.mps.synchronize()
    for _ in range(5): old(); new()
    sync()
    start=time.perf_counter(); old(); sync()
    loops=max(1,min(1000,int(.02/max(time.perf_counter()-start,1e-6))))
    samples=[[],[]]
    for block in range(repeats):
        for idx in ((0,1) if block%2==0 else (1,0)):
            fn=(old,new)[idx]
            sync(); start=time.perf_counter()
            for _ in range(loops): fn()
            sync(); samples[idx].append((time.perf_counter()-start)*1000/loops)
    a,b=map(statistics.median,samples)
    return {'legacy_ms':a,'candidate_ms':b,'speedup':a/b,'time_reduction_pct':100*(1-b/a),
            'samples_ms':samples,'legacy_min_ms':min(samples[0]),'legacy_max_ms':max(samples[0]),
            'candidate_min_ms':min(samples[1]),'candidate_max_ms':max(samples[1]),
            'candidate_faster_blocks':sum(b<a for a,b in zip(*samples)),'calls_per_block':loops}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/nn_training_review'/f'rank_scatter_{datetime.now():%Y%m%d_%H%M%S_%f}.json')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--suite',choices=('basic','full'),default='full')
    parser.add_argument('--sizes',type=int,nargs='+',default=[200,2000,4999,5000,10000])
    parser.add_argument('--no-email',action='store_true')
    parser.add_argument('--recipient',default=None)
    args=parser.parse_args()
    if any(n < 1 for n in args.sizes):
        parser.error('--sizes must be positive')
    if args.repeats < 3:
        parser.error('--repeats must be at least 3')
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if any(args.output.with_suffix(ext).exists() for ext in ('.json','.csv','.log')):
        parser.error('Output already exists; choose a new --output path')
    if args.output.suffix != '.json':
        parser.error('--output must end in .json')
    failure=None
    with args.output.with_suffix('.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print(f'Results: {args.output}',flush=True)
            try:
                execute(args)
            except BaseException as exc:
                failure=exc
                traceback.print_exc()
                if not args.output.exists():
                    save_report(args.output,{'status':'failed','device':args.device,'rows':[],
                                             'traceback':traceback.format_exc()})
    # Mail only after the log is closed; failure evidence is attached as well.
    mail_error=None
    if not args.no_email:
        try:
            send_results(args)
        except Exception as exc:
            mail_error=exc
            print(f'Mail failed: {exc}; results remain at {args.output}',file=sys.stderr)
    if failure is not None:
        raise failure
    if mail_error is not None:
        raise mail_error


def send_results(args):
    from src.proj.util.web.task_mail import TaskResultMail
    report=json.loads(args.output.read_text())
    rows=report.get('rows',[])
    body=(f"NN rank/scatter benchmark: {report['status']}\n"
          f"Device: {report.get('device')}; GPU: {report.get('gpu')}\n"
          f"Validation: {report.get('validation')}\nCompleted timing cases: {len(rows)}\n"
          f"Result: {args.output}\n"
          "speedup > 1 means faster; this measures functions, not end-to-end training.\n"
          f"Error: {report.get('traceback','none')}")
    directory=args.output.with_suffix('.mail')
    sender=TaskResultMail(directory)
    attachments=[args.output.with_suffix(ext) for ext in ('.json','.csv','.log')
                 if args.output.with_suffix(ext).is_file()]
    if not sender(f"NN rank benchmark - {report['status']}",body,args.recipient,attachments=attachments):
        raise RuntimeError(f'Mail not fully delivered; retry only delivery with: '
                           f'python -m src.proj.util.web.task_mail {directory}')
    print(f'Result mail sent; receipt: {directory / "delivery.json"}',flush=True)


def execute(args):
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type not in ('cpu','cuda'):
        raise ValueError('Supported benchmark devices: cpu, cuda, cuda:N')
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
        if device.index is None:
            device=torch.device('cuda',torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.manual_seed(20261029)
    report={'torch':torch.__version__,'device':str(device),'platform':platform.platform(),
            'threads':torch.get_num_threads(),'seed':20261029,'repeats':args.repeats,
            'cuda_available':torch.cuda.is_available(),
            'method':'5 warmups; 9 alternating old/new blocks by default; median milliseconds per call; data allocation excluded; CUDA synchronized; dtype recorded per case; Spearman nonfinite-row filtering excluded',
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'cuda_version':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'spearman_scatter_threshold':5000,'sample_labels':['legacy','candidate'],
            'status':'running','suite':args.suite,'sizes':args.sizes,'rows':[]}
    report['git_commit']=probe(['git','rev-parse','HEAD'])
    report['gpu_before']=probe(['nvidia-smi']) if device.type=='cuda' else None
    print(f"Device={device}; GPU={report['gpu']}; torch={torch.__version__}; threads={report['threads']}",flush=True)
    save_report(args.output,report)
    try:
        report['validation']=validate(device)
        print('Validation:',report['validation'],flush=True)
        with torch.no_grad():
            for case,x,y,weight in scenarios(device,args.sizes,args.suite):
                dim=case['dim']
                for op,old,new in (
                    ('rank',lambda:rank(x,dim=dim),lambda:rank_scatter(x,dim=dim)),
                    ('rank_pct',lambda:rank_pct(x,dim=dim),lambda:rank_pct_scatter(x,dim=dim)),
                    ('spearman',lambda:spearman(x,y,weight,dim,scatter_enabled=False),lambda:spearman(x,y,weight,dim))):
                    # Raw nonfinite inputs are useful for ranking; finite filtered
                    # Spearman is measured separately to avoid trivial NaN comparisons.
                    if op=='spearman' and not torch.isfinite(x).all():
                        continue
                    report['current_case']=dict(case,operation=op)
                    save_report(args.output,report)
                    torch.testing.assert_close(old(),new(),rtol=0,atol=0,equal_nan=True)
                    row=dict(case,shape=list(x.shape),dtype=str(x.dtype),strides=list(x.stride()),
                             contiguous=x.is_contiguous(),weighted=weight is not None,operation=op,
                             **measure(old,new,device,args.repeats))
                    row['sort_length']=x.numel() if dim is None else x.shape[dim]
                    report['rows'].append(row)
                    save_report(args.output,report)
                    print(f"{case['name']} {op}: {row['legacy_ms']:.4f} -> {row['candidate_ms']:.4f} ms "
                          f"({row['speedup']:.2f}x; time reduction {row['time_reduction_pct']:.1f}%)",flush=True)
        report['status']='success'
    except BaseException:
        report['status']='failed'
        report['traceback']=traceback.format_exc()
        raise
    finally:
        report['gpu_after']=probe(['nvidia-smi']) if device.type=='cuda' else None
        save_report(args.output,report)
        fields=['name','shape','dtype','strides','contiguous','weighted','dim','distribution','operation','sort_length','legacy_ms','candidate_ms','speedup','time_reduction_pct','calls_per_block']
        with args.output.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader(); writer.writerows(report['rows'])
    print('Complete: JSON raw samples, CSV summary and console log saved. Speedup > 1 means faster.',flush=True)


def probe(command):
    try:
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except OSError as exc:
        return {'error':str(exc)}
    except subprocess.TimeoutExpired:
        return {'error':'probe timed out'}


def save_report(path,report):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2))
    temp.replace(path)


class Tee:
    def __init__(self,*streams):
        self.streams=streams
    def write(self,text):
        for stream in self.streams:
            stream.write(text); stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


if __name__=='__main__':
    main()
