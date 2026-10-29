"""Isolated rank experiment; never patches production functions.

From repo root: python scripts/5_test/benchmark_nn_rank_scatter.py
Defaults to CUDA when available; --device cpu or --device cuda:1 overrides it.
Optional: --repeats 15 --output results/rank_server.json.
Writes JSON, CSV and a console log; does not change production code or send mail.
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
from src.func.tensor import rank, rank_pct
from src.func.metric import pearson, spearman


def rank_scatter(x, *, dim=0):
    assert x.ndim <= 3, x.shape
    if dim is None or x.ndim == 0:
        return rank_scatter(x.flatten(), dim=0).reshape(x.shape)
    order = x.argsort(dim=dim)
    shape = [1] * x.ndim
    shape[dim] = x.shape[dim]
    positions = torch.arange(x.shape[dim], device=x.device, dtype=order.dtype).reshape(shape).expand_as(order)
    inverse = torch.empty_like(order).scatter_(dim, order, positions)
    result = inverse.to(torch.float32) + 1
    result[x.isnan()] = torch.nan
    return result


def rank_pct_scatter(x, *, dim=0):
    # Preserve legacy flattening and the count of non-NaNs (including Inf).
    if dim is None:
        return rank_pct_scatter(x.flatten(), dim=0).reshape(x.shape)
    result = rank_scatter(x, dim=dim)
    return result / (~result.isnan()).sum(dim=dim, keepdim=True)


def spearman_scatter(x, y, w=None, dim=None):
    return pearson(rank_scatter(x, dim=dim), rank_scatter(y, dim=dim), w, dim)


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
                    a=spearman(x,y,w,dim); b=spearman_scatter(x,y,w,dim)
                    torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
                    if distribution != 'nonfinite':
                        assert torch.isfinite(a).all()
                    correlations+=1
    return {'rank_comparisons':count,'spearman_comparisons':correlations,'dtypes':dtype_names,'exact':True}


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
    return {'legacy_ms':a,'scatter_ms':b,'speedup':a/b,'time_reduction_pct':100*(1-b/a),
            'samples_ms':samples,'calls_per_block':loops}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/nn_training_review'/f'rank_scatter_{datetime.now():%Y%m%d_%H%M%S_%f}.json')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--repeats',type=int,default=9)
    args=parser.parse_args()
    if args.repeats < 3:
        parser.error('--repeats must be at least 3')
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if any(args.output.with_suffix(ext).exists() for ext in ('.json','.csv','.log')):
        parser.error('Output already exists; choose a new --output path')
    if args.output.suffix != '.json':
        parser.error('--output must end in .json')
    with args.output.with_suffix('.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print(f'Results: {args.output}',flush=True)
            try:
                execute(args)
            except BaseException:
                traceback.print_exc()
                raise


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
            'method':'5 warmups; 9 alternating old/new blocks by default; median milliseconds per call; data allocation excluded; CUDA synchronized; float32 timing; Spearman nonfinite-row filtering excluded',
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'cuda_version':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'status':'running','rows':[]}
    report['git_commit']=probe(['git','rev-parse','HEAD'])
    report['gpu_before']=probe(['nvidia-smi']) if device.type=='cuda' else None
    print(f"Device={device}; GPU={report['gpu']}; torch={torch.__version__}; threads={report['threads']}",flush=True)
    save_report(args.output,report)
    try:
        report['validation']=validate(device)
        print('Validation:',report['validation'],flush=True)
        with torch.no_grad():
            for n in (200,1000,5000,10000):
                for channels in (1,32):
                    for distribution in ('normal','ties_nan'):
                        x=torch.randn(n,channels,device=device)
                        y=torch.randn(n,1,device=device)
                        if distribution=='ties_nan':
                            x=(x*3).round(); y=(y*3).round()
                            x[::20]=torch.nan; y[::25]=torch.nan
                        mask=torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
                        xs,ys=x[mask],y[mask]  # Filtering is outside timing; benchmark finite correlations.
                        for op,old,new in (
                            ('rank',lambda:rank(x,dim=0),lambda:rank_scatter(x,dim=0)),
                            ('rank_pct',lambda:rank_pct(x,dim=0),lambda:rank_pct_scatter(x,dim=0)),
                            ('spearman',lambda:spearman(xs,ys,dim=0),lambda:spearman_scatter(xs,ys,dim=0))):
                            torch.testing.assert_close(old(),new(),rtol=0,atol=0,equal_nan=True)
                            row=dict(shape=[n,channels],effective_n=len(xs) if op=='spearman' else n,dim=0,distribution=distribution,operation=op,
                                     **measure(old,new,device,args.repeats))
                            report['rows'].append(row)
                            save_report(args.output,report)
                            print(f'{n}x{channels} {distribution} {op}: {row["legacy_ms"]:.4f} -> {row["scatter_ms"]:.4f} ms ({row["speedup"]:.2f}x)',flush=True)
        report['status']='success'
    except BaseException:
        report['status']='failed'
        report['traceback']=traceback.format_exc()
        raise
    finally:
        report['gpu_after']=probe(['nvidia-smi']) if device.type=='cuda' else None
        save_report(args.output,report)
        fields=['shape','effective_n','dim','distribution','operation','legacy_ms','scatter_ms','speedup','time_reduction_pct','calls_per_block']
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
