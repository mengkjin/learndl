"""Isolated HiddenCorr accuracy/gradient/speed experiment; production is unchanged.

uv run python scripts/5_test/benchmark_nn_hidden_corr.py --device cuda
Default: FP32 timings, FP32/FP64 validation, JSON/CSV/log and result email.
Precision summaries appear in the console and email; .accuracy.csv contains
per-check errors, failure counts and worst-location baseline/candidate values.
A numerical failure preserves all evidence, continues other cases, then exits nonzero.
"""
import copy
import math
import numpy as np
import statistics
import warnings

# Also support the project's file-based script launcher.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nn_benchmark_common import parser_for, run_cli, persist, timed_variants, summarize
import torch
from src.res.algo.nn.loss.loss import HiddenCorr

BASELINE = HiddenCorr()
TOLERANCES = {torch.float32:(1e-5,1e-6), torch.float64:(1e-9,1e-11)}


def unpack(hidden):
    return torch.cat(hidden, dim=-1) if isinstance(hidden,(tuple,list)) else hidden


def legacy(hidden):
    return BASELINE(hidden=hidden)


def joint_stats(hidden):
    hidden = unpack(hidden)
    std, mean = torch.std_mean(hidden, dim=0, correction=1, keepdim=True)
    z = (hidden-mean)/(std+1e-6)
    return z.T.cov().norm()


def centered_covariance(hidden):
    hidden = unpack(hidden)
    std, mean = torch.std_mean(hidden, dim=0, correction=1, keepdim=True)
    centered = hidden-mean
    residual = centered.sum(dim=0, keepdim=True)
    n = hidden.shape[0]
    covariance = (centered.T @ centered - residual.T @ residual / n) / max(n-1,0)
    denominator = std+1e-6
    return (covariance / (denominator.T @ denominator)).norm()


FUNCTIONS = {'legacy':legacy, 'joint_stats':joint_stats, 'centered_covariance':centered_covariance}


def compare(actual, expected, dtype, *, finite_required=True):
    """JSON-safe diagnostics: retain locations, never relax tolerances."""
    rtol, atol = TOLERANCES[dtype]
    a, b = actual.detach(), expected.detach()
    finite = torch.isfinite(a) & torch.isfinite(b)
    close = torch.isclose(a,b,rtol=rtol,atol=atol,equal_nan=not finite_required)
    if finite_required: close = close & finite
    difference = (a[finite]-b[finite]).abs().double()
    relative = difference / b[finite].abs().double().clamp_min(torch.finfo(dtype).tiny)
    bad = (~close).nonzero(as_tuple=False)
    # Scalar mismatch nonzero() has shape (1,0); retain [[]] as its location.
    def number(value):
        value=float(value)
        return value if math.isfinite(value) else str(value)
    total=a.numel()
    diff=(a.double()-b.double()).abs()
    allowed=atol+rtol*b.double().abs()
    ratio=diff/allowed
    worst=None
    if finite.any():
        flat=int(ratio.masked_fill(~finite,-1).reshape(-1).argmax())
        index=list(np.unravel_index(flat,tuple(a.shape))) if a.ndim else []
        worst=dict(index=[int(i) for i in index],baseline=number(b.reshape(-1)[flat]),
                   candidate=number(a.reshape(-1)[flat]),abs_error=number(diff.reshape(-1)[flat]),
                   allowed_error=number(allowed.reshape(-1)[flat]),tolerance_ratio=number(ratio.reshape(-1)[flat]))
    return dict(passed=bool(close.all()), max_abs_error=float(difference.max()) if difference.numel() else None,
                mean_abs_error=float(difference.mean()) if difference.numel() else None,
                max_rel_error=number(relative.max()) if relative.numel() else None,
                total_count=total, failed_count=int((~close).sum()),
                failed_pct=100*int((~close).sum())/total if total else 0.,
                worst_tolerance=worst, first_failed_indices=bad[:5].cpu().tolist(),
                actual_nonfinite=int((~torch.isfinite(a)).sum()), expected_nonfinite=int((~torch.isfinite(b)).sum()),
                rtol=rtol, atol=atol)


def accuracy_report(report):
    """Flatten accuracy checks for CSV and readable console/email summaries."""
    rows=[]
    for case in report.get('validation',[]):
        checks=case.get('checks') or {k:case[k] for k in ('loss','gradient') if k in case}
        if not checks:
            checks={'exception':dict(passed=case['passed'],baseline_error=case.get('baseline_error'),
                                     candidate_error=case.get('candidate_error'))}
        for metric,check in checks.items():
            row=dict(candidate=case['candidate'],dtype=case['dtype'],case=case['case'],
                     exceptional=case.get('exceptional',False),metric=metric,**check)
            worst=row.pop('worst_tolerance',None) or {}
            row.update({f'worst_{k}':v for k,v in worst.items()})
            rows.append(row)
    report['accuracy_rows']=rows
    groups={}
    for row in rows:
        metric=row['metric']
        category='parameter_gradient' if metric.startswith('gradient/') else (
            'updated_parameter' if metric.startswith('updated_parameter/') else metric)
        groups.setdefault((row['candidate'],row['dtype'],category,row['exceptional']),[]).append(row)
    lines=['Accuracy: |candidate - baseline| <= atol + rtol * |baseline|.',
           'Tolerance ratio <= 1 passes; relative error alone can be large near zero.',
           'Exceptional cases test NaN/Inf/error compatibility, not usable numerical accuracy.']
    def fmt(value):
        return f'{value:.3e}' if isinstance(value,(float,int)) else str(value if value is not None else 'n/a')
    for (candidate,dtype,category,exceptional),items in groups.items():
        failures=[r for r in items if not r['passed']]
        count=sum(r.get('failed_count',0) for r in items)
        total=sum(r.get('total_count',0) for r in items)
        max_abs=max((r['max_abs_error'] for r in items if r.get('max_abs_error') is not None),default=None)
        finite_ratios=[r for r in items if isinstance(r.get('worst_tolerance_ratio'),(float,int))]
        worst=max(finite_ratios,key=lambda r:r['worst_tolerance_ratio'],default=None)
        lines.append(f'{candidate} {dtype} {category}{" [exceptional]" if exceptional else ""}: '
                     f'{"FAIL" if failures else "PASS"}; failed checks={len(failures)}/{len(items)}, '
                     f'elements={count}/{total}; max_abs={fmt(max_abs)}; '
                     f'max_tolerance_ratio={fmt(worst.get("worst_tolerance_ratio") if worst else None)}')
        # Show the worst finite difference even on passing checks, not just a bool.
        if worst:
            lines.append(f'  worst: {worst["case"]} / {worst["metric"]}, index={worst["worst_index"]}; '
                         f'baseline={worst["worst_baseline"]}, candidate={worst["worst_candidate"]}, '
                         f'abs_diff={fmt(worst["worst_abs_error"])}, allowed={fmt(worst["worst_allowed_error"])}')
        nonfinite=sum(r.get('actual_nonfinite',0) for r in items)
        if nonfinite: lines.append(f'  candidate nonfinite elements={nonfinite}; consult accuracy CSV for compatibility results.')
        if failures:
            lines.append('  failed scenarios: '+', '.join(dict.fromkeys(r['case'] for r in failures)))
    report['accuracy_summary']='\n'.join(lines)
    return report['accuracy_summary']



def value_and_gradient(fn, value, container=None):
    x = value.detach().clone().requires_grad_(True)
    argument = x
    if container:
        argument = container(torch.tensor_split(x,2,dim=-1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        loss = fn(argument)
        gradient, = torch.autograd.grad(loss,x)
    return loss, gradient, [str(w.message) for w in caught]


def check_case(value, name, *, exceptional=False, container=None):
    results = {}
    for label, fn in FUNCTIONS.items():
        try:
            loss, grad, notices = value_and_gradient(fn,value,container)
            results[label] = dict(loss=loss,gradient=grad,warnings=notices)
        except Exception as exc:
            results[label] = dict(error=type(exc).__name__,message=str(exc))
    reference = results['legacy']
    rows = []
    for label in list(FUNCTIONS)[1:]:
        result = results[label]
        row = dict(case=name, candidate=label, dtype=str(value.dtype),shape=list(value.shape),
                   exceptional=exceptional, warnings=result.get('warnings',[]))
        if 'error' in reference or 'error' in result:
            row.update(passed=exceptional and result.get('error')==reference.get('error'),
                       baseline_error=reference.get('error'),candidate_error=result.get('error'),
                       candidate_message=result.get('message'))
        else:
            for field in ('loss','gradient'):
                row[field] = compare(result[field],reference[field],value.dtype,finite_required=not exceptional)
            row['passed'] = row['loss']['passed'] and row['gradient']['passed']
        rows.append(row)
    return rows


def validation_cases(device,dtype):
    x = torch.randn(97,32,device=device,dtype=dtype)
    yield 'random', x, False, None
    yield 'strong_correlation', x[:,:1].expand_as(x)+x*1e-3, False, None
    yield 'duplicate_columns', x[:,:1].expand_as(x), False, None
    y=x.clone(); y[:,0]=2
    yield 'constant_column',y,False,None
    yield 'all_constant',torch.ones_like(x),False,None
    yield 'near_zero_variance',x*1e-8,False,None
    yield 'large_offset',x+1e4,False,None
    for scale in (1e-4,1e4): yield f'scale_{scale}',x*scale,False,None
    yield 'noncontiguous',torch.randn(32,97,device=device,dtype=dtype).T,False,None
    for container in (list,tuple): yield container.__name__,x,False,container
    for n in (0,1,2): yield f'samples_{n}',x[:n],n<2,None
    for special in (float('nan'),float('inf')):
        y=x.clone(); y[0,0]=special
        yield str(special),y,True,None


def validate_network(device,dtype):
    torch.manual_seed(901)
    template=torch.nn.GRU(6,32,batch_first=True).to(device=device,dtype=dtype)
    data=torch.randn(17,5,6,device=device,dtype=dtype)
    outputs={}
    for name,fn in FUNCTIONS.items():
        net=copy.deepcopy(template)
        optimizer=torch.optim.Adam(net.parameters(),lr=.005)
        hidden=net(data)[0][:,-1,:]
        loss=fn(hidden)
        loss.backward()
        grads={k:p.grad.detach().clone() for k,p in net.named_parameters()}
        optimizer.step()
        outputs[name]=(loss.detach(),grads,{k:p.detach().clone() for k,p in net.named_parameters()})
    ref=outputs['legacy']; rows=[]
    for name in list(FUNCTIONS)[1:]:
        loss,grads,params=outputs[name]
        checks={'loss':compare(loss,ref[0],dtype)}
        for k in grads:
            checks['gradient/'+k]=compare(grads[k],ref[1][k],dtype)
            checks['updated_parameter/'+k]=compare(params[k],ref[2][k],dtype)
        rows.append(dict(case='gru_backward_adam_step',candidate=name,dtype=str(dtype),
                         checks=checks,passed=all(v['passed'] for v in checks.values())))
    return rows


def timed_call(fn,x,backward):
    if backward:
        # Fresh graph per call, no .grad accumulation and no retained graph.
        loss=fn(x)
        torch.autograd.grad(loss,x)
    else:
        with torch.no_grad(): fn(x)


def execute(args,report):
    if any(n<2 for n in args.sizes) or any(d<1 for d in args.hidden_dims):
        raise ValueError('Timing sizes must be >=2 and hidden dimensions positive')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type not in ('cpu','cuda'): raise ValueError('Supported devices: cpu, cuda, cuda:N')
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
    report.update(device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                  seed=20261030,method='5 warmups; rotated/reversed variant blocks; block-boundary synchronization; median ms/call',
                  timing_dtype='torch.float32',tolerance_policy='fixed; nonfinite normal-case values/gradients fail')
    torch.manual_seed(20261030)
    for dtype in TOLERANCES:
        for name,x,exceptional,container in validation_cases(device,dtype):
            report['validation'].extend(check_case(x,name,exceptional=exceptional,container=container))
        report['validation'].extend(validate_network(device,dtype))
        accuracy_report(report)
        persist(args.output,report)
    print(accuracy_report(report),flush=True)
    sync=(lambda:torch.cuda.synchronize(device)) if device.type=='cuda' else lambda:None
    for n in args.sizes:
        for d in args.hidden_dims:
            x=torch.randn(n,d,device=device,requires_grad=True)
            checks=check_case(x,f'timing_{n}x{d}')
            report['validation'].extend(checks)
            accuracy_report(report)
            for backward in (False,True):
                calls={label:lambda fn=fn:timed_call(fn,x,backward) for label,fn in FUNCTIONS.items()}
                samples,loops=timed_variants(calls,args.repeats,sync)
                baseline=statistics.median(samples['legacy'])
                for label,fn in calls.items():
                    peak=None
                    if device.type=='cuda':
                        sync(); before=torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        fn(); sync()
                        peak=torch.cuda.max_memory_allocated(device)-before
                    row=dict(samples=n,hidden_dim=d,mode='forward_backward' if backward else 'forward',
                             implementation=label,calls_per_block=loops,cuda_peak_extra_bytes=peak,
                             **summarize(samples[label],baseline))
                    report['rows'].append(row)
                    print(f'{n}x{d} {row["mode"]} {label}: {row["median_ms"]:.4f} ms; saved {row["saved_ms"]:.4f} ms',flush=True)
                persist(args.output,report)
    qualification={name:all(row['passed'] for row in report['validation'] if row['candidate']==name)
                   for name in list(FUNCTIONS)[1:]}
    report['validation_summary']=qualification
    for row in report['rows']:
        row['qualified']=row['implementation']=='legacy' or qualification[row['implementation']]
    print(accuracy_report(report),flush=True)
    persist(args.output,report)
    print(f'Accuracy details: {args.output.with_suffix(".accuracy.csv")}',flush=True)
    print('Candidate qualification:',qualification,flush=True)
    if not all(qualification.values()):
        raise RuntimeError('Candidate validation failed; timings are diagnostic only. See JSON validation entries.')


def main(argv=None):
    parser=parser_for('hidden_corr',__doc__)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--sizes',type=int,nargs='+',default=[200,1000,4096,5000])
    parser.add_argument('--hidden-dims',type=int,nargs='+',default=[32,64,128])
    return run_cli(parser,execute,'HiddenCorr',argv)


if __name__=='__main__': main()
