"""Isolated HiddenCorr accuracy/gradient/speed experiment; production is unchanged.

uv run python scripts/5_test/benchmark_nn_hidden_corr.py --device cuda
Default: FP32 timings, FP32/FP64 validation, JSON/CSV/log and result email.
A numerical failure preserves all evidence, continues other cases, then exits nonzero.
"""
import copy
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
    return dict(passed=bool(close.all()), max_abs_error=float(difference.max()) if difference.numel() else None,
                max_rel_error=float(relative.max()) if relative.numel() else None,
                failed_count=int((~close).sum()), first_failed_indices=bad[:5].cpu().tolist(),
                actual_nonfinite=int((~torch.isfinite(a)).sum()), expected_nonfinite=int((~torch.isfinite(b)).sum()),
                rtol=rtol, atol=atol)


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
        persist(args.output,report)
    sync=(lambda:torch.cuda.synchronize(device)) if device.type=='cuda' else lambda:None
    for n in args.sizes:
        for d in args.hidden_dims:
            x=torch.randn(n,d,device=device,requires_grad=True)
            checks=check_case(x,f'timing_{n}x{d}')
            report['validation'].extend(checks)
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
    persist(args.output,report)
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
