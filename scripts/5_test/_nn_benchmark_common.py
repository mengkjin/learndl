"""Output and delivery helpers for isolated NN experiments (not production code)."""
import argparse
import contextlib
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


def parser_for(name, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output', type=Path, default=ROOT/'results/nn_training_review'/f'{name}_{datetime.now():%Y%m%d_%H%M%S_%f}.json')
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--no-email', action='store_true')
    parser.add_argument('--recipient')
    return parser


def environment():
    import torch
    return dict(platform=platform.platform(), python=sys.version, torch=torch.__version__,
                threads=torch.get_num_threads(), cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                git_commit=subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip())


def persist(path, report):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False))
    temp.replace(path)
    rows = report.get('rows', [])
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def send_results(args, report):
    from src.proj.util.web.task_mail import TaskResultMail
    directory = args.output.with_suffix('.mail')
    body = (f"{report['experiment']}: {report['status']}\n"
            f"Completed rows: {len(report['rows'])}\n"
            f"Validation: {report.get('validation_summary', 'see JSON')}\n"
            f"Results: {args.output}\n"
            "Local function/storage costs only; not end-to-end training speedup.\n"
            f"Error: {report.get('traceback', 'none')}")
    attachments = [args.output.with_suffix(ext) for ext in ('.json','.csv','.log')]
    if not TaskResultMail(directory)(f"NN {report['experiment']} - {report['status']}", body,
                                     args.recipient, attachments=attachments):
        raise RuntimeError('Mail not fully delivered; retry only delivery with: '
                           f'python -m src.proj.util.web.task_mail {directory}')
    print(f'Mail receipt: {directory / "delivery.json"}', flush=True)


def run_cli(parser, execute, name, argv=None):
    args = parser.parse_args(argv)
    if args.repeats < 3: parser.error('--repeats must be at least 3')
    args.output = args.output.resolve()
    if args.output.suffix != '.json': parser.error('--output must end in .json')
    if any(args.output.with_suffix(ext).exists() for ext in ('.json','.csv','.log','.mail')):
        parser.error('Output exists; choose a new --output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(experiment=name, status='running', rows=[], validation=[], arguments={
        k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    failure = None
    with args.output.with_suffix('.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:
                persist(args.output, report)
                report['environment'] = environment()
                print(f'Results: {args.output}', flush=True)
                execute(args, report)
                report['status'] = 'success'
            except BaseException as exc:
                failure = exc
                report['status'] = 'failed'
                report['traceback'] = traceback.format_exc()
                traceback.print_exc()
            finally:
                persist(args.output, report)
                print(f"Complete: {report['status']}; {len(report['rows'])} timing rows", flush=True)
    # No attachment is modified after delivery begins, including failure logs.
    mail_error = None
    if not args.no_email:
        try: send_results(args, report)
        except Exception as exc:
            mail_error = exc
            print(f'Mail failed: {exc}; results remain at {args.output}', file=sys.stderr)
    if failure is not None: raise failure
    if mail_error is not None: raise mail_error
    return report


def summarize(samples, baseline):
    median = statistics.median(samples)
    return dict(samples_ms=samples, median_ms=median, min_ms=min(samples), max_ms=max(samples),
                saved_ms=baseline-median, speedup=baseline/median if median else None,
                time_reduction_pct=100*(1-median/baseline) if baseline else None)


def timed_variants(functions, repeats, sync=lambda:None, target_seconds=.02):
    """Warm every variant; rotate/reverse order; synchronize block boundaries."""
    for fn in functions.values():
        for _ in range(5): fn()
    sync()
    start = time.perf_counter()
    next(iter(functions.values()))()
    sync()
    loops = max(1,min(1000,int(target_seconds/max(time.perf_counter()-start,1e-6))))
    samples = {name:[] for name in functions}
    names = list(functions)
    for block in range(repeats):
        shift = block % len(names)
        order = names[shift:]+names[:shift]
        if block % 2: order.reverse()
        for name in order:
            sync()
            start = time.perf_counter()
            for _ in range(loops): functions[name]()
            sync()
            samples[name].append((time.perf_counter()-start)*1000/loops)
    return samples, loops
