"""Independent-process paired benchmarks for default generators."""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.proj import Save
from src.res.factor.fmp.builder import PortfolioGroupBuilder
from src.res.factor.fmp.generator_array import run_generator
from src.res.factor.fmp.prepared_market import prepare_market
from src.res.model.diagnostics.t50_benchmark import load_inputs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--category',choices=['top','screen','revscreen','reinforce'])
    p.add_argument('--mode',choices=['legacy','prior','prepared'])
    p.add_argument('--start',type=int,default=20240101)
    p.add_argument('--end',type=int,default=20241231)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--prior-source',type=Path,default=Path('results/nn_training_review/generator_stage2/t50_before.py'))
    args=p.parse_args()
    alpha, universe, metadata=load_inputs('gru_day_V1','top-1000',args.start,args.end)
    root=args.output/'market'
    if args.mode:
        begin=perf_counter()
        group=PortfolioGroupBuilder(args.category,alpha,[universe],analytic=False,attribution=False,
            trade_engine='yale',n_best=50,start=args.start,end=args.end,vb_level='never')
        if args.mode=='prior':
            path=args.prior_source
            spec=importlib.util.spec_from_file_location('prior_t50',path)
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            frame=module.run_t50(group,'array')
        else:
            frame=run_generator(group,'legacy' if args.mode=='legacy' else 'array',
                                prepared_root=root if args.mode=='prepared' else None)
        Save.async_wait_all(caller_name='generator_benchmark')
        elapsed=perf_counter()-begin
        frame.reset_index().to_feather(args.output/f'{args.category}-{args.mode}-account.feather')
        group.builders[0].portfolio.to_dataframe().reset_index(drop=True).to_feather(args.output/f'{args.category}-{args.mode}-portfolio.feather')
        print(json.dumps(dict(category=args.category,mode=args.mode,seconds=elapsed,
                             diagnostics=group.t50_diagnostics)))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(inputs=metadata,preparation=prepare_market(root,alpha.available_dates().dates),runs=[],
        note='Fresh process for each measured run; OS cache not flushed. Common prediction/universe input loading excluded. Includes market pack reads/validation, external signals, generation, accounting and normal output conversion. Preparation excluded and reported separately. Stored universe flags only; historical warnst incomplete.')
    for repeat in range(args.repeats):
        for category in ['top','screen','revscreen','reinforce']:
            modes=['legacy','prior','prepared'] if category=='top' and args.prior_source.exists() else ['legacy','prepared']
            if repeat%2:modes=modes[::-1]
            for mode in modes:
                proc=subprocess.run([sys.executable,'-m','src.res.model.diagnostics.generator_benchmark',
                    '--output',str(args.output),'--category',category,'--mode',mode,
                    '--start',str(args.start),'--end',str(args.end),'--prior-source',str(args.prior_source)],capture_output=True,text=True)
                if proc.returncode:
                    raise RuntimeError(proc.stdout[-2000:]+proc.stderr[-4000:])
                row=json.loads(proc.stdout.strip().splitlines()[-1]);row['repeat']=repeat
                report['runs'].append(row)
                print(category,mode,round(row['seconds'],3),flush=True)
            for mode in modes:
                for part in ['account','portfolio']:
                    old=pd.read_feather(args.output/f'{category}-legacy-{part}.feather')
                    new=pd.read_feather(args.output/f'{category}-{mode}-{part}.feather')
                    pd.testing.assert_frame_equal(old,new,check_exact=part=='account',rtol=1e-10,atol=1e-12)
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
    report['equivalent']=True
    report['medians']={category:{mode:statistics.median(r['seconds'] for r in report['runs'] if r['category']==category and r['mode']==mode)
        for mode in (['legacy','prior','prepared'] if category=='top' and args.prior_source.exists() else ['legacy','prepared'])}
        for category in ['top','screen','revscreen','reinforce']}
    report['ranges']={category:{mode:[min(r['seconds'] for r in report['runs'] if r['category']==category and r['mode']==mode),
        max(r['seconds'] for r in report['runs'] if r['category']==category and r['mode']==mode)]
        for mode in report['medians'][category]} for category in report['medians']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['medians'],indent=2))

if __name__=='__main__':main()
