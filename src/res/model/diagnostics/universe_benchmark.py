"""Isolated real-snapshot universe IO benchmark (no missing-flag synthesis)."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
from src.proj import DB, Load
from src.res.factor.util.classes.universe import Universe
from src.res.factor.util.classes.portfolio import Portfolio
from src.res.factor.util.classes.prepared_universe import PreparedUniverse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['old','new'])
    args=parser.parse_args()
    name='top-1000'; exclusions='st_lowprice_bse_loser'
    dates=[int(d) for d in DB.dates('universe',name) if 20240101<=d<=20241231]
    store=PreparedUniverse(args.output/'prepared',name)
    if args.mode:
        t=time.perf_counter()
        if args.mode=='old':
            frames=[Load.df(DB.path('universe',name,d)).assign(date=d) for d in dates]
        else:
            frames=[store.load(dates,exclusions)]
        read=time.perf_counter()-t
        t=time.perf_counter(); frame=Universe._combine_caches_with_exclusion(frames,exclusions)
        combine=time.perf_counter()-t
        t=time.perf_counter(); port=Portfolio.from_dataframe(frame,name=name)
        construct=time.perf_counter()-t
        repeat_begin=time.perf_counter()
        if args.mode == 'old':
            # Original callback preloads once; FMP.universe(load=True).get('all')
            # repeats the filtering and Portfolio construction twice.
            for _ in range(2):
                repeat_frame=Universe._combine_caches_with_exclusion(frames,exclusions)
                Portfolio.from_dataframe(repeat_frame,name=name)
        repeat_seconds=time.perf_counter()-repeat_begin
        frame.reset_index(drop=True).to_feather(args.output/f'{args.mode}.feather')
        print(json.dumps(dict(read=read,combine=combine,portfolio=construct,total=read+combine+construct,callback_universe_total=read+combine+construct+repeat_seconds,repeat_construct=repeat_seconds,days=len(port.ports))))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    preparation=store.prepare(dates[:-5])
    incremental=store.prepare(dates)
    report=dict(preparation=preparation,incremental=incremental,no_change=store.prepare(dates),dates=len(dates),
        exclusions=exclusions,limitation='Stored flags only; warnst coverage is incomplete. Not an end-to-end callback benchmark.',runs=[])
    for i in range(3):
        for mode in (['old','new'] if i%2==0 else ['new','old']):
            proc=subprocess.run([sys.executable,'-m',__name__ if __name__!='__main__' else 'src.res.model.diagnostics.universe_benchmark','--output',str(args.output),'--mode',mode],capture_output=True,text=True,check=True)
            row=json.loads(proc.stdout.strip().splitlines()[-1]);row['mode']=mode;report['runs'].append(row)
        pd.testing.assert_frame_equal(pd.read_feather(args.output/'old.feather'),pd.read_feather(args.output/'new.feather'))
    report['median']={mode:{key:statistics.median(r[key] for r in report['runs'] if r['mode']==mode) for key in ['read','combine','portfolio','total','repeat_construct','callback_universe_total']} for mode in ['old','new']}
    report['equivalent']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
