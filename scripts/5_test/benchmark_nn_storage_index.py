"""Isolated pandas/dictionary batch-index experiment; no production cache changes.

uv run python scripts/5_test/benchmark_nn_storage_index.py
All disk operations use a temporary directory. Default email includes JSON/CSV/log.
Metadata cases carry None (no tensor I/O); total cases use identical 32 KiB CPU tensors.
"""
import gc
import statistics
import tempfile
import time
import tracemalloc

# Also support the project's file-based script launcher.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nn_benchmark_common import parser_for, run_cli, persist, summarize
import pandas as pd
import torch
from src.proj.util.functional.device import Device
from src.res.model.util.storage.torch_file import TorchFileStorage


class IsolatedLegacy(TorchFileStorage):
    def __init__(self, mem_storage, directory):
        super().__init__(mem_storage)
        self.directory=Path(directory)
    def real_path(self,key):
        return self.directory/f'{key}.pt'


class DictStorage(IsolatedLegacy):
    """Experiment only: payload behavior copied verbatim, index updates replaced.

    records is a fresh query/export snapshot. External mutation is not supported;
    current production consumers only query it. Overwriting moves a key to the end.
    """
    def __init__(self,mem_storage,directory):
        self.is_mem=mem_storage
        self.memdisk={}
        self.directory=Path(directory)
        self.entries={}
        self.groups={}

    @property
    def records(self):
        if not self.entries:
            return pd.DataFrame(columns=pd.Index(['key','path','group']),dtype=str)
        return pd.DataFrame(list(self.entries.values()),columns=['key','path','group'],
                            index=[0]*len(self.entries))

    def index_save(self,key,path,group):
        if key in self.entries:
            old=self.entries.pop(key)['group']
            self.groups[old].pop(key)
            if not self.groups[old]: self.groups.pop(old)
        self.entries[key]=dict(key=key,path=str(path),group=group)
        self.groups.setdefault(group,{})[key]=None

    def save(self,obj,key,group='default',to_disk=None):
        if self.is_mem:
            path=key
            self.memdisk[key]=Device.send_to(obj,'cpu',copy=True)
        else:
            path=self.real_path(key)
            torch.save(obj,path)
            self.memdisk[key]=1
        self.index_save(key,path,group)

    def del_one(self,key):
        if not self.is_mem: self.real_path(key).unlink(missing_ok=True)
        self.memdisk.pop(key)  # Match legacy KeyError on absent key.
        old=self.entries.pop(key)
        self.groups[old['group']].pop(key)
        if not self.groups[old['group']]: self.groups.pop(old['group'])

    def del_group(self,group):
        for key in list(self.groups.get(group,{})): self.del_one(key)

    def del_all(self):
        for key in list(self.entries): self.del_one(key)


def validate(directory, checks=None):
    if checks is None: checks=[]
    for mem in (True,False):
        # Separate roots: compare relative paths, never let variants share files.
        roots=[Path(directory)/f'validate_{mem}_{i}' for i in range(2)]
        for root in roots: root.mkdir()
        stores=[IsolatedLegacy(mem,roots[0]),DictStorage(mem,roots[1])]
        operations=[('save',('a','g0')),('save',('b','g1')),('save',('c','g0')),
                    ('save',('a','g0')),('save',('b','g0')),('save',('c','g2')),
                    ('del_one',('missing',)),('del_group',('missing',)),
                    ('del_group',('g0',)),('del_group',('g0',)),('save',('a','g3')),
                    ('del_one',('c',)),('del_all',()),('del_all',())]
        for step,(operation,args) in enumerate(operations):
            errors=[]
            for store in stores:
                try:
                    if operation=='save': store.save(torch.tensor([float(step)]),args[0],args[1])
                    else: getattr(store,operation)(*args)
                    errors.append(None)
                except Exception as exc: errors.append(type(exc).__name__)
            assert errors[0]==errors[1],(operation,errors)
            frames=[]
            for store,root in zip(stores,roots):
                frame=store.records.copy()
                if not mem: frame['path']=frame['path'].map(lambda p:str(Path(p).relative_to(root)))
                frames.append(frame)
            pd.testing.assert_frame_equal(*frames)
            assert stores[0].keys==stores[1].keys
            for key in stores[0].keys:
                assert stores[0].exists(key)==stores[1].exists(key)
                torch.testing.assert_close(stores[0].load(key),stores[1].load(key),rtol=0,atol=0)
            if not mem:
                assert sorted(p.name for p in roots[0].iterdir())==sorted(p.name for p in roots[1].iterdir())
            checks.append(dict(mode='memory' if mem else 'disk',step=step,operation=operation,passed=True))
    return checks


def populate(store,n,payload):
    """Untimed fixture setup: construct equivalent records in bulk.

    Actual timed operations always use the implementation's save/delete/load APIs.
    This avoids repeatedly timing setup while preparing a deletion fixture.
    """
    rows=[]
    for i in range(n):
        key=f'b{i}'; group=f'g{i%4}'
        if store.is_mem:
            path=key
            store.memdisk[key]=Device.send_to(payload,'cpu',copy=True)
        else:
            path=store.real_path(key)
            torch.save(payload,path)
            store.memdisk[key]=1
        if isinstance(store,DictStorage): store.index_save(key,path,group)
        else: rows.append(dict(key=key,path=str(path),group=group))
    if not isinstance(store,DictStorage) and rows:
        store.records=pd.DataFrame(rows,columns=['key','path','group'],index=[0]*n)


def operation(store,name,n,payload):
    if name in ('add','overwrite','cross_group'):
        for i in range(n):
            store.save(payload,f'b{i}',f'g{i%4}' if name!='cross_group' else f'g{(i+1)%4}')
    elif name=='delete_one': store.del_one(f'b{n//2}')
    elif name=='delete_group': store.del_group('g0')
    elif name=='clear': store.del_all()
    elif name=='dataframe': return store.records
    elif name=='load':
        for i in range(n): store.load(f'b{i}')
    else: raise ValueError(name)


def measure_operation(directory,mode,name,n,repeats):
    classes={'legacy':IsolatedLegacy,'dict':DictStorage}
    samples={label:[] for label in classes}
    payload=None if mode=='metadata' else torch.arange(8192,dtype=torch.float32)
    mem=mode!='disk'
    # One untimed warmup for each variant; mutations get a fresh equivalent fixture.
    for repeat in range(-1,repeats):
        labels=list(classes) if repeat%2==0 else list(reversed(classes))
        for label in labels:
            with tempfile.TemporaryDirectory(dir=directory) as root:
                store=classes[label](mem,root)
                if name!='add': populate(store,n,payload)
                start=time.perf_counter()
                result=operation(store,name,n,payload)
                elapsed=(time.perf_counter()-start)*1000
                if repeat>=0: samples[label].append(elapsed)
                del result,store
    return samples


def index_memory(directory,n):
    """Traced live Python/index allocations; no tensor payload and no file I/O."""
    result={}
    for label,cls in (('legacy',IsolatedLegacy),('dict',DictStorage)):
        gc.collect()
        tracemalloc.start()
        before=tracemalloc.get_traced_memory()[0]
        store=cls(True,directory)
        populate(store,n,None)
        current,peak=tracemalloc.get_traced_memory()
        frame=store.records
        result[label]=dict(traced_live_bytes=current-before,traced_peak_bytes=peak-before,
                           dataframe_deep_bytes=int(frame.memory_usage(index=True,deep=True).sum()))
        tracemalloc.stop()
        del frame,store
    return result


def execute(args,report):
    if any(n<1 for n in args.sizes) or args.storage_size<1:
        raise ValueError('Sizes must be positive')
    report.update(payload_bytes=32768,method='fresh fixtures outside timing; one warmup; alternating variants; ms per entire operation',
                  memory_method='tracemalloc live/peak index allocations; None payload; not process RSS',
                  dataframe_contract='query snapshot; no external mutation support',memory=[])
    with tempfile.TemporaryDirectory(prefix='nn_storage_index_') as directory:
        validate(directory, report['validation'])
        report['validation_summary']=dict(passed=True,checks=len(report['validation']))
        persist(args.output,report)
        jobs=[('metadata',n,name) for n in args.sizes for name in
              ('add','overwrite','cross_group','delete_one','delete_group','clear','dataframe')]
        jobs += [(mode,args.storage_size,name) for mode in ('memory','disk') for name in
                 ('add','overwrite','cross_group','delete_one','delete_group','clear','dataframe','load')]
        for mode,n,name in jobs:
            samples=measure_operation(directory,mode,name,n,args.repeats)
            baseline=statistics.median(samples['legacy'])
            for label in samples:
                row=dict(mode=mode,records=n,operation=name,implementation=label,
                         **summarize(samples[label],baseline))
                report['rows'].append(row)
                print(f'{mode} {n} {name} {label}: {row["median_ms"]:.3f} ms; saved {row["saved_ms"]:.3f} ms',flush=True)
            persist(args.output,report)
        for n in args.sizes:
            report['memory'].append(dict(records=n,**index_memory(directory,n)))
        report['savings_480']=[{k:r[k] for k in ('mode','operation','saved_ms','median_ms')}
                               for r in report['rows'] if r['records']==480 and r['implementation']=='dict']
        print('480-record absolute savings:',report['savings_480'],flush=True)
        persist(args.output,report)


def main(argv=None):
    parser=parser_for('storage_index',__doc__)
    parser.add_argument('--sizes',type=int,nargs='+',default=[100,480,2400])
    parser.add_argument('--storage-size',type=int,default=480)
    return run_cli(parser,execute,'StorageIndex',argv)


if __name__=='__main__': main()
