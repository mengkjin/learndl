import torch
import numpy as np
import pandas as pd
import gc , time , os , psutil
from copy import deepcopy
class Timer:
    def __init__(self , *args):
        self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        print(self.key , '...', end='')
    def __exit__(self, type, value, trace):
        print(f'... cost {time.time()-self.start_time:.2f} secs')

class ProcessTimer:
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {} if record else None

    class ptimer:
        def __init__(self , target_dict = None , *args):
            self.target_dict = target_dict
            if self.target_dict is not None:
                self.key = '/'.join(args)
                if self.key not in self.target_dict.keys():
                    self.target_dict[self.key] = []
        def __enter__(self):
            if self.target_dict is not None:
                self.start_time = time.time()
        def __exit__(self, type, value, trace):
            if self.target_dict is not None:
                time_cost = time.time() - self.start_time
                self.target_dict[self.key].append(time_cost)

    def __call__(self , *args):
        return self.ptimer(self.recorder , *args)
    
    def print(self):
        if self.recorder is not None:
            keys = list(self.recorder.keys())
            num_calls = [len(self.recorder[k]) for k in keys]
            total_time = [np.sum(self.recorder[k]) for k in keys]
            tb = pd.DataFrame({'keys':keys , 'num_calls': num_calls, 'total_time': total_time})
            tb['avg_time'] = tb['total_time'] / tb['num_calls']
            print(tb.sort_values(by=['total_time'],ascending=False))
            
class MemoryPrinter:
    def __init__(self) -> None:
        pass
    def __repr__(self) -> str:
        return 'Used: {:.2f}G; Free {:.2f}G'.format(
            float(psutil.virtual_memory().used)/1024**3,
            float(psutil.virtual_memory().free)/1024**3)
    def print(self):
        print(self.__repr__())
        
class FilteredIterator:
    def __init__(self, iterable, condition):
        self.iterable  = iter(iterable)
        self.condition = condition if callable(condition) else iter(condition)
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: return item
        
class SwiftStorage():
    def __init__(self , *args):
        if len(args) > 0 and args[0] in ['disk' , 'mem']:
            self.activate(args[0])

    def activate(self , default = 'mem'):
        assert default in ['disk' , 'mem']
        self.default = default
        self.mem_disk = dict()
        self.file_record = list()
        self.file_group = dict()
    
    def save(self , obj , paths , to_disk = False , group = 'default'):
        for p in self._pathlist(paths): 
            self._saveone(obj , p , self.default == 'disk' or to_disk)
            self._addrecord(p , group)
            
    def load(self , path , from_disk = False):
        if self.default == 'disk' or from_disk:
            return torch.load(path) if os.path.exists(path) else None
        else:
            return self.mem_disk.get(path)

    def _pathlist(self , p):
        if p is None: 
            return []
        elif isinstance(p , str):
            p = [p]
        return p
    
    def _saveone(self , obj , p , to_disk = False):
        if to_disk:
            torch.save(obj , p)
        else:
            self.mem_disk[p] = deepcopy(obj)
    
    def _addrecord(self , p , group):
        self.file_record = np.union1d(self.file_record , p)
        if group not in self.file_group.keys(): 
            self.file_group[group] = [p]
        else:
            self.file_group[group] = np.union1d(self.file_group[group] , [p])
    
    def save_model_state(self , net , paths , to_disk = False , group = 'default'):
        if isinstance(net , torch.nn.Module):
            if self.default == 'disk' or to_disk:
                sd = net.state_dict() 
            else:
                sd = deepcopy(net).cpu().state_dict()
        elif isinstance(net , dict):
            sd = net
        self.save(sd , paths , to_disk , group)
        
    def load_model_state(self , net , path , from_disk = False):
        sd = self.load(path , from_disk)
        net.load_state_dict(sd)
        return net
            
    def valid_paths(self , paths):
        return np.intersect1d(self._pathlist(paths) ,  self.file_record).tolist()
    
    def del_path(self , *args):
        for paths in args:
            if self.default == 'disk':
                [os.remove(p) for p in self._pathlist(paths) if os.path.exists(p)]
            else:
                [self.mem_disk.__delitem__(p) for p in np.intersect1d(self._pathlist(paths) , list(self.mem_disk.keys()))]
            self.file_record = np.setdiff1d(self.file_record , paths)
        gc.collect()
        
    def del_group(self , clear_groups = []):
        for g in self._pathlist(clear_groups):
            paths = self.file_group.get(g)
            if paths is not None:
                self.del_path(paths)
                del self.file_group[g]
