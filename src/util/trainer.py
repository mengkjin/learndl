import os , time
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass , field
from torch import Tensor
from torch.optim.swa_utils import AveragedModel , update_bn
from typing import Any , ClassVar

from ..environ import DIR
from .config import TrainConfig
from ..model import model

@dataclass
class Info:
    config    : TrainConfig
    datas     : dict[str,Any] = field(default_factory=dict)
    count     : dict[str,int] = field(default_factory=dict)
    times     : dict[str,float] = field(default_factory=dict)
    texts     : dict[str,str] = field(default_factory=dict)

    result_path : ClassVar[str] = f'{DIR.result}/model_results.yaml'

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.result_path) , exist_ok=True)
        self.tic('init')
        self.new_count()

    def add_data(self , key : str , value : Any):
        self.datas[key] = value

    def new_count(self):
        self.count['model'] = 0
        self.count['epoch'] = 0

    def add_text(self , key : str , value : str):
        self.texts[key] = value

    def add_model(self): self.count['model'] += 1
    def add_epoch(self): self.count['epoch'] += 1

    def tic(self , key : str): 
        self.times[key] = time.time()
        return 'Start Process [{}] at {:s}!'.format(key.capitalize() , time.ctime(self.times[key]))

    def toc(self , key : str , count_avgs = False): 
        self.times[f'{key}_end'] = time.time()
        spent = self.times[f'{key}_end'] - self.times[key]
        if count_avgs:
            self.texts[f'{key}'] = 'Finish Process [{}], Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
                key.capitalize() , spent / 3600 , spent / 60 / max(self.count['model'],1) , spent / max(self.count['epoch'],1)
            )
        else:
            self.texts[f'{key}'] = 'Finish Process [{}], Cost {:.1f} Secs'.format(key.capitalize() , spent)
        return self.texts[f'{key}']

    @property
    def initial_models(self): return self.count['model'] < self.config.model_num

    def dump_result(self):
        result = {
            '0_model' : f'{self.config.model_name}(x{len(self.config.model_num_list)})',
            '1_start' : time.ctime(self.times['init']) ,
            '2_basic' : 'short' if self.config.short_test else 'full' , 
            '3_datas' : self.config.model_data_type ,
            '4_label' : ','.join(self.config.labels),
            '5_dates' : '-'.join([str(self.config.beg_date),str(self.config.end_date)]),
            '6_train' : self.texts['train'],
            '7_test'  : self.texts['test'],
            '8_result': self.datas['test_score_sum'],
        }
        DIR.dump_yaml(result , self.result_path)
@dataclass
class Output:
    outputs : Tensor | tuple | list

    def pred(self):
        if isinstance(self.outputs , (list , tuple)):
            return self.outputs[0]
        else:
            return self.outputs
    
    def hidden(self):
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            return self.outputs[1]
        else:
            return None

class PTimer:
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

class Filtered:
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
            
class SWAModel:
    def __init__(self , module , param , device = None) -> None:
        self.module   = module
        self.param    = param
        self.device   = device
        self.template = model.new(module , self.param)
        self.module   = AveragedModel(self.template).to(self.device)

    def update_sd(self , state_dict):
        self.template.load_state_dict(state_dict)
        self.module.update_parameters(self.template) 
        return self
    
    def update_bn(self , data_loader):
        update_bn(self.bn_loader(data_loader) , self.module) 
        return self
    
    def bn_loader(self , data_loader):
        for batch_data in data_loader: 
            yield (batch_data.x , batch_data.y , batch_data.w)

