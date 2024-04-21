import logging , os , time
import numpy as np
import pandas as pd
import torch

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass , field
from torch import nn , Tensor
from torch.optim.swa_utils import AveragedModel , update_bn
from typing import Any , ClassVar , Literal , Optional 

from .logger import Logger
from .config import TrainConfig
from .metric import Metrics , MetricList
from .store  import Storage

from ..environ import DIR

@dataclass
class Pipeline:
    config : TrainConfig
    logger : Logger | logging.Logger | logging.RootLogger
    stage  : str = ''

    epoch_counts : int = 0
    model_counts : int = 0

    nanloss : bool = False
    loop_status : str = 'model'
    conds : dict[str,Any] = field(default_factory=dict)
    texts : dict[str,str] = field(default_factory=dict)

    def __post_init__(self):
        self.nanloss_life = self.config.train_param['trainer']['nanloss']['retry']
        self.aggmetrics = AggMetrics()

    @property
    def initial_models(self): return self.model_counts < self.config.model_num

    def get(self , key , default = None): return self.__dict__.get(key , default)

    def set_stage(self , stage : Literal['data' , 'fit' , 'test']):
        self.stage = stage
        self.epoch_counts = 0
        self.model_counts = 0

    def set_dataset(self , dataset : Literal['train' , 'valid' , 'test']):
        self.dataset = dataset

    def new_loop(self):
        if self.loop_status == 'epoch':
            self.new_epoch()
        elif self.loop_status == 'attempt':
            self.new_attempt()
        elif self.loop_status == 'model':
            pass
        else:
            raise KeyError(self.loop_status)

    def new_epoch(self):
        self.epoch_i += 1
        self.epoch_all += 1
        self.epoch_counts += 1

    def new_attempt(self):
        '''Reset variables at attempt start'''
        self.nanloss = False

        self.epoch_i = -1
        self.epoch_attempt_best = -1
        self.score_attempt_best = -10000.
        self.loss_list  = {'train' : [] , 'valid' : [] , 'test' : []}
        self.score_list = {'train' : [] , 'valid' : [] , 'test' : []}
        self.lr_list    = []

        self.attempt_i += 1
        self.texts['attempt'] = f'FirstBite' if self.attempt_i == 0 else f'Retrain#{self.attempt_i}'

        self.new_epoch()

    def new_model(self , model_num , model_date):
        '''Reset variables of model start'''
        self.model_num , self.model_date = model_num , model_date

        self.model_counts += 1

        self.attempt_i = -1
        self.epoch_all = -1

        self.texts = {k : '' for k in ['model','attempt','epoch','exit','stat','time','trainer']}
        self.conds = {}

        self.texts['model'] = '{:s} #{:d} @{:4d}'.format(self.config.model_name , self.model_num , self.model_date)

        self.loop_status = 'attempt'
        
    def check_nanloss(self , is_nanloss):
        '''Deal with nanloss, life -1 and change nanloss condition to True'''
        if self.stage == 'fit' and is_nanloss: 
            self.logger.error(f'{self.texts["model"]} Attempt{self.attempt_i}, epoch{self.epoch_i} got nanloss!')
            if self.nanloss_life > 0:
                self.logger.error(f'Initialize a new model to retrain! Lives remaining {self.nanloss_life}')
                self.nanloss_life -= 1
                self.nanloss   = True
                self.attempt_i -= 1
                self.loop_status = 'attempt'
            else:
                raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')

    def new_metric(self , model_type='best'):
        self.model_type = model_type
        self.aggmetrics.new(self.dataset , self.model_num , self.model_date , self.epoch_i , model_type)
    
    def record_metric(self , metrics : Metrics):
        self.aggmetrics.record(metrics)

    def collect_metric(self):
        self.check_nanloss(self.aggmetrics.nanloss)
        self.aggmetrics.collect()
        self.loss_list[self.dataset].append(self.aggmetrics.loss) 
        self.score_list[self.dataset].append(self.aggmetrics.score)

    def collect_lr(self , last_lr): 
        if self.dataset == 'train': self.lr_list.append(last_lr)

    @property
    def losses(self): return self.aggmetrics.losses
    @property
    def scores(self): return self.aggmetrics.scores
    @property
    def aggloss(self): return self.aggmetrics.loss
    @property
    def aggscore(self): return self.aggmetrics.score


@dataclass
class Info:
    config    : TrainConfig

    times     : dict[str,float] = field(default_factory=dict)
    texts     : dict[str,str] = field(default_factory=dict)
    
    test_score : Any = None

    result_path : ClassVar[str] = f'{DIR.result}/model_results.yaml'

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.result_path) , exist_ok=True)
        self.tic('init')

    def tic(self , key : str): self.times[key] = time.time()

    def tic_str(self , key : str):
        self.tic(key)
        return 'Start Process [{}] at {:s}!'.format(key.capitalize() , time.ctime(self.times[key]))

    def toc(self , key : str): 
        return time.time() - self.times[key]
    
    def toc_str(self , key : str , model_counts = 0 , epoch_counts = 0): 
        toc = self.toc(key)
        if model_counts * epoch_counts:
            self.texts[key] = 'Finish Process [{}], Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
                key.capitalize() , toc / 3600 , toc / 60 / model_counts , toc / epoch_counts)
        else:
            self.texts[key] = 'Finish Process [{}], Cost {:.1f} Secs'.format(key.capitalize() , toc)
        return self.texts[f'{key}']

    def dump_result(self):
        result = {
            '0_model' : f'{self.config.model_name}(x{len(self.config.model_num_list)})',
            '1_start' : time.ctime(self.times['init']) ,
            '2_basic' : 'short' if self.config.short_test else 'full' , 
            '3_datas' : self.config.model_data_type ,
            '4_label' : ','.join(self.config.labels),
            '5_dates' : '-'.join([str(self.config.beg_date),str(self.config.end_date)]),
            '6_fit'   : self.texts.get('fit'),
            '7_test'  : self.texts.get('test'),
            '8_result': self.test_score,
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

class AggMetrics:
    def __init__(self) -> None:
        self.table : Optional[pd.DataFrame] = None
        self.new('init',0,0)
    def __len__(self): 
        return len(self._losses.values)
    def new(self , dataset , model_num , model_date , epoch = 0 , model_type = 'best'):
        self._params = [dataset , model_num , model_date , epoch , model_type]
        self._losses = MetricList(f'{dataset}.{model_num}.{model_date}.{epoch}.loss'  , 'loss')
        self._scores = MetricList(f'{dataset}.{model_num}.{model_date}.{epoch}.score' , 'score')
    def record(self , metrics): 
        self._losses.record(metrics)
        self._scores.record(metrics)
    def collect(self):
        df = pd.DataFrame([self._params + [self._losses.mean() , self._scores.mean()]] , 
                          columns = ['dataset','model_num','model_date','epoch','model_type','loss','score'])
        self.table = df if self.table is None else pd.concat([self.table , df]).reindex()
    @property
    def nanloss(self): return self._losses.any_nan()
    @property
    def loss(self):  return self._losses.mean()
    @property
    def score(self): return self._scores.mean()
    @property
    def losses(self): return self._losses.values
    @property
    def scores(self): return self._scores.values

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
            
class Checkpoints(Storage):
    n_epochs = 20

    def __init__(self, mem_storage: bool = True , ):
        super().__init__(mem_storage)

    def new_model(self , model_param : dict , model_date : int):
        if self.is_disk:
            self.dir = '{}/{}'.format(model_param.get('path') , model_date)
        else:
            self.dir = '{}/{}'.format(os.path.basename(str(model_param.get('path'))) , model_date)
        self.sources = [[] for _ in range(self.n_epochs)]
        self.del_all()

    def join(self , src : Any , epoch : int , net : nn.Module):
        if epoch > len(self.sources):
            self.sources += [[] for _ in range(self.n_epochs)]
        if len(self.sources[epoch]) == 0:  
            self.save_state_dict(net , self.epoch_path(epoch))
        if src not in self.sources[epoch]: 
            self.sources[epoch].append(src)

    def disjoin(self , src , epoch : int):
        if epoch is not None:
            self.sources[epoch] = [s for s in self.sources[epoch] if s is not src]
            if len(self.sources[epoch]) == 0:  self.del_path(self.epoch_path(epoch))

    def load_epoch(self , epoch):
        return self.load(self.epoch_path(epoch))
    
    def epoch_path(self , epoch):
        return f'{self.dir}/checkpoint.{epoch}.pt'
    
class SWAModel:
    def __init__(self , module : nn.Module) -> None:
        self.template = deepcopy(module)
        self.avgmodel = AveragedModel(self.template)

    def update_sd(self , state_dict):
        self.template.load_state_dict(state_dict)
        self.avgmodel.update_parameters(self.template) 
        return self
    
    def update_bn(self , data_loader , device = None):
        self.avgmodel = device(self.avgmodel) if callable(device) else self.avgmodel.to(device)
        update_bn(self.bn_loader(data_loader) , self.avgmodel) 
        return self
     
    def bn_loader(self , data_loader):
        for batch_data in data_loader: 
            yield (batch_data.x , batch_data.y , batch_data.w)

    @property
    def module(self) -> nn.Module: return self.avgmodel.module
    
class SeletedModel:
    def __init__(self, ckpt : Checkpoints , 
                 use : Literal['loss','score'] = 'score') -> None:
        self.ckpt   = ckpt
        self.use    = use

    @abstractmethod
    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        pass

    @abstractmethod
    def state_dict(self , *args , device = None) -> nn.Module | dict:
        pass

def choose_model(model_type : Literal['best' , 'swabest' , 'swalast']):
    if model_type == 'best': return BestModel
    elif model_type == 'swabest': return SWABest
    elif model_type == 'swalast': return SWALast
    else: raise KeyError(model_type)

class BestModel(SeletedModel):
    def __init__(self, ckpt : Checkpoints , use : Literal['loss','score'] = 'score') -> None:
        super().__init__(ckpt , use)
        self.epoch_fix  = -1
        self.metric_fix = None

    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        value = loss if self.use == 'loss' else score
        if self.metric_fix is None or (self.metric_fix < value if self.use == 'score' else self.metric_fix > value):
            self.ckpt.disjoin(self , self.epoch_fix)
            self.epoch_fix = epoch
            self.metric_fix = value
            self.ckpt.join(self , epoch , net)

    def state_dict(self , *args , device = None , **kwargs):
        return self.ckpt.load_epoch(self.epoch_fix)

class SWABest(SeletedModel):
    def __init__(self, ckpt : Checkpoints , use : Literal['loss','score'] = 'score' , n_best = 5) -> None:
        super().__init__(ckpt , use)
        assert n_best > 0, n_best
        self.n_best      = n_best
        self.metric_list = []
        self.candidates  = []
        
    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        value = loss if self.use == 'loss' else score
        if len(self.metric_list) == self.n_best :
            arg = np.argmin(self.metric_list) if self.use == 'score' else np.argmax(self.metric_list)
            if (self.metric_list[arg] < value if self.use == 'score' else self.metric_list[arg] > value):
                self.metric_list.pop(arg)
                candid = self.candidates.pop(arg)
                self.ckpt.disjoin(self , candid)

        if len(self.metric_list) < self.n_best:
            self.metric_list.append(value)
            self.candidates.append(epoch)
            self.ckpt.join(self , epoch , net)

    def state_dict(self , net , data_loader , *args , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: swa.update_sd(self.ckpt.load_epoch(epoch))
        swa.update_bn(data_loader , getattr(data_loader , 'device' , None))
        return swa.module.cpu().state_dict()
    
class SWALast(SeletedModel):
    def __init__(self, ckpt : Checkpoints , use : Literal['loss','score'] = 'score' ,
                 n_last = 5 , interval = 3) -> None:
        super().__init__(ckpt , use)
        assert n_last > 0 and interval > 0, (n_last , interval)
        self.n_last      = n_last
        self.interval    = interval
        self.left_epochs = (n_last // 2) * interval
        self.epoch_fix   = -1
        self.metric_fix  = None
        self.candidates  = []

    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        value = loss if self.use == 'loss' else score
        old_candidates = self.candidates
        if self.metric_fix is None or (self.metric_fix < value if self.use == 'score' else self.metric_fix > value):
            self.epoch_fix = epoch
            self.metric_fix = value
        self.candidates = list(range(self.epoch_fix - self.left_epochs , epoch + 1 , self.interval))[:self.n_last]
        for candid in old_candidates:
            if candid >= self.candidates[0]: break
            self.ckpt.disjoin(self , candid)
        self.ckpt.join(self , epoch , net)

    def state_dict(self , net , data_loader , *args , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: swa.update_sd(self.ckpt.load_epoch(epoch))
        swa.update_bn(data_loader , getattr(data_loader , 'device' , None))
        return swa.module.cpu().state_dict()



