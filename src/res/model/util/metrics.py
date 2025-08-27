import numpy as np
import pandas as pd

from dataclasses import dataclass , field
from torch import nn , Tensor
from typing import Any , Literal , Optional

from src.res.algo.nn.util import MetricsCalculator
from .batch import BatchMetric

@dataclass(slots=True)
class MetricList:
    name : str
    type : str
    values : list[Any] = field(default_factory=list) 

    def __post_init__(self): assert self.type in ['loss' , 'score']
    def record(self , metrics): self.values.append(metrics.loss_item if self.type == 'loss' else metrics.score)
    def last(self): self.values[-1]
    def mean(self): return np.mean(self.values)
    def any_nan(self): return np.isnan(self.values).any()

class Metrics:
    '''calculator of batch output'''
    VAL_DATASET : Literal['train','valid'] = 'valid'
    VAL_METRIC  : Literal['loss' ,'score'] = 'score'
    
    def __init__(self , 
                 module_type = 'nn' , nn_category = None ,
                 loss_type : Literal['mse', 'pearson', 'ccc'] = 'ccc' , 
                 score_type : Literal['mse', 'pearson', 'ccc', 'spearman'] = 'spearman',
                 penalty_kwargs : dict = {} ,
                 multilosses_type: Literal['ewa','hybrid','dwa','ruw','gls','rws'] | None = None ,
                 multilosses_param: dict[str,Any] = {} ,
                 **kwargs) -> None:
        assert 'loss' not in penalty_kwargs , 'loss is a reserved keyword for penalty'
        assert 'score' not in penalty_kwargs , 'score is a reserved keyword for penalty'
        
        self.module_type = module_type
        self.nn_category = nn_category
        self.loss_type = loss_type
        self.score_type = score_type
        self.penalty_kwargs = penalty_kwargs
        self.multi_type = multilosses_type
        self.multi_param = multilosses_param

        self.calculator    = MetricsCalculator(self.loss_type , self.score_type , self.penalty_kwargs ,
                                               multi_type = self.multi_type, multi_param = self.multi_param)
        self.output        = BatchMetric()
        self.metric_batchs = MetricsAggregator()
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}
        self.latest : dict[str,Any] = {}
        self.last_metric : Any = None
        self.best_metric : Any = None

    def __repr__(self):
        return f'{self.__class__.__name__}(loss={self.loss_type},metric={self.score_type},penalty={list(self.penalty_kwargs.keys())})'

    @property
    def loss(self): return self.output.loss
    @property
    def score(self): return self.output.score
    @property
    def loss_item(self): return self.output.loss_item
    @property
    def losses(self): return self.metric_batchs.losses
    @property
    def scores(self): return self.metric_batchs.scores
    
    def new_model(self , model_param : dict[str,Any] , net : nn.Module | None = None , **kwargs):
        self.model_param  = model_param

        self.calculator.new_model(net , penalty_conds = self.penalty_conds , num_head = self.num_output)
        
        self.new_attempt()
        return self
    
    @property
    def num_output(self): return self.model_param.get('num_output' , 1)
    @property
    def which_output(self): return self.model_param.get('which_output' , 0)
    @property
    def penalty_conds(self): 
        return {
            'hidden_corr':(self.nn_category == 'tra') or self.model_param.get('hidden_as_factors' , False)
        }

    def new_attempt(self):
        self.best_metric = None
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}

    def new_epoch(self , dataset : Literal['train','valid','test'] , model_num : int , model_date : int , 
                  model_submodel , epoch : int , **kwargs):
        self.dataset : Literal['train','valid','test'] = dataset
        self.metric_batchs.new(self.dataset , model_num , model_date , epoch , model_submodel)

    def collect_batch(self):
        self.metric_batchs.record(self.output)

    def collect_epoch(self):
        self.metric_batchs.collect()
        loss , score = self.metric_batchs.loss , self.metric_batchs.score
        self.latest[f'{self.dataset}.loss']  = loss
        self.latest[f'{self.dataset}.score'] = score
        self.metric_epochs[f'{self.dataset}.loss'].append(loss) 
        self.metric_epochs[f'{self.dataset}.score'].append(score)

        if self.dataset == self.VAL_DATASET:
            metric = loss if self.VAL_METRIC == 'loss' else score
            self.last_metric = metric
            if (self.best_metric is None or
                (self.VAL_METRIC == 'score' and self.best_metric < metric) or 
                (self.VAL_METRIC == 'loss' and self.best_metric > metric)): 
                self.best_metric = metric
    
    def better_epoch(self , old_best_epoch : Any) -> bool:
        if old_best_epoch is None:
            return True
        elif self.VAL_METRIC == 'score':
            return self.last_metric > old_best_epoch 
        else:
            return self.last_metric < old_best_epoch
        
    def better_attempt(self , old_best_attempt : Any) -> bool:
        if old_best_attempt is None:
            return True
        elif self.VAL_METRIC == 'score':
            return self.best_metric > old_best_attempt 
        else:
            return self.best_metric < old_best_attempt

    def calculate(self , dataset , label : Tensor , pred : Tensor , weight : Optional[Tensor] = None , **kwargs):
        '''Calculate loss(with gradient), penalty , score'''
        assert dataset in ['train','valid','test'] , dataset
        
        if label.ndim == 1: label = label[:,None]
        if pred.ndim  == 1: pred  = pred[:,None]
        label , pred = label[:,:self.num_output] , pred[:,:self.num_output]
        assert label.shape == pred.shape , (label.shape , pred.shape)

        inputs = {'label':label,'pred':pred,'weight':weight,**kwargs}

        self.output = BatchMetric()
        self.output.set_score(self.calculator.score(**inputs , which_head = self.which_output))
        if dataset == 'train' and self.module_type == 'nn':
            self.output.add_losses(self.calculator.loss_penalties(**inputs))
            self.output.add_losses(kwargs , prefix = ('penalty_' , 'loss_'))

        #if self.output.loss.isnan():
        #    print(self.output)
        #    raise Exception('nan loss here')
        return self

class MetricsAggregator:
    '''record a list of batch metric and perform agg operations, usually used in an epoch'''
    def __init__(self) -> None:
        self.table : Optional[pd.DataFrame] = None
        self.new('init',0,0)
    def __len__(self):  return len(self._record['loss'].values)
    def new(self , dataset , model_num , model_date , epoch = 0 , submodel = 'best'):
        self._params = [dataset , model_num , model_date , submodel , epoch]
        self._record = {m:MetricList(f'{dataset}.{model_num}.{model_date}.{submodel}.{epoch}.{m}',m) for m in ['loss','score']}
    def record(self , metrics): 
        [self._record[m].record(metrics) for m in ['loss','score']]
    def collect(self):
        df = pd.DataFrame([[*self._params , self.loss , self.score]] , columns=['dataset','model_num','model_date','submodel','epoch','loss','score'])
        self.table = df if self.table is None else pd.concat([self.table , df]).reindex()
    @property
    def nanloss(self): return self._record['loss'].any_nan()
    @property
    def loss(self):  return self._record['loss'].mean()
    @property
    def score(self): return self._record['score'].mean()
    @property
    def losses(self): return self._record['loss'].values
    @property
    def scores(self): return self._record['score'].values
