from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from torch import nn , Tensor
from typing import Any , Callable , Literal
from pathlib import Path

from src.proj import Logger
from src.proj.util.func import AsyncSaver
from src.res.model.util.core import BatchData , epoch_key , attempt_key
from src.res.model.util.config import ModelConfig

from .functions import LossFunction , AccuracyFunction , RankICFunction
from .aggregator import MetricAggregator , AggregatorType
from .metric_result import EpochMetricResult

__all__ = ['Metrics' , 'BatchMetrics' , 'EpochMetrics' , 'AttemptMetrics' , 'ModelMetrics']

MetricTypes = Literal['accuracy' , 'loss' , 'rankic']
MetricOptions : tuple[MetricTypes,...] = ('accuracy' , 'loss' , 'rankic')

def _get_net(model : nn.Module | Any) -> nn.Module | None:
    if isinstance(model , nn.Module):
        return model
    elif hasattr(model , 'net') and isinstance(getattr(model , 'net') , nn.Module):
        return getattr(model , 'net')
    else:
        return None

class Metrics:
    '''calculator of batch output'''
    VAL_DATASET : Literal['train','valid'] = 'valid'
    MetricOptions = MetricOptions
    
    def __init__(self , 
                 module_type = 'nn' , nn_category = None ,
                 criterion_loss : dict[str,dict[str,Any]] = {'mse':{}} , 
                 criterion_accuracy : dict[str,dict[str,Any]] = {'spearman':{}} , 
                 criterion_multilosses : dict[str,Any] = {} ,
                 **kwargs) -> None:
        assert len(criterion_loss) > 0 , f'{criterion_loss} should be not empty'
        assert len(criterion_accuracy) > 0 , f'{criterion_accuracy} should be only one'
        
        self.module_type = module_type
        self.nn_category = nn_category
        self.criterion_loss = criterion_loss
        self.criterion_accuracy = criterion_accuracy
        self.criterion_multilosses = criterion_multilosses

        self.rankic_function = RankICFunction()
        self.aggregator = MetricAggregator()
        self.stage_metrics = {}

        self.epoch_metrics = EpochMetrics(self.aggregator)

    def __repr__(self):
        return f'{self.__class__.__name__}(loss={self.criterion_loss},metric={self.criterion_accuracy})'

    @property
    def batch_metrics(self) -> BatchMetrics:
        if 'batch' not in self.stage_metrics:
            self.stage_metrics['batch'] = BatchMetrics(self.aggregator)
        return self.stage_metrics['batch']

    @property
    def epoch_train_metrics(self) -> EpochMetrics:
        if 'epoch_train' not in self.stage_metrics:
            self.stage_metrics['epoch_train'] = EpochMetrics(self.aggregator)
        return self.stage_metrics['epoch_train']

    @property
    def epoch_valid_metrics(self) -> EpochMetrics:
        if 'epoch_valid' not in self.stage_metrics:
            self.stage_metrics['epoch_valid'] = EpochMetrics(self.aggregator)
        return self.stage_metrics['epoch_valid']

    @property
    def epoch_test_metrics(self) -> EpochMetrics:
        if 'epoch_test' not in self.stage_metrics:
            self.stage_metrics['epoch_test'] = EpochMetrics(self.aggregator)
        return self.stage_metrics['epoch_test']

    @property
    def attempt_metrics(self) -> AttemptMetrics:
        if 'attempt' not in self.stage_metrics:
            self.stage_metrics['attempt'] = AttemptMetrics(self.aggregator)
        return self.stage_metrics['attempt']

    @property
    def model_metrics(self) -> ModelMetrics:
        if 'model' not in self.stage_metrics:
            self.stage_metrics['model'] = ModelMetrics(self.aggregator)
        return self.stage_metrics['model']

    @property
    def batch_accuracy(self): 
        return self.batch_metrics.total_accuracy
    @property
    def batch_loss(self): 
        return self.batch_metrics.total_loss_item
    
    @property
    def multilosses_kwargs(self): 
        return self.criterion_multilosses | {'num_head':self.model_param.get('num_output' , 1)}
    @property
    def which_output(self) -> int | list[int] | None: 
        return self.model_param.get('which_output' , None)
    @property
    def which_label(self) -> int | list[int] | None: 
        return self.model_param.get('which_label' , None)
    @property
    def ignore_loss(self) -> list[str]:
        if not hasattr(self , '_ignore_loss'):
            self._ignore_loss = []
            if (self.nn_category == 'tra') or self.model_param.get('hidden_as_factors' , False):
                self._ignore_loss.extend(['hidden_corr' , 'hidden_corr_deprecated'])
        return self._ignore_loss
    @property
    def ignore_accuracy(self) -> list[str]:
        if not hasattr(self , '_ignore_accuracy'):
            self._ignore_accuracy = []
        return self._ignore_accuracy

    def calculate(self , dataset : Literal['train','valid','test','predict'] , batch_key : Any , batch_data : BatchData):
        '''Calculate loss(with gradient), penalty , accuracy'''
        if dataset not in ['train' , 'valid' , 'test']:
            dataset = 'test'
        self.dataset = dataset
        self.new_batch(batch_key = batch_key)
        accuracies = self.accuracy_function.accuracies(batch_data, self.which_output , self.which_label , dataset = dataset)
        losses = self.loss_function.losses(batch_data , self.which_output , self.which_label , dataset = dataset)
        rankic = self.rankic_function.rankic(batch_data)
        self.batch_metrics.set_values(accuracies , losses , rankic)
        return self

    def collect_calculation(self):
        if self.dataset == 'train':
            self.epoch_train_metrics.append(self.batch_metrics)
        elif self.dataset == 'valid':
            self.epoch_valid_metrics.append(self.batch_metrics)
        elif self.dataset == 'test':
            self.epoch_test_metrics.append(self.batch_metrics)
        else:
            raise ValueError(f'invalid dataset: {self.dataset}')
        self.batch_metrics.close()
        return self

    def set_loss_weights(self , loss_weights : dict[str,float]):
        """
        set the weights of the losses to calculate the total loss
        used before collecting
        allow dynamic weight scheme for different loss parts based on external factors, e.g. epoch number, convergence status, etc.
        weights shall be reset when new attempt is set, which means in batch/epoch iteration the weights are sticky.
        """
        #assert self.initiated , 'BatchMetrics is not initiated , please call new(batch_key) before or call close() after'
        assert not self.batch_metrics.has_metrics('total_loss') , 'total loss is already calculated, please call set_loss_weights() before total_loss is used'
        self.aggregator.inject('loss' , loss_weights)

    def set_accuracy_aggregator(self , aggregator : dict[str,float] | Callable[[dict[str,float]],float] | None):
        """
        set the aggregator to calculate the total accuracy
        used before collecting
        allow dynamic accuracy scheme for different accuracy parts based on external factors, e.g. epoch number, convergence status, etc.
        aggregator shall be reset when new attempt is set, which means in batch/epoch iteration the aggregator is sticky.
        """
        #assert self.initiated , 'BatchMetrics is not initiated , please call new(batch_key) before or call close() after'
        assert not self.batch_metrics.has_metrics('total_accuracy') , 'total accuracy is already calculated, please call set_accuracy_computer() before total_accuracy is used'
        self.aggregator.inject('accuracy' , aggregator)

    def set_accuracy_verdict(self , verdict : dict[str,float] | Callable[[dict[str,float]],float] | None):
        """
        set the verdict to calculate the total accuracy of different attempts
        at most once in every model
        """
        #assert self.initiated , 'BatchMetrics is not initiated , please call new(batch_key) before or call close() after'
        self.model_metrics.set_accuracy_verdict(verdict)

    def new_model(self , model : nn.Module | Any , model_param : dict[str,Any] , **kwargs):
        self.model_param  = model_param
        net = _get_net(model)
        self.loss_function = LossFunction(self.criterion_loss , net , self.ignore_loss , self.multilosses_kwargs)
        self.accuracy_function = AccuracyFunction(self.criterion_accuracy , net , self.ignore_accuracy)
        
        self.model_metrics.new(**kwargs)
        self.new_attempt(**kwargs)
        return self

    def new_attempt(self , attempt : int , redo : int , next_attempt : int = 0 , next_redo : int = 0 , **kwargs):
        self.attempt_metrics.new(attempt = next_attempt , redo = next_redo , **kwargs)
        self.aggregator.reset()
        return self

    def new_fit_epoch(self , dataset , epoch : int = 0 , phase : int = 0 , **kwargs):
        self.epoch_train_metrics.new('train' , epoch , phase , **kwargs)
        self.epoch_valid_metrics.new('valid' , epoch , phase , **kwargs)
        return self
    
    def new_test_epoch(self , dataset , epoch : int = 0 , phase : int = 0 , **kwargs):
        self.epoch_test_metrics.new('test' , epoch , phase , **kwargs)
        return self

    def new_batch(self , batch_key : Any , **kwargs):
        self.batch_metrics.new(batch_key = batch_key , **kwargs)

    def collect_fit_epoch(self):
        self.attempt_metrics.append(self.epoch_train_metrics , self.epoch_valid_metrics)
        self.epoch_train_metrics.close()
        self.epoch_valid_metrics.close()
        
    def collect_test_epoch(self):
        self.epoch_test_metrics.close()
        return self

    def collect_attempt(self):
        self.model_metrics.append(self.attempt_metrics)
        self.attempt_metrics.close()

    def collect_model(self):
        self.model_metrics.close()

    def compare_epochs(self , epoch0 : EpochMetricResult | None , epoch1 : EpochMetricResult | None , aggregator : AggregatorType | None = None) -> bool:
        if epoch0 is None:
            return False
        if epoch1 is None:
            return True
        return self.aggregator.larger(epoch0 , epoch1 , 'accuracy' , aggregator)

    def argmax_epochs(self , epochs : list[EpochMetricResult] , aggregator : AggregatorType | None = None) -> int:
        return np.argmax(self.aggregator.compile_results(epochs , 'accuracy' , aggregator)).item()

    def argmin_epochs(self , epochs : list[EpochMetricResult] , aggregator : AggregatorType | None = None) -> int:
        return np.argmin(self.aggregator.compile_results(epochs , 'loss' , aggregator)).item()

    @classmethod
    def from_config(cls , config : ModelConfig) -> Metrics:
        return cls(
            config.module_type,
            config.nn_category,
            config.criterion_loss,
            config.criterion_accuracy,
            config.criterion_multilosses,
        )

class BatchMetrics:
    def __init__(self , aggregator : MetricAggregator) -> None:
        self.aggregator = aggregator
        self.metrics : dict[str,Any] = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key},accuracies={self.accuracies},losses={self.losses})'

    @property
    def initiated(self) -> bool:
        if not hasattr(self , '_initiated'):
            self._initiated = False
        return self._initiated

    @property
    def collected(self) -> bool:
        if not hasattr(self , '_collected'):
            self._collected = False
        return self._collected

    @property
    def key(self) -> dict[str,Any]:
        if not hasattr(self , '_key'):
            self._key = {}
        return self._key

    def set_initiated(self , initiated : bool):
        self._initiated = initiated

    def set_collected(self , collected : bool):
        self._collected = collected

    def set_key(self , key : dict[str,Any]):
        self.key.clear()
        self.key.update(key)

    def set_metrics(self , key : Literal['accuracies' , 'losses' , 'rankic' , 'total_loss' , 'total_accuracy' , 'total_loss_item' , 'loss_weights'] , metrics : Any):
        match key:
            case 'accuracies' | 'losses' | 'loss_weights':
                assert isinstance(metrics , dict) , f'{key} should be a dict[str,float], but got {metrics}'
            case 'rankic' | 'total_loss':
                assert isinstance(metrics , Tensor) , f'{key} should be a Tensor, but got {metrics}'
            case 'total_accuracy' | 'total_loss_item':
                assert isinstance(metrics , float) , f'{key} should be a float, but got {metrics}'
            case _:
                raise ValueError(f'Invalid metric key: {key}')
        self.metrics[key] = metrics

    def has_metrics(self , key : Literal['accuracies' , 'losses' , 'rankic' , 'total_loss' , 'total_accuracy' , 'total_loss_item' , 'loss_weights']) -> bool:
        return key in self.metrics

    def reset_metrics(self , key : str | list[str] | Literal['all']):
        if not key:
            return
        if key == 'all':
            self.metrics.clear()
            return
        key = [key] if isinstance(key , str) else key
        for k in key:
            self.metrics.pop(k , None)

    def new(self , batch_key : Any = None , **kwargs):
        assert not self.initiated , f'{self} is already initiated , please call close() first'
        self.set_initiated(True)
        self.set_collected(False)
        self.set_key({'batch':batch_key})
        self.reset_metrics(['total_loss' , 'total_accuracy' , 'total_loss_item'])

    def set_values(self , accuracies : dict[str,float] , losses : dict[str,Tensor] , rankic : Tensor):
        assert self.initiated , 'BatchMetrics is not initiated , please call new(batch_key) first'
        self.set_metrics('accuracies' , accuracies)
        self.set_metrics('losses' , losses)
        self.set_metrics('rankic' , rankic)

    def collect(self):
        self.set_collected(True)

    def close(self):
        assert self.collected , f'{self} is not collected before closing, please be appended to some metrics first'
        self.set_initiated(False)
        self.reset_metrics(['total_loss' , 'total_accuracy'])

    @property
    def losses(self) -> dict[str,Tensor]:
        return self.metrics.get('losses' , {})
    @property
    def accuracies(self) -> dict[str,float]:
        return self.metrics.get('accuracies' , {})
    @property
    def rankic(self) -> Tensor:
        return self.metrics.get('rankic' , torch.Tensor([0.]))
    @property
    def loss_weights(self) -> dict[str,float]:
        return self.aggregator.loss_weights(self.losses_items)

    @property
    def total_loss(self) -> Tensor:
        assert self.initiated , 'cannot access total loss when BatchMetrics is not initiated'
        losses = self.losses
        if not losses:
            return torch.Tensor([0.])
        if 'total_loss' not in self.metrics:
            total_loss = self.aggregator.total_loss(losses)
            self.set_metrics('total_loss' , total_loss)
            self.set_metrics('total_loss_item' , total_loss.item())
        return self.metrics['total_loss']

    @property
    def total_accuracy(self) -> float:
        if not self.accuracies:
            return np.nan
        if 'total_accuracy' not in self.metrics:
            total_accuracy = self.aggregator.total_accuracy(self.accuracies)
            self.set_metrics('total_accuracy' , total_accuracy)
        return self.metrics['total_accuracy']

    @property
    def total_loss_item(self) -> float: 
        return self.metrics.get('total_loss_item' , 0.)
    @property
    def losses_items(self) -> dict[str,float]:
        assert self.initiated , 'cannot access losses items when BatchMetrics is not initiated'
        return {key:value.item() for key,value in self.losses.items()}
    @property
    def table_index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_frame(pd.DataFrame(self.key , index = [0]))
    @property
    def table_accuracies(self) -> pd.DataFrame:
        return pd.DataFrame(self.accuracies , index = self.table_index)
    @property
    def table_losses(self) -> pd.DataFrame:
        return pd.DataFrame(self.losses_items , index = self.table_index)
    @property
    def table_weights(self) -> pd.DataFrame:
        return pd.DataFrame({key:self.loss_weights.get(key , 1.) for key in self.losses_items} , index = self.table_index)
    @property
    def table_total(self) -> pd.DataFrame:
        return pd.DataFrame({'rankic':self.rankic.item() , 'total_loss':self.total_loss_item , 'total_accuracy':self.total_accuracy} , index = self.table_index)
    @property
    def batch_key(self) -> int | str | Any:
        return self.key['batch']

class AggregatedMetrics:
    def __init__(self , aggregator : MetricAggregator) -> None:
        self.aggregator = aggregator
        self.indices : dict[str,list[str|Any]] = defaultdict(list)
        self.tables : dict[str,list[pd.DataFrame]] = defaultdict(list)
        self.collected_tables : dict[str,pd.DataFrame] = {}
    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key},tables={self.tables.keys()})'
    @property
    def initiated(self) -> bool:
        if not hasattr(self , '_initiated'):
            self._initiated = False
        return self._initiated
    @property
    def collected(self) -> bool:
        if not hasattr(self , '_collected'):
            self._collected = False
        return self._collected
    @property
    def key(self) -> dict[str,Any]:
        if not hasattr(self , '_key'):
            self._key = {}
        return self._key
    def set_initiated(self , initiated : bool):
        self._initiated = initiated
    def set_collected(self , collected : bool):
        self._collected = collected
    def set_key(self , key : dict[str,Any]):
        self.key.clear()
        self.key.update(key)
    def new(self , **kwargs):
        assert not self.initiated , f'{self} is already initiated , please call close() first'
        self.set_initiated(True)
        self.set_collected(False)
        self.set_key(kwargs)
        self.indices.clear()
        self.tables.clear()
        self.collected_tables.clear()
    def append(self , *metrics : BatchMetrics | AggregatedMetrics):
        assert self.initiated , f'{self} is not initiated , please call new() first'
        assert not any(metric.collected for metric in metrics), f'{metrics} has already collected , please call new() first'
        for metric in metrics:
            metric.collect()
        # define specific append method for different levels

    def collect(self):
        for name in self.tables:
            self.collected_tables[name] = self.get_table(name)
        self.set_collected(True)
        
    def close(self):
        assert self.collected , f'{self} is not collected before closing, please be appended to some metrics first'
        self.set_initiated(False)

    def get_table(self , name : str) -> pd.DataFrame:
        if name in self.collected_tables:
            return self.collected_tables[name]
        if name not in self.tables:
            Logger.only_once(f'{name} not found in {self.tables.keys()}', object = self , mark = 'get_table' , printer = Logger.alert1 , vb_level = 'max')
        if not self.tables[name]:
            return pd.DataFrame()
        df = pd.concat(self.tables[name])
        cols = [*self.key.keys() , *df.index.names]
        return df.assign(**self.key).reset_index(drop = False).set_index(cols)

class EpochMetrics(AggregatedMetrics):
    '''record a list of batch metric and perform agg operations, usually used in an epoch'''
    def new(self , dataset , epoch = 0 , phase = 0 , **kwargs):
        super().new(dataset = dataset , epoch = epoch , phase = phase)
    
    def append(self , batch : BatchMetrics):
        super().append(batch)
        self.indices['batch'].append(batch.batch_key)
        self.tables['accuracies'].append(batch.table_accuracies)
        self.tables['losses'].append(batch.table_losses)
        self.tables['weights'].append(batch.table_weights)
        self.tables['totals'].append(batch.table_total)

    def collect(self):
        for name in self.tables:
            self.collected_tables[name] = self.get_table(name)
        for name in ['accuracies' , 'losses' , 'weights' , 'totals']:
            self.collected_tables[f'epoch_{name}'] = self.get_epoch_table(name)
        self.set_collected(True)

    def get_epoch_table(self , name : Literal['accuracies' , 'losses' , 'weights' , 'totals'] | str) -> pd.DataFrame:
        if f'epoch_{name}' in self.collected_tables:
            return self.collected_tables[f'epoch_{name}']
        batch_table = self.get_table(name)
        epoch_table = batch_table.mean(axis=0).to_frame().T
        epoch_table.index = batch_table.index.droplevel('batch').drop_duplicates()
        return epoch_table

    @property
    def dataset(self):
        return self.key['dataset']
    @property
    def epoch(self):
        return self.key['epoch']
    @property
    def phase(self):
        return self.key['phase']
    @property
    def epoch_key(self):
        return epoch_key(self.epoch , self.phase)
    @property
    def nanloss(self): 
        return any(np.isnan(df.to_numpy()).any() for df in self.tables['totals'])

    @property
    def table_accuracies(self) -> pd.DataFrame:
        return self.get_epoch_table('accuracies')
    @property
    def table_losses(self) -> pd.DataFrame:
        return self.get_epoch_table('losses')
    @property
    def table_weights(self) -> pd.DataFrame:
        return self.get_epoch_table('weights')
    @property
    def table_totals(self) -> pd.DataFrame:
        return self.get_epoch_table('totals')

    @property
    def accuracies_dict(self) -> dict:
        return self.table_accuracies.iloc[0].to_dict()
    @property
    def losses_dict(self) -> dict:
        return self.table_losses.iloc[0].to_dict()
    @property
    def rankic(self) -> float:
        return self.table_totals['rankic'].item()
    @property
    def total_accuracy(self) -> dict[str,Any]:
        return self.table_totals['total_accuracy'].item()
    @property
    def total_loss(self) -> float:
        return self.table_totals['total_loss'].item()

    @property
    def metrics_dict(self) -> dict[str,Any]:
        return {
            f'{self.dataset}_accuracies':self.accuracies_dict,
            f'{self.dataset}_losses':self.losses_dict,
            f'{self.dataset}_rankic':self.rankic,
            f'{self.dataset}_total_accuracy':self.total_accuracy,
            f'{self.dataset}_total_loss':self.total_loss,
        }

class AttemptMetrics(AggregatedMetrics):
    '''record a list of dataset metric and perform agg operations, usually used in an attempt'''
    def new(self , attempt : int = 0 , redo : int = 0 , **kwargs):
        super().new(attempt = attempt , redo = redo)
        self.epoch_metric_results : list[EpochMetricResult] = []

    def append(self , train : EpochMetrics , valid : EpochMetrics):
        super().append(train , valid)
        assert train.epoch_key == valid.epoch_key , f'train and valid have different epoch keys: {train.epoch_key} != {valid.epoch_key}'
        self.indices[f'train_epoch'].append(train.epoch_key)
        self.indices[f'valid_epoch'].append(valid.epoch_key)
        for name in ['accuracies' , 'losses' , 'weights' , 'totals']:
            self.tables[f'train_batch_{name}'].append(train.get_table(name))
            self.tables[f'valid_batch_{name}'].append(valid.get_table(name))
            self.tables[f'train_epoch_{name}'].append(train.get_epoch_table(name))
            self.tables[f'valid_epoch_{name}'].append(valid.get_epoch_table(name))

        self.epoch_metric_results.append(EpochMetricResult(
            epoch = train.epoch, phase = train.phase, **valid.metrics_dict, **train.metrics_dict,
        ))

    def total_metrics(self , dataset : Literal['train','valid'] , metric : MetricTypes) -> list[float]:
        col = 'rankic' if metric == 'rankic' else f'total_{metric}'
        return [df[col].item() for df in self.tables[f'{dataset}_epoch_totals']]

    def latest(self , dataset : Literal['train','valid'] , metric : MetricTypes) -> float:
        metrics = self.total_metrics(dataset , metric)
        return metrics[-1] if metrics else 0.

    def latest_epoch(self) -> None | EpochMetricResult:
        if len(self.indices['valid_epoch']) != len(self.indices['train_epoch']):
            raise ValueError('valid_epoch and train_epoch have different lengths, must call after valid epoch is collected')
        if not self.epoch_metric_results:
            return None
        return self.epoch_metric_results[-1]

    def best_epoch(self , aggregator : AggregatorType | None = None) -> None | EpochMetricResult:
        """return the index of the best epoch, the IC and the accuracies of the best epoch (determined by the accuracy)"""
        metrics = self.get_table(f'valid_epoch_accuracies')
        if metrics.empty:
            return None
        argbest = self.aggregator.argbest(metrics , aggregator = aggregator)
        return self.epoch_metric_results[argbest]
            
    @property
    def attempt(self) -> int:
        return self.key['attempt']

    @property
    def redo(self) -> int:
        return self.key['redo']

    @property
    def attempt_key(self) -> str:
        return attempt_key(self.attempt , self.redo)

    @property
    def total_accuracies(self) -> list[float]:
        return [df['total_accuracy'].item() for df in self.tables['valid_totals']]

    @property
    def total_losses(self) -> list[float]:
        return [df['total_loss'].item() for df in self.tables['valid_totals']]

class ModelMetrics(AggregatedMetrics):
    def new(self , stage : Literal['fit' , 'test' , 'predict'] = 'fit' , model_num : int = 0 , model_date : int = 0 , submodel : str  = 'best' , **kwargs):
        super().new(stage = stage , model_num = model_num , model_date = model_date , submodel = submodel)
        self.attempt_metric_results : list[EpochMetricResult | None] = []
        self.accuracy_verdict : dict[str,float] | Callable[[dict[str,float]],float] | None = None

    def append(self , attempt : AttemptMetrics):
        super().append(attempt)
        self.indices['attempt'].append(attempt.attempt_key)
        for name in attempt.tables:
            self.tables[name].append(attempt.get_table(name))
        self.attempt_metric_results.append(attempt.best_epoch())

    def close(self):
        self.collect()
        super().close()

    def set_accuracy_verdict(self , verdict : dict[str,float] | Callable[[dict[str,float]],float] | None):
        self.accuracy_verdict = {} if verdict is None else verdict

    @property
    def best_ic(self) -> float:
        rankics = [metric.valid_rankic for metric in self.attempt_metric_results if metric]
        return max(rankics) if rankics else -1.

    def best_attempt_index(self) -> int:
        assert self.attempt_metric_results , 'no attempt metric results found'
        attempt_results = [metric.valid_accuracies for metric in self.attempt_metric_results if metric]
        assert attempt_results , 'no valid accuracies found'
        if len(attempt_results[0]) > 1:
            assert self.accuracy_verdict is not None , f'accuracy_verdict is not set, you should at least set one accuracy verdict for multiple accuracies {attempt_results[0]}'
        argbest = self.aggregator.argbest(self.attempt_metric_results , aggregator = self.accuracy_verdict)
        metrics = self.attempt_metric_results[argbest]
        assert metrics is not None , 'best attempt metric result is None'
        return argbest

    def best_attempt(self) -> str:
        """return the index of the best attempt, the IC and the accuracies of the best attempt (determined by the accuracy)"""
        argbest = self.best_attempt_index()
        return self.indices['attempt'][argbest]

    def best_attempt_metrics(self) -> EpochMetricResult:
        argbest = self.best_attempt_index()
        metrics = self.attempt_metric_results[argbest]
        assert metrics is not None , 'best attempt metric result is None'
        return metrics

    def export(self , export_path : Path):
        assert self.collected , f'{self} is not collected before exporting, please call close() first'
        assert export_path.suffix == '.xlsx' , f'{export_path} is not a valid excel file'
        export_path.parent.mkdir(parents = True , exist_ok = True)
        dfs = {name:self.get_table(name).reset_index(drop = False).drop(columns = ['stage' , 'model_date' , 'model_num' , 'submodel']) for name in self.tables}
        AsyncSaver.dfs(dfs , export_path)