from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from torch import Tensor
from typing import Any , Callable , Literal
from pathlib import Path

from src.proj.util.io.async_save import AsyncSaver
from src.res.model.util.core import epoch_key , attempt_key

from .aggregator import MetricAggregator , AggregatorType
from .metric_result import EpochMetricResult

__all__ = ['BatchMetrics' , 'EpochMetrics' , 'AttemptMetrics' , 'ModelMetrics']

MetricTypes = Literal['accuracy' , 'loss' , 'rankic']

class BatchMetrics:
    def __init__(self , aggregator : MetricAggregator) -> None:
        self.aggregator = aggregator
        self.metrics : dict[str,Any] = {}
        self.initiated = False
        self.collected = False
        self.key = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key},accuracies={self.accuracies},losses={self.losses})'

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
        self.initiated = True
        self.collected = False
        self.key = {'batch':batch_key}
        self.reset_metrics(['total_loss' , 'total_accuracy' , 'total_loss_item'])

    def set_values(self , losses : dict[str,Tensor] , accuracies : dict[str,float] , rankic : Tensor):
        assert self.initiated , 'BatchMetrics is not initiated , please call new(batch_key) first'
        self.set_metrics('losses' , losses)
        self.set_metrics('accuracies' , accuracies)
        self.set_metrics('rankic' , rankic)

    def collect(self):
        self.collected = True

    def close(self):
        assert self.collected , f'{self} is not collected before closing, please be appended to some metrics first'
        self.initiated = False
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
        self.initiated = False
        self.collected = False
        self.key = {}
    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key},tables={self.tables.keys()})'

    def new(self , **kwargs):
        assert not self.initiated , f'{self} is already initiated , please call close() first'
        self.initiated = True
        self.collected = False
        self.key = kwargs
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
        self.collected = True
        
    def close(self):
        assert self.collected , f'{self} is not collected before closing, please be appended to some metrics first'
        self.initiated = False

    def get_table(self , name : str) -> pd.DataFrame:
        if name in self.collected_tables:
            return self.collected_tables[name]
        if name not in self.tables:
            from src.proj import Logger
            Logger.only_once(f'{name} not found in {self.tables.keys()}', object = self , mark = 'get_table' , printer = 'alert1' , vb_level = 'max')
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
        self.collected = True

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
        return [df['total_accuracy'].item() for df in self.tables['valid_epoch_totals']]

    @property
    def total_losses(self) -> list[float]:
        return [df['total_loss'].item() for df in self.tables['valid_epoch_totals']]

class ModelMetrics(AggregatedMetrics):
    def new(self , stage : Literal['fit' , 'test'] | Any = 'fit' , model_num : int = 0 , model_date : int = 0 , submodel : str  = 'best' , **kwargs):
        assert stage in ['fit' , 'test'] , f'[{stage}] stage is not allowed to be used for model metrics'
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
        export_path.mkdir(parents = True , exist_ok = True)
        export_path = export_path.joinpath(f'model.{self.key["model_num"]}.{self.key["model_date"]}.xlsx')
        assert export_path.suffix == '.xlsx' , f'{export_path} is not a valid excel file'
        dfs = {name:self.get_table(name).reset_index(drop = False).drop(columns = ['stage' , 'model_date' , 'model_num' , 'submodel']) for name in self.tables}
        AsyncSaver.dfs(dfs , export_path)