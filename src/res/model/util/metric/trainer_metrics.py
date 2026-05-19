from __future__ import annotations
import numpy as np

from functools import cached_property
from torch import nn
from typing import Any , Callable , Literal

from src.res.model.util.core import BatchData
from src.res.model.util.config import ModelConfig
from src.res.model.util.trainer import BaseTrainer , TrainerPipeline

from .functions import LossFunction , AccuracyFunction , RankICFunction
from .aggregator import MetricAggregator , AggregatorType
from .metric_result import EpochMetricResult
from .stage_metrics import BatchMetrics , EpochMetrics , AttemptMetrics , ModelMetrics

__all__ = ['TrainerMetrics']

MetricTypes = Literal['accuracy' , 'loss' , 'rankic']

def _get_net(model : nn.Module | Any) -> nn.Module | None:
    if isinstance(model , nn.Module):
        return model
    elif hasattr(model , 'net') and isinstance(getattr(model , 'net') , nn.Module):
        return getattr(model , 'net')
    else:
        return None
class TrainerMetrics(TrainerPipeline):
    '''calculator of batch output'''
    MetricOptions : tuple[MetricTypes,...] = ('accuracy' , 'loss' , 'rankic')
    
    def __init__(self , trainer : BaseTrainer | ModelConfig , **kwargs) -> None:
        self.bound_with(trainer)

        assert self.config.criterion_loss , 'criterion_loss cannot be empty'
        assert self.config.criterion_accuracy , 'criterion_accuracy cannot be empty'

        self.rankic_function = RankICFunction()
        self.aggregator = MetricAggregator()
        self.stage_metrics = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(loss={self.config.criterion_loss},metric={self.config.criterion_accuracy})'

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
        return self.config.criterion_multilosses | {'num_head':self.model_param.get('num_output' , 1)}
    @property
    def which_output(self) -> int | list[int] | None: 
        return self.model_param.get('which_output' , None)
    @property
    def which_label(self) -> int | list[int] | None: 
        return self.model_param.get('which_label' , None)
    @cached_property
    def ignore_loss(self) -> list[str]:
        ignore_loss = []
        if (self.config.nn_category == 'tra') or self.model_param.get('hidden_as_factors' , False):
            ignore_loss.extend(['hidden_corr' , 'hidden_corr_deprecated'])
        return ignore_loss
    @cached_property
    def ignore_accuracy(self) -> list[str]:
        return []
    def calculate(self , dataset : Literal['train','valid','test','predict'] , batch_key : Any , batch_data : BatchData):
        '''Calculate loss(with gradient), penalty , accuracy'''
        if dataset not in ['train' , 'valid' , 'test']:
            dataset = 'test'
        self.new_batch(batch_key = batch_key)
        
        if self.config.module_type == 'nn':
            losses = self.loss_function.losses(batch_data , self.which_output , self.which_label , dataset = dataset)
        else:
            losses = {}
        accuracies = self.accuracy_function.accuracies(batch_data, self.which_output , self.which_label , dataset = dataset)
        rankic = self.rankic_function.rankic(batch_data)
        self.batch_metrics.set_values(losses , accuracies , rankic)
        return self

    def collect_calculation(self):
        if self.status.dataset == 'train':
            self.epoch_train_metrics.append(self.batch_metrics)
        elif self.status.dataset == 'valid':
            self.epoch_valid_metrics.append(self.batch_metrics)
        elif self.status.dataset == 'test':
            self.epoch_test_metrics.append(self.batch_metrics)
        else:
            raise ValueError(f'invalid dataset: {self.status.dataset}')
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

    def new_model(self , model : nn.Module | Any , model_param : dict[str,Any]):
        self.model_param  = model_param
        net = _get_net(model)
        self.loss_function = LossFunction(self.config.criterion_loss , net , self.ignore_loss , self.multilosses_kwargs)
        self.accuracy_function = AccuracyFunction(self.config.criterion_accuracy , net , self.ignore_accuracy)
        
        self.model_metrics.new(self.status.stage , self.status.model_num , self.status.model_date , self.status.model_submodel)
        self.new_attempt()
        return self

    def new_attempt(self):
        self.attempt_metrics.new(attempt = self.status.next_attempt , redo = self.status.next_redo)
        self.aggregator.reset()
        return self

    def new_fit_epoch(self):
        self.epoch_train_metrics.new('train' , self.status.epoch , self.status.phase)
        self.epoch_valid_metrics.new('valid' , self.status.epoch , self.status.phase)
        return self
    
    def new_test_epoch(self):
        self.epoch_test_metrics.new('test' , self.status.epoch , self.status.phase)
        return self

    def new_batch(self , batch_key : Any):
        self.batch_metrics.new(batch_key = batch_key)

    def new_in_test_mode(self , model : nn.Module | Any , model_param : dict[str,Any] , batch_key : Any = 'test'):
        self.model_param  = model_param
        net = _get_net(model)
        self.loss_function = LossFunction(self.config.criterion_loss , net , self.ignore_loss , self.multilosses_kwargs)
        self.accuracy_function = AccuracyFunction(self.config.criterion_accuracy , net , self.ignore_accuracy)
        self.model_metrics.new()
        self.attempt_metrics.new()
        self.epoch_train_metrics.new('train')
        self.epoch_valid_metrics.new('valid')
        self.batch_metrics.new(batch_key = 'test')
        return self

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
        self.model_metrics.export(self.config.base_path.snapshot('metrics'))

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

    def on_train_epoch_start(self):
        self.new_fit_epoch()

    def on_train_batch_end(self):
        self.collect_calculation()

    def on_validation_batch_end(self):
        self.collect_calculation()

    def on_validation_epoch_end(self):
        self.collect_fit_epoch()