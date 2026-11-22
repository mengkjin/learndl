"""
Trainer metrics for the project , includes:
- calculate batch metrics
- collect batch / epoch / attempt / model metrics
"""
from __future__ import annotations
import numpy as np

from copy import copy, deepcopy
from functools import cached_property
import torch
from torch import nn
from typing import Any

from src.proj import Base
from src.res.model.util.advance.torch_compile import TorchCompiler
from src.res.model.util.core import BatchData, BatchOutput
from src.res.model.util.config import ModelConfig
from src.res.model.util.trainer import BaseTrainer , TrainerPipeline

from .functions import LossFunction, AccuracyFunction, RankICFunction, NonFiniteAccuracyError
from .aggregator import MetricAggregator , AggregatorType
from .metric_result import EpochMetricResult
from .fast_metrics import FastBatchMetrics, FastEpochMetrics, ReusingAccuracyFunction
from .stage_metrics import BatchMetrics , EpochMetrics , AttemptMetrics , ModelMetrics

__all__ = ['TrainerMetrics', 'copy_metric_function']

def _get_net(model : nn.Module | Any) -> nn.Module | None:
    net = model if isinstance(model, nn.Module) else getattr(model, 'net', None)
    return TorchCompiler.unwrap_module(net) if isinstance(net, nn.Module) else None


def copy_metric_function(function, net):
    """Copy mutable metric state and bind any network references to the candidate."""
    original = function.net
    memo = {}
    if original is not None and net is not None:
        for source, target in ((original.named_modules(), net.named_modules()),
                               (original.named_parameters(), net.named_parameters()),
                               (original.named_buffers(), net.named_buffers())):
            by_name = dict(target)
            memo.update({id(value): by_name[name] for name, value in source})
    return deepcopy(function, memo)


class TrainerMetrics(TrainerPipeline):
    """calculator of batch output"""
    
    def __init__(self , trainer : BaseTrainer | ModelConfig , **kwargs) -> None:
        self.bound_with(trainer)

        assert self.config.criterion_loss , 'criterion_loss cannot be empty'
        assert self.config.criterion_accuracy , 'criterion_accuracy cannot be empty'

        self.experimental_fastpath = bool(self.config['train.experimental_metrics_fastpath'])
        self.rankic_function = RankICFunction()
        self.aggregator = MetricAggregator()
        self.stage_metrics = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(loss={self.config.criterion_loss},metric={self.config.criterion_accuracy})'

    @property
    def batch_metrics(self) -> BatchMetrics:
        if 'batch' not in self.stage_metrics:
            cls = FastBatchMetrics if self.experimental_fastpath else BatchMetrics
            self.stage_metrics['batch'] = cls(self.aggregator)
        return self.stage_metrics['batch']

    @property
    def epoch_train_metrics(self) -> EpochMetrics:
        if 'epoch_train' not in self.stage_metrics:
            cls = FastEpochMetrics if self.experimental_fastpath else EpochMetrics
            self.stage_metrics['epoch_train'] = cls(self.aggregator)
        return self.stage_metrics['epoch_train']

    @property
    def epoch_valid_metrics(self) -> EpochMetrics:
        if 'epoch_valid' not in self.stage_metrics:
            cls = FastEpochMetrics if self.experimental_fastpath else EpochMetrics
            self.stage_metrics['epoch_valid'] = cls(self.aggregator)
        return self.stage_metrics['epoch_valid']

    @property
    def epoch_test_metrics(self) -> EpochMetrics:
        if 'epoch_test' not in self.stage_metrics:
            cls = FastEpochMetrics if self.experimental_fastpath else EpochMetrics
            self.stage_metrics['epoch_test'] = cls(self.aggregator)
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
    def which_output(self) -> Base.intNums | None: 
        return self.model_param.get('which_output' , None)
    @property
    def which_label(self) -> Base.intNums | None: 
        return self.model_param.get('which_label' , None)
    @cached_property
    def ignore_loss(self) -> list[str]:
        ignore_loss = []
        if (self.config.nn_category == 'tra'):
            ignore_loss.extend(['hidden_corr' , 'hidden_corr_deprecated'])
        return ignore_loss
    @cached_property
    def ignore_accuracy(self) -> list[str]:
        return []
    def calculate(self , dataset : Base.lit.DatasetAll , batch_key : Any , batch_data : BatchData):
        """Calculate loss(with gradient), penalty , accuracy"""
        if dataset not in ['train' , 'valid' , 'test']:
            dataset = 'test'
        self.new_batch(batch_key = batch_key)
        
        if self.config.module_type == 'nn':
            losses = self.loss_function.losses(batch_data , self.which_output , self.which_label , dataset = dataset)
        else:
            losses = {}
        accuracies = self.accuracy_function.accuracies(batch_data, self.which_output , self.which_label , dataset = dataset)
        reused_rankic = getattr(self.accuracy_function, 'last_rankic', None) if self.experimental_fastpath else None
        rankic = reused_rankic if reused_rankic is not None else self.rankic_function.rankic(batch_data)
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

    def set_accuracy_aggregator(self , aggregator : AggregatorType):
        """
        set the aggregator to calculate the total accuracy
        used before collecting
        allow dynamic accuracy scheme for different accuracy parts based on external factors, e.g. epoch number, convergence status, etc.
        aggregator shall be reset when new attempt is set, which means in batch/epoch iteration the aggregator is sticky.
        """
        #assert self.initiated , 'BatchMetrics is not initiated , please call new(batch_key) before or call close() after'
        assert not self.batch_metrics.has_metrics('total_accuracy') , 'total accuracy is already calculated, please call set_accuracy_computer() before total_accuracy is used'
        self.aggregator.inject('accuracy' , aggregator)

    def set_accuracy_verdict(self , verdict : AggregatorType):
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
        if self.experimental_fastpath:
            self.accuracy_function = ReusingAccuracyFunction(
                self.config.criterion_accuracy, net, self.ignore_accuracy,
                model_param=self.model_param, label_count=len(self.config.labels))
        else:
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
        if self.experimental_fastpath:
            self.accuracy_function = ReusingAccuracyFunction(
                self.config.criterion_accuracy, net, self.ignore_accuracy,
                model_param=self.model_param, label_count=len(self.config.labels))
        else:
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

    def evaluate_split(self, net, forward, dataset, *, loss_function, loss_weights):
        """Measure a candidate without changing live metrics, model modes, or RNG."""
        from src.res.model.util.trainer.phase_transition import (
            auxiliary_batch, evaluate_batches, preserve_module_modes, preserve_rng,
        )

        trainer = self.trainer
        evaluator = copy(self)
        evaluator.stage_metrics = {}
        evaluator.aggregator = MetricAggregator()
        evaluator.aggregator.inject('loss', loss_weights)
        evaluator.aggregator.inject('accuracy', self.model_metrics.accuracy_verdict)
        evaluator.loss_function = copy_metric_function(loss_function, net)
        evaluator.accuracy_function = copy_metric_function(self.accuracy_function, net)

        def batch_data(batch):
            batch = auxiliary_batch(trainer, batch, training=False)
            return BatchData(batch, BatchOutput(forward(batch) if len(batch) else None))

        if dataset == 'train':
            loader, epoch_metrics = trainer.data.train_dataloader, evaluator.epoch_train_metrics
        else:
            loader, epoch_metrics = trainer.data.val_dataloader, evaluator.epoch_valid_metrics
        with preserve_rng(), preserve_module_modes(net), torch.no_grad():
            if net is not None:
                net.eval()
            result = evaluate_batches(trainer, evaluator, loader(), batch_data, dataset=dataset,
                                      epoch_metrics=epoch_metrics)
            return result.metrics_dict

    def evaluate_submodel(self, predictor, submodel, source_epochs, ensembler=None):
        anchor = getattr(ensembler, 'best_epoch', None)
        record = {'attempt': self.status.attempt_key, 'submodel': submodel, 'source_epochs': source_epochs,
                  'anchor_epoch': anchor.epoch if anchor is not None else None,
                  'anchor_phase': anchor.phase if anchor is not None else None}
        record.update(getattr(ensembler, 'continuation', {}))
        loss_function = getattr(ensembler, 'loss_function', None) or self.loss_function
        loss_aggregator = getattr(ensembler, 'loss_aggregator', None)
        record['loss_objective'] = 'anchor' if loss_aggregator is not None else 'attempt_end'
        record['loss_weights'] = deepcopy((loss_aggregator or self.aggregator.loss_aggregator()).aggregator)
        record['loss_criteria'] = deepcopy(loss_function.criterions)
        for dataset in ('train', 'valid'):
            try:
                result = self.evaluate_split(predictor.net, predictor.forward, dataset,
                                             loss_function=loss_function, loss_weights=record['loss_weights'])
            except NonFiniteAccuracyError as exc:
                record.update(score=float('nan'), invalid_reason=f'{dataset}: {exc}')
                return record
            record.update(result)
        record['score'] = self.aggregator.total_accuracy(
            record['valid_accuracies'], self.model_metrics.accuracy_verdict or {})
        record['train_eval_score'] = self.aggregator.total_accuracy(
            record['train_accuracies'], self.model_metrics.accuracy_verdict or {})
        return record

    def collect_model(self):
        self.model_metrics.close()
        self.model_metrics.export(self.config.base_path.snapshot('metrics'))

    def compare_epochs(
        self , epoch0 : EpochMetricResult | None , epoch1 : EpochMetricResult | None , 
        aggregator : AggregatorType = None
    ) -> bool:
        if epoch0 is None:
            return False
        if epoch1 is None:
            return True
        return self.aggregator.larger(epoch0 , epoch1 , 'accuracy' , aggregator)

    def argmax_epochs(self , epochs : list[EpochMetricResult] , aggregator : AggregatorType = None) -> int:
        return np.argmax(self.aggregator.compile_results(epochs , 'accuracy' , aggregator)).item()

    def argmin_epochs(self , epochs : list[EpochMetricResult] , aggregator : AggregatorType = None) -> int:
        return np.argmin(self.aggregator.compile_results(epochs , 'loss' , aggregator)).item()

    def on_train_epoch_start(self):
        self.new_fit_epoch()

    def on_train_batch_end(self):
        self.collect_calculation()

    def on_validation_batch_end(self):
        self.collect_calculation()

    def on_validation_epoch_end(self):
        self.collect_fit_epoch()
        for submodel in getattr(self.model, "submodels", {}).values():
            submodel.assess(self.status, self)
