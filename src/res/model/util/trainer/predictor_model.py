from __future__ import annotations

import torch

from abc import abstractmethod
from functools import cached_property
from typing import Any

from src.res.algo.nn.loss import MultiHeadLosses
from src.res.model.util.core import ModelDict , BatchInput , BatchOutput , BatchData , ModelFile
from src.res.model.util.config import ModelConfig
from .pipeline import TrainerPipeline
from .base_trainer import BaseTrainer

class PredictorModel(TrainerPipeline):
    """a group of ensemble models , of same net structure"""
    COMPULSARY_CALLBACKS = ['BasicTestResult' , 'DetailedAlphaAnalysis' , 'StatusDisplay' , 'SummaryWriter']
    
    def __init__(self, *args , vb_level : Any = 1 , **kwargs) -> None:
        self.reset()
        self.net : torch.nn.Module | Any = None
        self.model_dict = ModelDict()

    def __call__(self , input : BatchInput | torch.Tensor | Any | None , *args , **kwargs):
        if input is None or len(input) == 0:
            output = None
        else:
            output = self.forward(input , *args , **kwargs)
        return BatchOutput(output)

    def get_batch_data(self , input : BatchInput , *args , **kwargs):
        output = BatchOutput(self.forward(input , *args , **kwargs))
        return BatchData(input , output)

    def __repr__(self): 
        if not self.is_bounded:
            return f'{self.__class__.__name__}(not bounded to trainer or config)'
        return f'{self.__class__.__name__}(binder={self.binder})'

    @classmethod
    def initialize(cls , trainer_or_config : BaseTrainer | ModelConfig , **kwargs):
        from src.res.model.model_module.module import get_predictor_module
        binder = trainer_or_config
        if isinstance(trainer_or_config , BaseTrainer):
            config = trainer_or_config.config
            kwargs = trainer_or_config.input_model_kwargs | kwargs
        else:
            config = trainer_or_config
            kwargs = kwargs
        model = get_predictor_module(config , **kwargs).bound_with(binder)
        return model

    def print_out(self , vb_level : Any = 2 , min_key_len = 30):
        infos = {'Module Type' : self.__class__.__name__}
        self.logger.stdout_pairs(infos , title = f'Predictor Model Initiated:' , vb_level = vb_level , min_key_len = min_key_len)
    
    def multiloss_params(self): 
        return MultiHeadLosses.get_params(getattr(self , 'net' , None))

    @property
    def model_full_name(self):
        return f'{self.config.model_name}@{self.model_num}@{self.model_date}@{self.model_submodel}'
    @property
    def model_num(self):
        return self.trainer.model_num if self.bounded_with_trainer else self._model_num
    @property
    def model_date(self):
        return self.trainer.model_date if self.bounded_with_trainer else self._model_date
    @property
    def model_submodel(self):
        return self.trainer.model_submodel if self.bounded_with_trainer else self._model_submodel
    @property
    def model_param(self): return self.config.model_param[self.model_num]
    @property
    def complete_model_param(self) -> dict[str,Any]:
        param = self.cached_properties.query('complete_model_param' , lambda: None)
        return self.model_param if param is None else param
    @complete_model_param.setter
    def complete_model_param(self , value : dict[str,Any] | None):
        self.cached_properties.set('complete_model_param' , value)    
    @cached_property
    def current_model_file(self):
        return ModelFile(None)
    @property
    def param_has_nan(self) -> bool:
        if self.net is None:
            return False
        return any(param.isnan().any() for param in self.net.parameters())
    @property
    def grad_has_nan(self) -> bool:
        if self.net is None:
            return False
        return any(param.grad is not None and param.grad.isnan().any() for param in self.net.parameters())

    def load_model_file(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        """call when fitting/testing new model"""
        if model_num is not None: 
            self._model_num  = model_num
        else: 
            model_num = self.model_num
        if model_date is not None: 
            self._model_date = model_date
        else: 
            model_date = self.model_date
        if submodel is not None: 
            self._model_submodel = submodel
        else: 
            submodel = self.model_submodel
        assert self.deposition.exists(model_num , model_date , submodel) , (model_num , model_date , submodel)
        return self.deposition.load_model(model_num , model_date , submodel)
    
    @abstractmethod
    def reload_model(self , *args , **kwargs):
        """call when fitting new model or having new attempt, reload model parameters and initialize weights and optimizer"""
        from src.res.model.util.trainer.optimizer import Optimizer
        self.optimizer : Optimizer
        return self
    
    @abstractmethod
    def load_model(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        """call when testing new model"""
        return self
    @abstractmethod
    def ckpt_state_dict(self) -> dict[str , Any]:
        """state dict of model at epoch to be saved in checkpoint"""
    @abstractmethod
    def load_state_dict(self , state_dict : dict):
        """load state dict of model from checkpoint"""
        return self
    @abstractmethod
    def forward(self , batch_input : BatchInput | torch.Tensor , *args , **kwargs) -> Any: 
        """model object that can be called to forward"""
    @abstractmethod
    def fit(self) -> None:
        """fit the model inside"""
    @abstractmethod
    def collect(self , submodel = 'best' , *args) -> ModelDict: 
        """collect model params, called before stacking model"""

    def new_model(self , *args , **kwargs):
        """call when fitting new model"""
        self.reload_model(*args , **kwargs)
        return self
    
    def stack_model(self):
        """temporaly save self to somewhere"""
        self.metrics.collect_attempt()
        if self.metrics.epoch_train_metrics.nanloss:
            # skip saving model when nan loss is encountered
            return
        self.trainer.on_before_stack_model()
        for _ in self.trainer.iter_model_submodels_on_save():
            model_dict = self.collect(self.model_submodel)
            self.deposition.stack_model(model_dict , self.status.attempt_key , self.model_num , self.model_date , self.model_submodel) 

    def dump_model(self):
        """dump model to somewhere"""
        self.metrics.collect_model()
        self.trainer.on_before_dump_model()
        best_attempt = self.metrics.model_metrics.best_attempt()
        for _ in self.trainer.iter_model_submodels_on_save():
            self.deposition.dump_stacked_model(best_attempt , self.model_num , self.model_date , self.model_submodel) 

    def test(self):
        """test the model inside"""
        if self.data.no_date_to_test:
            return
        for _ in self.trainer.iter_model_submodels():
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()

    def batch_forward(self) -> None: 
        if self.batch_idx >= self.trainer.batch_resumed: 
            self.batch_output = self(self.batch_input)
            if self.batch_idx < self.trainer.batch_warm_up or self.batch_idx >= self.trainer.batch_aftermath:
                self.batch_output = BatchOutput()

    def batch_metrics(self) -> None:
        if self.batch_output.empty: 
            return
        self.trainer.on_batch_metrics_before()
        self.metrics.calculate(self.status.dataset , self.batch_key , self.batch_data)
        self.trainer.on_batch_metrics_after()

    def batch_backward(self) -> None:
        if self.batch_output.empty: 
            return
        assert self.status.dataset == 'train' , self.status.dataset
        self.trainer.on_batch_backward_before()
        self.optimizer.backward(self.metrics.batch_metrics)
        self.trainer.on_batch_backward_after()
