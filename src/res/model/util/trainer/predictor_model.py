from __future__ import annotations

import torch

from abc import abstractmethod
from typing import Any

from src.proj import Logger
from src.res.algo.nn.loss import MultiHeadLosses
from src.res.model.util.core import ModelDict , BatchInput , BatchOutput , ModelConfig
from .base_trainer import BaseTrainer , ModelStreamLineWithTrainer

class BasePredictorModel(ModelStreamLineWithTrainer):
    '''a group of ensemble models , of same net structure'''
    AVAILABLE_CALLBACKS = []
    COMPULSARY_CALLBACKS = ['BasicTestResult' , 'DetailedAlphaAnalysis' , 'StatusDisplay' , 'SummaryWriter']
    
    def __init__(self, *args , **kwargs) -> None:
        self.reset()
        self.net : torch.nn.Module | Any = None
        self.model_dict = ModelDict()

    def __call__(self , input : BatchInput | torch.Tensor | Any | int | None , *args , **kwargs):
        if isinstance(input , int):
            from src.res.model.data_module import DataModule
            input = DataModule.get_date_batch_data(self.config , input)
            output = self.forward(input , *args , **kwargs)
        elif input is None or len(input) == 0:
            output = None
        else:
            output = self.forward(input , *args , **kwargs)
        return BatchOutput(output)
    
    def __repr__(self): 
        if self.trainer is None and self._config is None:
            return f'{self.__class__.__name__}(not bounded to trainer or config)'
        return f'{self.__class__.__name__}(config={self.config})'

    @classmethod
    def initialize(cls , config : ModelConfig , trainer : BaseTrainer | None = None , * , vb_level : Any = 2 , min_key_len = -1 , **kwargs):
        from src.res.model.model_module.module import get_predictor_module
        binder = config if trainer is None else trainer
        model = get_predictor_module(config , **kwargs).bound_with(binder)
        infos = {'Module Type' : model.__class__.__name__}
        Logger.stdout_pairs(infos , title = f'Predictor Model Initiated:' , vb_level = vb_level , min_key_len = min_key_len)
        return model
    
    def multiloss_params(self): 
        return MultiHeadLosses.get_params(getattr(self , 'net' , None))

    def reset(self):
        self.trainer : BaseTrainer | Any = None
        self._config : ModelConfig | Any = None
        return self

    def bound_with(self , binder : ModelConfig | BaseTrainer):
        if isinstance(binder , ModelConfig):
            return self.bound_with_config(binder)
        else:
            return self.bound_with_trainer(binder)

    def bound_with_config(self , config : ModelConfig):
        assert self.trainer is None , 'Cannot bound with config if bound with trainer first'
        self._config = config
        return self

    def bound_with_trainer(self , trainer : BaseTrainer):
        self.reset()
        self.trainer = trainer
        return self

    @classmethod
    def create_from_trainer(cls , trainer : BaseTrainer):
        return cls().bound_with_trainer(trainer)

    @property
    def config(self):
        return self.trainer.config if self.trainer else self._config
    @property
    def model_full_name(self):
        return f'{self.config.model_name}@{self.model_num}@{self.model_date}@{self.model_submodel}'
    @property
    def model_num(self):
        return self.trainer.model_num if self.trainer else self._model_num
    @property
    def model_date(self):
        return self.trainer.model_date if self.trainer else self._model_date
    @property
    def model_submodel(self):
        return self.trainer.model_submodel if self.trainer else self._model_submodel
    @property
    def model_param(self): return self.config.model_param[self.model_num]
    @property
    def complete_model_param(self) -> dict[str,Any]:
        if not hasattr(self , '_complete_model_param') or self._complete_model_param is None:
            return self.model_param
        else:
            return self._complete_model_param
    @complete_model_param.setter
    def complete_model_param(self , value : dict[str,Any] | None):
        self._complete_model_param : dict[str,Any] | None = value
    
    def load_model_file(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when fitting/testing new model'''
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
    def new_model(self , *args , **kwargs):
        '''call when fitting new model'''
        self.optimizer : Any
        return self
    @abstractmethod
    def load_model(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when testing new model'''
        return self
    @abstractmethod
    def forward(self , batch_input : BatchInput | torch.Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
    @abstractmethod
    def fit(self) -> None:
        '''fit the model inside'''
    @abstractmethod
    def collect(self , submodel = 'best' , *args) -> ModelDict: 
        '''collect model params, called before stacking model'''

    def stack_model(self):
        '''temporaly save self to somewhere'''
        self.trainer.on_before_save_model()
        for submodel in self.trainer.model_submodels:
            self.deposition.stack_model(self.collect(submodel) , self.model_num , self.model_date , submodel) 

    def dump_model(self):
        '''dump model to somewhere'''
        for submodel in self.trainer.model_submodels:
            self.deposition.dump_stacked_model(self.model_num , self.model_date , submodel) 

    def test(self):
        '''test the model inside'''
        Logger.note(f'model {self.model_str} test start' , vb_level = 'max')

        for _ in self.trainer.iter_model_submodels():
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

        Logger.note(f'model {self.model_str} test done' , vb_level = 'max')
    
    def batch_forward(self) -> None: 
        if self.batch_idx >= self.trainer.batch_resumed and self.batch_idx < self.trainer.batch_aftermath: 
            self.batch_output = self(self.batch_input)

    def batch_metrics(self) -> None:
        if self.batch_output.empty or self.batch_idx < self.trainer.batch_warm_up: 
            return
        batch_key = self.batch_idx if self.status.stage == 'fit' else self.trainer.batch_dates[self.batch_idx]
        self.metrics.calculate(self.status.dataset , batch_key , self.batch_data)

    def batch_backward(self) -> None:
        if self.batch_input.empty: 
            return
        assert self.status.dataset == 'train' , self.status.dataset
        self.trainer.on_before_backward()
        self.optimizer.backward(self.metrics.batch_metrics)
        self.trainer.on_after_backward()