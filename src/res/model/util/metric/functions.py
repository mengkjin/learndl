from __future__ import annotations
import torch

from torch import nn , Tensor
from typing import Any , Literal

from src.proj import Logger
from src.proj.util.torch import RequireGrad
from src.res.algo.nn.loss import Accuracy , Loss , MultiHeadLosses
from src.res.model.util.core import BatchData
from .components import MetricComponent , LossComponent , AccuracyComponent

__all__ = ['MetricFunction' , 'LossFunction' , 'AccuracyFunction']

class MetricFunction:
    SearchList : list[str] = []
    ExcludeNan : bool = True
    def __init__(
        self , 
        criterions : dict[str,dict[str,Any]] , 
        net : nn.Module | None = None , 
        ignores : list[str] | None = None ,
    ) -> None:
        self.criterions = criterions
        self.net = net
        self.ignores = ignores or []
        self.components : dict[str,MetricComponent] = {}
 
    def __repr__(self):
        return f'{self.__class__.__name__}(components={{{', '.join([f"{key}: {value.lamb}" for key, value in self.components.items()])}}})'

    def __call__(
        self , data : BatchData , 
        which_output : int | list[int] | None = None , 
        which_label : int | list[int] | None = None , 
        require_grad : bool = True
    ) -> dict[str,Tensor]:
        with RequireGrad(require_grad):
            inputs = data.loss_inputs(exclude_nan = self.ExcludeNan)
            results : dict[str,Tensor] = {}
            for criterion , component in self.components.items():
                Logger.only_once(f'{criterion} calculated!' , object = self , mark = criterion , printer = 'success' , vb_level = 'max')
                value = component(which_output = which_output , which_label = which_label , **inputs)
                if isinstance(value , dict):
                    results.update({k:v.sum() for k,v in value.items()})
                else:
                    results[criterion] = value.sum()
            
        return results

class LossFunction(MetricFunction):
    SearchList : list[str] = ['loss' , 'Loss' , 'LossFunction' , 'loss_function' , 'losses' , 'Losses']
    ExcludeNan : bool = False
    def __init__(
        self , 
        criterions : dict[str,dict[str,Any]] , 
        net : nn.Module | None = None , 
        ignores : list[str] | None = None ,
        multilosses_kwargs : dict[str,Any] | None = None ,
    ) -> None:
        super().__init__(criterions , net , ignores)
        self.multilosses_kwargs = multilosses_kwargs or {}
        self.init_components()

    def init_components(self):
        self.components : dict[str,LossComponent] = {}
        if self.net is not None:
            for name in self.SearchList:
                if hasattr(self.net , name):
                    calculator = getattr(self.net , name)
                    self.components['net_specific'] = LossComponent(calculator)
                    return self

        for i , (criterion , kwargs) in enumerate(self.criterions.items()):
            if criterion in self.ignores:
                continue
            lamb = kwargs.pop('lamb' , 1.0)
            calculator = Loss.get(criterion , **kwargs)
            if i == 0 and calculator.multiheadlosses_capable:
                multilosses = MultiHeadLosses(**self.multilosses_kwargs , mt_param = MultiHeadLosses.get_params(self.net))
            else:
                multilosses = None
            self.components[criterion] = LossComponent(calculator , lamb = lamb , multilosses = multilosses , **kwargs)

    def losses(self , data : BatchData , which_output : int | list[int] | None = None , which_label : int | list[int] | None = None ,
               prefix : str | tuple[str,...] | None = ('penalty_' , 'loss_') , dataset : Literal['train','valid'] | Any = 'train') -> dict[str,Tensor]:
        if dataset not in ['train','valid']:
            return {}
        losses = self(data, which_output = which_output , which_label = which_label , require_grad = dataset == 'train')
        if prefix is not None:
            losses.update({key:value for key , value in data.output.other.items() if key.lower().startswith(prefix)})
        return losses

class AccuracyFunction(MetricFunction):
    SearchList : list[str] = ['accuracy' , 'Accuracy' , 'AccuracyFunction' , 'accuracy_function' , 'accuracies' , 'Accuracies']
    ExcludeNan : bool = True
    def __init__(
        self , 
        criterions : dict[str,dict[str,Any]] , 
        net : nn.Module | None = None , 
        ignores : list[str] | None = None ,
    ) -> None:
        super().__init__(criterions , net , ignores)
        self.init_components()

    def init_components(self):
        self.components : dict[str,AccuracyComponent] = {}
        if self.net is not None:
            for name in self.SearchList:
                if hasattr(self.net , name):
                    self.components['net_specific'] = AccuracyComponent(getattr(self.net , name))
                    return self

        for i , (criterion , kwargs) in enumerate(self.criterions.items()):
            if criterion in self.ignores:
                continue
            lamb = kwargs.pop('lamb' , 1.0)
            calculator = Accuracy.get(criterion , **kwargs)
            self.components[criterion] = AccuracyComponent(calculator , lamb = lamb , **kwargs)

    def accuracies(self , data : BatchData , which_output : int | list[int] | None = None , which_label : int | list[int] | None = None , 
                   dataset : Literal['train','valid'] | Any = 'train') -> dict[str,float]:
        if dataset not in ['train','valid']:
            return {}
        accuracies = self(data, which_output = which_output , which_label = which_label , require_grad = False)
        accuracies = {key:value.item() if isinstance(value , Tensor) else value for key,value in accuracies.items()}
        return accuracies

class RankICFunction:
    ExcludeNan : bool = True
    def __init__(
        self
    ) -> None:
        self.rankic_calculator = AccuracyComponent(Accuracy.get('spearman') , lamb = 1.0 , which_label=0)
 
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __call__(
        self , data : BatchData , 
        which_output : int | list[int] | None = None , 
        which_label : int | list[int] | None = None , 
        **kwargs
    ) -> Tensor:
        with RequireGrad(False):
            inputs = data.loss_inputs(exclude_nan = self.ExcludeNan)
            value = self.rankic_calculator(which_output = which_output , which_label = which_label , **inputs)
            assert isinstance(value , Tensor) , f'rankic value should be a Tensor, but got {value}'
        return value

    def rankic(self , data : BatchData) -> Tensor:
        value = self(data, which_output = None , which_label = 0).mean()
        return value
        
    def hidden_rankic(self , data : BatchData , which_label : int = 0 , require_grad : bool = False) -> torch.Tensor:
        assert not require_grad , 'hidden accuracies do not support require_grad'
        hidden = data.output.hidden
        label = data.input.y[...,which_label:which_label+1]
        with RequireGrad(require_grad):
            accu = self.rankic_calculator(pred = hidden , label = label , dim = 0)
            assert isinstance(accu , Tensor) , f'hidden rankic value should be a Tensor, but got {accu}'
            return accu