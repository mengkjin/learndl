import torch
import torch.nn as nn
from typing import Callable , Any , Literal

from src.proj import Logger
from src.func import mse , pearson , ccc , spearman
from .multiloss import MultiHeadLosses

class MetricsCalculator:
    def __init__(self , 
                 loss_type : Literal['mse', 'pearson', 'ccc'] = 'ccc' , 
                 score_type : Literal['mse', 'pearson', 'ccc', 'spearman'] = 'spearman',
                 penalty_kwargs : dict = {} , 
                 multi_type : Literal['ewa','hybrid','dwa','ruw','gls','rws'] | None = None ,
                 multi_param : dict[str,Any] = {} ,
                 **kwargs):
        self.loss_type = loss_type
        self.score_type = score_type
        self.penalty_kwargs = penalty_kwargs
        self.multi_type = multi_type
        self.multi_param = multi_param
    
    def __call__(self , *args , **kwargs) -> torch.Tensor:
        ...
    
    def new_model(self , net : nn.Module | None = None , penalty_conds : dict[str,bool] | None = None , 
                  num_head : int = 1 , **kwargs):
        self.calculators : dict[str,MetricCalculator] = {}
        self.calculators['loss']    = MetricCalculator('loss' , self.loss_type , net = net)
        self.calculators['score']   = MetricCalculator('score' , self.score_type , net = net)
        if net is not None and hasattr(net , 'penalty'):
            self.calculators['penalty.net_specific'] = MetricCalculator('penalty' , 'net_specific' , net = net)
        else:
            for k , v in self.penalty_kwargs.items():
                self.calculators[f'penalty.{k}'] = MetricCalculator('penalty' , k , **v)

        penalty_conds = penalty_conds or {}
        for k,v in penalty_conds.items():
            if f'penalty.{k}' in self.calculators:
                self.calculators[f'penalty.{k}'].set_cond(v)

        self.multilosses = MultiHeadLosses(num_head , self.multi_type, **self.multi_param)    

        return self
    
    def _calc(self , type : str , *args , **kwargs) -> torch.Tensor:
        if type == 'loss':
            return self.calculators['loss'](*args , **kwargs)
        elif type == 'score':
            return self.calculators['score'](*args , **kwargs)
        else:
            return self.calculators[f'penalty.{type}'](*args , **kwargs)
    def losses(self , label : torch.Tensor , pred : torch.Tensor , weight : torch.Tensor | None = None , 
               dim : int = 0 , which_head : int | None = None, **kwargs) -> torch.Tensor:
        return self._calc('loss' , label , pred , weight , dim , which_head , **kwargs)
    def score(self , label : torch.Tensor , pred : torch.Tensor , weight : torch.Tensor | None = None , 
              dim : int = 0 , which_head : int | None = None, **kwargs) -> torch.Tensor:
        return self._calc('score' , label , pred , weight , dim , which_head , **kwargs)
    def penalties(self , label : torch.Tensor , pred : torch.Tensor , weight : torch.Tensor | None = None , 
                  dim : int = 0 , which_head : int | None = None, **kwargs) -> dict[str,torch.Tensor]:
        penalties : dict[str,torch.Tensor] = {}
        for k,v in self.calculators.items():
            if k.startswith('penalty.'):
                penalties[k] = v(label , pred , weight , dim , which_head , **kwargs)
        return penalties
    def loss_penalties(self , label : torch.Tensor , pred : torch.Tensor , weight : torch.Tensor | None = None , 
                      dim : int = 0 , which_head : int | None = None, **kwargs) -> dict[str,torch.Tensor]:
        lps : dict[str,torch.Tensor] = {}
        losses = self.losses(label , pred , weight , dim , which_head , **kwargs)
        lps[self.loss_criterion] = self.multilosses(losses , mt_param = kwargs)
        for k,v in self.calculators.items():
            if k.startswith('penalty.'):
                lps[k] = v(label , pred , weight , dim , which_head , **kwargs)
        return lps
    
    @property
    def loss_criterion(self): 
        if hasattr(self , 'multilosses') and self.multilosses:
            return '.'.join([self.calculators['loss'].criterion , str(self.multilosses.multi_type)])
        else:
            return self.calculators['loss'].criterion
    @property
    def score_criterion(self): 
        return self.calculators['score'].criterion
        

class MetricCalculator:
    DISPLAY_LOG : dict[str,bool] = {}

    def __init__(self , metric_type : Literal['loss' , 'score' , 'penalty'] , 
                 criterion : str , 
                 net : nn.Module | None = None , 
                 lamb : float = 1. , **kwargs):
        '''
        metric_type : 'loss' , 'score' , 'penalty'
        criterion : metric function
        collapse  : aggregate last dimension if which_col < -1
        '''
        self.metric_type = metric_type
        self.init_calc(criterion , net)
        self.lamb = lamb
        self.kwargs = kwargs
        self.cond = True

    def set_cond(self , cond : bool):
        self.cond = cond
        return self

    def __call__(self, label : torch.Tensor , pred : torch.Tensor , weight : torch.Tensor | None = None , 
                 dim : int = 0 , which_head : int | None = None, **kwargs) -> torch.Tensor:
        '''calculate the resulting metric'''
        if not self.cond: 
            return torch.Tensor([0.])
        args , kwargs = self.slice_inputs(label , pred , weight , dim , which_head , **kwargs)
        v = self.lamb * self.calc(*args , **kwargs)
        self.display()
        return v
    
    def slice_inputs(self, label : torch.Tensor , pred : torch.Tensor , weight : torch.Tensor | None = None , 
                     dim : int = 0 , which_head : int | None = None, **kwargs):
        if self.metric_type == 'penalty':
            return (label , pred , weight , dim) , kwargs
        
        if self.metric_type == 'loss':
            label , pred , weight = self._slice_lpw(label , pred , weight , which_head = which_head , 
                                                    nan_check = False, training = True)
        elif self.metric_type == 'score':
            label , pred , weight = self._slice_lpw(label , pred , weight , which_head = which_head ,
                                                    nan_check = True, training = False)

        return (label , pred , weight , dim) , kwargs
    
    def init_calc(self , criterion : str , net : nn.Module | None = None):
        
        if net is not None and hasattr(net , self.metric_type):
            self.calc = getattr(net , self.metric_type)
            self.criterion = 'net_specific'
        elif criterion != getattr(self , 'criterion' , None):
            if self.metric_type == 'loss':
                func = LossMetrics.get_func(criterion)
            elif self.metric_type == 'score':
                func = torch.no_grad()(ScoreMetrics.get_func(criterion))
            elif self.metric_type == 'penalty':
                func = PenaltyMetrics.get_func(criterion)
            self.calc = func
            self.criterion = criterion

        assert hasattr(self , 'calc') and hasattr(self , 'criterion') , \
            f'{self.metric_type} function of [{self.criterion}] not found'
        
    def display(self):
        if self.DISPLAY_LOG.get(f'{self.metric_type}.{self.criterion}' , False): 
            return
        Logger.success(f'{self.__class__.__name__}: {self.metric_type} {self.criterion} calculated and success!')
        self.DISPLAY_LOG[f'{self.metric_type}.{self.criterion}'] = True

    @classmethod
    def _slice_lpw(cls , label : torch.Tensor , pred : torch.Tensor , weight : torch.Tensor | None = None , 
                   which_head : int | None = None , nan_check : bool = False , training = False) -> tuple[torch.Tensor , torch.Tensor , torch.Tensor | None]:
        '''each element return ith column, if negative then return raw element'''
        if nan_check: 
            label , pred , weight = cls._slice_nan(label , pred , weight)
        label  = cls._slice_col(label , None if training else 0)
        pred   = cls._slice_col(pred  , None if training else which_head)
        weight = cls._slice_col(weight, None if training else which_head)

        if pred.ndim > label.ndim:
            pred = pred.nanmean(dim = -1)
            if weight is not None: 
                weight = weight.nanmean(dim = -1)
        assert label.shape == pred.shape , (label.shape , pred.shape)
        return label , pred , weight

    @staticmethod
    def _slice_col(data : torch.Tensor | None , col : int | None = None) -> Any:
        return data if data is None or col is None else data[...,col]
        
    @staticmethod
    def _slice_nan(*args , print_all_nan = False) -> tuple[torch.Tensor , torch.Tensor , torch.Tensor | None] | Any:
        nanpos = False
        for arg in args:
            if arg is not None: 
                nanpos += arg.isnan()
        if isinstance(nanpos , torch.Tensor) and nanpos.any():
            if nanpos.ndim > 1:
                nanpos = nanpos.sum(tuple(range(1 , nanpos.ndim))) > 0
            if print_all_nan and nanpos.all(): 
                Logger.error('Encountered all nan inputs in metric calculation!')
                [Logger.stdout(arg) for arg in args]
            args = [None if arg is None else arg[~nanpos] for arg in args]
        return args

class LossMetrics:
    @classmethod
    def get_func(cls , name : str , **kwargs) -> Callable[...,torch.Tensor]:
        if name == 'mse':
            return cls.mse
        elif name == 'pearson':
            return cls.pearson
        elif name == 'ccc':
            return cls.ccc
        elif name == 'quantile':
            return cls.quantile
        else:
            raise ValueError(f'Invalid loss name: {name}')

    @staticmethod
    def mse(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = mse(label , pred , w , dim)
        return v
    
    @staticmethod
    def pearson(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = pearson(label , pred , w , dim)
        return torch.exp(-v)
    
    @staticmethod
    def ccc(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = ccc(label , pred , w , dim)
        return torch.exp(-v)
    
    @staticmethod
    def quantile(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , 
                 quantiles : list[float] = [0.1,0.5,0.9] , predictions : torch.Tensor | None = None , **kwargs):
        assert predictions is not None , f'predictions should be provided'
        assert predictions.shape[-1] == len(quantiles) , f'shape of predictions {predictions.shape} should be (...,{len(quantiles)})'
        if predictions.ndim == label.ndim + 1: 
            predictions = predictions.squeeze(-2)
        assert predictions.ndim == label.ndim == 2 , f'shape of predictions {predictions.shape} and label {label.shape} should be (...,1)'
        if w is None:
            w1 = 1.
        else:
            w1 = w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
        
        losses = []
        label = label.expand_as(predictions)
        
        for i, q in enumerate(quantiles):
            pred_q = predictions[..., i:i+1]
            error = label - pred_q
            valid = ~error.isnan()
            loss = torch.max(q * error[valid], (q - 1) * error[valid])
            losses.append((w1 * loss).mean(dim=dim,keepdim=True))
        
        v = torch.stack(losses,dim=-1).mean(dim=-1)
        return v

class ScoreMetrics:
    @classmethod
    def get_func(cls , name : str , **kwargs) -> Callable[...,torch.Tensor]:
        if name == 'mse':
            return cls.mse
        elif name == 'pearson':
            return cls.pearson
        elif name == 'ccc':
            return cls.ccc
        elif name == 'spearman':
            return cls.spearman
        else:
            raise ValueError(f'Invalid score name: {name}')

    @staticmethod
    def mse(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = mse(label , pred , w , dim)
        return -v

    @staticmethod
    def pearson(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        return pearson(label , pred , w , dim)
    
    @staticmethod
    def ccc(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        return ccc(label , pred , w , dim)
    
    @staticmethod
    def spearman(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        return spearman(label , pred , w , dim)
    
class PenaltyMetrics:
    @classmethod
    def get_func(cls , name : str , **kwargs) -> Callable[...,torch.Tensor]:
        if name == 'hidden_corr':
            return cls.hidden_corr
        else:
            raise ValueError(f'Invalid penalty name: {name}')

    @staticmethod
    def hidden_corr(*args , hidden : torch.Tensor | list | tuple , **kwargs) -> torch.Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)): 
            hidden = torch.cat(hidden,dim=-1)
        h = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        # pen = h.T.cov().norm().square() / (h.shape[-1] ** 2)
        pen = h.T.cov().square().mean()
        return pen