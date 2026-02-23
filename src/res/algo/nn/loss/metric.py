import torch
from typing import Callable

from src.math import mse , pearson , ccc , spearman

def align_shape(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None):
    if label.shape[-1] != pred.shape[-1]:
        last_dim = min(label.shape[-1] , pred.shape[-1])
        label = label[...,:last_dim]
        pred = pred[...,:last_dim]
        if w is not None:
            w = w[...,:last_dim]
    return label , pred , w

class CommonLoss:
    '''Common loss functions'''
    options = ['mse' , 'pearson' , 'ccc']
    @classmethod
    def get(cls , name : str , **kwargs) -> Callable[...,torch.Tensor]:
        if name in cls.options:
            return getattr(cls , name)
        else:
            raise ValueError(f'Invalid loss name: {name}, options: {cls.options}')

    @staticmethod
    def mse(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = mse(*align_shape(label , pred , w) , dim)
        return v
    
    @staticmethod
    def pearson(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = pearson(*align_shape(label , pred , w) , dim)
        return torch.exp(-v)
    
    @staticmethod
    def ccc(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = ccc(*align_shape(label , pred , w) , dim)
        return torch.exp(-v)
    
class CommonScore:
    options = ['mse' , 'pearson' , 'ccc' , 'spearman']
    @classmethod
    def get(cls , name : str , **kwargs) -> Callable[...,torch.Tensor]:
        if name in cls.options:
            return getattr(cls , name)
        else:
            raise ValueError(f'Invalid loss name: {name}, options: {cls.options}')

    @staticmethod
    def mse(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        v = mse(*align_shape(label , pred , w) , dim)
        return -v

    @staticmethod
    def pearson(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        return pearson(*align_shape(label , pred , w) , dim)
    
    @staticmethod
    def ccc(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        return ccc(*align_shape(label , pred , w) , dim)
    
    @staticmethod
    def spearman(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs):
        return spearman(*align_shape(label , pred , w) , dim)
    
class SpecialLoss:
    '''Special loss functions'''
    options = ['hidden_corr' , 'hidden_corr_deprecated' , 'quantile']
    @classmethod
    def get(cls , name : str , **kwargs) -> Callable[...,torch.Tensor | dict[str,torch.Tensor]]:
        if name in cls.options:
            return getattr(cls , name)
        else:
            raise ValueError(f'Invalid loss name: {name}, options: {cls.options}')

    @staticmethod
    def hidden_corr_deprecated(*args , hidden : torch.Tensor | list | tuple , **kwargs) -> torch.Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)): 
            hidden = torch.cat(hidden,dim=-1)
        h = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        # pen = h.T.cov().norm().square() / (h.shape[-1] ** 2)
        pen = h.T.cov().square().mean()
        return pen

    @staticmethod
    def hidden_corr(*args , hidden : torch.Tensor | list | tuple , **kwargs) -> torch.Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)): 
            hidden = torch.cat(hidden,dim=-1)
        h = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        # pen = h.T.cov().norm().square() / (h.shape[-1] ** 2)
        pen = h.T.cov().norm()
        return pen

    @staticmethod
    def quantile(label : torch.Tensor , predictions : torch.Tensor | None = None , w : torch.Tensor | None = None , dim = None , 
                 quantiles : list[float] = [0.1,0.5,0.9] , **kwargs):
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