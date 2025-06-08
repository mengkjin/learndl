import torch
import numpy as np
import pandas as pd

from torch import Tensor
from typing import Any , Literal , Optional

from src.func import mse , pearson , ccc , spearman

class LossMetrics:
    @staticmethod
    def mse(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , **kwargs):
        v = mse(label , pred , w , dim)
        return v
    
    @staticmethod
    def pearson(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , **kwargs):
        v = pearson(label , pred , w , dim)
        return torch.exp(-v)
    
    @staticmethod
    def ccc(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , **kwargs):
        v = ccc(label , pred , w , dim)
        return torch.exp(-v)
    
    @staticmethod
    def quantile(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , 
                 quantiles : list[float] = [0.1,0.5,0.9] , predications : Tensor | None = None , **kwargs):
        assert predications is not None , f'predications should be provided'
        assert predications.shape[-1] == len(quantiles) , f'shape of predications {predications.shape} should be (...,{len(quantiles)})'
        if w is None:
            w1 = 1.
        else:
            w1 = w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
        
        losses = []
        label = label.expand_as(predications)
        
        for i, q in enumerate(quantiles):
            pred_q = predications[:, :, i:i+1]
            error = label - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            losses.append((w1 * loss).mean(dim=dim,keepdim=True))
        
        v = torch.stack(losses,dim=-1).mean(dim=-1)
        return v

class ScoreMetrics:
    @staticmethod
    def mse(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , **kwargs):
        v = mse(label , pred , w , dim)
        return -v

    @staticmethod
    def pearson(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , **kwargs):
        return pearson(label , pred , w , dim)
    
    @staticmethod
    def ccc(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , **kwargs):
        return ccc(label , pred , w , dim)
    
    @staticmethod
    def spearman(label : Tensor , pred : Tensor , w : Tensor | None = None , dim = None , **kwargs):
        return spearman(label , pred , w , dim)
    
class PenaltyMetrics:
    @staticmethod
    def hidden_corr(*args , hidden : Tensor | list | tuple , **kwargs) -> Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)): hidden = torch.cat(hidden,dim=-1)
        h = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        # pen = h.T.cov().norm().square() / (h.shape[-1] ** 2)
        pen = h.T.cov().square().mean()
        return pen