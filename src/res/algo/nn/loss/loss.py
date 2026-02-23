import torch
import torch.nn as nn

from src.math import mse , pearson , ccc

from .abc import align_shape

__all__ = ['Loss']

class BaseLoss(nn.Module):
    """Base class for all loss functions"""
    key : str = ''
    multiheadlosses_capable : bool = False

    def __init__(self , **kwargs):
        super().__init__()

    def __call__(self , *args , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        return self.forward(*args , **kwargs)

    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        raise NotImplementedError

class LossMSE(BaseLoss):
    key = 'mse'
    multiheadlosses_capable = True
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return mse(*align_shape(label , pred , w) , dim)

class LossPearson(BaseLoss):
    key = 'pearson'
    multiheadlosses_capable = True
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return torch.exp(-pearson(*align_shape(label , pred , w) , dim))

class LossCCC(BaseLoss):
    key = 'ccc'
    multiheadlosses_capable = True
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return torch.exp(-ccc(*align_shape(label , pred , w) , dim))

class LossHiddenCorrDeprecated(BaseLoss):
    key = 'hidden_corr_deprecated'
    def forward(self , *args , hidden : torch.Tensor | list | tuple , **kwargs) -> torch.Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)): 
            hidden = torch.cat(hidden,dim=-1)
        std_hidden = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        return std_hidden.T.cov().square().mean()

class LossHiddenCorr(BaseLoss):
    key = 'hidden_corr'
    def forward(self , *args , hidden : torch.Tensor | list | tuple , **kwargs) -> torch.Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)): 
            hidden = torch.cat(hidden,dim=-1)
        std_hidden = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        return std_hidden.T.cov().norm()

class LossQuantile(BaseLoss):
    key = 'quantile'
    def forward(self , label : torch.Tensor , predictions : torch.Tensor | None = None , w : torch.Tensor | None = None , dim = None , 
                quantiles : list[float] = [0.1,0.5,0.9] , **kwargs) -> torch.Tensor:
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

class Loss:
    options = {cls.key : cls for cls in BaseLoss.__subclasses__() if cls.key != ''}
    @classmethod
    def get(cls , name : str , **kwargs) -> BaseLoss:
        return cls.options[name](**kwargs)