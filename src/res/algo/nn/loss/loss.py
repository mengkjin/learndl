import torch
import torch.nn as nn
import torch.nn.functional as F

from src.func.metric import mse , pearson , ccc

from .basic import align_shape

__all__ = ['Loss']

class BaseLoss(nn.Module):
    """Base class for all loss functions"""
    key : str = ''
    multiheadlosses_capable : bool = False

    def __init__(self , **kwargs):
        super().__init__()

    def __call__(self , *args , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        output = self.forward(*args , **kwargs)
        assert isinstance(output,torch.Tensor) or isinstance(output,dict) , f'output of {self.key} should be a tensor or a dict , but got {output}'
        return output

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

class CCCHiddenCorrLoss(BaseLoss):
    key = 'ccc_hcorr'
    def __init__(self, lamb : float = 0.01):
        super().__init__()
        self.lamb = lamb
        self.ccc_loss = LossCCC()
        self.hidden_corr_loss = LossHiddenCorr()

    def forward(self , label : torch.Tensor , pred : torch.Tensor , hidden : torch.Tensor , **kwargs) -> dict[str,torch.Tensor]:
        return {
            'ccc' : self.ccc_loss.forward(label , pred) ,
            'hidden_corr' : self.hidden_corr_loss.forward(hidden = hidden) * self.lamb
        }

class ABCMLoss(BaseLoss):
    """ABCM:基于神经网络的alpha因子和beta因子协同挖掘模型"""
    key = 'abcm'
    def __init__(self, lamb : float = 0.1):
        super().__init__()
        self.lamb = lamb

    def forward(self, pred : torch.Tensor , label : torch.Tensor , alphas : torch.Tensor , betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        assert label.shape[-1] == 2 , label.shape
        mse = F.mse_loss(pred.squeeze() , label[...,0].squeeze())
        rsquare = self.rsquare_loss(alphas , label[...,1])
        corr = self.corr_loss(betas)
        turnover = self.turnover_loss(betas , betas_peer)
        
        return mse + rsquare + self.lamb * corr + turnover

    def rsquare_loss(self, hiddens : torch.Tensor , label : torch.Tensor , **kwargs):
        assert hiddens.ndim == 2 , hiddens.shape
        y_norm = label.norm()
        pred = hiddens @ (hiddens.T @ hiddens).inverse() @ hiddens.T @ label
        res_norm = (label - pred).norm()
        return 1 - res_norm / y_norm

    def corr_loss(self, hiddens : torch.Tensor , **kwargs):
        h = (hiddens - hiddens.mean(dim=0,keepdim=True)) / (hiddens.std(dim=0,keepdim=True) + 1e-6)
        pen = h.T.cov().norm()
        return pen

    def turnover_loss(self, betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        return (betas - betas_peer).norm()

class Loss:
    options = {cls.key : cls for cls in BaseLoss.__subclasses__() if cls.key != ''}
    @classmethod
    def get(cls , name : str , **kwargs) -> BaseLoss:
        return cls.options[name](**kwargs)