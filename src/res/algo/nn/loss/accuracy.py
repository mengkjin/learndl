import torch
import torch.nn as nn

from src.func.metric import mse , pearson , ccc , spearman

from .abc import align_shape

__all__ = ['Accuracy']

class BaseAccuracy(nn.Module):
    key : str = ''
    def __init__(self , **kwargs):
        super().__init__()

    def __call__(self , *args , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        return self.forward(*args , **kwargs)

    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        raise NotImplementedError

class AccuracyMSE(BaseAccuracy):
    key = 'mse'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return -mse(*align_shape(label , pred , w) , dim)

class AccuracyPearson(BaseAccuracy):
    key = 'pearson'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return pearson(*align_shape(label , pred , w) , dim)

class AccuracyCCC(BaseAccuracy):
    key = 'ccc'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return ccc(*align_shape(label , pred , w) , dim)

class AccuracySpearman(BaseAccuracy):
    key = 'spearman'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return spearman(*align_shape(label , pred , w) , dim)

class Accuracy:
    options = {cls.key : cls for cls in BaseAccuracy.__subclasses__() if cls.key != ''}
    @classmethod
    def get(cls , name : str , **kwargs) -> BaseAccuracy:
        return cls.options[name](**kwargs)