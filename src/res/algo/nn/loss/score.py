import torch
import torch.nn as nn

from src.math import mse , pearson , ccc , spearman

from .abc import align_shape

__all__ = ['Score']

class BaseScore(nn.Module):
    key : str = ''
    def __init__(self , **kwargs):
        super().__init__()

    def __call__(self , *args , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        return self.forward(*args , **kwargs)

    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        raise NotImplementedError

class ScoreMSE(BaseScore):
    key = 'mse'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return -mse(*align_shape(label , pred , w) , dim)

class ScorePearson(BaseScore):
    key = 'pearson'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return pearson(*align_shape(label , pred , w) , dim)

class ScoreCCC(BaseScore):
    key = 'ccc'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return ccc(*align_shape(label , pred , w) , dim)

class ScoreSpearman(BaseScore):
    key = 'spearman'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return spearman(*align_shape(label , pred , w) , dim)

class Score:
    options = {cls.key : cls for cls in BaseScore.__subclasses__() if cls.key != ''}
    @classmethod
    def get(cls , name : str , **kwargs) -> BaseScore:
        return cls.options[name](**kwargs)