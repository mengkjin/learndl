from typing import Literal, Any
from src.res.algo.nn.loss.loss import ProgressiveGlobal2Top as g2t_loss
from src.res.algo.nn.loss.accuracy import ProgressiveGlobal2Top as g2t_accuracy
from .RNN import gru

class gru_global2top(gru):
    """GRU-based univariate model.  Registry key: ``'gru'``."""
    def __init__(
        self , input_dim , hidden_dim , 
        global_loss : Literal['pearson', 'ccc' , 'mse'] = 'ccc', 
        top_loss : Literal['soft_topk'] = 'soft_topk', 
        global_loss_kwargs : dict[str,Any] | None = None , 
        top_loss_kwargs : dict[str,Any] | None = None , 
        top_loss_lambda : float = 1. ,
        global_accuracy : Literal['mse' , 'pearson', 'ccc' , 'spearman'] = 'spearman', 
        top_accuracy : Literal['long_avg' , 'long_short'] = 'long_avg', 
        global_accuracy_kwargs : dict[str,Any] | None = None , 
        top_accuracy_kwargs : dict[str,Any] | None = None , 
        **kwargs
    ):
        super().__init__(input_dim , hidden_dim , **kwargs)
        self.global2top_loss = g2t_loss(global_loss , top_loss , global_loss_kwargs , top_loss_kwargs , top_loss_lambda)
        self.global2top_accuracy = g2t_accuracy(global_accuracy , top_accuracy , global_accuracy_kwargs , top_accuracy_kwargs)

    def loss(self , **kwargs):
        losses = self.global2top_loss(**kwargs)
        return losses

    def accuracy(self , **kwargs):
        accuracies = self.global2top_accuracy(**kwargs)
        return accuracies