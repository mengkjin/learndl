"""ABCM (Astgnn): Alpha-Beta Co-Mining neural network model.

Filename: ABCM.py  |  Main class: Astgnn
Note: The file is named ABCM but contains the ``Astgnn`` class.  This naming
inconsistency is documented in ``TODO_res_algo.md``.

The loss logic here largely duplicates ``ABCMLoss`` in ``loss/loss.py``.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.proj import Logger
from src.res.algo.nn import layer as Layer

from .RNN import mod_gru

__all__ = ['Astgnn' , 'AstgnnIdiosyncratic']

class _AstgnnLoss(nn.Module):
    """Abstract base class for Astgnn loss functions."""
    def __init__(self, loss_corr_lamb = 0.1, **kwargs):
        super().__init__()
        self.loss_corr_lamb = loss_corr_lamb

    def loss(self, pred : torch.Tensor , label : torch.Tensor , alphas : torch.Tensor , betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        """Composite ABCM loss: MSE + R² + corr penalty + turnover penalty.

        Args:
            pred:        Scalar predictions ``[bs, 1]``.
            label:       Two-column label ``[bs, 2]`` where ``[...,0]`` is the
                         return target and ``[...,1]`` is the R² target. (std and rtn)
            alphas:      Alpha factors ``[bs, alpha_num]``.
            betas:       Current-step beta factors ``[bs, beta_num]``.
            betas_peer:  Previous-step beta factors ``[bs, beta_num]`` for
                         turnover penalty.
        """
        assert label.shape[-1] == 2 , label.shape
        mse = F.mse_loss(pred.squeeze() , label[...,:1].squeeze())
        rsquare = self.rsquare_loss(alphas , label[...,1])
        corr = self.corr_loss(betas)
        turnover = self.turnover_loss(betas , betas_peer)
        all_losses = {
            'mse': mse,
            'rsquare': rsquare,
            'corr': self.loss_corr_lamb * corr,
            'turnover': self.loss_corr_lamb * turnover,
        }
        return all_losses

    def rsquare_loss(self, hiddens : torch.Tensor , label : torch.Tensor , **kwargs):
        """Compute ``1 - R²`` (projection residual fraction)."""
        assert hiddens.ndim == 2 , hiddens.shape
        y_norm = label.norm()
        pred = hiddens @ (hiddens.T @ hiddens).inverse() @ hiddens.T @ label
        res_norm = (label - pred).norm()
        return res_norm / y_norm

    def corr_loss(self, hiddens : torch.Tensor , **kwargs):
        """Frobenius norm of the standardized beta covariance matrix."""
        h = (hiddens - hiddens.mean(dim=0,keepdim=True)) / (hiddens.std(dim=0,keepdim=True) + 1e-6)
        pen = h.T.cov().norm()
        return pen

    def turnover_loss(self, betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        """L2 distance between window-end and window-start betas."""
        return (betas - betas_peer).norm()


class Astgnn(_AstgnnLoss):
    """Alpha-Beta Co-Mining GRU model (ABCM).  Registry key: ``'abcm'``

    Two-branch GRU architecture:
    * **alpha_net** — produces ``alpha_num`` alpha factors; the mean across
      factors is the final scalar prediction
    * **beta_net** — produces ``beta_num`` beta/risk factors; the loss
      penalizes factor collinearity (``corr_loss``) and temporal turnover
      (``turnover_loss``)

    The combined loss is::

        MSE(pred, label[...,:1])
        + R²_loss(alphas, label[...,1])
        + loss_corr_lamb * corr_loss(betas)
        + turnover_loss(betas, betas_peer)

    Args:
        input_dim:       Input feature dimension.
        hidden_dim:      GRU hidden dimension (default ``128``).
        dropout:         Dropout rate (default ``0.1``).
        rnn_layers:      Number of GRU layers (default ``2``).
        enc_in_dim:      Input projection dimension (default ``64``).
        alpha_num:       Number of alpha factors (default ``60``).
        beta_num:        Number of beta factors (default ``10``).
        loss_corr_lamb:  Coefficient for the beta correlation penalty
                         (default ``0.1``).

    Forward:
        Input: ``[bs, seq_len, input_dim]`` or tuple of tensors
        Output: ``([bs, 1], {'alphas': [bs, alpha_num], 'betas': [bs, beta_num],
                              'betas_peer': [bs, beta_num]})``
    """
    def __init__(self,input_dim,hidden_dim = 64,dropout = 0.1,rnn_layers = 2,enc_in=None,enc_in_dim=64,
                 act_type='leaky',dec_mlp_layers=2,dec_mlp_dim=64,
                 alpha_num = 60 , beta_num = 10 , loss_corr_lamb = 0.1,
                 beta_into_pred = False,
                 **kwargs):
        super().__init__(loss_corr_lamb = loss_corr_lamb)
        if not isinstance(input_dim, int):
            input_dim = sum(input_dim)
        self.fc_enc_in = nn.Sequential(nn.Linear(input_dim, enc_in_dim),nn.Tanh())

        rnn_kwargs = {'input_dim':enc_in_dim,'output_dim':hidden_dim,'num_layers':rnn_layers, 'dropout':dropout}
        self.fc_rnn = mod_gru(**rnn_kwargs)

        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim , alpha_num), 
            Layer.Act.get_activation_fn(act_type), 
        )
        self.beta_net = nn.Sequential(
            nn.Linear(hidden_dim , beta_num), 
            Layer.Act.get_activation_fn(act_type), 
        )
        self.alpha_map_out = Layer.MeanPool()
        self.beta_into_pred = beta_into_pred
        if beta_into_pred:
            self.beta_map_out = nn.Linear(beta_num , 1)
        else:
            self.beta_map_out = None

    def forward(self, input : Tensor | tuple[Tensor, ...] | list[Tensor]):
        """
        in: [bs x seq_len x input_dim]
        out:[bs x hidden_dim]
        """
        x = input if isinstance(input , Tensor) else torch.concat(input , dim = -1) 
        x = self.fc_enc_in(x)
        x = self.fc_rnn(x)
        x , x_0 = x[:,-1] , x[: , 0]
        alphas = self.alpha_net(x)
        betas = self.beta_net(x)
        betas_peer = self.beta_net(x_0)

        pred_alpha = self.alpha_map_out(alphas) 
        if self.beta_into_pred and self.beta_map_out is not None:
            pred_beta = self.beta_map_out(betas)
        else:
            pred_beta = 0
        pred = pred_alpha + pred_beta
        if False:
            Logger.stdout(f'input shape: {x.shape}')
            Logger.stdout(f'enc_in shape: {x.shape}')
            Logger.stdout(f'rnn outpur shape: {x.shape}')
            Logger.stdout(f'last rnn output shape: {x.shape}')
            Logger.stdout(f'initial rnn output shape: {x_0.shape}')
            Logger.stdout(f'alphas shape: {alphas.shape}')
            Logger.stdout(f'betas shape: {betas.shape}')
            Logger.stdout(f'pred shape: {pred.shape}')
        return pred , {'alphas':alphas , 'betas':betas , 'betas_peer':betas_peer , 'pred_alpha':pred_alpha , 'pred_beta':pred_beta}

class AstgnnIdiosyncratic(_AstgnnLoss):
    """
    Astgnn model with idiosyncratic input. For example, 15m+day+style or mincr+day+style.
    If the first input is of higher dim (15m etc.), use resnet_block to project it to the same dim as the rest.
    Alpha-Beta Co-Mining GRU model (ABCM).  Registry key: ``'abcm'``

    Two-branch GRU architecture:
    * **alpha_net** — produces ``alpha_num`` alpha factors; the mean across
      factors is the final scalar prediction
    * **beta_net** — produces ``beta_num`` beta/risk factors; the loss
      penalizes factor collinearity (``corr_loss``) and temporal turnover
      (``turnover_loss``)

    The combined loss is::

        MSE(pred, label[...,:1])
        + R²_loss(alphas, label[...,1])
        + loss_corr_lamb * corr_loss(betas)
        + turnover_loss(betas, betas_peer)

    Args:
        input_dim:       Input feature dimension , list / tuple of ints.
        hidden_dim:      GRU hidden dimension (default ``128``).
        dropout:         Dropout rate (default ``0.1``).
        rnn_layers:      Number of GRU layers (default ``2``).
        enc_in_dim:      Input projection dimension (default ``64``).
        alpha_num:       Number of alpha factors (default ``60``).
        beta_num:        Number of beta factors (default ``10``).
        loss_corr_lamb:  Coefficient for the beta correlation penalty
                         (default ``0.1``).

    Forward:
        Input: tuple of tensors, where the first tensor is the alpha input, and the rest are beta inputs.
        Output: ``([bs, 1], {'alphas': [bs, alpha_num], 'betas': [bs, beta_num],
                              'betas_peer': [bs, beta_num]})``
    """
    def __init__(
        self,input_dim,hidden_dim = 64,dropout = 0.1,rnn_layers = 2,enc_in=None,enc_in_dim=64,
        act_type='leaky',dec_mlp_layers=2,dec_mlp_dim=64,
        alpha_num = 60 , beta_num = 10 , loss_corr_lamb = 0.1,
        beta_into_pred = False, resnet_projector = False,
        **kwargs
    ):
        super().__init__(loss_corr_lamb = loss_corr_lamb)
        assert isinstance(input_dim, list | tuple) , f'input_dim ({input_dim}) must be a list or tuple, but got {type(input_dim)}'
        self.alpha_input_dim = input_dim[0]
        self.beta_input_dims = sum(input_dim[1:])
        assert self.alpha_input_dim > 0 , f'alpha_input_dim ({self.alpha_input_dim}) must be greater than 0'
        assert self.beta_input_dims > 0 , f'beta_input_dims ({self.beta_input_dims}) must be greater than 0'

        if resnet_projector:
            from src.res.algo.nn.model.CNN import mod_resnet_1d
            res_kwargs = {k:v for k,v in kwargs.items() if k != 'seq_len'}
            self.alpha_fc_enc_in = mod_resnet_1d(kwargs['inday_dim'][0] , self.alpha_input_dim , enc_in_dim , **res_kwargs) 
        else:
            self.alpha_fc_enc_in = nn.Sequential(nn.Linear(self.alpha_input_dim, enc_in_dim),nn.Tanh())
        self.beta_fc_enc_in = nn.Sequential(nn.Linear(self.beta_input_dims, enc_in_dim),nn.Tanh())

        rnn_kwargs = {'input_dim':enc_in_dim,'output_dim':hidden_dim,'num_layers':rnn_layers, 'dropout':dropout}
        self.alpha_fc_rnn = mod_gru(**rnn_kwargs)
        self.beta_fc_rnn = mod_gru(**rnn_kwargs)

        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim , alpha_num), 
            Layer.Act.get_activation_fn(act_type), 
        )
        self.beta_net = nn.Sequential(
            nn.Linear(hidden_dim , beta_num), 
            Layer.Act.get_activation_fn(act_type), 
        )
        self.alpha_map_out = Layer.MeanPool()
        self.beta_into_pred = beta_into_pred
        if beta_into_pred:
            self.beta_map_out = nn.Linear(beta_num , 1)
        else:
            self.beta_map_out = None

    def forward(self, input : Tensor | tuple[Tensor, ...] | list[Tensor]):
        """
        in: [bs x seq_len x input_dim]
        out:[bs x hidden_dim]
        """
        assert isinstance(input , tuple | list) , f'input ({input}) must be a tuple or list, but got {type(input)}'
        
        x_alpha = input[0]
        x_beta = torch.concat(input[1:] , dim = -1)
        
        x_alpha = self.alpha_fc_enc_in(x_alpha)
        x_beta = self.beta_fc_enc_in(x_beta)

        x_alpha = self.alpha_fc_rnn(x_alpha)
        x_beta = self.beta_fc_rnn(x_beta)

        x_alpha = x_alpha[:,-1]
        x_beta , x_beta_0 = x_beta[:,-1] , x_beta[: , 0]
        
        alphas = self.alpha_net(x_alpha)
        betas = self.beta_net(x_beta)
        betas_peer = self.beta_net(x_beta_0)

        pred_alpha = self.alpha_map_out(alphas) 
        if self.beta_into_pred and self.beta_map_out is not None:
            pred_beta = self.beta_map_out(betas)
        else:
            pred_beta = 0
        pred = pred_alpha + pred_beta
        return pred , {'alphas':alphas , 'betas':betas , 'betas_peer':betas_peer , 'pred_alpha':pred_alpha , 'pred_beta':pred_beta}
