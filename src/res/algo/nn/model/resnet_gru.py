"""
ResNet-GRU model.
Use resnet to encode the input features, and then use gru to process the features.
"""

from __future__ import annotations
from torch import nn , Tensor

from .. import layer as Layer
from .Attention import TimeWiseAttention
from .CNN import mod_resnet_1d
from .RNN import mod_gru

class resnet_gru(nn.Module):
    """GRU with intra-day ResNet-1D encoder.  Registry key: ``'resnet_gru'``.

    Requires ``inday_dim`` (intra-day bar count) in kwargs.
    """
    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**6,
        dropout         = 0.1,
        act_type        = 'leaky',
        enc_in          = None,
        enc_in_dim      = None ,
        enc_att         = False,
        rnn_type        = 'gru',
        rnn_layers      = 2,
        loss_corr_lamb = 0.1,
        **kwargs
    ):
        super().__init__()
        assert rnn_type == 'gru' , f'only gru is supported : {rnn_type}'
        self.loss_corr_lamb = loss_corr_lamb

        if enc_in_dim is None:
            enc_in_dim = hidden_dim
        
        res_kwargs = {k:v for k,v in kwargs.items() if k != 'seq_len'}
        self.fc_enc_in = mod_resnet_1d(kwargs['inday_dim'] , input_dim , enc_in_dim , **res_kwargs) 

        self.fc_rnn = mod_gru(
            input_dim = enc_in_dim,
            output_dim = hidden_dim,
            num_layers = rnn_layers,
            dropout = dropout,
        )

        if enc_att:
            self.fc_enc_att = TimeWiseAttention(hidden_dim,hidden_dim,dropout=dropout) 
        else:
            self.fc_enc_att = None
        
        self.fc_hid_out = nn.Sequential(nn.Linear(hidden_dim , hidden_dim) , nn.BatchNorm1d(hidden_dim)) 
        self.fc_map_out = nn.Sequential(Layer.MeanPool() , nn.BatchNorm1d(1))

        
    def forward(self , x : Tensor) -> tuple[Tensor,dict]:
        """
        in: [bs x seq_len x input_dim]
        out:[bs x 1] , [bs x hidden_dim]
        """
        x = self.fc_enc_in(x)
        x = self.fc_rnn(x)
        x = self.fc_enc_att(x) if self.fc_enc_att is not None else x[:,-1]
        x = self.fc_hid_out(x)
        o = self.fc_map_out(x)
        return o , {'hidden' : x}

    # def loss(self, pred : Tensor , label : Tensor , hidden : Tensor , weight : float = 1.0, **kwargs):
    #     """Composite ABCM loss: MSE + R² + corr penalty + turnover penalty.

    #     Args:
    #         pred:        Scalar predictions ``[bs, 1]``.
    #         label:       Two-column label ``[bs, 2]`` where ``[...,0]`` is the
    #                      return target and ``[...,1]`` is the R² target. (std and rtn)
    #         hidden:      Hidden states ``[bs, hidden_dim]``.
    #     """
    #     from torch.nn import functional as F
    #     mse = F.mse_loss(pred.squeeze() , label.squeeze())
    #     corr = self.corr_loss(hidden)
    #     all_losses = {
    #         'mse': mse,
    #         'corr': self.loss_corr_lamb * corr,
    #     }
    #     return all_losses

    # def corr_loss(self, hiddens : Tensor , **kwargs):
    #     """Frobenius norm of the standardized beta covariance matrix."""
    #     h = (hiddens - hiddens.mean(dim=0,keepdim=True)) / (hiddens.std(dim=0,keepdim=True) + 1e-6)
    #     pen = h.T.cov().norm()
    #     return pen
