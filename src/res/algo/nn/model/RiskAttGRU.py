"""RiskAttGRU: GRU with risk-factor cross-attention (style + industry factors).

The model uses style and industry risk factors as attention keys and queries
to produce a context vector that attends over the GRU hidden state.
"""
import torch
import torch.nn as nn

from src.proj import CONST

from .. import layer as Layer

__all__ = ['risk_att_gru']

class risk_att_gru(nn.Module):
    """GRU with explicit risk-factor cross-attention.  Registry key: ``'risk_att_gru'``.

    ``_default_data_type = 'day+style+indus'`` signals to the training loop
    that this model requires a 3-tuple input: ``(trade, style, indus)``.

    Architecture:
    1. GRU encodes the daily trading feature sequence → ``H = [bs, hidden_dim]``
    2. Industry features are optionally embedded: ``indus → [bs, indus_dim]``
    3. Style + embedded indus are concatenated to form risk vector ``r``
    4. Explicit Q/K/V attention: ``A = softmax(Qr · (Kr)ᵀ / sqrt(h))``;
       context = ``A @ V(H)``
    5. Concatenate ``H`` and context → MLP → scalar output

    Note: This is an explicit Q/K/V attention computation, *not* the standard
    multi-head attention module.

    Args:
        input_dim:    Tuple ``(trade_dim, style_dim, indus_dim)`` — defaults
                      use the project's CONST config for style/indus counts.
        hidden_dim:   GRU hidden dimension (default ``32``).
        att_dim:      Attention key/query dimension (default ``128``).
        dropout:      Dropout rate (default ``0.1``).
        act_type:     Activation key for the MLP (default ``'leaky'``).
        rnn_layers:   Number of GRU layers (default ``2``).
        indus_dim:    Industry embedding dimension (default ``8``).
        indus_embed:  If True, embed the industry one-hot encoding with a
                      Linear layer before attention (default ``True``).

    Forward input:
        x: Tuple ``(trade, style, indus)``
           * ``trade``: ``[bs, seq_len, trade_dim]``
           * ``style``: ``[bs, 1, style_dim]``
           * ``indus``: ``[bs, 1, indus_dim]``

    Returns:
        Scalar prediction ``[bs, 1]``.
    """
    _default_data_type = 'day+style+indus'

    def __init__(
            self, 
            input_dim    = (6,len(CONST.Conf.Factor.RISK.style),len(CONST.Conf.Factor.RISK.indus)),
            hidden_dim   = 2**5,
            att_dim      = 2**7 ,
            dropout      = 0.1,
            act_type     = 'leaky',
            rnn_layers   = 2 ,
            indus_dim    = 2**3 ,
            indus_embed  = True , 
            **kwargs) -> None:
        super().__init__()
        
        self.trade_gru = nn.GRU(input_dim[0] , hidden_dim , num_layers = rnn_layers , dropout = dropout , batch_first = True)
        assert input_dim[1] == len(CONST.Conf.Factor.RISK.style) , \
            (input_dim , (len(CONST.Conf.Factor.RISK.style),len(CONST.Conf.Factor.RISK.indus)))
        assert input_dim[2] == len(CONST.Conf.Factor.RISK.indus) , \
            (input_dim , (len(CONST.Conf.Factor.RISK.style),len(CONST.Conf.Factor.RISK.indus)))

        if indus_embed:
            self.indus_embedding = nn.Linear(input_dim[-1] , indus_dim)
            h = input_dim[1] + indus_dim
        else:
            self.indus_embedding = nn.Sequential()
            h = sum(input_dim[1:])

        self.Q = nn.Linear(h , att_dim)
        self.K = nn.Linear(h , att_dim)
        self.V = nn.Linear(hidden_dim , hidden_dim)
        self.softmax = nn.Softmax(dim = -1)
        self.mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim , hidden_dim), 
                Layer.Act.get_activation_fn(act_type), 
                nn.Dropout(dropout) , 
                nn.BatchNorm1d(hidden_dim) ,
                nn.Linear(hidden_dim , 1), )
        self.sqrt_H = h ** 0.5

    def forward(self , x):
        trade , style , indus = x
        indus = self.indus_embedding(indus)
        risk = torch.concat([style , indus], dim = -1).squeeze(1)
        q = self.Q(risk)
        k = self.K(risk)
        A = self.softmax(q.mm(k.T) / self.sqrt_H)
        H = self.trade_gru(trade)[0][:,-1]
        v = self.V(H)
        z = torch.concat([H,A.mm(v)] , dim = -1)
        z = self.mlp(z)
        return z
    
if __name__ == '__main__' :
    from src.res.model.data_module import get_realistic_batch_data
    batch_input = get_realistic_batch_data('day+style+indus')

    rau = risk_att_gru(indus_embed=True)
    rau(batch_input.x).shape