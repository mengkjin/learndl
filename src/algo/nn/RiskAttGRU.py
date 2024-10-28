import torch
import torch.nn as nn

from . import layer as Layer
from ...basic.conf import RISK_INDUS , RISK_STYLE

__all__ = ['risk_att_gru']

class risk_att_gru(nn.Module):
    def __init__(
            self, 
            input_dim    = (6,len(RISK_STYLE),len(RISK_INDUS)),
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
        assert input_dim[1] == len(RISK_STYLE) , (input_dim , (len(RISK_STYLE),len(RISK_INDUS)))
        assert input_dim[2] == len(RISK_INDUS) , (input_dim , (len(RISK_STYLE),len(RISK_INDUS)))

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
    from src.api import DataAPI
    batch_data = DataAPI.get_realistic_batch_data('day+style+indus')

    rau = risk_att_gru(indus_embed=True)
    rau(batch_data.x).shape