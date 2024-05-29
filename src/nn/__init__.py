from torch import nn , Tensor
from typing import Literal
from . import (
    attention , cnn , rnn , modernTCN , patchTST , TSMixer
)

from .tra import tra
from .factorVAE import FactorVAE as factor_vae

def get_nn_category(module_name : str) -> Literal['vae' , 'tra' , '']:
    if module_name == 'factor_vae':
        return 'vae'
    elif module_name == 'tra':
        return 'tra'
    else:
        return ''

class simple_lstm(nn.Module):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim , hidden_dim , num_layers = 1 , dropout = 0 , batch_first = True)
        self.fc = nn.Linear(hidden_dim , 1)
    def forward(self, x : Tensor) -> tuple[Tensor , dict]:
        o = self.lstm(x)[0][:,-1]
        return self.fc(o) , {'hidden' : o}
    
class gru(rnn.rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'gru'})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class lstm(rnn.rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'lstm' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class resnet_lstm(lstm):
    def __init__(self, input_dim , hidden_dim , inday_dim , **kwargs) -> None:
        kwargs.update({
            'enc_in' : 'resnet' , 
            'enc_in_dim' : kwargs.get('enc_in_dim') if kwargs.get('enc_in_dim') else hidden_dim // 4 , 
        })
        super().__init__(input_dim , hidden_dim , inday_dim = inday_dim , **kwargs)

class resnet_gru(gru):
    def __init__(self, input_dim , hidden_dim , inday_dim , **kwargs):
        kwargs.update({
            'enc_in' : 'resnet' , 
            'enc_in_dim' : kwargs.get('enc_in_dim') if kwargs.get('enc_in_dim') else hidden_dim // 4 ,
        })
        super().__init__(input_dim , hidden_dim , inday_dim = inday_dim , **kwargs)
        
class transformer(rnn.rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'transformer' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class tcn(rnn.rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'tcn' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class rnn_ntask(rnn.rnn_univariate):
    def __init__(self , input_dim , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(input_dim , hidden_dim , num_output = num_output , **kwargs)

class rnn_general(rnn.rnn_multivariate):
    def __init__(self , input_dim , **kwargs):
        super().__init__(input_dim , **kwargs)

class patch_tst(patchTST.PatchTST):
    def __init__(self , input_dim , seq_len , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(nvars = input_dim , seq_len = seq_len , d_model = hidden_dim , 
                         predict_steps = num_output , head_type = 'prediction' , **kwargs)
        
class modern_tcn(modernTCN.ModernTCN):
    def __init__(self , input_dim , seq_len , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(nvars = input_dim , seq_len = seq_len , d_model = hidden_dim , 
                         predict_steps = num_output , head_type = 'prediction' , **kwargs)
        
class ts_mixer(TSMixer.TSMixer):
    def __init__(self , input_dim , seq_len , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(nvars = input_dim , seq_len = seq_len , d_model = hidden_dim , 
                         predict_steps = num_output , head_type = 'prediction' , **kwargs)