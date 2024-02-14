import torch
import torch.nn as nn
from ..function.basic import *
from .tra import mod_tra
from .rnn import rnn_univariate , rnn_multivariate

"""
class TRA_LSTM(TRA):
    def __init__(self , input_dim , hidden_dim , tra_num_states=1, tra_horizon = 20 , 
                 tra_hidden_size=8, tra_tau=1.0, tra_rho = 0.999 , tra_lamb = 0.0):
        base_model = mod_lstm(input_dim , hidden_dim , dropout=0.0 , num_layers = 2)
        super().__init__(base_model , hidden_dim , num_states = tra_num_states, 
                         horizon = tra_horizon , hidden_size = tra_hidden_size, 
                         tau = tra_tau, rho = tra_rho , lamb = tra_lamb)

class ResNet_LSTM(nn.Module):
    def __init__(self, input_dim , inday_dim , hidden_dim = 64 , **kwargs) -> None:
        super().__init__()
        self.resnet = mod_resnet_1d(inday_dim , input_dim , hidden_dim // 4 , **kwargs) 
        self.lstm   = LSTM(hidden_dim // 4 , hidden_dim = hidden_dim , **kwargs)
    def forward(self , x):
        hidden = self.resnet(x)
        output = self.lstm(hidden)
        return output
"""     

class gru(rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'gru' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class lstm(rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'lstm' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class resnet_lstm(lstm):
    def __init__(self, input_dim , hidden_dim , inday_dim , **kwargs) -> None:
        kwargs.update({'enc_in' : 'resnet' , 'enc_in_dim' : kwargs.get('enc_in_dim') if kwargs.get('enc_in_dim') else hidden_dim // 4})
        super().__init__(input_dim , hidden_dim , inday_dim = inday_dim , **kwargs)

class tra_lstm(mod_tra):
    def __init__(self , input_dim , hidden_dim , tra_num_states=1, tra_horizon = 20 , num_output = 1 , **kwargs):
        temp_model = lstm(input_dim , hidden_dim = hidden_dim , num_output = 1 , **kwargs)
        base_model = nn.Sequential(temp_model.encoder , temp_model.decoder)
        super().__init__(base_model , hidden_dim , num_states = tra_num_states,  horizon = tra_horizon)
        
class transformer(rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'transformer' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class tcn(rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'tcn' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class rnn_ntask(rnn_univariate):
    def __init__(self , input_dim , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(input_dim , hidden_dim , num_output = num_output , **kwargs)

class rnn_general(rnn_multivariate):
    def __init__(self , input_dim , **kwargs):
        super().__init__(input_dim , **kwargs)