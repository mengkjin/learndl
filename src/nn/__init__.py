from torch import nn , Tensor
from . import (
    attention , cnn , rnn , tra , modernTCN , patchTST , TSMixer
)

class simple_lstm(nn.Module):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim , hidden_dim , num_layers = 1 , dropout = 0 , batch_first = True)
        self.fc = nn.Linear(hidden_dim , 1)
    def forward(self, x : Tensor) -> tuple[Tensor , Tensor]:
        o = self.lstm(x)[0][:,-1]
        return self.fc(o) , o
    
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

@tra.tra_component('mapping')
class tra_lstm(lstm):
    def __init__(self , input_dim , hidden_dim , tra_num_states=1, tra_horizon = 20 , num_output = 1 , **kwargs):
        super().__init__(input_dim , hidden_dim , **kwargs)
        self.mapping = tra.block_tra(hidden_dim , num_states = tra_num_states, horizon = tra_horizon)
        self.set_multiloss_params()

    def forward(self, inputs):
        # inputs.shape : (bat_size, seq, input_dim)
        hidden = self.encoder(inputs) # hidden.shape : (bat_size, hidden_dim)
        hidden = self.decoder(hidden) # hidden.shape : tuple of (bat_size, hidden_dim) , len is num_output
        if isinstance(hidden , tuple): hidden = hidden[0]
        output , hidden2 = self.mapping(hidden) # output.shape : (bat_size, num_output)   
        return output , hidden2
        
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
                         predict_steps = num_output , head_type = "prediction" , **kwargs)
        
class modern_tcn(modernTCN.ModernTCN):
    def __init__(self , input_dim , seq_len , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(nvars = input_dim , seq_len = seq_len , d_model = hidden_dim , 
                         predict_steps = num_output , head_type = "prediction" , **kwargs)
        
class ts_mixer(TSMixer.TSMixer):
    def __init__(self , input_dim , seq_len , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(nvars = input_dim , seq_len = seq_len , d_model = hidden_dim , 
                         predict_steps = num_output , head_type = "prediction" , **kwargs)