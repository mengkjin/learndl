import torch
import torch.nn as nn
#from ..function.basic import *
from .basic import *
from .cnn import mod_resnet_1d , mod_tcn
from .attention import mod_transformer,TimeWiseAttention,ModuleWiseAttention

class mod_lstm(nn.Module):
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_layers = min(3,num_layers)
        self.lstm = nn.LSTM(input_dim , output_dim , num_layers = num_layers , dropout = dropout , batch_first = True)
    def forward(self, inputs):
        return self.lstm(inputs)[0]

class mod_gru(nn.Module):
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_layers = min(3,num_layers)
        self.gru = nn.GRU(input_dim , output_dim , num_layers = num_layers , dropout = dropout , batch_first = True)
    def forward(self, inputs):
        return self.gru(inputs)[0]

class rnn_univariate(nn.Module):
    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**5,
        dropout         = 0.1,
        act_type        = 'LeakyReLU',
        enc_in          = None,
        enc_in_dim      = None ,
        enc_att         = False,
        rnn_type        = 'gru',
        rnn_layers      = 2,
        dec_mlp_layers  = 2,
        dec_mlp_dim     = None,
        num_output      = 1 ,
        output_as_factors  = True,
        hidden_as_factors  = False,
        **kwargs
    ):
        super().__init__()
        self.num_output = num_output

        self.kwargs = {
            'input_dim':        input_dim,
            'hidden_dim':       hidden_dim,
            'dropout':          dropout,
            'act_type':         act_type,
            'enc_in':           enc_in,
            'enc_in_dim':       enc_in_dim,
            'enc_att':          enc_att,
            'rnn_type':         rnn_type,
            'rnn_layers':       rnn_layers,
            'dec_mlp_layers':   dec_mlp_layers,
            'dec_mlp_dim':      dec_mlp_dim,
            'num_output':       num_output,
            'hidden_as_factors':hidden_as_factors,
            'output_as_factors':output_as_factors,
            **kwargs,
        }

        self.encoder = mod_parallel(uni_rnn_encoder(**self.kwargs) , num_mod = 1 , feedforward = False , concat_output = True)
        self.decoder = mod_parallel(uni_rnn_decoder(**self.kwargs) , num_mod = num_output , feedforward = False , concat_output = False)
        self.mapping = mod_parallel(uni_rnn_mapping(**self.kwargs) , num_mod = num_output , feedforward = True , concat_output = True)
        self.set_multiloss_params()

    def forward(self, inputs):
        # inputs.shape : (bat_size, seq, input_dim)
        hidden = self.encoder(inputs) # hidden.shape : (bat_size, hidden_dim)
        hidden = self.decoder(hidden) # hidden.shape : tuple of (bat_size, hidden_dim) , len is num_output
        output = self.mapping(hidden) # output.shape : (bat_size, num_output)   
        return output , hidden[0]
        
    def set_multiloss_params(self):
        self.multiloss_alpha = torch.nn.Parameter((torch.rand(self.num_output) + 1e-4).requires_grad_())
        
    def get_multiloss_params(self):
        return {'alpha':self.multiloss_alpha}
    
class rnn_multivariate(nn.Module):
    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**5,
        dropout         = 0.1,
        act_type        = 'LeakyReLU',
        enc_in          = None,
        enc_in_dim      = None ,
        enc_att         = False,
        rnn_type        = 'gru',
        rnn_layers      = 2 ,
        rnn_att         = False,
        num_heads       = None,
        dec_mlp_layers  = 2,
        dec_mlp_dim     = None,
        num_output      = 1 ,
        ordered_param_group = False,
        output_as_factors   = True,
        hidden_as_factors   = False,
        **kwargs,
    ):
        super().__init__()
        self.num_output = num_output
        self.num_rnn = len(input_dim) if isinstance(input_dim , (list,tuple)) else 1
        self.ordered_param_group = ordered_param_group

        self.kwargs = {
            'input_dim':        input_dim,
            'hidden_dim':       hidden_dim,
            'dropout':          dropout,
            'act_type':         act_type,
            'enc_in':           enc_in,
            'enc_in_dim':       enc_in_dim,
            'enc_att':          enc_att,
            'rnn_type':         rnn_type,
            'rnn_layers':       rnn_layers,
            'rnn_att':          rnn_att,
            'num_heads':        num_heads,
            'dec_mlp_layers':   dec_mlp_layers,
            'dec_mlp_dim':      dec_mlp_dim,
            'num_output':       num_output,
            'ordered_param_group':ordered_param_group,
            'hidden_as_factors':hidden_as_factors,
            'output_as_factors':output_as_factors,
            'num_rnn':          self.num_rnn,
            **kwargs,
        }

        mod_encoder = multi_rnn_encoder if self.num_rnn > 1 else uni_rnn_encoder
        mod_decoder = multi_rnn_decoder if self.num_rnn > 1 else uni_rnn_decoder
        mod_mapping = multi_rnn_mapping if self.num_rnn > 1 else uni_rnn_mapping

        self.encoder = mod_parallel(mod_encoder(**self.kwargs) , num_mod = 1 , feedforward = False , concat_output = True)
        self.decoder = mod_parallel(mod_decoder(**self.kwargs) , num_mod = num_output , feedforward = False , concat_output = False)
        self.mapping = mod_parallel(mod_mapping(**self.kwargs) , num_mod = num_output , feedforward = True , concat_output = True)

        self.set_multiloss_params()
        self.set_param_groups()
    
    def forward(self, inputs):
        # inputs.shape : tuple of (bat_size, seq , input_dim[i_rnn]) , len is num_rnn
        hidden = self.encoder(inputs) # hidden.shape : tuple of (bat_size, hidden_dim) , len is num_rnn
        hidden = self.decoder(hidden) # hidden.shape : tuple of (bat_size, num_rnn * hidden_dim) , len is num_output
        output = self.mapping(hidden) # output.shape : (bat_size, 1)      
        return output , hidden[0]
    
    def max_round(self):
        return len(self.param_groups)
    
    def set_param_groups(self):
        self.param_groups = []
        if self.ordered_param_group and self.num_rnn > 1:
            for i in range(self.num_rnn):
                _exclude_strings = [f'enc_list.{j}.' for j in range(self.num_rnn) if j!=i] + [f'dec_list.{j}.' for j in range(self.num_rnn) if j!=i]
                self.param_groups.append([param for k,param in self.named_parameters() if all([k.find(_str) < 0 for _str in _exclude_strings])]) 
                assert len(self.param_groups[-1]) > 0
        else:
            self.param_groups.append(list(self.parameters())) 
    
    def training_round(self , round_num):
        [par.requires_grad_(round_num >= self.max_round()) for par in self.parameters()]
        [par.requires_grad_(True) for par in self.param_groups[round_num]]
        
    def set_multiloss_params(self):
        self.multiloss_alpha = torch.nn.Parameter((torch.rand(self.num_output) + 1e-4).requires_grad_())
        
    def get_multiloss_params(self):
        return {'alpha':self.multiloss_alpha}

class uni_rnn_encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,dropout,rnn_type,rnn_layers,enc_in=None,enc_in_dim=None,enc_att=False,**kwargs):
        super().__init__()
        self.mod_rnn = {'transformer':mod_transformer,'lstm':mod_lstm,'gru':mod_gru,'tcn':mod_tcn,}[rnn_type]
        if self.mod_rnn == mod_transformer: 
            enc_in , enc_in_dim , enc_att = None , input_dim , False
        else:
            enc_in , enc_in_dim , enc_att = enc_in , enc_in_dim if enc_in_dim else hidden_dim , enc_att
        
        if enc_in == 'linear' or enc_in == True:
            self.fc_enc_in = nn.Sequential(nn.Linear(input_dim, enc_in_dim),nn.Tanh())
        elif enc_in == 'resnet':
            self.fc_enc_in = mod_resnet_1d(kwargs['inday_dim'] , input_dim , enc_in_dim , **kwargs) 
        else:
            enc_in_dim = input_dim
            self.fc_enc_in = nn.Sequential()

        rnn_kwargs = {'input_dim':enc_in_dim,'output_dim':hidden_dim,'num_layers':rnn_layers, 'dropout':dropout}
        if rnn_type == 'tcn': rnn_kwargs['kernel_size'] = kwargs['kernel_size']
        self.fc_rnn = self.mod_rnn(**rnn_kwargs)

        if enc_att:
            self.fc_enc_att = TimeWiseAttention(hidden_dim,hidden_dim,dropout=dropout) 
        else:
            self.fc_enc_att = None

    def forward(self, inputs):
        # inputs.shape : (bat_size, seq, input_dim)
        # output.shape : (bat_size, hidden_dim)
        output = self.fc_enc_in(inputs)
        output = self.fc_rnn(output)
        output = self.fc_enc_att(output) if self.fc_enc_att else output[:,-1]
        return output
    
class uni_rnn_decoder(nn.Module):
    def __init__(self,hidden_dim,act_type,dec_mlp_layers,dec_mlp_dim,dropout,hidden_as_factors,map_to_one=False,**kwargs):
        super().__init__()
        self.mod_act = getattr(nn , act_type)
        self.fc_dec_mlp = nn.Sequential()
        mlp_dim = dec_mlp_dim if dec_mlp_dim else hidden_dim
        for i in range(dec_mlp_layers): 
            self.fc_dec_mlp.append(nn.Sequential(nn.Linear(hidden_dim if i == 0 else mlp_dim , mlp_dim), self.mod_act(), nn.Dropout(dropout)))
        if hidden_as_factors:
            self.fc_hid_out = nn.Sequential(nn.Linear(mlp_dim , 1 if map_to_one else hidden_dim) , nn.BatchNorm1d(1 if map_to_one else hidden_dim)) 
        else:
            self.fc_hid_out = nn.Linear(mlp_dim , 1 if map_to_one else hidden_dim)
    def forward(self, inputs):
        # inputs.shape : (bat_size, hidden_dim)
        # output.shape : (bat_size, out_dim/hidden_dim)
        output = self.fc_dec_mlp(inputs)
        output = self.fc_hid_out(output)
        return output
    
class uni_rnn_mapping(nn.Module):
    def __init__(self,hidden_dim,hidden_as_factors,output_as_factors,**kwargs):
        super().__init__()
        self.fc_map_out = nn.Sequential(mod_ewlinear()) if hidden_as_factors else nn.Sequential(nn.Linear(hidden_dim, 1))
        if output_as_factors: self.fc_map_out.append(nn.BatchNorm1d(1))
    def forward(self, inputs):
        # inputs.shape : (bat_size, hidden_dim)
        # output.shape : (bat_size, 1)
        return self.fc_map_out(inputs)

class multi_rnn_encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,**kwargs):
        super().__init__()
        self.enc_list = nn.ModuleList([uni_rnn_encoder(input_dim=indim,hidden_dim=hidden_dim,**kwargs) for indim in input_dim])
    def forward(self, inputs):
        # inputs.shape : tuple of (bat_size, seq , input_dim[i_rnn]) , seq can be different 
        # output.shape : tuple of (bat_size, hidden_dim) or tuple of (bat_size, 1) if ordered_param_group
        output = [mod(inp) for inp , mod in zip(inputs , self.enc_list)]
        return output
    
class multi_rnn_decoder(nn.Module):
    def __init__(self,hidden_dim,dropout,num_rnn,rnn_att,num_heads,ordered_param_group,hidden_as_factors,**kwargs):
        super().__init__()
        num_rnn = num_rnn
        self.dec_list = nn.ModuleList([uni_rnn_decoder(hidden_dim , hidden_as_factors = False , map_to_one = ordered_param_group , **kwargs) for _ in range(num_rnn)])
        self.fc_mod_att = nn.Sequential()
        if ordered_param_group:
            self.fc_hid_out =  nn.BatchNorm1d(num_rnn)
        else:
            if rnn_att: 
                self.fc_mod_att = ModuleWiseAttention(hidden_dim,num_rnn , num_heads=num_heads , dropout=dropout , seperate_output=True)
            if hidden_as_factors:
                self.fc_hid_out = nn.Sequential(nn.Linear(num_rnn*hidden_dim , hidden_dim) , nn.BatchNorm1d(hidden_dim))
            else:
                self.fc_hid_out = nn.Linear(num_rnn*hidden_dim , hidden_dim)
    def forward(self, inputs):
        # inputs.shape : tuple of (bat_size, hidden_dim) , len is num_rnn
        # output.shape : (bat_size, hidden_dim) or (bat_size, num_rnn) if ordered_param_group
        output = [mod(inp) for inp , mod in zip(inputs , self.dec_list)]
        output = torch.cat(self.fc_mod_att(output) , dim = -1)
        output = self.fc_hid_out(output)
        return output
    
class multi_rnn_mapping(nn.Module):
    def __init__(self,hidden_dim,ordered_param_group,hidden_as_factors,output_as_factors,**kwargs):
        super().__init__()
        if ordered_param_group or hidden_as_factors: 
            self.fc_map_out = nn.Sequential(mod_ewlinear())
        else:
            self.fc_map_out = nn.Sequential(nn.Linear(hidden_dim, 1))
        if output_as_factors:  self.fc_map_out.append(nn.BatchNorm1d(1))
    def forward(self, inputs):
        # inputs.shape : (bat_size, hidden_dim) or (bat_size, num_rnn) if ordered_param_group
        # output.shape : (bat_size, 1)
        return self.fc_map_out(inputs)
