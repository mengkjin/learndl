import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from copy import deepcopy
from ..function.basic import *
from .TRA import TRA
from .ResNet import resnet_1d , resnet_2d

"""
class TRA_LSTM(TRA):
    def __init__(self , input_dim , hidden_dim , tra_num_states=1, tra_horizon = 20 , 
                 tra_hidden_size=8, tra_tau=1.0, tra_rho = 0.999 , tra_lamb = 0.0):
        base_model = mod_lstm(input_dim , hidden_dim , dropout=0.0 , num_layers = 2)
        super().__init__(base_model , hidden_dim , num_states = tra_num_states, 
                         horizon = tra_horizon , hidden_size = tra_hidden_size, 
                         tau = tra_tau, rho = tra_rho , lamb = tra_lamb)
"""
class MyTRA_LSTM(TRA):
    def __init__(self , input_dim , hidden_dim , tra_num_states=1, tra_horizon = 20 , num_output = 1 , **kwargs):
        temp_model = MyLSTM(input_dim , hidden_dim = hidden_dim , num_output = 1 , **kwargs)
        base_model = nn.Sequential(temp_model.encoder , temp_model.decoder)
        super().__init__(base_model , hidden_dim , num_states = tra_num_states,  horizon = tra_horizon)

class MyResNet_LSTM(nn.Module):
    def __init__(self, input_dim , inday_dim , hidden_dim = 64 , **kwargs) -> None:
        super().__init__(**kwargs)
        self.resnet = resnet_1d(inday_dim , input_dim , hidden_dim // 4 , *kwargs) 
        self.lstm   = MyLSTM(hidden_dim // 4 , hidden_dim , **kwargs)
    def forward(self , x):
        hidden = self.resnet(x)
        output = self.lstm(hidden)[:,-1,:]
        return output
        
class mod_tcn_block(nn.Module):
    def __init__(self, input_dim , output_dim , dilation, dropout=0.0 , kernel_size=3):
        super().__init__()
        padding = (kernel_size-1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(input_dim , output_dim, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation))
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
        self.net = nn.Sequential(self.conv1, self._chomp(padding), nn.ReLU(), nn.Dropout(dropout), 
                                 self.conv2, self._chomp(padding), nn.ReLU(), nn.Dropout(dropout))
        
        if input_dim != output_dim:
            self.residual = nn.Conv1d(input_dim , output_dim, 1)
            self.residual.weight.data.normal_(0, 0.01)
        else:
            self.residual = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.net(inputs)
        output = self.relu(output + self.residual(inputs))
        return output
    
    class _chomp(nn.Module):
        def __init__(self, padding):
            super().__init__()
            self.padding = padding
        def forward(self, x):
            return x[:, :, :-self.padding] # .contiguous()

class mod_tcn(nn.Module):
    def __init__(self, input_dim , output_dim , dropout=0.0 , num_layers = 2 , kernel_size = 3):
        super().__init__()
        num_layers = max(2 , num_layers)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            inp_d , out_dim = (input_dim , output_dim) if i == 0 else (output_dim , output_dim)
            layers += [mod_tcn_block(inp_d, out_dim, dilation=dilation, dropout=dropout , kernel_size = kernel_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        output = self.net(inputs.permute(0,2,1)).permute(0,2,1)
        return output
    
class mod_transformer(nn.Module):
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_heads , ffn_dim = 8 , 4 * output_dim
        assert output_dim % num_heads == 0
        num_layers = max(2,num_layers)
        self.fc_in = nn.Sequential(nn.Linear(input_dim, output_dim),nn.Tanh())
        self.pos_enc = PositionalEncoding(output_dim,dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(output_dim , num_heads, dim_feedforward=ffn_dim , dropout=dropout , batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer , num_layers)
    def forward(self, inputs):
        hidden = self.fc_in(inputs)
        hidden = self.pos_enc(hidden)
        return self.trans(hidden)

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
    
class mod_ewlinear(nn.Module):
    def __init__(self, dim = -1 , keepdim = True):
        super().__init__()
        self.dim , self.keepdim = dim , keepdim
    def forward(self, inputs):
        return inputs.mean(dim = self.dim , keepdim = self.keepdim)
    
class mod_parallel(nn.Module):
    def __init__(self, sub_mod , num_mod , feedforward = True , concat_output = False):
        super().__init__()
        self.mod_list = nn.ModuleList([deepcopy(sub_mod) for _ in range(num_mod)])
        self.feedforward = feedforward
        self.concat_output = concat_output
    def forward(self, inputs):
        output = tuple([mod(inputs[i] if self.feedforward else inputs) for i,mod in enumerate(self.mod_list)])
        if self.concat_output:
            if isinstance(output[0] , (list,tuple)):
                output = tuple([torch.cat([out[i] for out in output] , dim = -1) for i in range(len(output[0]))])  
            else:
                output = torch.cat(output , dim = -1)
        return output

class rnn_univariate(nn.Module):
    def __init__(
        self,
        input_dim ,
        hidden_dim: int = 2**5,
        rnn_layers: int = 2,
        mlp_layers: int = 2,
        dropout:  float = 0.1,
        fc_att:    bool = False,
        fc_in:     bool = False,
        type_rnn:   str = 'gru',
        type_act:   str = 'LeakyReLU',
        num_output: int = 1 ,
        dec_mlp_dim:int = None,
        output_as_factors: bool = True,
        hidden_as_factors: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_output = num_output
        self.kwargs = kwargs
        self.kwargs.update({'input_dim':input_dim,'hidden_dim':hidden_dim,'rnn_layers':rnn_layers,'mlp_layers':mlp_layers,'dropout':dropout,
                            'fc_att':fc_att,'fc_in':fc_in,'type_rnn':type_rnn,'type_act':type_act,'num_output':num_output,'dec_mlp_dim':dec_mlp_dim,
                            'output_as_factors':output_as_factors,'hidden_as_factors':hidden_as_factors, 
                           })
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
        hidden_dim: int = 2**5,
        rnn_layers: int = 2,
        mlp_layers: int = 2,
        dropout:  float = 0.1,
        fc_att:    bool = False,
        fc_in:     bool = False,
        type_rnn:   str = 'gru',
        type_act:   str = 'LeakyReLU',
        num_output: int = 1 ,
        rnn_att:   bool = False,
        num_heads:  int = None,
        dec_mlp_dim:int = None,
        ordered_param_group: bool = False,
        output_as_factors:   bool = True,
        hidden_as_factors:   bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_output = num_output
        self.num_rnn = len(input_dim) if isinstance(input_dim , (list,tuple)) else 1
        self.ordered_param_group = ordered_param_group
        self.kwargs = kwargs
        self.kwargs.update({'input_dim':input_dim,'hidden_dim':hidden_dim,'rnn_layers':rnn_layers,'mlp_layers':mlp_layers,'dropout':dropout,
                            'fc_att':fc_att,'fc_in':fc_in,'type_rnn':type_rnn,'type_act':type_act,'num_output':num_output,'dec_mlp_dim':dec_mlp_dim,
                            'rnn_att':rnn_att,'num_heads':num_heads,'num_rnn':self.num_rnn,
                            'ordered_param_group':ordered_param_group,'output_as_factors':output_as_factors,'hidden_as_factors':hidden_as_factors,
                           })
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
                _exclude_strings = np.array([[f'enc_list.{j}.',f'dec_list.{j}.'] for j in range(self.num_rnn) if j!=i]).flatten()
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
    
class MyGRU(rnn_univariate):
    def __init__(self , input_dim , type_rnn = 'gru' , num_output = 1 , **kwargs):
        super().__init__(input_dim , type_rnn = 'gru' , num_output = 1 , **kwargs)
        
class MyLSTM(rnn_univariate):
    def __init__(self , input_dim , type_rnn = 'lstm' , num_output = 1 , **kwargs):
        super().__init__(input_dim , type_rnn = 'lstm' , num_output = 1 , **kwargs)
        
class MyTransformer(rnn_univariate):
    def __init__(self , input_dim , type_rnn = 'transformer' , num_output = 1 , **kwargs):
        super().__init__(input_dim , type_rnn = 'transformer' , num_output = 1 , **kwargs)
        
class MyTCN(rnn_univariate):
    def __init__(self , input_dim , type_rnn = 'tcn' , num_output = 1 , **kwargs):
        super().__init__(input_dim , type_rnn = 'tcn' , num_output = 1 , **kwargs)
        
class MynTaskRNN(rnn_univariate):
    def __init__(self , input_dim , num_output = 1 , **kwargs):
        super().__init__(input_dim , num_output = num_output , **kwargs)

class MyGeneralRNN(rnn_multivariate):
    def __init__(self , input_dim , **kwargs):
        super().__init__(input_dim , **kwargs)

class uni_rnn_encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,rnn_layers,dropout,fc_att,fc_in,type_rnn,**kwargs):
        super().__init__()
        self.mod_rnn = {'transformer':mod_transformer,'lstm':mod_lstm,'gru':mod_gru,'tcn':mod_tcn,}[type_rnn]
        if type_rnn == 'transformer': fc_in , fc_att = False , False
        
        self.rnn_kwargs = {'input_dim':hidden_dim if fc_in else input_dim, 'output_dim':hidden_dim,'num_layers':rnn_layers, 'dropout':dropout}
        if 'kernel_size' in kwargs.keys() and type_rnn == 'tcn': self.rnn_kwargs['kernel_size'] = kwargs['kernel_size']
        
        self.fc_in = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.Tanh()) if fc_in else nn.Sequential()
        self.fc_rnn = self.mod_rnn(**self.rnn_kwargs)
        self.fc_enc_att = TimeWiseAttention(hidden_dim,hidden_dim,dropout=dropout) if fc_att else None
    def forward(self, inputs):
        # inputs.shape : (bat_size, seq, input_dim)
        # output.shape : (bat_size, hidden_dim)
        output = self.fc_in(inputs)
        output = self.fc_rnn(output)
        output = self.fc_enc_att(output) if self.fc_enc_att else output[:,-1]
        return output
    
class uni_rnn_decoder(nn.Module):
    def __init__(self,hidden_dim,dec_mlp_dim,mlp_layers,dropout,type_act,hidden_as_factors,map_to_one=False,**kwargs):
        super().__init__()
        assert type_act in ['LeakyReLU' , 'ReLU']
        self.mod_act = getattr(nn , type_act)
        self.fc_dec_mlp = nn.Sequential()
        mlp_dim = dec_mlp_dim if dec_mlp_dim else hidden_dim
        for i in range(mlp_layers): 
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
    def __init__(self,hidden_dim,output_as_factors,hidden_as_factors,**kwargs):
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
        self.enc_list = nn.ModuleList([uni_rnn_encoder(d_inp,hidden_dim,**kwargs) for d_inp in input_dim])
    def forward(self, inputs):
        # inputs.shape : tuple of (bat_size, seq , input_dim[i_rnn]) , seq can be different 
        # output.shape : tuple of (bat_size, hidden_dim) or tuple of (bat_size, 1) if ordered_param_group
        output = [mod(inp) for inp , mod in zip(inputs , self.enc_list)]
        return output
    
class multi_rnn_decoder(nn.Module):
    def __init__(self, hidden_dim,num_rnn,rnn_att,ordered_param_group,hidden_as_factors,**kwargs):
        super().__init__()
        self.dec_list = nn.ModuleList([uni_rnn_decoder(hidden_dim , hidden_as_factors = False , map_to_one = ordered_param_group , **kwargs) for _ in range(num_rnn)])
        self.fc_mod_att = nn.Sequential()
        if ordered_param_group:
            self.fc_hid_out =  nn.BatchNorm1d(num_rnn)
        else:
            if rnn_att: 
                self.fc_mod_att = ModuleWiseAttention(hidden_dim,num_rnn , num_heads=kwargs['num_heads'] , dropout=kwargs['dropout'] , seperate_output=True)
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
    def __init__(self,hidden_dim,num_rnn,ordered_param_group,output_as_factors, hidden_as_factors,**kwargs):
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
    
class TimeWiseAttention(nn.Module):
    def __init__(self , input_dim, output_dim=None, att_dim = None, dropout = 0.0):
        super().__init__()
        if output_dim is None: output_dim = input_dim
        if att_dim is None: att_dim = output_dim
        self.fc_in = nn.Linear(input_dim, att_dim)
        self.att_net = nn.Sequential(nn.Dropout(dropout),nn.Tanh(),nn.Linear(att_dim,1,bias=False),nn.Softmax(dim=0))
        self.fc_out = nn.Linear(2*att_dim,output_dim)

    def forward(self, inputs):
        inputs = self.fc_in(inputs)
        att_score = self.att_net(inputs)  # [batch, seq_len, 1]
        output = torch.mul(inputs, att_score).sum(dim=1)
        output = torch.cat((inputs[:, -1], output), dim=1)
        return self.fc_out(output)
    
class ModuleWiseAttention(nn.Module):
    def __init__(self , input_dim , mod_num = None , att_dim = None , num_heads = None , dropout=0.0 , seperate_output = True):
        super().__init__()
        if isinstance(input_dim , (list,tuple)):
            assert mod_num == len(input_dim)
        else:
            input_dim = [input_dim for _ in range(mod_num)]
        
        att_dim = max(input_dim) if att_dim is None else att_dim
        num_heads = att_dim // 8 if num_heads is None else num_heads
        
        self.in_fc = nn.ModuleList([nn.Linear(inp_d , att_dim) for inp_d in input_dim])
        self.task_mha = nn.MultiheadAttention(att_dim, num_heads = num_heads, batch_first=True , dropout = dropout)
        self.seperate_output = seperate_output
    def forward(self, inputs):
        hidden = torch.stack([f(x) for x,f in zip(inputs,self.in_fc)],dim=-2)
        hidden = self.task_mha(hidden , hidden , hidden)[0] + hidden
        if self.seperate_output:
            return tuple([hidden.select(-2,i) for i in range(hidden.shape[-2])])
        else:
            return hidden
        
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, dropout=0.0, max_len=1000,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.seq_len = max_len
        self.P = torch.zeros(1 , self.seq_len, input_dim)
        X = torch.arange(self.seq_len, dtype=torch.float).reshape(-1,1) / torch.pow(10000,torch.arange(0, input_dim, 2 ,dtype=torch.float) / input_dim)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X[:,:input_dim//2])
    def forward(self, inputs):
        return self.dropout(inputs + self.P[:,:inputs.shape[1],:].to(inputs.device))

class SampleWiseTranformer(nn.Module):
    def __init__(self , hidden_dim , ffn_dim = None , num_heads = 8 , encoder_layers = 2 , dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        ffn_dim = 4 * hidden_dim if ffn_dim is None else ffn_dim
        self.fc_att = TimeWiseAttention(hidden_dim,hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=ffn_dim , dropout=dropout , batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer , encoder_layers)
    def forward(self, inputs , pad_mask = None):
        if inputs.isnan().any():
            pad_mask = self.pad_mask_nan(inputs) if pad_mask is None else (self.pad_mask_nan(inputs) + pad_mask) > 0
            inputs = inputs.nan_to_num()
        hidden = hidden.unsqueeze(0) if hidden.dim() == 2 else self.fc_att(inputs).unsqueeze(0)
        return self.trans(hidden , src_key_padding_mask = pad_mask).squeeze(0)
    def pad_mask_rand(self , inputs , mask_ratio = 0.1):
        return (torch.rand(1,inputs.shape[0]) < mask_ratio).to(inputs.device)
    def pad_mask_nan(self , inputs):
        return inputs.sum(dim = tuple(torch.arange(inputs.dim())[1:])).isnan().unsqueeze(0)    

class TimeWiseTranformer(nn.Module):
    def __init__(self , input_dim , hidden_dim , ffn_dim = None , num_heads = 8 , encoder_layers = 2 , dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        ffn_dim = 4 * hidden_dim if ffn_dim is None else ffn_dim
        self.pos_enc = PositionalEncoding(hidden_dim,dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim , num_heads, dim_feedforward=ffn_dim , dropout=dropout , batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer , encoder_layers)
    def forward(self, inputs):
        hidden = self.pos_enc(hidden)
        return self.trans(hidden)
