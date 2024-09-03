import torch

from torch import nn , Tensor
from typing import Optional

from . import layer as Layer
from .Attention import mod_transformer,TimeWiseAttention,ModuleWiseAttention
from .CNN import mod_resnet_1d , mod_tcn
from .util import add_multiloss_params

def get_rnn_mod(rnn_type):
    return {'transformer':mod_transformer,'lstm':mod_lstm,'gru':mod_gru,'tcn':mod_tcn,}[rnn_type]

class simple_lstm(nn.Module):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim , hidden_dim , num_layers = 1 , dropout = 0 , batch_first = True)
        self.fc = nn.Linear(hidden_dim , 1)
    def forward(self, x : Tensor) -> tuple[Tensor , dict]:
        o = self.lstm(x)[0][:,-1]
        return self.fc(o) , {'hidden' : o}
    

class mod_lstm(nn.Module):
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_layers = min(3,num_layers)
        self.lstm = nn.LSTM(input_dim , output_dim , num_layers = num_layers , dropout = dropout , batch_first = True)
    def forward(self, x : Tensor) -> Tensor:
        return self.lstm(x)[0]

class mod_gru(nn.Module):
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_layers = min(3,num_layers)
        self.gru = nn.GRU(input_dim , output_dim , num_layers = num_layers , dropout = dropout , batch_first = True)
    def forward(self, x : Tensor) -> Tensor:
        return self.gru(x)[0]

class rnn_univariate(nn.Module):
    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**5,
        dropout         = 0.1,
        act_type        = 'leaky',
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

        self.encoder = Layer.Parallel(uni_rnn_encoder(**self.kwargs) , num_mod = 1 , feedforward = False , concat_output = True)
        self.decoder = Layer.Parallel(uni_rnn_decoder(**self.kwargs) , num_mod = num_output , feedforward = False , concat_output = False)
        self.mapping = Layer.Parallel(uni_rnn_mapping(**self.kwargs) , num_mod = num_output , feedforward = True , concat_output = True)

        add_multiloss_params(self , num_output)

    def forward(self , x : Tensor) -> tuple[Tensor,dict]:
        '''
        in: [bs x seq_len x input_dim]
        out:[bs x 1] , [bs x hidden_dim]
        '''
        x = self.encoder(x) # [bs x hidden_dim]
        x = self.decoder(x) # tuple of [bs x hidden_dim] , len is num_output
        o = self.mapping(x) # [bs x num_output]
        return o , {'hidden' : x[0]}
        
        
class rnn_multivariate(nn.Module):
    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**5,
        dropout         = 0.1,
        act_type        = 'leaky',
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
        output_as_factors   = True,
        hidden_as_factors   = False,
        **kwargs,
    ):
        super().__init__()
        self.num_output = num_output
        self.num_rnn = len(input_dim) if isinstance(input_dim , (list,tuple)) else 1

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
            'hidden_as_factors':hidden_as_factors,
            'output_as_factors':output_as_factors,
            'num_rnn':          self.num_rnn,
            **kwargs,
        }

        mod_encoder = multi_rnn_encoder if self.num_rnn > 1 else uni_rnn_encoder
        mod_decoder = multi_rnn_decoder if self.num_rnn > 1 else uni_rnn_decoder
        mod_mapping = multi_rnn_mapping if self.num_rnn > 1 else uni_rnn_mapping

        self.encoder = Layer.Parallel(mod_encoder(**self.kwargs) , num_mod = 1 , feedforward = False , concat_output = True)
        self.decoder = Layer.Parallel(mod_decoder(**self.kwargs) , num_mod = num_output , feedforward = False , concat_output = False)
        self.mapping = Layer.Parallel(mod_mapping(**self.kwargs) , num_mod = num_output , feedforward = True , concat_output = True)

        add_multiloss_params(self , num_output)
    
    def forward(self, x : Tensor) -> tuple[Tensor,dict]:
        '''
        in: tuple of [bs x seq_len x input_dim[i]] , len is num_rnn
        out:[bs x 1] , [bs x num_rnn * hidden_dim]
        '''
        x = self.encoder(x) # tuple of [bs x hidden_dim] , len is num_rnn
        x = self.decoder(x) # tuple of [bs x num_rnn * hidden_dim] , len is num_output
        o = self.mapping(x) # [bs, 1]
        return o , {'hidden' : x[0]}

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
            res_kwargs = {k:v for k,v in kwargs.items() if k != 'seq_len'}
            self.fc_enc_in = mod_resnet_1d(kwargs['inday_dim'] , input_dim , enc_in_dim , **res_kwargs) 
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

    def forward(self, x : Tensor) -> Tensor:
        '''
        in: [bs x seq_len x input_dim]
        out:[bs x hidden_dim]
        '''
        x = self.fc_enc_in(x)
        x = self.fc_rnn(x)
        x = self.fc_enc_att(x) if self.fc_enc_att else x[:,-1]
        return x
    
class uni_rnn_decoder(nn.Module):
    def __init__(self,hidden_dim,act_type,dec_mlp_layers,dec_mlp_dim,dropout,hidden_as_factors,map_to_one=False,**kwargs):
        super().__init__()
        self.fc_dec_mlp = nn.Sequential()
        mlp_dim = dec_mlp_dim if dec_mlp_dim else hidden_dim
        for i in range(dec_mlp_layers): 
            self.fc_dec_mlp.append(nn.Sequential(
                nn.Linear(hidden_dim if i == 0 else mlp_dim , mlp_dim), 
                Layer.Act.get_activation_fn(act_type), 
                nn.Dropout(dropout)))
        if hidden_as_factors:
            self.fc_hid_out = nn.Sequential(nn.Linear(mlp_dim , 1 if map_to_one else hidden_dim) , nn.BatchNorm1d(1 if map_to_one else hidden_dim)) 
        else:
            self.fc_hid_out = nn.Linear(mlp_dim , 1 if map_to_one else hidden_dim)

    def forward(self, x : Tensor) -> Tensor:
        '''
        in: [bs x hidden_dim]
        out:[bs x out_dim/hidden_dim]
        '''
        x = self.fc_dec_mlp(x)
        return self.fc_hid_out(x)
    
class uni_rnn_mapping(nn.Module):
    def __init__(self,hidden_dim,hidden_as_factors,output_as_factors,**kwargs):
        super().__init__()
        self.fc_map_out = nn.Sequential(Layer.EwLinear()) if hidden_as_factors else nn.Sequential(nn.Linear(hidden_dim, 1))
        if output_as_factors: self.fc_map_out.append(nn.BatchNorm1d(1))
    def forward(self , x : Tensor) -> Tensor:
        '''
        in: [bs x hidden_dim]
        out:[bs x 1]
        '''
        return self.fc_map_out(x)

class multi_rnn_encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,**kwargs):
        super().__init__()
        self.enc_list = nn.ModuleList([uni_rnn_encoder(input_dim=indim,hidden_dim=hidden_dim,**kwargs) for indim in input_dim])

    def forward(self, x : Tensor) -> list | tuple:
        '''
        in: tuple of [bs x seq_len x input_dim[i]] , seq can be different 
        out:tuple of [bs x hidden_dim]
        '''
        return [mod(inp) for inp , mod in zip(x , self.enc_list)]
    
class multi_rnn_decoder(nn.Module):
    def __init__(self,hidden_dim,dropout,num_rnn,rnn_att,num_heads,hidden_as_factors,**kwargs):
        super().__init__()
        num_rnn = num_rnn
        self.dec_list = nn.ModuleList([uni_rnn_decoder(hidden_dim , hidden_as_factors = False , **kwargs) for _ in range(num_rnn)])
        self.fc_mod_att = nn.Sequential()
        if rnn_att: 
            self.fc_mod_att = ModuleWiseAttention(hidden_dim,num_rnn , num_heads=num_heads , dropout=dropout)
        else:
            self.fc_mod_att = Layer.Pass()
        if hidden_as_factors:
            self.fc_hid_out = nn.Sequential(nn.Linear(num_rnn*hidden_dim , hidden_dim) , nn.BatchNorm1d(hidden_dim))
        else:
            self.fc_hid_out = nn.Linear(num_rnn*hidden_dim , hidden_dim)

    def forward(self, x : list[Tensor] | tuple[Tensor]) -> Tensor:
        '''
        in: tuple of [bs x hidden_dim] , len is num_rnn
        out:[bs x hidden_dim]
        '''
        x = [mod(inp) for inp , mod in zip(x , self.dec_list)]
        o = torch.cat(self.fc_mod_att(x) , dim = -1)
        return self.fc_hid_out(o)
    
class multi_rnn_mapping(nn.Module):
    def __init__(self,hidden_dim,hidden_as_factors,output_as_factors,**kwargs):
        super().__init__()
        if hidden_as_factors: 
            self.fc_map_out = nn.Sequential(Layer.EwLinear())
        else:
            self.fc_map_out = nn.Sequential(nn.Linear(hidden_dim, 1))
        if output_as_factors:  self.fc_map_out.append(nn.BatchNorm1d(1))
    def forward(self, x : Tensor) -> Tensor:
        '''
        in: [bs x hidden_dim]
        out:[bs x 1]
        '''
        return self.fc_map_out(x)
    
class gru(rnn_univariate):
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'gru'})
        super().__init__(input_dim , hidden_dim , **kwargs)
        
class lstm(rnn_univariate):
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

class gru_dsize(gru):
    def __init__(self , input_dim , hidden_dim , num_output = 1 , **kwargs):
        kwargs.update({'rnn_type' : 'gru'})
        super().__init__(input_dim , hidden_dim , num_output = num_output , **kwargs)
        self.residual = Layer.Lin.HardLinearRegression()
        self.residual_bn = nn.BatchNorm1d(num_output)
    def forward(self, x: Tensor , size : Optional[Tensor]) -> tuple[Tensor, dict]:
        x , o = super().forward(x)
        if self.training: 
            x = self.residual(x , size)
            x = self.residual_bn(x)
        return x , o

