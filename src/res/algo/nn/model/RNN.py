"""RNN-based model architectures for sequential stock feature processing.

Univariate models (single input stream): simple_lstm, gru, lstm, transformer,
    tcn, rnn_ntask, gru_dsize
Multivariate models (multiple input streams): rnn_general
Named entry points all inherit from rnn_univariate or rnn_multivariate.
"""
import torch

from torch import nn , Tensor

from .. import layer as Layer
from .Attention import mod_transformer,TimeWiseAttention,ModuleWiseAttention
from .CNN import mod_resnet_1d , mod_tcn
from ..loss import MultiHeadLosses

def get_rnn_mod(rnn_type):
    """Return the RNN sub-module constructor for a given type string.

    Supported keys: ``'transformer'``, ``'lstm'``, ``'gru'``, ``'tcn'``.
    """
    return {'transformer':mod_transformer,'lstm':mod_lstm,'gru':mod_gru,'tcn':mod_tcn,}[rnn_type]

class simple_lstm(nn.Module):
    """Minimal 1-layer LSTM for quick experiments.

    Single-layer LSTM that returns the last hidden state mapped to a scalar.

    Args:
        input_dim:  Input feature dimension.
        hidden_dim: LSTM hidden size.

    Returns:
        ``(pred, {'hidden': o})``
        * ``pred``: ``[bs, 1]``
        * ``o``:    ``[bs, hidden_dim]`` — last step hidden state
    """
    def __init__(self , input_dim , hidden_dim , **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim , hidden_dim , num_layers = 1 , dropout = 0 , batch_first = True)
        self.fc = nn.Linear(hidden_dim , 1)
    def forward(self, x : Tensor) -> tuple[Tensor , dict]:
        o = self.lstm(x)[0][:,-1]
        return self.fc(o) , {'hidden' : o}
    
class mod_lstm(nn.Module):
    """Multi-layer LSTM sequence-to-sequence sub-module (no output head).

    Returns the full output sequence for all time steps.  Used as an encoder
    component inside ``uni_rnn_encoder`` / ``rnn_univariate``.

    Args:
        input_dim:  Input feature dimension.
        output_dim: LSTM hidden/output dimension.
        dropout:    Dropout between LSTM layers.
        num_layers: Number of LSTM layers (capped at 3).

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``[bs, seq_len, output_dim]``
    """
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_layers = min(3,num_layers)
        self.lstm = nn.LSTM(input_dim , output_dim , num_layers = num_layers , dropout = dropout , batch_first = True)
    def forward(self, x : Tensor) -> Tensor:
        return self.lstm(x)[0]

class mod_gru(nn.Module):
    """Multi-layer GRU sequence-to-sequence sub-module (no output head).

    Returns the full output sequence.  Used as an encoder component.

    Args:
        input_dim:  Input feature dimension.
        output_dim: GRU hidden/output dimension.
        dropout:    Dropout between GRU layers.
        num_layers: Number of GRU layers (capped at 3).

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``[bs, seq_len, output_dim]``
    """
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_layers = min(3,num_layers)
        self.gru = nn.GRU(input_dim , output_dim , num_layers = num_layers , dropout = dropout , batch_first = True)
    def forward(self, x : Tensor) -> Tensor:
        return self.gru(x)[0]

class rnn_univariate(nn.Module):
    """Encoder-decoder-mapping RNN pipeline for a single input stream.

    Three-stage pipeline:
    1. **Encoder** (``uni_rnn_encoder``) — optional input projection /
       ResNet encoding, followed by the RNN, optionally with time-wise
       attention pooling.
    2. **Decoder** (``uni_rnn_decoder``) — per-head MLP with optional
       BatchNorm for factor normalization.
    3. **Mapping** (``uni_rnn_mapping``) — per-head scalar projection with
       optional BatchNorm.

    When ``num_output > 1``, ``num_output`` independent decoder and mapping
    branches are created via ``Layer.Parallel``.  A learnable
    ``multiloss_alpha`` parameter is injected for multi-task loss weighting.

    Args:
        input_dim:        Number of daily input features.
        hidden_dim:       RNN hidden dimension.
        dropout:          Dropout rate throughout.
        act_type:         Activation function key for decoder MLP.
        enc_in:           Input encoder type: ``None`` (identity),
                          ``'linear'`` / ``True`` (Linear), ``'resnet'``
                          (``mod_resnet_1d``).
        enc_in_dim:       Dimension of the input encoder output.
        enc_att:          If True, use ``TimeWiseAttention`` pooling instead
                          of taking the last time step.
        rnn_type:         RNN backbone: ``'gru'``, ``'lstm'``,
                          ``'transformer'``, ``'tcn'``.
        rnn_layers:       Number of RNN layers.
        dec_mlp_layers:   Number of MLP layers in the decoder.
        dec_mlp_dim:      Hidden dimension of the decoder MLP.
        num_output:       Number of output heads (default ``1``).
        output_as_factors: If True, apply ``BatchNorm1d(1)`` to normalize
                          each head's scalar output.
        hidden_as_factors: If True, project hidden state to factors instead
                          of direct scalar regression.

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``([bs, num_output], {'hidden': [bs, hidden_dim]})``
    """
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

        MultiHeadLosses.add_params(self , num_output)

    def forward(self , x : Tensor) -> tuple[Tensor,dict]:
        '''
        in: [bs x seq_len x input_dim]
        out:[bs x 1] , [bs x hidden_dim]
        '''
        #print(f'input shape: {x.shape}')
        x = self.encoder(x) # [bs x hidden_dim]
        #print(f'encoder output shape: {x.shape}')
        x = self.decoder(x) # tuple of [bs x hidden_dim] , len is num_output
        #print(f'decoder output shape: {len(x)} {x[0].shape}')
        o = self.mapping(x) # [bs x num_output]
        #print(f'mapping output shape: {o.shape}')
        return o , {'hidden' : x[0]}
        
        
class rnn_multivariate(nn.Module):
    """Encoder-decoder-mapping RNN pipeline for multiple input streams.

    Extends ``rnn_univariate`` to handle a list of input sequences (e.g.
    daily features + intra-day features).  Each stream is encoded by its own
    ``uni_rnn_encoder``; the stream representations can optionally interact
    via ``ModuleWiseAttention`` (``rnn_att=True``) before decoding.

    Args:
        input_dim: List of feature dimensions, one per input stream.  If a
                   scalar, falls back to single-stream (``rnn_univariate``)
                   behavior.
        rnn_att:   If True, apply cross-stream multi-head attention in the
                   decoder to allow streams to attend to each other.
        num_heads: Number of attention heads for the module-wise attention.
        (other args same as ``rnn_univariate``)

    Shapes:
        Input:  Tuple of ``[bs, seq_len_i, input_dim[i]]`` for each stream.
        Output: ``([bs, num_output], {'hidden': [bs, hidden_dim]})``
    """
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

        MultiHeadLosses.add_params(self , num_output)
    
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
    """Input encoder for a single RNN stream.

    Optionally applies an input projection or ResNet encoding, runs the RNN,
    then optionally applies time-wise attention pooling to select the summary
    vector.

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``[bs, hidden_dim]``
    """
    def __init__(self,input_dim,hidden_dim,dropout,rnn_type,rnn_layers,enc_in=None,enc_in_dim=None,enc_att=False,**kwargs):
        super().__init__()
        self.mod_rnn = {'transformer':mod_transformer,'lstm':mod_lstm,'gru':mod_gru,'tcn':mod_tcn,}[rnn_type]
        if self.mod_rnn == mod_transformer: 
            enc_in , enc_in_dim , enc_att = None , input_dim , False
        else:
            enc_in , enc_in_dim , enc_att = enc_in , enc_in_dim if enc_in_dim else hidden_dim , enc_att
        
        if enc_in == 'linear' or enc_in is True:
            self.fc_enc_in = nn.Sequential(nn.Linear(input_dim, enc_in_dim),nn.Tanh())
        elif enc_in == 'resnet':
            res_kwargs = {k:v for k,v in kwargs.items() if k != 'seq_len'}
            self.fc_enc_in = mod_resnet_1d(kwargs['inday_dim'] , input_dim , enc_in_dim , **res_kwargs) 
        else:
            enc_in_dim = input_dim
            self.fc_enc_in = nn.Sequential()

        rnn_kwargs = {'input_dim':enc_in_dim,'output_dim':hidden_dim,'num_layers':rnn_layers, 'dropout':dropout}
        if rnn_type == 'tcn': 
            rnn_kwargs['kernel_size'] = kwargs['kernel_size']
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
    """MLP decoder for a single output head.

    Applies ``dec_mlp_layers`` fully-connected + activation + dropout layers,
    then projects to a ``hidden_dim``-sized output (or scalar when
    ``map_to_one=True``).  Optional ``BatchNorm1d`` when
    ``hidden_as_factors=True``.

    Shapes:
        Input:  ``[bs, hidden_dim]``
        Output: ``[bs, hidden_dim]`` or ``[bs, 1]`` when ``map_to_one=True``
    """
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
    """Final scalar mapping head for a single output.

    Maps ``hidden_dim`` → scalar (1).  When ``hidden_as_factors``, uses
    temporal mean pooling instead of Linear.  When ``output_as_factors``,
    applies ``BatchNorm1d(1)`` for cross-sectional normalization.

    Shapes:
        Input:  ``[bs, hidden_dim]``
        Output: ``[bs, 1]``
    """
    def __init__(self,hidden_dim,hidden_as_factors,output_as_factors,**kwargs):
        super().__init__()
        self.fc_map_out = nn.Sequential(Layer.EwLinear()) if hidden_as_factors else nn.Sequential(nn.Linear(hidden_dim, 1))
        if output_as_factors: 
            self.fc_map_out.append(nn.BatchNorm1d(1))
    def forward(self , x : Tensor) -> Tensor:
        '''
        in: [bs x hidden_dim]
        out:[bs x 1]
        '''
        return self.fc_map_out(x)

class multi_rnn_encoder(nn.Module):
    """Parallel independent encoders for multiple input streams.

    Runs ``len(input_dim)`` independent ``uni_rnn_encoder`` instances, one
    per stream.

    Shapes:
        Input:  Tuple of ``[bs, seq_len_i, input_dim[i]]``
        Output: List of ``[bs, hidden_dim]``, one per stream
    """
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
    """Cross-stream decoder that aggregates multiple RNN stream representations.

    Each stream is decoded by an independent ``uni_rnn_decoder``, then all
    decoded representations are optionally attended via
    ``ModuleWiseAttention`` before concatenation and a final projection.

    Shapes:
        Input:  List of ``[bs, hidden_dim]``, one per stream
        Output: ``[bs, hidden_dim]``
    """
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
    """Final scalar mapping head for a multivariate output.

    Same as ``uni_rnn_mapping`` but used in the multivariate pipeline.
    Maps ``hidden_dim`` → scalar (1).

    Shapes:
        Input:  ``[bs, hidden_dim]``
        Output: ``[bs, 1]``
    """
    def __init__(self,hidden_dim,hidden_as_factors,output_as_factors,**kwargs):
        super().__init__()
        if hidden_as_factors: 
            self.fc_map_out = nn.Sequential(Layer.EwLinear())
        else:
            self.fc_map_out = nn.Sequential(nn.Linear(hidden_dim, 1))
        if output_as_factors:  
            self.fc_map_out.append(nn.BatchNorm1d(1))
    def forward(self, x : Tensor) -> Tensor:
        '''
        in: [bs x hidden_dim]
        out:[bs x 1]
        '''
        return self.fc_map_out(x)
    
class gru(rnn_univariate):
    """GRU-based univariate model.  Registry key: ``'gru'``."""
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'gru'})
        super().__init__(input_dim , hidden_dim , **kwargs )

class lstm(rnn_univariate):
    """LSTM-based univariate model (single output head).  Registry key: ``'lstm'``."""
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'lstm' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)

class resnet_lstm(lstm):
    """LSTM with intra-day ResNet-1D encoder.  Registry key: ``'resnet_lstm'``.

    Requires ``inday_dim`` (intra-day bar count) in kwargs.
    """
    def __init__(self, input_dim , hidden_dim , inday_dim , **kwargs) -> None:
        kwargs.update({
            'enc_in' : 'resnet' ,
            'enc_in_dim' : kwargs.get('enc_in_dim') if kwargs.get('enc_in_dim') else hidden_dim // 4 ,
        })
        super().__init__(input_dim , hidden_dim , inday_dim = inday_dim , **kwargs)

class resnet_gru(gru):
    """GRU with intra-day ResNet-1D encoder.  Registry key: ``'resnet_gru'``.

    Requires ``inday_dim`` (intra-day bar count) in kwargs.
    """
    def __init__(self, input_dim , hidden_dim , inday_dim , **kwargs):
        kwargs.update({
            'enc_in' : 'resnet' ,
            'enc_in_dim' : kwargs.get('enc_in_dim') if kwargs.get('enc_in_dim') else hidden_dim // 4 ,
        })
        super().__init__(input_dim , hidden_dim , inday_dim = inday_dim , **kwargs)

class transformer(rnn_univariate):
    """Transformer-encoder-based univariate model.  Registry key: ``'transformer'``."""
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'transformer' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)

class tcn(rnn_univariate):
    """TCN-based univariate model.  Registry key: ``'tcn'``."""
    def __init__(self , input_dim , hidden_dim , **kwargs):
        kwargs.update({'rnn_type' : 'tcn' , 'num_output' : 1})
        super().__init__(input_dim , hidden_dim , **kwargs)

class rnn_ntask(rnn_univariate):
    """Multi-task GRU with configurable output heads.  Registry key: ``'rnn_ntask'``."""
    def __init__(self , input_dim , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(input_dim , hidden_dim , num_output = num_output , **kwargs)

class rnn_general(rnn_multivariate):
    """General multi-stream RNN model.  Registry key: ``'rnn_general'``."""
    def __init__(self , input_dim , **kwargs):
        super().__init__(input_dim , **kwargs)

class gru_dsize(gru):
    """GRU with size-factor neutralization during training.  Registry key: ``'gru_dsize'``.

    After the base GRU forward pass, during training only, regresses out the
    size factor from the output via ``HardLinearRegression`` and applies
    ``BatchNorm1d`` to the residuals.  At eval time the raw GRU output is
    returned unchanged.

    Additional forward argument:
        size: ``[bs, num_output]`` size factor values to residualize against.
    """
    def __init__(self , input_dim , hidden_dim , num_output = 1 , **kwargs):
        kwargs.update({'rnn_type' : 'gru'})
        super().__init__(input_dim , hidden_dim , num_output = num_output , **kwargs)
        self.residual = Layer.Lin.HardLinearRegression()
        self.residual_bn = nn.BatchNorm1d(num_output)
    def forward(self, x: Tensor , size : Tensor | None) -> tuple[Tensor, dict]:
        x , o = super().forward(x)
        if self.training:
            x = self.residual(x , size)
            x = self.residual_bn(x)
        return x , o

