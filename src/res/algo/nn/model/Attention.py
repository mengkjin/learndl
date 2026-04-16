"""Attention-based building blocks: transformer encoder, time-wise attention,
module-wise attention, and positional encoding modules.
"""
import torch
from torch import nn , Tensor

class mod_transformer(nn.Module):
    """Standard Transformer encoder sub-module.

    Pipeline: ``Linear(input→output) → Tanh → SinusoidalPE → TransformerEncoder``

    Args:
        input_dim:  Input feature dimension.
        output_dim: Output (model) dimension.  Must be divisible by 8
                    (``num_heads=8`` is hard-coded).
        dropout:    Dropout rate for the encoder layers.
        num_layers: Number of ``TransformerEncoderLayer`` layers (min 2).

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``[bs, seq_len, output_dim]``
    """
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_heads , ffn_dim = 8 , 4 * output_dim
        assert output_dim % num_heads == 0 , (output_dim , num_heads)
        num_layers = max(2,num_layers)
        self.fc_in = nn.Sequential(nn.Linear(input_dim, output_dim),nn.Tanh())
        self.pos_enc = PositionalEncoding(output_dim,dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(output_dim , num_heads, dim_feedforward=ffn_dim , dropout=dropout , batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer , num_layers)
    def forward(self, x : Tensor) -> Tensor:
        '''
        in: [bs x seq_len x input_dim]
        out:[bs x seq_len x output_dim]
        '''
        x = self.fc_in(x)
        x = self.pos_enc(x)
        return self.trans(x)
  
class TimeWiseAttention(nn.Module):
    """Soft attention pooling over the time dimension.

    Computes a weighted sum of all time steps (attention summary), then
    concatenates it with the last time step hidden state and projects the
    result to ``output_dim``.

    Args:
        input_dim:  Input feature dimension.
        output_dim: Output dimension (defaults to ``input_dim``).
        att_dim:    Internal attention dimension (defaults to ``output_dim``).
        dropout:    Dropout applied before the attention score computation.

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``[bs, output_dim]``
    """
    def __init__(self , input_dim, output_dim=None, att_dim = None, dropout = 0.0):
        super().__init__()
        if output_dim is None: 
            output_dim = input_dim
        if att_dim is None: 
            att_dim = output_dim
        self.fc_in = nn.Linear(input_dim, att_dim)
        self.att_net = nn.Sequential(nn.Dropout(dropout),nn.Tanh(),nn.Linear(att_dim,1,bias=False),nn.Softmax(dim=0))
        self.fc_out = nn.Linear(2*att_dim,output_dim)

    def forward(self, x : Tensor) -> Tensor:
        '''
        in: [bs x seq_len x input_dim]
        out:[bs x seq_len x output_dim]
        '''
        x = self.fc_in(x)
        scores = self.att_net(x)  # [batch, seq_len, 1]
        o = torch.mul(x , scores).sum(dim=1)
        o = torch.cat((x[:, -1], o), dim=1)
        return self.fc_out(o)
    
class ModuleWiseAttention(nn.Module):
    """Cross-module multi-head self-attention for aggregating multiple RNN streams.

    Projects each of ``mod_num`` input streams to a shared ``att_dim``,
    stacks them as a sequence, and applies multi-head self-attention across
    the module dimension.  A residual connection preserves the original
    projections.  Used in ``multi_rnn_decoder`` to allow streams to
    communicate information to each other.

    Args:
        input_dim: Feature dimension of each input stream, or a list of
                   per-stream dimensions (length must equal ``mod_num``).
        mod_num:   Number of parallel streams.
        att_dim:   Attention model dimension (defaults to ``max(input_dim)``).
        num_heads: Number of attention heads (defaults to ``att_dim // 8``).
        dropout:   Attention dropout.

    Shapes:
        Input:  list/tuple of ``mod_num`` tensors each ``[bs, att_dim]``
        Output: tuple of ``mod_num`` tensors each ``[bs, att_dim]``
    """
    def __init__(self , input_dim , mod_num , att_dim = None , num_heads = None , dropout=0.0):
        super().__init__()
        if isinstance(input_dim , (list,tuple)):
            assert mod_num == len(input_dim) , (mod_num , len(input_dim))
        else:
            input_dim = [input_dim for _ in range(mod_num)]
        
        att_dim = max(input_dim) if att_dim is None else att_dim
        num_heads = att_dim // 8 if num_heads is None else num_heads
        
        self.in_fc = nn.ModuleList([nn.Linear(inp_d , att_dim) for inp_d in input_dim])
        self.task_mha = nn.MultiheadAttention(att_dim, num_heads = num_heads, batch_first=True , dropout = dropout)

    def forward(self, x : list | tuple) -> list | tuple:
        hidden = torch.stack([f(xx) for xx,f in zip(x , self.in_fc)],dim=-2)
        hidden = self.task_mha(hidden , hidden , hidden)[0] + hidden
        return tuple([hidden.select(-2,i) for i in range(hidden.shape[-2])])
        
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding stored as a non-trainable buffer.

    Adds standard ``sin/cos`` positional embeddings to the input tensor.
    The encoding is computed once and stored; it is not learned.

    Args:
        input_dim: Model embedding dimension.
        dropout:   Dropout applied after adding the positional encoding.
        max_len:   Maximum sequence length supported (default ``1000``).

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``  (``seq_len <= max_len``)
        Output: ``[bs, seq_len, input_dim]``
    """
    def __init__(self, input_dim, dropout=0.0, max_len=1000,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.seq_len = max_len
        self.P = torch.zeros(1 , self.seq_len, input_dim)
        X = torch.arange(self.seq_len, dtype=torch.float).reshape(-1,1) / torch.pow(10000,torch.arange(0, input_dim, 2 ,dtype=torch.float) / input_dim)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X[:,:input_dim//2])
    def forward(self, x : Tensor) -> Tensor:
        return self.dropout(x + self.P[:,:x.shape[1],:].to(x.device))

class SampleWiseTranformer(nn.Module):
    """Cross-sample transformer (operates over the batch/sample dimension).

    Used in TRA to share information across different stock samples within a
    mini-batch.  Handles NaN inputs by building a padding mask from non-finite
    values.  When the input is 3-D, collapses the time dimension first with
    ``TimeWiseAttention`` before applying the transformer.

    Args:
        hidden_dim:     Hidden and model dimension (must be divisible by
                        ``num_heads``).
        ffn_dim:        Feedforward network dimension (defaults to
                        ``4 * hidden_dim``).
        num_heads:      Number of attention heads (default ``8``).
        encoder_layers: Number of transformer encoder layers (default ``2``).
        dropout:        Dropout rate.

    Shapes:
        Input:  ``[n_stocks, seq_len, hidden_dim]`` or
                ``[n_stocks, hidden_dim]``
        Output: ``[n_stocks, hidden_dim]``
    """
    def __init__(self , hidden_dim , ffn_dim = None , num_heads = 8 , encoder_layers = 2 , dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0 , (hidden_dim , num_heads)
        ffn_dim = 4 * hidden_dim if ffn_dim is None else ffn_dim
        self.fc_att = TimeWiseAttention(hidden_dim,hidden_dim)
        enc_layer  = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=ffn_dim , dropout=dropout , batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer , encoder_layers)
    def forward(self, x : Tensor , pad_mask = None) -> Tensor:
        if x.isnan().any():
            pad_mask = self.pad_mask_nan(x) if pad_mask is None else (self.pad_mask_nan(x) + pad_mask) > 0
            x = x.nan_to_num()
        if x.dim() != 2: 
            x = self.fc_att(x)
        x = self.trans(x.unsqueeze(0) , src_key_padding_mask = pad_mask)
        return x.squeeze(0)
    def pad_mask_rand(self , x : Tensor , mask_ratio = 0.1) -> Tensor:
        return (torch.rand(1,x.shape[0]) < mask_ratio).to(x.device)
    def pad_mask_nan(self , x : Tensor) -> Tensor:
        return x.sum(dim = list(range(x.ndim)[1:])).isnan().unsqueeze(0)    

class TimeWiseTranformer(nn.Module):
    """Standard time-axis transformer encoder with sinusoidal positional encoding.

    Args:
        input_dim:      Input feature dimension (not used after projection;
                        assumed equal to ``hidden_dim`` — projection is not
                        included in this module).
        hidden_dim:     Model dimension (must be divisible by ``num_heads``).
        ffn_dim:        Feedforward network dimension (defaults to
                        ``4 * hidden_dim``).
        num_heads:      Number of attention heads (default ``8``).
        encoder_layers: Number of transformer encoder layers (default ``2``).
        dropout:        Dropout rate.

    Shapes:
        Input:  ``[bs, seq_len, hidden_dim]``
        Output: ``[bs, seq_len, hidden_dim]``
    """
    def __init__(self , input_dim , hidden_dim , ffn_dim = None , num_heads = 8 , encoder_layers = 2 , dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0 , (hidden_dim , num_heads)
        ffn_dim = 4 * hidden_dim if ffn_dim is None else ffn_dim
        self.pos_enc = PositionalEncoding(hidden_dim,dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim , num_heads, dim_feedforward=ffn_dim , dropout=dropout , batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer , encoder_layers)
    def forward(self, x : Tensor) -> Tensor:
        return self.trans(self.pos_enc(x))
