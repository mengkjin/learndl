import torch
import torch.nn as nn
#from ..function.basic import *
from .basic import *
    
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
    def __init__(self , input_dim , mod_num , att_dim = None , num_heads = None , dropout=0.0 , seperate_output = True):
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
        enc_layer  = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=ffn_dim , dropout=dropout , batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer , encoder_layers)
    def forward(self, inputs , pad_mask = None):
        if inputs.isnan().any():
            pad_mask = self.pad_mask_nan(inputs) if pad_mask is None else (self.pad_mask_nan(inputs) + pad_mask) > 0
            inputs = inputs.nan_to_num()
        hidden = inputs.unsqueeze(0) if inputs.dim() == 2 else self.fc_att(inputs).unsqueeze(0)
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
        hidden = self.pos_enc(inputs)
        return self.trans(hidden)
