import torch
import torch.nn as nn

class MyATF(nn.Module):
    def __init__(
        self , 
        input_dim: int = 6,
        hidden_dim: int = 2 ** 5,
        num_blocks: int = 3,
        num_heads: int = 4,
        d_ff: int = None,
        d_selfatt: int = None,
        dropout: float = 0.,
        input_shape = None,
        **kwargs,
    ):
        super().__init__()
        if d_ff is None: d_ff = hidden_dim * 4
        if d_selfatt is None: d_selfatt = hidden_dim * 2
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.input_shape = input_shape
        self.MyATF_masks(**kwargs)
        
        # Define the input layer
        self.input_layer = nn.Sequential(PositionalEncoding(input_dim),nn.Linear(input_dim, hidden_dim) , nn.Tanh())
        # Define the multihead attention layer
        self.mha_layer = nn.ModuleList([nn.MultiheadAttention(hidden_dim,num_heads,batch_first=True) for _ in range(num_blocks)])
        
        # Define the position-wise feed forward network
        self.ffn_layer = nn.Sequential(nn.Linear(hidden_dim, d_ff), nn.LeakyReLU(),nn.Dropout(dropout),nn.Linear(d_ff,self.hidden_dim))
        # Define the temporal self attention mechanism
        self.self_att = nn.Sequential(nn.Linear(hidden_dim, d_selfatt), nn.Tanh(), nn.Linear(d_selfatt, 1 , bias=False), nn.Softmax(dim=0))
        self.last_layer = nn.Linear(self.hidden_dim , 1)

    def forward(self, inputs):     
        # 4-D mask : num_blocks , num_head(1 if not gaussian_prior) , seq_len , seq_len
        batch , seq_len = inputs.shape[0] , inputs.shape[1]
        output = self.input_layer(inputs)    
        for i, mod in enumerate(self.mha_layer):
            output , _ = mod(output , output , output , attn_mask=self.mask[i,:self.num_heads*batch,:seq_len,:seq_len].squeeze(0))
        output = self.ffn_layer(output)
        hidden = self.self_att(output)
        hidden = torch.sum(output * hidden, dim=1 , keepdim=True)
        output = self.last_layer(hidden)
        return output.squeeze(1), hidden.squeeze(1)
    
    def MyATF_masks(
        self,
        gaussian_prior_param: int = [5,10,20,40],
        trading_gap_splitter_window: int = [16,80,1000],
        max_len: int = 60 * 16,
        **kwargs,
    ):
        # Define the masks
        # if shape is not giving, use max_batch and max_len instead, 
        # 4-D mask : num_blocks , num_head * batch_size(1 if not gaussian_prior) , seq_len , seq_len
        # first check if ATF_masking_param exists, if so use its elements to replace default ones
        mask_dict = {'causal':True,'gaussian':False,'tradegap':False} 
        if kwargs.get('ATF_mask') is not None: mask_dict.update(kwargs.get('ATF_mask'))
        
        batch_size = 10000
        SEQ_LEN = 30
        seq_len = max(max_len , max(SEQ_LEN)) if kwargs.get('sequence_len') is None else kwargs.get('sequence_len')
        causal_mask , gaussian_mask , tradegap_mask = None , None , None
        if self.input_shape is not None:
            batch_size , seq_len = self.input_shape[0] , self.input_shape[1]

        self.mask = torch.zeros(self.num_blocks,1,seq_len,seq_len) 
        m1 , m2 , m3 = None , None , None
        if sum(mask_dict.values()) > 0:
            # 4-D mask : num_blocks , num_head(1 if not gaussian_prior) , seq_len , seq_len
            if mask_dict.get('causal'):
                m1 = torch.triu(torch.ones(seq_len,seq_len).fill_(-1e6) , diagonal=1)
                self.mask = self.mask + m1

            if mask_dict.get('gaussian'):
                # expand prior_param if num_heads too much
                gpp = (gaussian_prior_param + torch.zeros(self.num_heads,dtype=int).tolist())[:self.num_heads]
                _a = torch.arange(1,seq_len+1).repeat(seq_len).reshape(seq_len,-1)
                _b = torch.tril(torch.ones(seq_len,seq_len))
                m2 = torch.stack([torch.exp(-torch.pow(_a - _a.t(),2)/2/d**2).nan_to_num() * _b for d in gpp])
                self.mask = self.mask + m2

            if mask_dict.get('tradegap'):
                # expand trading_gap_splitter_window if num_blocks too much
                tgsw = (trading_gap_splitter_window + 1000*torch.ones(self.num_blocks,dtype=int).tolist())[:self.num_blocks]
                m3 = torch.zeros(self.num_blocks,1,seq_len,seq_len).fill_(-1e6)
                for i_block , sub_len in enumerate(tgsw):
                    for sub_start in range(0,seq_len,sub_len): 
                        m3[i_block , 0 , sub_start:(sub_start+sub_len) , sub_start:(sub_start+sub_len)] = 0
                self.mask = self.mask + m3
            if self.mask.shape[1] != 1: self.mask = self.mask.repeat(1,batch_size,1,1)        
        # self.submask = (m1 , m2 , m3)
        
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