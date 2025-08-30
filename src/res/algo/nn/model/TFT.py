import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Any
import math

from src.basic import CONF
class GatedResidualNetwork(nn.Module):
    """门控残差网络"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 残差连接的投影层
        if input_dim != output_dim:
            self.skip_connection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_connection = None
    
    def forward(self, x):
        # 主路径
        h = F.elu(self.linear1(x))
        h = self.dropout(h)
        output = self.linear2(h)
        
        # 门控机制
        gate = torch.sigmoid(self.gate(h))
        output = output * gate
        
        # 残差连接
        if self.skip_connection is not None:
            x = self.skip_connection(x)
        
        return self.layer_norm(output + x)

class VariableSelectionNetwork(nn.Module):
    """变量选择网络"""
    def __init__(self, input_dim: int, num_vars: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        
        # 变量选择权重
        self.variable_weights = GatedResidualNetwork(
            input_dim, hidden_dim, num_vars, dropout
        )
        
        # 每个变量的处理网络
        self.variable_networks = nn.ModuleList([
            GatedResidualNetwork(input_dim // num_vars, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 计算变量选择权重
        weights = F.softmax(self.variable_weights(x), dim=-1)  # (batch_size, seq_len, num_vars)
        
        # 分割输入变量
        var_dim = self.input_dim // self.num_vars
        variables = x.view(batch_size, seq_len, self.num_vars, var_dim)
        
        # 处理每个变量
        processed_vars = []
        for i, net in enumerate(self.variable_networks):
            var_input = variables[:, :, i, :]  # (batch_size, seq_len, var_dim)
            processed_var = net(var_input)  # (batch_size, seq_len, hidden_dim)
            processed_vars.append(processed_var)
        
        processed_vars = torch.stack(processed_vars, dim=2)  # (batch_size, seq_len, num_vars, hidden_dim)
        
        # 加权组合
        weights = weights.unsqueeze(-1)  # (batch_size, seq_len, num_vars, 1)
        output = torch.sum(processed_vars * weights, dim=2)  # (batch_size, seq_len, hidden_dim)
        
        return output, weights.squeeze(-1)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape
        
        # 线性变换 - 修复：分别处理不同长度的序列
        Q = self.w_q(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        
        output = self.w_o(context)
        
        # 残差连接和层归一化
        return self.layer_norm(output + query)

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer模型"""
    _default_data_type = 'day+style+indus'
    
    def __init__(
        self,
        input_dim : tuple[int,int] | tuple[int,int,int] = \
            (6 , len(CONF.RISK['style']),len(CONF.RISK['indus'])), # aka , known dynamic dim , static dim
        hidden_dim: int = 16,
        label_dim: int = 0, # aka , unknown dynamic dim, should be 0 if not used
        num_heads: int = 4,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
        quantiles: list[float] = [0.1, 0.5, 0.9] ,
        indus_dim    = 2**3 ,
        indus_embed  = True , 
        lstm_layers  = 1,
        max_len      = 120,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        if len(input_dim) == 2:
            self.known_dynamic_dim , self.static_dim = input_dim
        elif len(input_dim) == 3:
            _d0 , _d1 , self.static_dim = input_dim
            self.known_dynamic_dim = _d0 + _d1
        else:
            raise ValueError(f'input_dim should be a tuple of length 2 or 3, but got {len(input_dim)}')
        
        self.unknown_dynamic_dim = label_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.pred_len = pred_len
        self.quantiles = quantiles

        self.indus_dim = indus_dim
        self.indus_embed = indus_embed

        assert self.static_dim == len(CONF.RISK['indus']) , (input_dim , len(CONF.RISK['indus']))
        
        if indus_embed:
            self.indus_embedding = nn.Linear(self.static_dim , indus_dim)
            static_dim = indus_dim
        else:
            self.indus_embedding = nn.Sequential()
            static_dim = self.static_dim
        
        # 静态变量编码器 - 修复维度问题
        self.static_encoder = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout  # 输入是hidden_dim而不是static_dim
        )
        
        # 动态变量编码器
        total_dynamic_dim = self.known_dynamic_dim + self.unknown_dynamic_dim
        self.dynamic_encoder = GatedResidualNetwork(
            total_dynamic_dim, hidden_dim, hidden_dim, dropout
        )
        
        # 变量选择网络
        self.static_vsn = VariableSelectionNetwork(
            static_dim, static_dim, hidden_dim, dropout
        )
        
        self.historical_vsn = VariableSelectionNetwork(
            total_dynamic_dim, total_dynamic_dim, hidden_dim, dropout
        )
        
        self.future_vsn = VariableSelectionNetwork(
            self.known_dynamic_dim, self.known_dynamic_dim, hidden_dim, dropout
        )
        
        # LSTM编码器
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout if lstm_layers > 1 else 0, num_layers=lstm_layers
        )
        
        self.lstm_decoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout if lstm_layers > 1 else 0, num_layers=lstm_layers
        )
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 位置编码
        self.position_encoding_encoder = nn.Parameter(
            torch.randn(max_len, hidden_dim)
        )
        self.position_encoding_decoder = nn.Parameter(
            torch.randn(self.pred_len, hidden_dim)
        )
        
        # 输出层
        self.output_projection = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in quantiles
        ])
        
        self.dropout = nn.Dropout(dropout)
    

    def show_shape(self , name : str , x : Any , print_shape : bool = False):
        if not print_shape: return
        print(f"{name}: {x.shape}")

    def forward(
        self, x : tuple[torch.Tensor,...] | Any
    ):
        """
        x:
            historical_features: (batch_size, seq_len, known_dim + unknown_dim) 历史特征
            static_features: (batch_size, static_dim) 静态特征
            future_features: (batch_size, pred_len, known_dim) 未来知特征
        """
        assert len(x) in [len(self.input_dim) , len(self.input_dim) + 1] , f'x should be a tuple of length {len(self.input_dim)} or {len(self.input_dim) + 1}, but got {len(x)}'
        if len(self.input_dim) == 2:
            if len(x) == len(self.input_dim):
                historical_features , static_features = x
                future_features = None
            else:
                historical_features , static_features , future_features = x
        elif len(self.input_dim) == 3:
            if len(x) == len(self.input_dim):
                historical_features , additional_features , static_features = x
                future_features = None
            else:
                historical_features , additional_features , static_features , future_features = x
            historical_features = torch.cat([historical_features , additional_features], dim=-1)

        seq_len = historical_features.shape[1]
        
        # 1. 静态特征处理
        if static_features.ndim == historical_features.ndim and static_features.shape[1] > 1:
            static_features = static_features[:,-1]
        self.show_shape('static_features' , static_features)
        static_embed = self.indus_embedding(static_features)
        static_encoded, static_weights = self.static_vsn(
            static_embed.unsqueeze(1).repeat(1, seq_len, 1)
        )
        # static_encoded现在是(batch_size, seq_len, hidden_dim)
        static_context = self.static_encoder(static_encoded.mean(dim=1))  # (batch_size, hidden_dim)
        
        self.show_shape('static_embed' , static_embed)
        self.show_shape('static_encoded' , static_encoded)
        self.show_shape('static_weights' , static_weights)
        self.show_shape('static_context' , static_context)

        # 2. 历史特征处理
        historical_encoded, hist_weights = self.historical_vsn(historical_features)

        self.show_shape('historical_features' , historical_features)
        self.show_shape('historical_encoded' , historical_encoded)
        self.show_shape('hist_weights' , hist_weights)
        
        # 3. 未来特征处理
        if future_features is not None:
            future_encoded, future_weights = self.future_vsn(future_features)
            self.show_shape('future_features' , future_features)
            self.show_shape('future_encoded' , future_encoded)
            self.show_shape('future_weights' , future_weights)
        else:
            future_encoded = 0.
            future_weights = 0.

        # 4. LSTM编码
        lstm_input = historical_encoded + static_encoded
        lstm_input = lstm_input + self.position_encoding_encoder[:seq_len].unsqueeze(0)

        self.show_shape('lstm_input' , lstm_input)
        
        encoder_output, (h_n, c_n) = self.lstm_encoder(lstm_input)

        self.show_shape('encoder_output' , encoder_output)
        self.show_shape('h_n' , h_n)
        self.show_shape('c_n' , c_n)
        
        # 5. 解码器
        decoder_input = future_encoded + static_context.unsqueeze(1).repeat(1, self.pred_len, 1)
        self.show_shape('decoder_input' , decoder_input)

        decoder_input = decoder_input + self.position_encoding_decoder.unsqueeze(0)
        
        decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))

        self.show_shape('decoder_output' , decoder_output)
        
        # 6. 注意力机制
        attention_output = decoder_output
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(
                attention_output, encoder_output, encoder_output
            )

        self.show_shape('attention_output' , attention_output)
        
        # 7. 分位数预测
        quantile_outputs = []
        for projection in self.output_projection:
            quantile_pred = projection(attention_output)  # (batch_size, pred_len, 1)
            quantile_outputs.append(quantile_pred)
        
        predictions = torch.cat(quantile_outputs, dim=-1)  # (batch_size, pred_len, num_quantiles)
        
        self.show_shape('predictions' , predictions)

        y_hat = predictions[...,self.quantiles.index(0.5)][:,:1] # only use the first step as y_hat
        return y_hat , {
            'predictions': predictions,
            'static_weights': static_weights,
            'historical_weights': hist_weights,
            'future_weights': future_weights
        }
    
    @staticmethod
    def loss(label : torch.Tensor , pred : torch.Tensor | None = None , w : torch.Tensor | None = None , dim = None , 
             quantiles : list[float] = [0.1,0.5,0.9] , predictions : torch.Tensor | None = None , **kwargs):
        assert predictions is not None , f'predictions should be provided'
        assert predictions.shape[-1] == len(quantiles) , f'shape of predictions {predictions.shape} should be (...,{len(quantiles)})'
        if predictions.ndim == label.ndim + 1: predictions = predictions.squeeze(-2)
        assert predictions.ndim == label.ndim == 2 , f'shape of predictions {predictions.shape} and label {label.shape} should be (...,1)'
        if w is None:
            w1 = 1.
        else:
            w1 = w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
        
        losses = []
        label = label.expand_as(predictions)
        
        for i, q in enumerate(quantiles):
            pred_q = predictions[..., i:i+1]
            error = label - pred_q
            valid = ~error.isnan()
            loss = torch.max(q * error[valid], (q - 1) * error[valid])
            losses.append((w1 * loss).mean(dim=dim,keepdim=True))
        
        v = torch.stack(losses,dim=-1).mean(dim=-1)
        return v


if __name__ == '__main__' :
    from src.res.model.data_module import get_realistic_batch_data
    batch_data = get_realistic_batch_data('day+style+indus')

    rau = TemporalFusionTransformer(indus_embed=True)
    rau(batch_data.x).shape