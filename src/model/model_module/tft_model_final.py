import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math

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
    def __init__(
        self,
        static_dim: int,
        known_dynamic_dim: int,
        unknown_dynamic_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        seq_len: int = 20,
        pred_len: int = 5,
        dropout: float = 0.1,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.static_dim = static_dim
        self.known_dynamic_dim = known_dynamic_dim
        self.unknown_dynamic_dim = unknown_dynamic_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.quantiles = quantiles
        
        # 静态变量编码器 - 修复维度问题
        self.static_encoder = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout  # 输入是hidden_dim而不是static_dim
        )
        
        # 动态变量编码器
        total_dynamic_dim = known_dynamic_dim + unknown_dynamic_dim
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
            known_dynamic_dim, known_dynamic_dim, hidden_dim, dropout
        )
        
        # LSTM编码器
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout
        )
        
        self.lstm_decoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout
        )
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 位置编码
        self.position_encoding = nn.Parameter(
            torch.randn(seq_len + pred_len, hidden_dim)
        )
        
        # 输出层
        self.output_projection = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in quantiles
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        static_features: torch.Tensor,
        historical_features: torch.Tensor,
        future_features: torch.Tensor
    ):
        """
        Args:
            static_features: (batch_size, static_dim) 静态特征
            historical_features: (batch_size, seq_len, known_dim + unknown_dim) 历史特征
            future_features: (batch_size, pred_len, known_dim) 未来已知特征
        """
        batch_size = static_features.shape[0]
        
        # 1. 静态特征处理
        static_encoded, static_weights = self.static_vsn(
            static_features.unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        # static_encoded现在是(batch_size, seq_len, hidden_dim)
        static_context = self.static_encoder(static_encoded.mean(dim=1))  # (batch_size, hidden_dim)
        
        print(f"static_features: {static_features.shape}")
        print(f"static_encoded: {static_encoded.shape}")
        print(f"static_weights: {static_weights.shape}")
        print(f"static_context: {static_context.shape}")

        # 2. 历史特征处理
        historical_encoded, hist_weights = self.historical_vsn(historical_features)

        print(f"historical_features: {historical_features.shape}")
        print(f"historical_encoded: {historical_encoded.shape}")
        print(f"hist_weights: {hist_weights.shape}")
        
        # 3. 未来特征处理
        future_encoded, future_weights = self.future_vsn(future_features)

        print(f"future_features: {future_features.shape}")
        print(f"future_encoded: {future_encoded.shape}")
        print(f"future_weights: {future_weights.shape}")
        
        # 4. LSTM编码
        lstm_input = historical_encoded + static_encoded
        lstm_input = lstm_input + self.position_encoding[:self.seq_len].unsqueeze(0)

        print(f"lstm_input: {lstm_input.shape}")
        
        encoder_output, (h_n, c_n) = self.lstm_encoder(lstm_input)

        print(f"encoder_output: {encoder_output.shape}")
        print(f"h_n: {h_n.shape}")
        print(f"c_n: {c_n.shape}")
        
        # 5. 解码器
        decoder_input = future_encoded + static_context.unsqueeze(1).repeat(1, self.pred_len, 1)
        decoder_input = decoder_input + self.position_encoding[self.seq_len:].unsqueeze(0)
        
        decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))

        print(f"decoder_input: {decoder_input.shape}")
        print(f"decoder_output: {decoder_output.shape}")
        
        # 6. 注意力机制
        attention_output = decoder_output
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(
                attention_output, encoder_output, encoder_output
            )

        print(f"attention_output: {attention_output.shape}")
        
        # 7. 分位数预测
        quantile_outputs = []
        for projection in self.output_projection:
            quantile_pred = projection(attention_output)  # (batch_size, pred_len, 1)
            quantile_outputs.append(quantile_pred)
        
        predictions = torch.cat(quantile_outputs, dim=-1)  # (batch_size, pred_len, num_quantiles)
        
        print(f"predictions: {predictions.shape}")
        
        return {
            'predictions': predictions,
            'static_weights': static_weights,
            'historical_weights': hist_weights,
            'future_weights': future_weights
        }

class QuantileLoss(nn.Module):
    """分位数损失函数"""
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            predictions: (batch_size, pred_len, num_quantiles)
            targets: (batch_size, pred_len, 1)
        """
        losses = []
        targets = targets.expand_as(predictions)
        
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, :, i:i+1]
            error = targets - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)

# 数据生成和预处理函数
def generate_synthetic_data(
    num_stocks: int = 50,
    num_days: int = 200,
    seq_len: int = 20,
    pred_len: int = 5,
    num_industries: int = 10
):
    """生成合成股票数据"""
    np.random.seed(42)
    
    # 生成行业标签
    industries = np.random.randint(0, num_industries, num_stocks)
    industry_onehot = np.eye(num_industries)[industries]  # (num_stocks, num_industries)
    
    # 生成价格和成交量数据
    prices = np.random.randn(num_stocks, num_days) * 0.02 + 1.0
    prices = np.cumprod(1 + prices, axis=1) * 100  # 累积收益率转价格
    
    volumes = np.random.lognormal(10, 1, (num_stocks, num_days))
    
    # 计算日收益率
    returns = np.diff(prices, axis=1) / prices[:, :-1]  # (num_stocks, num_days-1)
    
    # 计算未来m日收益率 - 简化版本
    future_returns = []
    for i in range(num_days - pred_len):
        future_ret = (prices[:, i + pred_len] - prices[:, i]) / prices[:, i]
        future_returns.append(future_ret)
    future_returns = np.array(future_returns).T  # (num_stocks, num_days - pred_len)
    
    return {
        'prices': prices,
        'volumes': volumes,
        'returns': returns,
        'future_returns': future_returns,
        'industries': industry_onehot
    }

def create_sequences(data: Dict, seq_len: int = 20, pred_len: int = 5):
    """创建训练序列 - 修复版本"""
    prices = data['prices']
    volumes = data['volumes']
    returns = data['returns']
    future_returns = data['future_returns']
    industries = data['industries']
    
    num_stocks, num_days = prices.shape
    
    sequences = []
    
    for stock_idx in range(num_stocks):
        # 确保有足够的数据进行序列创建
        start_idx = seq_len
        end_idx = min(num_days - pred_len, future_returns.shape[1])
        
        for day_idx in range(start_idx, end_idx):
            # 静态特征：行业
            static_feat = industries[stock_idx]
            
            # 历史特征：价格、成交量、历史日收益率
            hist_prices = prices[stock_idx, day_idx - seq_len:day_idx]
            hist_volumes = volumes[stock_idx, day_idx - seq_len:day_idx]
            
            # 使用日收益率作为历史特征，而不是未来收益率
            if day_idx - seq_len > 0:
                hist_returns = returns[stock_idx, day_idx - seq_len:day_idx]
            else:
                # 如果没有足够的历史收益率，用零填充
                available_returns = returns[stock_idx, :day_idx]
                padding_length = seq_len - len(available_returns)
                hist_returns = np.concatenate([np.zeros(padding_length), available_returns])
            
            # 标准化
            hist_prices = (hist_prices - hist_prices.mean()) / (hist_prices.std() + 1e-8)
            hist_volumes = (hist_volumes - hist_volumes.mean()) / (hist_volumes.std() + 1e-8)
            hist_returns = (hist_returns - hist_returns.mean()) / (hist_returns.std() + 1e-8)
            
            historical_features = np.column_stack([hist_prices, hist_volumes, hist_returns])
            
            # 未来已知特征：未来价格、成交量
            future_prices = prices[stock_idx, day_idx:day_idx + pred_len]
            future_volumes = volumes[stock_idx, day_idx:day_idx + pred_len]
            
            # 标准化
            future_prices = (future_prices - future_prices.mean()) / (future_prices.std() + 1e-8)
            future_volumes = (future_volumes - future_volumes.mean()) / (future_volumes.std() + 1e-8)
            
            future_features = np.column_stack([future_prices, future_volumes])
            
            # 目标：未来收益率
            if day_idx < future_returns.shape[1]:
                target = future_returns[stock_idx, day_idx:day_idx + pred_len]
                
                # 确保target长度正确
                if len(target) == pred_len:
                    sequences.append({
                        'static': static_feat,
                        'historical': historical_features,
                        'future': future_features,
                        'target': target
                    })
    
    return sequences

# 训练函数
def train_model(
    model: TemporalFusionTransformer,
    train_data: List[Dict],
    val_data: List[Dict],
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """训练TFT模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = QuantileLoss(model.quantiles)
    
    # 数据加载器
    def create_dataloader(data, batch_size, shuffle=True):
        dataset = []
        for item in data:
            dataset.append((
                torch.FloatTensor(item['static']),
                torch.FloatTensor(item['historical']),
                torch.FloatTensor(item['future']),
                torch.FloatTensor(item['target']).unsqueeze(-1)
            ))
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        return dataloader
    
    train_loader = create_dataloader(train_data, batch_size, True)
    val_loader = create_dataloader(val_data, batch_size, False)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        
        for batch_idx, (static, historical, future, target) in enumerate(train_loader):
            static = static.to(device)
            historical = historical.to(device)
            future = future.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(static, historical, future)
            loss = criterion(outputs['predictions'], target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for static, historical, future, target in val_loader:
                static = static.to(device)
                historical = historical.to(device)
                future = future.to(device)
                target = target.to(device)
                
                outputs = model(static, historical, future)
                loss = criterion(outputs['predictions'], target)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
    
    return train_losses, val_losses

# 推理函数
def predict(
    model: TemporalFusionTransformer,
    static_features: np.ndarray,
    historical_features: np.ndarray,
    future_features: np.ndarray,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """模型推理"""
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        static = torch.FloatTensor(static_features).unsqueeze(0).to(device)
        historical = torch.FloatTensor(historical_features).unsqueeze(0).to(device)
        future = torch.FloatTensor(future_features).unsqueeze(0).to(device)
        
        outputs = model(static, historical, future)
        predictions = outputs['predictions'].cpu().numpy()
        
        return {
            'predictions': predictions[0],  # (pred_len, num_quantiles)
            'static_weights': outputs['static_weights'].cpu().numpy()[0],
            'historical_weights': outputs['historical_weights'].cpu().numpy()[0],
            'future_weights': outputs['future_weights'].cpu().numpy()[0]
        }

def evaluate_predictions(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """评估预测结果"""
    # 计算各分位数的准确率
    results = {}
    
    for i, q in enumerate(quantiles):
        pred_q = predictions[:, i]
        
        if q == 0.5:  # 中位数预测的MAE和RMSE
            mae = np.mean(np.abs(pred_q - targets))
            rmse = np.sqrt(np.mean((pred_q - targets) ** 2))
            results[f'MAE_q{q}'] = mae
            results[f'RMSE_q{q}'] = rmse
        
        # 分位数覆盖率
        if q < 0.5:
            coverage = np.mean(targets >= pred_q)
        else:
            coverage = np.mean(targets <= pred_q)
        
        results[f'Coverage_q{q}'] = coverage
    
    return results 