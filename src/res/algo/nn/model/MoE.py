"""Mixture-of-Experts GRU for single-task return prediction with optional market tower."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .. import layer as Layer
from .RNN import uni_rnn_encoder, uni_rnn_decoder, uni_rnn_mapping

__all__ = ['moe_gru', 'MoEGatingNetwork', 'load_balance_loss', 'diversity_loss']


def load_balance_loss(gate : Tensor) -> Tensor:
    """Encourage uniform expert usage within a batch (KL to uniform)."""
    mean_gate = gate.mean(dim = 0)
    uniform = torch.full_like(mean_gate , 1.0 / mean_gate.numel())
    return (mean_gate * (mean_gate / uniform).log()).sum()


def diversity_loss(expert_hiddens : list[Tensor]) -> Tensor:
    """Penalize high pairwise cosine similarity between expert representations."""
    h = torch.stack(expert_hiddens , dim = 1)
    h_norm = F.normalize(h , dim = -1)
    sim = torch.bmm(h_norm , h_norm.transpose(1 , 2))
    num_experts = sim.size(1)
    mask = ~torch.eye(num_experts , dtype = torch.bool , device = sim.device)
    return sim[:, mask].pow(2).mean()


class MoEGatingNetwork(nn.Module):
    """GRU gate over stock input; softmax-weights expert hidden vectors."""

    def __init__(
        self ,
        selector_dim : int ,
        num_experts : int ,
        num_layers : int = 1 ,
        dropout : float = 0.1 ,
    ) -> None:
        super().__init__()
        self.selector = nn.GRU(
            selector_dim , num_experts , num_layers ,
            batch_first = True , dropout = dropout if num_layers > 1 else 0.0,
        )
        self.softmax = nn.Softmax(dim = -1)

    def forward(self , stock : Tensor , expert_hiddens : list[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Args:
            stock: ``[bs, seq_len, input_dim]``
            expert_hiddens: ``num_experts`` tensors of shape ``[bs, hidden_dim]``

        Returns:
            fused hidden ``[bs, hidden_dim]`` and gate weights ``[bs, num_experts]``
        """
        gate = self.softmax(self.selector(stock)[0][:, -1])
        stacked = torch.stack(expert_hiddens , dim = 1)
        fused = (gate.unsqueeze(-1) * stacked).sum(dim = 1)
        return fused , gate


class MarketAwareExpert(nn.Module):
    """Dual-tower expert: stock GRU + market GRU fused to hidden_dim."""

    def __init__(self , stock_dim : int , market_dim : int , **encoder_kwargs) -> None:
        super().__init__()
        hidden_dim = encoder_kwargs['hidden_dim']
        self.stock_tower = uni_rnn_encoder(input_dim = stock_dim , **encoder_kwargs)
        self.market_tower = uni_rnn_encoder(input_dim = market_dim , **encoder_kwargs)
        act_type = encoder_kwargs.get('act_type' , 'leaky')
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim , hidden_dim) ,
            Layer.Act.get_activation_fn(act_type) ,
        )

    def forward(self , stock : Tensor , market : Tensor) -> Tensor:
        h_s = self.stock_tower(stock)
        h_m = self.market_tower(market)
        return self.fuse(torch.cat([h_s , h_m] , dim = -1))


class moe_gru(nn.Module):
    """MOE-GRU with stock-only and market-aware experts.  Registry key: ``'moe_gru'``.

    Expects ``_default_data_type = 'day+market'`` and tuple input
    ``(stock, market)`` from the training loop.
    """

    _default_data_type = 'day+market'

    def __init__(
        self ,
        input_dim ,
        hidden_dim : int = 2**5 ,
        dropout : float = 0.1 ,
        act_type : str = 'leaky',
        enc_in = None ,
        enc_in_dim = None ,
        enc_att : bool = False ,
        rnn_type : str = 'gru' ,
        rnn_layers : int = 2 ,
        gate_rnn_layers : int = 1 ,
        dec_mlp_layers : int = 2 ,
        dec_mlp_dim = None ,
        num_experts : int = 3 ,
        num_market_aware_experts : int = 1 ,
        num_output : int = 1 ,
        output_as_factors : bool = True ,
        hidden_as_factors = False ,
        hidden_as_factor = None ,
        load_balance_weight : float = 0.01 ,
        diversity_weight : float = 0.001 ,
        **kwargs ,
    ) -> None:
        super().__init__()
        if hidden_as_factor is not None:
            hidden_as_factors = hidden_as_factor
        if isinstance(input_dim , (list , tuple)):
            stock_dim , market_dim = int(input_dim[0]) , int(input_dim[1])
        else:
            stock_dim , market_dim = int(input_dim) , int(kwargs.get('market_input_dim' , 12))

        assert num_experts > num_market_aware_experts >= 0
        assert num_market_aware_experts <= num_experts

        self.num_experts = num_experts
        self.num_market_aware_experts = num_market_aware_experts
        self.num_stock_experts = num_experts - num_market_aware_experts
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        self.hidden_as_factors = hidden_as_factors

        encoder_kwargs = {
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'act_type': act_type,
            'enc_in': enc_in,
            'enc_in_dim': enc_in_dim,
            'enc_att': enc_att,
            'rnn_type': rnn_type,
            'rnn_layers': rnn_layers,
            **kwargs,
        }

        self.stock_experts = nn.ModuleList([
            uni_rnn_encoder(input_dim = stock_dim , **encoder_kwargs)
            for _ in range(self.num_stock_experts)
        ])
        self.market_aware_experts = nn.ModuleList([
            MarketAwareExpert(stock_dim , market_dim , **encoder_kwargs)
            for _ in range(num_market_aware_experts)
        ])
        self.gate = MoEGatingNetwork(stock_dim , num_experts , gate_rnn_layers , dropout)

        decoder_kwargs = {
            'hidden_dim': hidden_dim,
            'act_type': act_type,
            'dec_mlp_layers': dec_mlp_layers,
            'dec_mlp_dim': dec_mlp_dim,
            'dropout': dropout,
            'hidden_as_factors': hidden_as_factors,
        }
        self.decoder = uni_rnn_decoder(**decoder_kwargs)
        self.mapping = uni_rnn_mapping(
            hidden_dim = hidden_dim ,
            hidden_as_factors = hidden_as_factors ,
            output_as_factors = output_as_factors ,
        )

    def _expert_hiddens(self , stock : Tensor , market : Tensor) -> list[Tensor]:
        hiddens = [expert(stock) for expert in self.stock_experts]
        hiddens.extend([expert(stock , market) for expert in self.market_aware_experts])
        return hiddens

    def forward(self , x : Tensor | tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, dict]:
        if isinstance(x , (list , tuple)):
            stock , market = x[0] , x[1]
        else:
            raise TypeError(f'moe_gru expects (stock, market) tuple input, got {type(x)}')

        expert_hiddens = self._expert_hiddens(stock , market)
        fused , gate = self.gate(stock , expert_hiddens)
        hidden = self.decoder(fused)
        pred = self.mapping(hidden)

        other : dict[str, Tensor] = {
            'hidden': hidden,
            'gate': gate,
        }
        if self.load_balance_weight > 0:
            other['loss_load_balance'] = load_balance_loss(gate) * self.load_balance_weight
        if self.diversity_weight > 0:
            other['loss_diversity'] = diversity_loss(expert_hiddens) * self.diversity_weight
        return pred , other


if __name__ == '__main__':
    bs , seq_day , seq_mkt = 8 , 30 , 60
    stock = torch.randn(bs , seq_day , 6)
    market = torch.randn(bs , seq_mkt , 12)
    model = moe_gru(input_dim = [6 , 12] , hidden_dim = 64 , enc_in = True)
    pred , other = model((stock , market))
    assert pred.shape == (bs , 1) , pred.shape
    assert other['hidden'].shape == (bs , 64)
    assert other['gate'].shape == (bs , 3)
    assert 'loss_load_balance' in other and 'loss_diversity' in other
