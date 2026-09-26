"""Astgnn: Alpha-Beta Co-Mining GRU.

One encoder when ``ab_split_input`` is false. When it is true, inputs before
``ab_split_pos`` feed the alpha GRU and the rest feed the beta GRU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.res.algo.nn import layer as Layer
from src.res.algo.nn.loss.loss import Loss

from .RNN import mod_gru

__all__ = ['Astgnn']

_FIT_LOSSES = ('mse', 'pearson', 'ccc')


def _concat_streams(streams: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    if len(streams) == 1:
        return streams[0]
    return torch.concat(list(streams), dim=-1)


class _AstgnnLoss(nn.Module):
    """ABCM loss: a selectable fit term plus R², correlation, and turnover."""

    def __init__(self, loss_corr_lamb: float = 0.1, fit_loss: str = 'mse', **kwargs):
        super().__init__()
        assert fit_loss in _FIT_LOSSES, f'fit_loss must be one of {_FIT_LOSSES}, got {fit_loss}'
        self.loss_corr_lamb = loss_corr_lamb
        self.fit_loss = fit_loss
        self.fit_criterion = Loss.get(fit_loss)

    def loss(
        self,
        pred: Tensor,
        label: Tensor,
        alphas: Tensor,
        betas: Tensor,
        betas_peer: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Fit ``pred`` to ``label[..., 0]``; regularize beta span, correlation, and turnover.

        Args:
            pred: Scalar predictions ``[bs, 1]``.
            label: Two-column label ``[bs, 2]``. Column 0 is the fit target
                (standardized return). Column 1 is the R² target (raw return).
            alphas: Alpha factors ``[bs, alpha_num]``.
            betas: Window-end beta factors ``[bs, beta_num]``.
            betas_peer: Window-start beta factors ``[bs, beta_num]``.
            weight: Optional sample weight. Only the fit term uses it. A
                two-column weight is sliced to column 0 so it matches the fit label.
        """
        assert label.shape[-1] == 2, label.shape
        fit = self.fit_criterion(pred, label[..., :1], weight=_fit_weight(weight))
        assert isinstance(fit, Tensor), f'fit loss should be a tensor, got {type(fit)}'
        rsquare = self.rsquare_loss(betas, label[..., 1])
        corr = self.corr_loss(torch.concat([alphas, betas], dim=-1))
        turnover = self.turnover_loss(betas, betas_peer)
        return {
            self.fit_loss: fit,
            'rsquare': rsquare,
            'corr': self.loss_corr_lamb * corr,
            'turnover': self.loss_corr_lamb * turnover,
        }

    def rsquare_loss(self, hiddens: Tensor, label: Tensor, **kwargs) -> Tensor:
        """Residual fraction ``||y - Py|| / ||y||`` of the column space of ``hiddens``."""
        assert hiddens.ndim == 2, hiddens.shape
        y_norm = label.norm()
        pred = hiddens @ (hiddens.T @ hiddens).inverse() @ hiddens.T @ label
        res_norm = (label - pred).norm()
        return res_norm / y_norm

    def corr_loss(self, hiddens: Tensor, **kwargs) -> Tensor:
        """Frobenius norm of the standardized factor covariance."""
        h = (hiddens - hiddens.mean(dim=0, keepdim=True)) / (hiddens.std(dim=0, keepdim=True) + 1e-6)
        return h.T.cov().norm()

    def turnover_loss(self, betas: Tensor, betas_peer: Tensor, **kwargs) -> Tensor:
        """Mean squared distance between window-end and window-start betas."""
        return (betas - betas_peer).square().mean()


def _fit_weight(weight: Tensor | None) -> Tensor | None:
    """Keep the weight column aligned with ``label[..., 0]``."""
    if weight is None:
        return None
    if weight.ndim == 1:
        weight = weight[:, None]
    if weight.shape[-1] > 1:
        weight = weight[..., :1]
    return weight


class Astgnn(_AstgnnLoss):
    """Alpha-Beta Co-Mining GRU. Registry key: ``'astgnn'``.

    ``ab_split_input=False`` concatenates every input stream and runs one GRU.
    ``ab_split_input=True`` sends ``input[:ab_split_pos]`` through the alpha
    encoder and ``input[ab_split_pos:]`` through the beta encoder.

    The fit term is ``fit_loss`` (``mse``, ``pearson``, or ``ccc``) against
    ``label[..., 0]``. R², correlation, and turnover stay on the factor outputs.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        rnn_layers: int = 2,
        enc_in=None,
        enc_in_dim: int = 64,
        act_type: str = 'leaky',
        dec_mlp_layers: int = 2,
        dec_mlp_dim: int = 64,
        alpha_num: int = 60,
        beta_num: int = 10,
        loss_corr_lamb: float = 0.1,
        beta_into_pred: bool = False,
        ab_split_input: bool = False,
        ab_split_pos: int = 1,
        resnet_projector: bool = False,
        fit_loss: str = 'mse',
        **kwargs,
    ):
        super().__init__(loss_corr_lamb=loss_corr_lamb, fit_loss=fit_loss)
        assert isinstance(ab_split_input, bool), f'ab_split_input must be bool, got {type(ab_split_input)}'
        assert isinstance(resnet_projector, bool), f'resnet_projector must be bool, got {type(resnet_projector)}'
        assert type(ab_split_pos) is int, f'ab_split_pos must be int, got {type(ab_split_pos)}'
        self.ab_split_input = ab_split_input
        self.ab_split_pos = ab_split_pos
        self.resnet_projector = resnet_projector

        rnn_kwargs = {'input_dim': enc_in_dim, 'output_dim': hidden_dim, 'num_layers': rnn_layers, 'dropout': dropout}
        if ab_split_input:
            self._init_split_encoders(input_dim, enc_in_dim, rnn_kwargs, kwargs)
        else:
            if not isinstance(input_dim, int):
                input_dim = sum(input_dim)
            self.fc_enc_in = nn.Sequential(nn.Linear(input_dim, enc_in_dim), nn.Tanh())
            self.fc_rnn = mod_gru(**rnn_kwargs)

        self.alpha_net = nn.Linear(hidden_dim, alpha_num)
        self.beta_net = nn.Linear(hidden_dim, beta_num)
        self.alpha_map_out = Layer.MeanPool()
        self.beta_into_pred = beta_into_pred
        self.beta_map_out = nn.Linear(beta_num, 1) if beta_into_pred else None

    def _init_split_encoders(self, input_dim, enc_in_dim: int, rnn_kwargs: dict, kwargs: dict) -> None:
        assert isinstance(input_dim, (list, tuple)), f'input_dim must be a list or tuple when ab_split_input, got {type(input_dim)}'
        assert all(isinstance(dim, int) and dim > 0 for dim in input_dim), f'input_dim entries must be positive ints, got {input_dim}'
        n_input = len(input_dim)
        assert 1 <= self.ab_split_pos <= n_input - 1, (
            f'ab_split_pos must be in [1, {n_input - 1}] for {n_input} inputs, got {self.ab_split_pos}'
        )
        if self.resnet_projector:
            assert self.ab_split_pos == 1, (
                f'resnet_projector requires ab_split_pos == 1 so the 4D stream stays alone on the alpha side, got {self.ab_split_pos}'
            )
        self._n_input = n_input
        alpha_dim = sum(input_dim[:self.ab_split_pos])
        beta_dim = sum(input_dim[self.ab_split_pos:])

        if self.resnet_projector:
            from src.res.algo.nn.model.CNN import mod_resnet_1d
            inday_dim = kwargs['inday_dim']
            if isinstance(inday_dim, (list, tuple)):
                inday_dim = inday_dim[0]
            res_kwargs = {k: v for k, v in kwargs.items() if k != 'seq_len'}
            self.alpha_fc_enc_in = mod_resnet_1d(inday_dim, input_dim[0], enc_in_dim, **res_kwargs)
        else:
            self.alpha_fc_enc_in = nn.Sequential(nn.Linear(alpha_dim, enc_in_dim), nn.Tanh())
        self.beta_fc_enc_in = nn.Sequential(nn.Linear(beta_dim, enc_in_dim), nn.Tanh())
        self.alpha_fc_rnn = mod_gru(**rnn_kwargs)
        self.beta_fc_rnn = mod_gru(**rnn_kwargs)

    def forward(self, inputs: Tensor | tuple[Tensor, ...] | list[Tensor]):
        """
        Shared path in: ``[bs, seq, input_dim]`` or a sequence of streams.
        Split path in: a sequence of streams cut at ``ab_split_pos``.
        Out: ``[bs, 1]`` and factor dict.
        """
        if self.ab_split_input:
            h_alpha, h_beta, h_beta_0 = self._encode_split(inputs)
        else:
            h_alpha, h_beta, h_beta_0 = self._encode_shared(inputs)
        return self._heads(h_alpha, h_beta, h_beta_0)

    def _encode_shared(self, inputs: Tensor | tuple[Tensor, ...] | list[Tensor]):
        x = inputs if isinstance(inputs, Tensor) else _concat_streams(inputs)
        x = self.fc_rnn(self.fc_enc_in(x))
        h, h_0 = x[:, -1], x[:, 0]
        return h, h, h_0

    def _encode_split(self, inputs: Tensor | tuple[Tensor, ...] | list[Tensor]):
        assert isinstance(inputs, (tuple, list)), f'ab_split_input expects a sequence of tensors, got {type(inputs)}'
        assert len(inputs) == self._n_input, f'expected {self._n_input} input streams, got {len(inputs)}'
        pos = self.ab_split_pos
        x_alpha = _concat_streams(inputs[:pos])
        x_beta = _concat_streams(inputs[pos:])
        h_alpha = self.alpha_fc_rnn(self.alpha_fc_enc_in(x_alpha))[:, -1]
        h_beta_seq = self.beta_fc_rnn(self.beta_fc_enc_in(x_beta))
        return h_alpha, h_beta_seq[:, -1], h_beta_seq[:, 0]

    def _heads(self, h_alpha: Tensor, h_beta: Tensor, h_beta_0: Tensor):
        alphas = self.alpha_net(h_alpha)
        betas = self.beta_net(h_beta)
        betas_peer = self.beta_net(h_beta_0)
        pred_alpha = self.alpha_map_out(alphas)
        pred_beta = self.beta_map_out(betas) if self.beta_into_pred and self.beta_map_out is not None else 0
        pred = pred_alpha + pred_beta
        return pred, {
            'alphas': alphas,
            'betas': betas,
            'betas_peer': betas_peer,
            'pred_alpha': pred_alpha,
            'pred_beta': pred_beta,
        }
