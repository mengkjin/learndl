"""Loss function registry for NN training.

All losses inherit from ``BaseLoss``.  The ``Loss`` class is the factory used
by the training loop.  Loss values should be minimized (lower is better).

Registry keys: 'mse', 'pearson', 'ccc', 'hidden_corr_deprecated',
               'hidden_corr', 'quantile', 'ccc_hcorr', 'abcm'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.func.metric import mse , pearson , ccc

from .basic import align_shape

__all__ = ['Loss']

class BaseLoss(nn.Module):
    """Abstract base class for all loss functions.

    Subclasses must set the class attribute ``key`` (a non-empty string that
    acts as the registry key) and implement ``forward()``.

    Class attributes:
        key:                      Registry string used by ``Loss.get()``.
        multiheadlosses_capable:  If ``True``, the loss can be decomposed per
                                  output head and used with
                                  ``MultiHeadLosses``.
    """
    key : str = ''
    multiheadlosses_capable : bool = False

    def __init__(self , **kwargs):
        super().__init__()

    def __call__(self , *args , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        """Call forward and validate the output type."""
        output = self.forward(*args , **kwargs)
        assert isinstance(output,torch.Tensor) or isinstance(output,dict) , f'output of {self.key} should be a tensor or a dict , but got {output}'
        return output

    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        """Compute and return the loss value.

        Args:
            label: Ground-truth tensor.
            pred:  Model prediction tensor.
            w:     Optional per-sample weight tensor.
            dim:   Reduction dimension (``None`` = global reduction).
            **kwargs: Extra keyword arguments forwarded by some composite losses.

        Returns:
            A scalar ``Tensor`` or a ``dict`` mapping sub-loss names to tensors.
        """
        raise NotImplementedError

class LossMSE(BaseLoss):
    """Mean squared error loss.  Lower is better."""
    key = 'mse'
    multiheadlosses_capable = True
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return mse(*align_shape(label , pred , w) , dim)

class LossPearson(BaseLoss):
    """Pearson correlation loss: ``exp(-pearson)``.  Lower is better."""
    key = 'pearson'
    multiheadlosses_capable = True
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return torch.exp(-pearson(*align_shape(label , pred , w) , dim))

class LossCCC(BaseLoss):
    """Concordance Correlation Coefficient (CCC) loss: ``exp(-ccc)``.  Lower is better."""
    key = 'ccc'
    multiheadlosses_capable = True
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return torch.exp(-ccc(*align_shape(label , pred , w) , dim))

class LossHiddenCorrDeprecated(BaseLoss):
    """Deprecated hidden-state correlation penalty.  Use ``LossHiddenCorr`` instead.

    Difference: this version uses ``square().mean()`` (Frobenius² mean);
    ``LossHiddenCorr`` uses ``norm()`` (Frobenius norm), which is the
    correct formulation.
    """
    key = 'hidden_corr_deprecated'
    def forward(self , *args , hidden : torch.Tensor | list | tuple , **kwargs) -> torch.Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)):
            hidden = torch.cat(hidden,dim=-1)
        std_hidden = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        return std_hidden.T.cov().square().mean()

class LossHiddenCorr(BaseLoss):
    """Hidden-state correlation penalty (Frobenius norm of the covariance matrix).

    Penalizes the Frobenius norm of the column-standardized covariance matrix
    of the hidden representation, encouraging disentangled / decorrelated
    latent features.
    """
    key = 'hidden_corr'
    def forward(self , *args , hidden : torch.Tensor | list | tuple , **kwargs) -> torch.Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)):
            hidden = torch.cat(hidden,dim=-1)
        std_hidden = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        return std_hidden.T.cov().norm()

class LossQuantile(BaseLoss):
    """Quantile (pinball) loss for probabilistic forecasting.

    Computes the mean pinball loss across all requested quantile levels.

    Args (forward):
        label:       Ground-truth tensor of shape ``(bs, T)`` or ``(bs, 1)``.
        predictions: Model output tensor of shape ``(bs, T, n_quantiles)`` or
                     ``(bs, n_quantiles)``.  Must have ``predictions.shape[-1]
                     == len(quantiles)``.
        quantiles:   List of quantile levels to predict (default
                     ``[0.1, 0.5, 0.9]``).
        w:           Optional per-sample weight tensor.
    """
    key = 'quantile'
    def forward(self , label : torch.Tensor , predictions : torch.Tensor | None = None , w : torch.Tensor | None = None , dim = None ,
                quantiles : list[float] = [0.1,0.5,0.9] , **kwargs) -> torch.Tensor:
        assert predictions is not None , f'predictions should be provided'
        assert predictions.shape[-1] == len(quantiles) , f'shape of predictions {predictions.shape} should be (...,{len(quantiles)})'
        if predictions.ndim == label.ndim + 1: 
            predictions = predictions.squeeze(-2)
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

class CCCHiddenCorrLoss(BaseLoss):
    """Composite CCC + hidden-correlation penalty loss.

    Total loss = ``CCC_loss(label, pred) + lamb * HiddenCorr_loss(hidden)``

    Args:
        lamb: Weighting coefficient for the hidden correlation penalty
              (default ``0.01``).
    """
    key = 'ccc_hcorr'
    def __init__(self, lamb : float = 0.01):
        super().__init__()
        self.lamb = lamb
        self.ccc_loss = LossCCC()
        self.hidden_corr_loss = LossHiddenCorr()

    def forward(self , label : torch.Tensor , pred : torch.Tensor , hidden : torch.Tensor , **kwargs) -> dict[str,torch.Tensor]:
        return {
            'ccc' : self.ccc_loss.forward(label , pred) ,
            'hidden_corr' : self.hidden_corr_loss.forward(hidden = hidden) * self.lamb
        }

class ABCMLoss(BaseLoss):
    """Alpha-Beta Co-Mining (ABCM) composite loss.

    Combines four components:
    1. **MSE** — prediction error of the alpha net output vs. ``label[...,0]``
    2. **R² loss** — ``1 - R²`` of the alpha hidden states regressed on
       ``label[...,1]`` (maximizes explained variance)
    3. **Correlation penalty** — Frobenius norm of the beta hidden-state
       covariance (scaled by ``lamb``); discourages redundant beta factors
    4. **Turnover penalty** — L2 distance between current and peer betas;
       encourages temporal stability of the risk factors

    Total: ``MSE + R²_loss + lamb * corr_loss + turnover_loss``

    Args:
        lamb: Weight for the beta correlation penalty (default ``0.1``).

    Reference: 基于神经网络的alpha因子和beta因子协同挖掘模型
    """
    key = 'abcm'
    def __init__(self, lamb : float = 0.1):
        super().__init__()
        self.lamb = lamb

    def forward(self, pred : torch.Tensor , label : torch.Tensor , alphas : torch.Tensor , betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        assert label.shape[-1] == 2 , label.shape
        mse = F.mse_loss(pred.squeeze() , label[...,0].squeeze())
        rsquare = self.rsquare_loss(alphas , label[...,1])
        corr = self.corr_loss(betas)
        turnover = self.turnover_loss(betas , betas_peer)

        return mse + rsquare + self.lamb * corr + turnover

    def rsquare_loss(self, hiddens : torch.Tensor , label : torch.Tensor , **kwargs):
        """Compute ``1 - R²`` by projecting label onto the column space of hiddens."""
        assert hiddens.ndim == 2 , hiddens.shape
        y_norm = label.norm()
        pred = hiddens @ (hiddens.T @ hiddens).inverse() @ hiddens.T @ label
        res_norm = (label - pred).norm()
        return 1 - res_norm / y_norm

    def corr_loss(self, hiddens : torch.Tensor , **kwargs):
        """Frobenius norm of the standardized hidden-state covariance matrix."""
        h = (hiddens - hiddens.mean(dim=0,keepdim=True)) / (hiddens.std(dim=0,keepdim=True) + 1e-6)
        pen = h.T.cov().norm()
        return pen

    def turnover_loss(self, betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        """L2 distance between current beta factors and peer (lagged) betas."""
        return (betas - betas_peer).norm()

class Loss:
    """Factory for ``BaseLoss`` instances.

    ``options`` is populated at class definition time by scanning all direct
    subclasses of ``BaseLoss`` that have a non-empty ``key``.

    Usage::

        loss_fn = Loss.get('ccc')
        loss_fn = Loss.get('abcm', lamb=0.05)
    """
    options = {cls.key : cls for cls in BaseLoss.__subclasses__() if cls.key != ''}
    @classmethod
    def get(cls , name : str , **kwargs) -> BaseLoss:
        """Return an instantiated loss function by registry key.

        Args:
            name:    Registry key string (e.g. ``'mse'``, ``'ccc'``).
            **kwargs: Forwarded to the loss constructor.

        Returns:
            An instantiated ``BaseLoss`` subclass.
        """
        return cls.options[name](**kwargs)