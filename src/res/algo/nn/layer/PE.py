"""Positional encoding utilities for patch-based time-series models."""
import math , torch

from torch import nn , Tensor

def positional_encoding(pe, learn_pe, q_len, d_model) -> Tensor:
    """Build a positional encoding tensor and wrap it as ``nn.Parameter``.

    Args:
        pe:       Encoding type string — one of:
                  * ``None``      — random uniform in ``[-0.02, 0.02]``, shape
                                    ``(q_len, d_model)``; ``learn_pe`` is
                                    forced to ``False``
                  * ``'zero'``    — random uniform, shape ``(q_len, 1)``
                  * ``'zeros'``   — random uniform, shape ``(q_len, d_model)``
                  * ``'normal'``/``'gauss'`` — normal ``N(0, 0.1)``, shape
                                    ``(q_len, 1)``
                  * ``'uniform'`` — uniform ``U(0, 0.1)``, shape ``(q_len, 1)``
                  * ``'sincos'``  — deterministic sinusoidal PE (normalized),
                                    shape ``(q_len, d_model)``
        learn_pe: If ``True``, the returned parameter is trainable.
        q_len:    Sequence / patch length.
        d_model:  Model embedding dimension.

    Returns:
        ``nn.Parameter`` of shape ``(q_len, d_model)`` or ``(q_len, 1)``
        depending on ``pe`` type.  Broadcast-compatible with patch embeddings.
    """
    # Positional encoding
    if pe is None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': 
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: 
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

def PositionalEncoding(q_len, d_model, normalize=True) -> Tensor:
    """Compute a deterministic sinusoidal positional encoding tensor.

    Note: this is a factory *function*, not a class.  It returns a plain
    ``Tensor`` (not an ``nn.Module``); callers wrap the result in
    ``nn.Parameter`` via :func:`positional_encoding`.

    Args:
        q_len:     Sequence length.
        d_model:   Embedding dimension.
        normalize: If ``True``, subtract the mean and divide by
                   ``std * 10`` to keep magnitudes small.

    Returns:
        Tensor of shape ``(q_len, d_model)``.
    """
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe