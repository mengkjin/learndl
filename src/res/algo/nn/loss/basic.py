"""Shared utilities for loss and accuracy modules."""
import torch

def align_shape(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None):
    """Truncate label, pred, and optional weight to a consistent last dimension.

    When a model produces fewer output steps than the label (e.g. multi-horizon
    prediction with partial output), this function trims all three tensors to
    ``min(label.shape[-1], pred.shape[-1])`` along the last axis so that loss
    computation does not fail due to shape mismatch.

    Args:
        label: Ground-truth tensor of arbitrary shape ``(..., T_label)``.
        pred:  Model prediction tensor of arbitrary shape ``(..., T_pred)``.
        w:     Optional sample weight tensor of shape ``(..., T_w)``.

    Returns:
        ``(label, pred, w)`` trimmed to the same last dimension size.
        ``w`` is returned unchanged (``None``) when not provided.
    """
    if label.shape[-1] != pred.shape[-1]:
        last_dim = min(label.shape[-1] , pred.shape[-1])
        label = label[...,:last_dim]
        pred = pred[...,:last_dim]
        if w is not None:
            w = w[...,:last_dim]
    return label , pred , w
