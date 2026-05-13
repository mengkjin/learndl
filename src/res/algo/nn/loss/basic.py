"""Shared utilities for loss and accuracy modules."""
import torch

def mask_topx(tensor : torch.Tensor , x_percent : float , dim : int = 0 , ascending : bool = True , fill_nan : bool = False):
    """
    mask the top x% of the tensor in the specified dimension, ignoring NaN.
    """
    nan_pos = tensor.isnan()
    non_nan_counts = (~nan_pos).sum(dim=dim)
    
    clean_tensor = torch.where(tensor.isnan() , -torch.inf if ascending else torch.inf , tensor)
    ranks = clean_tensor.argsort(dim=dim, descending=ascending).argsort(dim=dim)
    k_threshold = (non_nan_counts * x_percent).long()
    mask = (ranks < k_threshold.unsqueeze(dim)) & (~tensor.isnan())
    
    if fill_nan:
        mask = torch.where(mask , 1. , torch.nan)
    else:
        mask = mask.float()
    return mask

def first_output(x : torch.Tensor):
    """Get the first output of a tensor."""
    if x.ndim == 1:
        return x[0]
    else:
        return x[...,0]