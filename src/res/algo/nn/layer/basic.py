"""Basic reusable nn.Module building blocks.

Classes:
    Pass       — identity / no-op layer
    Transpose  — dimension transposition with optional contiguous copy
    MeanPool   — temporal mean pooling along a specified dimension
    Parallel   — runs N independent copies of a sub-module in parallel
"""
import torch

from torch import nn , Tensor
from copy import deepcopy

class Pass(nn.Module):
    """Identity layer; passes input through unchanged.

    Useful as a placeholder in nn.Sequential or conditional module graphs.
    """
    def forward(self , x): return x

class Transpose(nn.Module):
    """Transpose two dimensions of a tensor.

    Args:
        *dims: Two dimension indices to transpose (e.g. ``1, 2``).
        contiguous: If True, calls ``.contiguous()`` after transpose so the
            result is stored in contiguous memory (required by some ops).
    """
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

class MeanPool(nn.Module):
    """Temporal mean pooling (element-wise mean) along a specified dimension.

    Despite the name, this is not a trainable linear layer — it computes the
    arithmetic mean of the input along ``dim``.  Typically used to pool a
    sequence of hidden states into a single summary vector.

    Args:
        dim:     Dimension to reduce over.  Default ``-1`` (last dim).
        keepdim: Whether to retain the reduced dimension.  Default ``True``.
    """
    def __init__(self, dim = -1 , keepdim = True):
        super().__init__()
        self.dim , self.keepdim = dim , keepdim
    def forward(self, x : Tensor):
        return x.mean(dim = self.dim , keepdim = self.keepdim)

class Parallel(nn.Module):
    """Run N independent copies of a sub-module in parallel.

    Creates ``num_mod`` deep copies of ``sub_mod`` and runs them in parallel
    during each forward pass.

    Args:
        sub_mod:       The template module to copy ``num_mod`` times.
        num_mod:       Number of parallel branches.
        feedforward:   If ``True`` (default), branch ``i`` receives
                       ``inputs[i]`` as input — each branch gets a distinct
                       slice of the input sequence.
                       If ``False``, every branch receives the same ``inputs``.
        concat_output: If ``True``, concatenate all branch outputs along the
                       last dimension before returning.  If the branch outputs
                       are tuples/lists, each position is concatenated
                       separately, returning a tuple of concatenated tensors.
                       If ``False`` (default), returns a tuple of raw outputs.

    Returns:
        Tuple of branch outputs, or a concatenated tensor / tuple of tensors
        when ``concat_output=True``.
    """
    def __init__(self, sub_mod , num_mod , feedforward = True , concat_output = False):
        super().__init__()
        self.mod_list = nn.ModuleList([deepcopy(sub_mod) for _ in range(num_mod)])
        self.feedforward = feedforward
        self.concat_output = concat_output
    def forward(self, inputs):
        output = tuple([mod(inputs[i] if self.feedforward else inputs) for i,mod in enumerate(self.mod_list)])
        if self.concat_output:
            if isinstance(output[0] , (list,tuple)):
                output = tuple([torch.cat([out[i] for out in output] , dim = -1) for i in range(len(output[0]))])
            else:
                output = torch.cat(output , dim = -1)
        return output