"""NumPy/torch helpers: index alignment, forward/backward fill, and merge utilities for panel keys.

Module attribute ``DIV_TOL`` is a small denominator tolerance shared with ``func.tensor`` and ``func.metric``.
"""
from __future__ import annotations
import torch , sys
import numpy as np
from typing import Any , Literal , overload , TypeVar , TypeAlias

DIV_TOL = 1e-6

T = TypeVar('T')

ArrayAny : TypeAlias = np.ndarray | Any
ArrayTensorAny : TypeAlias = np.ndarray | torch.Tensor | Any
IndexMergeMethod : TypeAlias = Literal['union' , 'intersect' , 'check' , 'stack']


def alert_message(message : str , color : str = 'yellow'):
    """Emit a one-line message to stderr, optionally with ANSI color."""
    if color == 'yellow':
        sys.stderr.write(f'\u001b[93m\u001b[1m{message}\u001b[0m')
    elif color == 'lightred':
        sys.stderr.write(f'\u001b[91m\u001b[1m{message}\u001b[0m')
    else:
        sys.stderr.write(message)

def allna(x : torch.Tensor | np.ndarray | None , inf_as_na = True):
    """Return whether ``x`` has no usable values.

    Args:
        x: Tensor, array, or None.
        inf_as_na: If True, treat non-finite values as missing. If False, only NaN counts as missing
            and "all missing" means every element is NaN.

    Returns:
        True if ``x`` is None or all values are missing per the rule above; otherwise False.
    """
    if x is None:
        return True
    elif inf_as_na:
        return not x.isfinite().any() if isinstance(x , torch.Tensor) else not np.isfinite(x).any()
    else:
        return x.isnan().all() if isinstance(x , torch.Tensor) else np.isnan(x).all()

def exact(x , y):
    """Return whether ``x`` and ``y`` are the same object (``is``)."""
    return x is y

def average_params(params_list : tuple[dict] | list[dict]):
    """Elementwise average of float tensors in aligned optimizer state dicts; keys must match across dicts."""
    n = len(params_list)
    if n == 1: 
        return params_list[0]
    new_params = {k:v / n for k,v in params_list[0].items()}
    for i, params in enumerate(params_list[1:]):
        for k, v in params.items():
            if k not in new_params.keys(): 
                raise KeyError(f'the {i}-th model has different params named {k}')
            else:
                new_params[k] += v / n
    return new_params

def as_array(values):
    """Convert list, scalar, tensor, or array-like to a NumPy array (tensor moved to CPU)."""
    if not isinstance(values , np.ndarray): 
        if isinstance(values , torch.Tensor): 
            values = values.cpu().numpy()
        elif isinstance(values , list):
            values = np.asarray(values)
        else: 
            values = np.asarray(values).reshape(-1)
    return np.atleast_1d(values)

def match_values(values , src_arr , ambiguous = 0):
    """Map each element of ``values`` to a sorted index in ``src_arr``.

    Args:
        values: Query values (coerced via ``to_numpy``).
        src_arr: Sorted reference values (coerced via ``to_numpy``).
        ambiguous: If 0, only values present in ``src_arr`` get a real index; others get ``len(src_arr)``.
            If non-zero, values ``<= max(src_arr)`` also get an index via ``searchsorted``.

    Returns:
        Integer array of the same shape as ``values`` with positions or sentinel ``len(src_arr)``.
    """
    values = as_array(values)
    src_arr = as_array(src_arr)
    sorter = np.argsort(src_arr)
    index = np.tile(len(src_arr) , values.shape)
    if ambiguous == 0:
        index[np.isin(values , src_arr)] = sorter[np.searchsorted(src_arr, values[np.isin(values , src_arr)], sorter=sorter)]
    else:
        index[values <= max(src_arr)] = sorter[np.searchsorted(src_arr, values[values <= max(src_arr)], sorter=sorter)]
    return index

def convert_to_slice(index : np.ndarray | list) -> slice | np.ndarray:
    """Compress contiguous integer indices into a Python ``slice`` when possible.

    Args:
        index: 1-D integer positions.

    Returns:
        ``slice(0,0,1)`` if empty; ``slice(i, i+1, 1)`` if length 1; ``slice(start, end, step)`` if
        indices form ``arange``; otherwise the original indices as an array.
    """
    if len(index) == 0:
        return slice(0,0,1)
    elif len(index) == 1:
        return slice(index[0],index[0]+1,1)
    else:
        start = min(index)
        end = max(index) + 1
        step = index[1] - index[0]
        if np.array_equal(index , np.arange(start,end,step)):
            return slice(start,end,step)
        else:
            return as_array(index)

def match_slice(values , src_arr , ambiguous = 0) -> slice | np.ndarray:
    """``match_values`` followed by ``convert_to_slice``.

    Args:
        values: Query values.
        src_arr: Reference sorted array.
        ambiguous: Passed to ``match_values``.

    Returns:
        ``slice(0,0,1)`` if ``values`` is empty; ``slice(None)`` if ``values`` equals ``src_arr``;
        otherwise ``convert_to_slice(match_values(...))``.
    """
    values = as_array(values)
    src_arr = as_array(src_arr)
    if len(values) == 0:
        return slice(0,0,1)
    elif np.array_equal(values , src_arr):
        return slice(None,None,1)
    else:
        return convert_to_slice(match_values(values , src_arr , ambiguous))

def array_to_tensor_slice(array : np.ndarray | list) -> slice | torch.Tensor:
    """``convert_to_slice`` with tensor output for non-slice results.

    Args:
        array: Integer indices.

    Returns:
        A ``slice`` or ``torch.from_numpy`` of the dense index array.
    """
    array_slice = convert_to_slice(array)
    if isinstance(array_slice , slice):
        return array_slice
    else:
        return torch.from_numpy(array_slice)
    
def forward_fillna_np(arr , axis = 0 , * , force_value = None):
    """[ARCHIVED] NumPy-only last-observation-carried-forward along one axis.

    Kept for reference/regression checks. New code should use :func:`forward_fillna`,
    which also accepts ``torch.Tensor`` directly and uses a faster path for ``force_value``.

    Args:
        arr: Input array (any shape).
        axis: Axis along which to propagate last valid values forward.
        force_value: If given, NaNs at/after the first valid observation are set to this
            value (leading NaNs before any valid observation stay NaN).

    Returns:
        Array of the same shape with NaNs filled from the left along the chosen axis.
    """
    shape = arr.shape
    if axis < 0: 
        axis = len(shape) + axis
    new_axes  = list(range(len(arr.shape)))
    new_axes[0] , new_axes[axis] = axis , 0
    
    filled_arr = np.nan_to_num(arr , nan = force_value) if force_value is not None else None

    arr = np.transpose(arr , new_axes)
    new_shape = arr.shape

    arr = arr.reshape(shape[axis],-1).transpose(1,0)
    idx = np.where(np.isnan(arr) == 0 , np.arange(arr.shape[1]), 0)
    idx = np.maximum.accumulate(idx, axis=1)
    out = arr[np.arange(idx.shape[0])[:,None], idx].transpose(1,0)
    out = np.transpose(out.reshape(new_shape) , new_axes)
    if filled_arr is not None:
        out = np.where(np.isnan(out) , np.nan , filled_arr)
    return out

def backward_fillna_np(arr, axis = 0 , * , force_value = None):
    """[ARCHIVED] NumPy-only next-observation-carried-backward along one axis.

    Kept for reference/regression checks. New code should use :func:`backward_fillna`,
    which also accepts ``torch.Tensor`` directly and uses a faster path for ``force_value``.

    Args:
        arr: Input array.
        axis: Axis along which to propagate next valid values backward.
        force_value: If given, NaNs at/before the last valid observation are set to this
            value (trailing NaNs after the last valid observation stay NaN).

    Returns:
        Array of the same shape with NaNs filled from the right along the chosen axis.
    """
    shape = arr.shape
    if axis < 0: 
        axis = len(shape) + axis
    new_axes  = list(range(len(arr.shape)))
    new_axes[0] , new_axes[axis] = axis , 0

    filled_arr = np.nan_to_num(arr , nan = force_value) if force_value is not None else None

    arr = np.transpose(arr , new_axes)
    new_shape = arr.shape

    arr = arr.reshape(shape[axis],-1).transpose(1,0)
    idx = np.where(np.isnan(arr) == 0 , np.arange(arr.shape[1]), arr.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1].copy()

    out = arr[np.arange(idx.shape[0])[:,None], idx].transpose(1,0)
    out = np.transpose(out.reshape(new_shape) , new_axes)
    if filled_arr is not None:
        out = np.where(np.isnan(out) , np.nan , filled_arr)
    return out

def _active_mask_np(notnan : np.ndarray , axis : int , backward : bool) -> np.ndarray:
    """Boolean mask of positions at/after (forward) or at/before (backward) the first/last valid value."""
    if backward:
        return np.flip(np.maximum.accumulate(np.flip(notnan , axis) , axis = axis) , axis)
    return np.maximum.accumulate(notnan , axis = axis)

def _active_mask_torch(notnan : torch.Tensor , axis : int , backward : bool) -> torch.Tensor:
    """Boolean mask of positions at/after (forward) or at/before (backward) the first/last valid value."""
    m = notnan.to(torch.long)
    if backward:
        m = torch.cummax(m.flip(axis) , dim = axis).values.flip(axis)
    else:
        m = torch.cummax(m , dim = axis).values
    return m.bool()

def _fillna(arr , axis : int , force_value , backward : bool):
    """Shared engine for forward/backward NaN fill supporting both NumPy arrays and torch tensors.

    When ``force_value`` is given, only the cheap active-mask path is used (no carry-forward
    gather): NaNs inside the active span are set to ``force_value`` while leading/trailing NaNs
    outside the span are left untouched. Otherwise the last/next valid value is carried along ``axis``.
    """
    if axis < 0:
        axis = arr.ndim + axis

    if isinstance(arr , torch.Tensor):
        notnan = ~torch.isnan(arr)
        if force_value is not None:
            active = _active_mask_torch(notnan , axis , backward)
            out = arr.clone()
            out[(~notnan) & active] = force_value
            return out
        n = arr.shape[axis]
        view = [1] * arr.ndim
        view[axis] = n
        ar = torch.arange(n , device = arr.device).reshape(view).expand(arr.shape)
        if backward:
            idx = torch.where(notnan , ar , torch.full_like(ar , n - 1))
            idx = torch.cummin(idx.flip(axis) , dim = axis).values.flip(axis)
        else:
            idx = torch.where(notnan , ar , torch.zeros_like(ar))
            idx = torch.cummax(idx , dim = axis).values
        return torch.gather(arr , axis , idx)
    elif isinstance(arr , np.ndarray):
        notnan = ~np.isnan(arr)
        if force_value is not None:
            active = _active_mask_np(notnan , axis , backward)
            out = arr.copy()
            out[(~notnan) & active] = force_value
            return out
        n = arr.shape[axis]
        view = [1] * arr.ndim
        view[axis] = n
        ar = np.broadcast_to(np.arange(n).reshape(view) , arr.shape)
        if backward:
            idx = np.where(notnan , ar , n - 1)
            idx = np.flip(np.minimum.accumulate(np.flip(idx , axis) , axis = axis) , axis)
        else:
            idx = np.where(notnan , ar , 0)
            idx = np.maximum.accumulate(idx , axis = axis)
        return np.take_along_axis(arr , idx , axis = axis)
    else:
        raise TypeError(f'Unsupported type: {type(arr)}')

@overload
def forward_fillna(arr : torch.Tensor , axis : int = 0 , * , force_value = None) -> torch.Tensor: ...
@overload
def forward_fillna(arr : np.ndarray , axis : int = 0 , * , force_value = None) -> np.ndarray: ...
def forward_fillna(arr , axis = 0 , * , force_value = None):
    """Last-observation-carried-forward along one axis (NumPy array or torch tensor).

    Accepts ``np.ndarray`` or ``torch.Tensor`` and returns the same type, preserving the tensor's
    device and dtype (no NumPy round-trip required for tensors).

    Args:
        arr: Input array/tensor (any shape).
        axis: Axis along which to propagate last valid values forward.
        force_value: If given, take the fast path: NaNs at/after the first valid observation are
            set to this value (leading NaNs before any valid observation stay NaN); no carry-forward.

    Returns:
        Array/tensor of the same shape with NaNs filled from the left along the chosen axis.
    """
    return _fillna(arr , axis , force_value , backward = False)

@overload
def backward_fillna(arr : torch.Tensor , axis : int = 0 , * , force_value = None) -> torch.Tensor: ...
@overload
def backward_fillna(arr : np.ndarray , axis : int = 0 , * , force_value = None) -> np.ndarray: ...
def backward_fillna(arr , axis = 0 , * , force_value = None):
    """Next-observation-carried-backward along one axis (NumPy array or torch tensor).

    Accepts ``np.ndarray`` or ``torch.Tensor`` and returns the same type, preserving the tensor's
    device and dtype (no NumPy round-trip required for tensors).

    Args:
        arr: Input array/tensor (any shape).
        axis: Axis along which to propagate next valid values backward.
        force_value: If given, take the fast path: NaNs at/before the last valid observation are
            set to this value (trailing NaNs after the last valid observation stay NaN); no carry-backward.

    Returns:
        Array/tensor of the same shape with NaNs filled from the right along the chosen axis.
    """
    return _fillna(arr , axis , force_value , backward = True)

def trim_index(index : ArrayAny , min_value = None , max_value = None) -> np.ndarray:
    """Clip a 1-D index array by inclusive bounds.

    Args:
        index: 1-D index values.
        min_value: If set, drop elements ``< min_value``.
        max_value: If set, drop elements ``> max_value``.

    Returns:
        Filtered 1-D array.
    """
    if min_value is not None:
        index = index[index >= min_value]
    if max_value is not None:
        index = index[index <= max_value]
    return index

def index_intersect(idxs , min_value = None , max_value = None) -> np.ndarray:
    """Intersection of several 1-D index arrays.

    Args:
        idxs: Iterable of 1-D arrays; ``None`` entries are skipped when merging.
        min_value: Optional lower bound passed to ``trim_index``.
        max_value: Optional upper bound passed to ``trim_index``.

    Returns:
        Sorted intersection, then trimmed.
    """
    new_idx : ArrayTensorAny = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.intersect1d(new_idx , idx)
    new_idx = np.sort(new_idx)
    return trim_index(new_idx , min_value , max_value)

def index_union(idxs , min_value = None , max_value = None) -> np.ndarray:
    """Union of several 1-D index arrays.

    Args:
        idxs: Iterable of 1-D arrays; ``None`` skipped as in ``index_intersect``.
        min_value: Optional lower bound for ``trim_index``.
        max_value: Optional upper bound for ``trim_index``.

    Returns:
        Sorted union, then trimmed.
    """
    new_idx : ArrayAny = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.union1d(new_idx , idx)
    return trim_index(new_idx , min_value , max_value)

def index_stack(idxs , min_value = None , max_value = None) -> np.ndarray:
    """Concatenate index arrays that are pairwise disjoint or nested.

    Args:
        idxs: Iterable of 1-D arrays.
        min_value: Optional bound for ``trim_index``.
        max_value: Optional bound for ``trim_index``.

    Returns:
        Concatenated / deduplicated index, trimmed.

    Raises:
        ValueError: If a new array neither is a subset of the accumulator nor disjoint from it.
    """
    new_idx : ArrayAny = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        elif np.isin(idx , new_idx).all():
            ...
        else:
            new_idx = np.concatenate([new_idx , [i for i in idx if i not in new_idx]])
    return trim_index(new_idx , min_value , max_value)

def index_check(idxs , min_value = None , max_value = None) -> np.ndarray:
    """Verify all index arrays are identical to the first.

    Args:
        idxs: Non-empty iterable of 1-D arrays.
        min_value: Optional bound for ``trim_index``.
        max_value: Optional bound for ``trim_index``.

    Returns:
        ``trim_index`` of the first array.

    Raises:
        ValueError: If any array differs from the first.
    """
    new_idx : ArrayAny = None
    for i , idx in enumerate(idxs):
        if i == 0:
            new_idx = idx
        elif np.array_equal(idx , new_idx):
            pass
        else:
            raise ValueError(f'idx at {i} is {idx}, does not equal to idx at 0 {new_idx} , index_check inputs must be identical')
    return trim_index(new_idx , min_value , max_value)

def index_merge(
    idxs , * , method : IndexMergeMethod = 'intersect' , 
    min_value = None , max_value = None) -> np.ndarray:
    """Merge index lists via intersect, union, stack, or equality check.

    Args:
        idxs: Non-empty list of 1-D index arrays.
        method: Which merge rule to apply.
        min_value: Optional lower trim bound.
        max_value: Optional upper trim bound.

    Returns:
        Merged 1-D index array.

    Raises:
        AssertionError: If ``idxs`` is empty.
        ValueError: If ``method`` is invalid or underlying helper raises.
    """
    assert len(idxs) > 0 , 'index_merge: idxs must be non-empty'
    if method == 'intersect':
        return index_intersect(idxs , min_value , max_value)
    elif method == 'union':
        return index_union(idxs , min_value , max_value)
    elif method == 'stack':
        return index_stack(idxs , min_value , max_value)
    elif method == 'check':
        return index_check(idxs , min_value , max_value)
    else:
        raise ValueError(f'Invalid method: {method}')

def intersect_pos_tensor(target_index : ArrayAny , source_index : ArrayAny) -> tuple[torch.Tensor , torch.Tensor]:
    """Positions of common labels in two 1-D index arrays.

    Args:
        target_index: First index (e.g. panel dates).
        source_index: Second index.

    Returns:
        Tuple ``(target_pos, source_pos)`` as ``torch.Tensor`` integer indices into each input.
    """
    _ , target_pos , source_pos = np.intersect1d(target_index , source_index , return_indices=True)
    return torch.from_numpy(target_pos), torch.from_numpy(source_pos)

def intersect_pos_slice(target_index : ArrayAny , source_index : ArrayAny) -> tuple[slice | np.ndarray , slice | np.ndarray]:
    """Like ``intersect_pos_tensor`` but compress contiguous runs to ``slice`` objects.

    Args:
        target_index: First 1-D index.
        source_index: Second 1-D index.

    Returns:
        Pair of ``slice`` or dense index arrays for target and source positions.
    """
    _ , target_pos , source_pos = np.intersect1d(target_index , source_index , return_indices=True)
    return convert_to_slice(target_pos), convert_to_slice(source_pos)

def intersect_meshgrid(target_indices : list[ArrayTensorAny] , source_indices : list[ArrayTensorAny]) -> tuple[tuple[torch.Tensor , ...] , tuple[torch.Tensor , ...]]:
    """Meshgrid of intersect positions for multiple paired axes.

    Args:
        target_indices: List of target 1-D indices, length ``K``.
        source_indices: List of source 1-D indices, same length ``K``.

    Returns:
        ``(target_mesh, source_mesh)`` from ``torch.meshgrid(..., indexing='ij')`` on each axis pair.

    Raises:
        AssertionError: If list lengths differ.
    """
    assert len(target_indices) == len(source_indices) , f'target_indices and source_indices must have the same length , but got {len(target_indices)} and {len(source_indices)}'
    target_pos = []
    source_pos = []
    for target_index , source_index in zip(target_indices, source_indices):
        tarpos , srcpos = intersect_pos_tensor(target_index , source_index)
        target_pos.append(tarpos)
        source_pos.append(srcpos)
    return torch.meshgrid(*target_pos, indexing='ij'), torch.meshgrid(*source_pos, indexing='ij')

