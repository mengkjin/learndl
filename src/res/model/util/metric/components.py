"""
Metric components for the project
"""
from __future__ import annotations

from torch import Tensor
from typing import Callable

from src.proj import Base
from src.proj.core import as_int_array
from src.res.model.util import TensorReturnType
from src.res.algo.nn.loss import MultiHeadLosses

__all__ = ['MetricComponent' , 'LossComponent' , 'AccuracyComponent']

def align_shape(pred : Tensor , label : Tensor , weight : Tensor | None = None , dim : int | None = None):
    """Adjust pred, label, and optional weight to a consistent last dimension.

    When a model produces fewer output steps than the label (e.g. multi-horizon
    prediction with partial output), this function trims all three tensors to
    ``min(pred.shape[-1], label.shape[-1])`` along the last axis so that loss
    computation does not fail due to shape mismatch.

    Args:
        pred:   Model prediction tensor of arbitrary shape ``(..., T_pred)``.
        label:  Ground-truth tensor of arbitrary shape ``(..., T_label)``.
        weight: Optional sample weight tensor of shape ``(..., T_w)``.
        dim:    Reduction dimension (``None`` = global reduction). If provided, the last dimension of label should match pred or 1.

    Returns:
        ``(pred, label, weight)`` trimmed to the same last dimension size.
        ``weight`` is returned unchanged (``None``) when not provided.
    """
    assert pred.ndim == label.ndim == 2 , \
        f'pred and label should have dimensions of 2, got {pred.ndim} and {label.ndim}'
    assert pred.shape[0] == label.shape[0] , \
        f'pred and label should have the same length of rows, got {pred.shape} and {label.shape}'
    if weight is not None:
        if weight.ndim == 1:
            weight = weight.unsqueeze(-1)
        assert weight.ndim == 2 , f'weight should have dimensions of 2, got {weight.ndim}'
    if dim is None:
        # if pred.shape[-1] != label.shape[-1]:
        #     last_dim = min(pred.shape[-1] , label.shape[-1])
        #     label = label[...,:last_dim]
        #     pred = pred[...,:last_dim]
        if weight is not None:
            weight = weight[...,:label.shape[-1]]
    else:
        assert dim == 0 , f'dim should be 0 when provided, got {dim}'
        assert label.shape[-1] == pred.shape[-1] or label.shape[-1] == 1 , \
            f'the last dimension of label should match pred or is 1 , got label shape {label.shape} and pred shape {pred.shape}'
        if weight is not None:
            assert weight.shape[-1] == pred.shape[-1] or weight.shape[-1] == 1 , f'the last dimension of weight ({weight.shape[-1]}) should match label ({pred.shape[-1]}) or 1'
    return pred , label , weight

class MetricComponent:
    def __init__(
        self , 
        calculator : Callable[...,TensorReturnType] , 
        lamb : float = 1. , 
        which_output : Base.intNums | None = None ,
        which_label : Base.intNums | None = None ,
        **kwargs
    ):
        self.calculator = calculator
        self.lamb = lamb
        self.which_output = which_output
        self.which_label = which_label
        self.kwargs = kwargs

    def __call__(self , **kwargs) -> TensorReturnType:
        kwargs = self.filter_inputs(**kwargs)
        output = self.calculator(**self.kwargs , **kwargs)
        return self.apply_lamb(output)

    def __repr__(self):
        return f'{self.__class__.__name__}(calculator={self.calculator},lamb={self.lamb},which_output={self.which_output},which_label={self.which_label},kwargs={self.kwargs})'

    def apply_lamb(self , output : TensorReturnType) -> TensorReturnType:
        if isinstance(output , Tensor):
            return self.lamb * output
        else:
            return {k:self.lamb * v for k,v in output.items()}

    def filter_inputs(self , which_output : Base.intNums | None = None , which_label : Base.intNums | None = None , **kwargs):
        label , pred , weight = kwargs.get('label' , None) , kwargs.get('pred' , None) , kwargs.get('weight' , None)
        which_output = self.which_output if which_output is None else which_output
        which_label = self.which_label if which_label is None else which_label
        if which_output is not None:
            if pred is not None:
                pred = pred[...,as_int_array(which_output)]
            if weight is not None:
                weight = weight[...,as_int_array(which_output)]
        if which_label is not None:
            if label is not None:
                label = label[...,as_int_array(which_label)]
        if pred is not None and pred.ndim == 1:
            pred = pred[:,None]
        if label is not None and label.ndim == 1:
            label = label[:,None]
        if weight is not None and weight.ndim == 1:
            weight = weight[:,None]
        dim = kwargs.pop('dim' , None)
        if pred is not None and label is not None:
            pred , label , weight = align_shape(pred , label , weight , dim = dim)
        return kwargs | {'pred':pred , 'label':label , 'weight':weight , 'dim':dim}

class LossComponent(MetricComponent):
    def __init__(
        self , 
        calculator : Callable[...,TensorReturnType] , 
        lamb : float = 1. , 
        which_output : Base.intNums | None = None ,
        which_label : Base.intNums | None = None ,
        multilosses : MultiHeadLosses | None = None ,
        **kwargs
    ):
        super().__init__(calculator , lamb , which_output , which_label , **kwargs)
        self.multilosses = multilosses

    def __call__(self , **kwargs) -> TensorReturnType:
        output = super().__call__(**kwargs)
        if self.multilosses:
            assert isinstance(output , Tensor) , f'loss output should be a Tensor when multilosses is applied, but got {output}'
            output = self.multilosses(output , mt_param = kwargs.get('mt_param' , {}))
        return output
        
class AccuracyComponent(MetricComponent):
    def __call__(self , **kwargs) -> TensorReturnType:
        output = super().__call__(**kwargs)
        return output