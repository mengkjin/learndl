"""Compare trained models without rerunning factor/portfolio backtests.

Example::

    result = compare_models(['models/nn/gru@a', 'models/nn/gru@b'])
    result = compare_models([CompareModelSpec('a.xlsx', name='A'),
                             CompareModelSpec('b.xlsx', name='B')],
                            config=CompareConfig(start=20200101, analyze_pred=False))
"""
from .types import CompareConfig, CompareModelSpec, CompareResult
from .engine import compare_models

__all__ = ['CompareConfig', 'CompareModelSpec', 'CompareResult', 'compare_models']
