"""Primitive building-block layers used across all NN model files.

Public API (re-exported from basic.py):
    Pass, Transpose, EwLinear, Parallel

Sub-modules:
    Act       — activation function registry and factory
    Attention — multi-head scaled dot-product attention (with Realformer / LSA options)
    Embed     — reserved for future embedding layer implementations
    Lin       — closed-form OLS residualization layer
    MLP       — multi-layer perceptron
    PE        — positional encoding utilities
    RevIN     — reversible instance normalization
"""
from .basic import *
from . import (
    Attention , Embed , Lin , MLP , PE , RevIN , Act , basic
)
