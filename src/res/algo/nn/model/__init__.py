"""NN model implementations.

Each sub-module defines one or more model architectures that are registered
in ``nn/api.py:AVAILABLE_NNS``.

Sub-modules:
    Attention  — Transformer building blocks (mod_transformer, SampleWiseTranformer, ...)
    CNN        — Convolutional blocks (TCN, ResNet-1D, ResNet-2D)
    RNN        — RNN-based models (gru, lstm, rnn_general, etc.)
    PatchTST   — Patch-based Time Series Transformer
    TSMixer    — Time-Series MLP-Mixer
    ModernTCN  — Modern Temporal Convolutional Network
    TRA        — Temporal Routing Adaptor (multi-state)
    FactorVAE  — VAE-based factor model
    PLE        — Progressive Layered Extraction multi-task GRU
    MoE        — Mixture-of-Experts GRU with market-aware expert
    RiskAttGRU — GRU with risk-factor cross-attention
    TFT        — Temporal Fusion Transformer
    ABCM       — Alpha-Beta Co-Mining GRU
    ResNetGru  — ResNet-GRU
"""
from . import (
    CNN, TFT , ModernTCN , PatchTST , PLE , RNN , RiskAttGRU , MoE ,
    TSMixer , TRA , FactorVAE ,
    Attention , ABCM , resnet_gru
)