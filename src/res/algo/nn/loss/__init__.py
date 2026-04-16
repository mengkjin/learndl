"""Loss function and accuracy metric registries for NN training.

Public exports:
    MultiHeadLosses — multi-task loss combination orchestrator
    Accuracy        — accuracy metric factory (higher is better)
    Loss            — loss function factory (lower is better)
"""
from .multiloss import MultiHeadLosses
from .accuracy import Accuracy
from .loss import Loss