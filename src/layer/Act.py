import torch
from torch import nn

def get_activation_fn(activation) -> nn.Module:
    if callable(activation): return activation()
    elif activation.lower() == 'relu': return nn.ReLU()
    elif activation.lower() == 'gelu': return nn.GELU()
    elif activation.lower() == 'leaky': return nn.LeakyReLU()
    raise ValueError(f'{activation} is not available')