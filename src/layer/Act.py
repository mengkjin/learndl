import torch
import torch.nn as nn

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == 'relu': return nn.ReLU()
    elif activation.lower() == 'gelu': return nn.GELU()
    elif activation.lower() == 'Leaky': return nn.LeakyReLU()
    raise ValueError(f'{activation} is not available')