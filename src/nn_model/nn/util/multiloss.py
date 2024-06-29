import torch
from typing import Any

def add_multiloss_params(module : torch.nn.Module , num_of_heads : int):
    if num_of_heads > 1:
        module.multiloss_alpha = torch.nn.Parameter((torch.ones(num_of_heads) + 1e-4).requires_grad_())
    
def get_multiloss_params(module : torch.nn.Module | Any):
    if hasattr(module , 'multiloss_alpha'):
        return {'alpha':module.multiloss_alpha}
    else:
        return {}