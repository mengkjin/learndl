"""Activation function registry and factory.

Supported string keys: 'relu', 'gelu', 'leaky' (LeakyReLU), 'softplus'.
"""
from torch import nn

# Mapping from lower-case string key to nn.Module constructor.
# Supported keys: 'relu', 'gelu', 'leaky' (LeakyReLU), 'softplus'.
_ACT = {
    'relu': nn.ReLU ,
    'gelu': nn.GELU ,
    'leaky': nn.LeakyReLU ,
    'softplus': nn.Softplus ,
}

def get_activation_fn(activation) -> nn.Module:
    """Return an instantiated activation nn.Module.

    Args:
        activation: Either a string key (case-insensitive) from the ``_ACT``
            registry (``'relu'``, ``'gelu'``, ``'leaky'``, ``'softplus'``),
            or a zero-argument callable that returns an ``nn.Module`` instance
            (e.g. ``lambda: nn.ReLU()``).

    Returns:
        An instantiated ``nn.Module`` activation function.

    Raises:
        AssertionError: If a callable is given but does not produce an
            ``nn.Module``, or if a non-string is given.
        KeyError: If the string key is not in the registry.
    """
    if callable(activation):
        act_fn = activation()
        assert isinstance(act_fn , nn.Module) , act_fn
        return act_fn
    assert isinstance(activation , str) , activation
    return _ACT[activation.lower()]()