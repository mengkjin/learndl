from torch import nn

_ACT = {
    'relu': nn.ReLU , 
    'gelu': nn.GELU ,
    'leaky': nn.LeakyReLU ,
    'softplus': nn.Softplus ,
}

def get_activation_fn(activation) -> nn.Module:
    if callable(activation): return activation()
    assert isinstance(activation , str) , activation
    return _ACT[activation.lower()]()