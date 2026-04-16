"""Simple multi-layer perceptron."""
from torch import nn
from .Act import get_activation_fn

class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers.

    Two construction modes depending on the type of ``hidden_size``:

    * **Scalar** (``int``): builds a two-layer network::

          Linear(input → hidden) → activation → Linear(hidden → output) → out_activation

    * **List** (``list[int]``): builds a deeper network with one hidden layer
      per element.  The inter-layer activations are **not** added in this path
      (see TODO); only ``out_activation`` is applied at the end, making the
      network effectively linear for depth > 1.  This is a known limitation.

    Args:
        input_size:     Number of input features.
        output_size:    Number of output features.
        hidden_size:    Hidden dimension(s).  Scalar → single hidden layer;
                        list → one hidden layer per element.
        activation:     Activation used between input and hidden layers
                        (scalar mode only).  String key or callable factory.
        out_activation: Activation applied after the final linear layer.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: list[int] | int = 32,
        activation = 'leaky',
        out_activation= 'leaky',
    ):
        super().__init__()
        if isinstance(hidden_size , list):
            num_hidden_layer = len(hidden_size)
            self.net = nn.Sequential()
            self.net.add_module('input', nn.Linear(input_size, hidden_size[0]))
            for i in range(num_hidden_layer - 1):
                self.net.add_module(f'hidden_{i}', nn.Linear(hidden_size[i], hidden_size[i + 1]))
                self.net.add_module(f'hidden_{i}_activation', get_activation_fn(activation))
            self.net.add_module('out', nn.Linear(hidden_size[num_hidden_layer - 1], output_size))
            self.net.add_module('out_activ', get_activation_fn(out_activation))
        else:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                get_activation_fn(activation),
                nn.Linear(hidden_size, output_size),
                get_activation_fn(out_activation)
            )

    def forward(self, x):
        return self.net(x)
