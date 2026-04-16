"""PLE: Progressive Layered Extraction multi-task GRU.

Reference: Tang et al. (2021) "Progressive Layered Extraction (PLE):
A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations."
"""
import torch
import torch.nn as nn

from .. import layer as Layer
from ..loss import MultiHeadLosses

__all__ = ['ple_gru']

class ple_gru(nn.Module):
    """Progressive Layered Extraction multi-task GRU.  Registry key: ``'ple_gru'``.

    Stacks ``expert_layers`` PLE layers, each with one shared expert and
    ``num_output`` task-specific experts.  Each layer uses gated attention
    to combine its experts.  A separate prediction head per task produces the
    final scalar output.  A learnable ``multiloss_alpha`` parameter is injected
    for multi-task loss weighting.

    Args:
        input_dim:     Input feature dimension (default ``6``).
        hidden_dim:    GRU hidden dimension (default ``32``).
        dropout:       Dropout rate (default ``0.1``).
        act_type:      Activation key for the prediction head (default
                       ``'leaky'``).
        expert_layers: Number of PLE layers, must be ``>= 2`` (default ``2``).
        rnn_layers:    Number of GRU layers per expert (default ``2``).
        num_output:    Number of tasks / output heads (must be ``2 <= n <= 4``).

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``[bs, num_output]``
    """
    def __init__(
            self, 
            input_dim    = 6 ,
            hidden_dim   = 2**5,
            dropout      = 0.1,
            act_type     = 'leaky',
            expert_layers = 2 ,
            rnn_layers   = 2 ,
            num_output   = 2 ,
            **kwargs) -> None:
        super().__init__()
        assert num_output > 1 and num_output < 5 , num_output
        assert expert_layers >= 2 , expert_layers
        
        self.expert_layers = expert_layers
        self.num_output = num_output

        layers = []
        for i in range(expert_layers):
            layers.append(ExpertLayer(
                input_dim = input_dim , 
                hidden_dim = hidden_dim , 
                first_layer = i == 0 , 
                dropout = dropout , 
                rnn_layers = rnn_layers , 
                num_output = num_output))
        self.layers = nn.ModuleList(layers)

        self.heads = nn.ModuleList([
            nn.Sequential(
                Layer.Act.get_activation_fn(act_type),
                nn.Linear(hidden_dim , 1) , 
                nn.BatchNorm1d(1)
            ) for _ in range(num_output)
        ])

        MultiHeadLosses.add_params(self , num_output)

    def forward(self , x):
        shared_output , task_outputs = None , []
        for layer in self.layers:
            shared_output , task_outputs = layer(x , shared_output , task_outputs)
        z = torch.concat([head(output[:,-1]) for head , output in zip(self.heads , task_outputs)] , dim = -1)
        return z
    
class ExpertLayer(nn.Module):
    """Single PLE layer with one shared and ``num_output`` task-specific experts.

    For the first layer (``first_layer=True``), all experts receive the raw
    input ``x``.  For subsequent layers, experts receive the shared/task
    output vectors from the previous layer.

    Args:
        input_dim:    Input feature dimension (used for raw input expert inputs
                      in the first layer).
        hidden_dim:   GRU expert output dimension.
        first_layer:  Whether this is the first PLE layer.
        dropout:      Dropout rate.
        rnn_layers:   Number of GRU layers per expert.
        num_output:   Number of task-specific expert branches.

    Returns:
        ``(shared_output, [task_output_1, ..., task_output_num_output])``
        where each tensor has shape ``[bs, seq_len, hidden_dim]``.
    """
    def __init__(
            self, 
            input_dim    = 6 ,
            hidden_dim   = 2**5,
            first_layer  = True ,
            dropout      = 0.1,
            rnn_layers   = 2 ,
            num_output   = 2 ,
            **kwargs) -> None:
        super().__init__()
        self.first_layer = first_layer
        expert_dim = input_dim if first_layer else hidden_dim

        self.shared_expert = ExpertNetwork(expert_dim , hidden_dim , rnn_layers , dropout)
        self.task_experts = nn.ModuleList([ExpertNetwork(expert_dim , hidden_dim , rnn_layers , dropout) for i in range(num_output)])
        self.shared_gate = GatingNetwork(input_dim , 1 + num_output , rnn_layers , dropout)
        self.task_gates = nn.ModuleList([GatingNetwork(input_dim , 2 , rnn_layers , dropout) for i in range(num_output)])
        
    def forward(self , x , *vecs):
        if self.first_layer:
            shared_vector = self.shared_expert(x)
            task_vectors  = [expert(x) for expert in self.task_experts]
        else:
            shared_input , tasks_inputs = vecs
            shared_vector = self.shared_expert(shared_input)
            task_vectors  = [expert(input) for expert , input in zip(self.task_experts , tasks_inputs)]

        shared_output = self.shared_gate(x , shared_vector , *task_vectors)
        task_outputs  = [gate(x , shared_vector , vector) for gate , vector  in zip(self.task_gates , task_vectors)]
        return shared_output , task_outputs
    
class GatingNetwork(nn.Module):
    """GRU-based gating network for expert selection.

    Processes the raw input ``x`` with a GRU to produce ``feature_dim``
    softmax attention weights, then computes a weighted sum over the provided
    expert output vectors.

    Args:
        selector_dim: Input feature dimension for the GRU (raw ``x``).
        feature_dim:  Number of experts to select from (``1 + num_output``
                      for shared gate, ``2`` for task gates).
        num_layers:   GRU layers.
        dropout:      Dropout rate.

    Shapes (forward):
        x:     ``[bs, seq_len, selector_dim]``
        *vecs: ``feature_dim`` tensors of shape ``[bs, seq_len, hidden_dim]``
        Output: weighted sum ``[bs, seq_len, hidden_dim]``
    """
    def __init__(
            self, 
            selector_dim = 6 ,
            feature_dim  = 2 ,
            num_layers = 1 , 
            dropout = 0.1 ,
            **kwargs) -> None:
        super().__init__()
        self.selector = nn.GRU(selector_dim , feature_dim , num_layers , batch_first=True , dropout=dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self , x , *vecs):
        v = self.selector(x)[0][:,-1]
        v = self.softmax(v)
        v = torch.stack([v[:,i].reshape(-1,1,1) * vec for i , vec in enumerate(vecs)] , dim = 0).sum(0)
        return v
    
class ExpertNetwork(nn.Module):
    """Single GRU expert that returns the full output sequence.

    Args:
        input_dim:  Input feature dimension.
        hidden_dim: GRU hidden/output dimension.
        num_layers: Number of GRU layers.
        dropout:    Dropout rate.

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``[bs, seq_len, hidden_dim]``
    """
    def __init__(
            self, 
            input_dim = 6 ,
            hidden_dim  = 2**5 ,
            num_layers = 1 , 
            dropout = 0.1 ,
            **kwargs) -> None:
        super().__init__()
        self.expert = nn.GRU(input_dim , hidden_dim , num_layers , batch_first=True , dropout=dropout)

    def forward(self , x):
        return self.expert(x)[0]
    
if __name__ == '__main__' :
    from src.res.model.data_module import get_realistic_batch_data
    batch_input = get_realistic_batch_data('day')

    rau = ple_gru(indus_embed=True)
    rau(batch_input.x).shape