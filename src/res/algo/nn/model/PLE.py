import torch
import torch.nn as nn

from .. import layer as Layer
from ..loss import MultiHeadLosses

__all__ = ['ple_gru']

class ple_gru(nn.Module):
    '''
    Progressive Layered Extraction
    '''
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