from scripts.dataset import ModelData
from scripts.my_models import *
import torch
import torch.nn as nn

class TRA(nn.Module):
    """Temporal Routing Adaptor (TRA)

    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
        rho (float): calculate Optimal Transport penalty
    """

    def __init__(self, base_model , input_size , num_states=1, hidden_size=8, tau=1.0, rho = 0.999 , lamb = 0.0 ,
                 horizon = 20 , src_info = 'LR_TPE' , hist_loss_source = None):
        super().__init__()
        self.base_model = base_model
        self.num_states = num_states
        self.tau = tau
        self.rho = rho
        self.lamb = lamb
        self.horizon = horizon
        self.src_info = src_info
        self.hist_loss_source = hist_loss_source

        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size + input_size, num_states)
        self.predictors = nn.Linear(input_size, num_states)
        self.new_model()

    def forward(self, inputs):
        if self.num_states == 1:
            hidden = self.base_model(inputs)
            if isinstance(hidden , (list , tuple)): hidden = hidden[0]
            preds = self.predictors(hidden[:,-1])
            final_pred = preds
            probs = None
        else:
            x , hist_loss = inputs
            hidden = self.base_model(x)
            if isinstance(hidden , (list , tuple)): hidden = hidden[0]
            preds = self.predictors(hidden[:,-1])
        
            # information type
            router_out, _ = self.router(hist_loss[:,:-self.horizon])
            if "LR" in self.src_info:
                latent_representation = hidden[:,-1]
            else:
                latent_representation = torch.randn(hidden[:,-1].shape).to(hidden)
            if "TPE" in self.src_info:
                temporal_pred_error = router_out[:, -1]
            else:
                temporal_pred_error = torch.randn(router_out[:, -1].shape).to(hidden)

            out = self.fc(torch.cat([temporal_pred_error, latent_representation], dim=-1))
            probs = nn.functional.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)

            if self.training:
                final_pred = (preds * probs).sum(dim=-1 , keepdim = True)
            else:
                final_pred = preds[range(len(preds)), probs.argmax(dim=-1)].unsqueeze(-1)

        self.preds = preds
        self.probs = probs
        return final_pred , hidden
    
    def hist_pred(self):
        return self.preds
    
    def new_model(self):
        self.global_steps = 0
        self.preds = None
        self.probs = None
    
    def modifier_inputs(self , inputs , batch_data , ModelData):
        if self.num_states > 1:
            x = batch_data['x']
            i = batch_data['i']
            d = ModelData.buffer['hist_loss']
            rw = ModelData.seqs['hist_loss']
            hist_loss = torch.stack([d[i[:,0],i[:,1]+j+1-rw] for j in range(rw)],dim=-2).nan_to_num(1)
            return x , hist_loss
        else:
            return inputs
    
    def modifier_metric(self , metric , batch_data , ModelData):
        if self.training and self.probs is not None and self.lamb != 0 and self.num_states > 1:
            label = batch_data['y']
            square_error = (self.preds - label).square()
            square_error -= square_error.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
            P = sinkhorn(-square_error, epsilon=0.01)  # sample assignment matrix
            lamb = self.lamb * (self.rho ** self.global_steps)
            reg = self.probs.log().mul(P).sum(dim=-1).mean()
            self.global_steps += 1
            metric['loss'] = metric['loss'] - lamb * reg
            return metric
        else:
            return metric
    
    def modifier_update(self , batch_data , ModelDate):
        if self.num_states > 1 and self.preds is not None:
            i = batch_data['i']
            v = self.preds.detach().to(ModelDate.buffer['hist_preds'])
            ModelDate.buffer['hist_preds'][i[:,0],i[:,1]] = v[:]
            ModelDate.buffer['hist_loss'][i[:,0],i[:,1]] = (v - ModelDate.buffer['hist_labels'][i[:,0],i[:,1]]).square()

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q

def buffer_init(tra_num_states):
    def wrapper(container , *args, **kwargs):
        buffer = dict()
        if tra_num_states > 1:
            hist_loss_shape = list(container.y.shape)
            hist_loss_shape[2] = tra_num_states
            buffer['hist_labels'] = container.y
            buffer['hist_preds'] = torch.randn(hist_loss_shape)
            buffer['hist_loss']  = (buffer['hist_preds'] - buffer['hist_labels']).square()
        return buffer
    return wrapper

def buffer_process(tra_num_states):
    def wrapper(container , *args, **kwargs):
        buffer = dict()
        if tra_num_states > 1:
            buffer['hist_loss']  = (container.buffer['hist_preds'] - container.buffer['hist_labels']).square()
        return buffer
    return wrapper