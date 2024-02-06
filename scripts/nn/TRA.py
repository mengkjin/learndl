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

    def __init__(self, base_model , base_dim , num_states=1, horizon = 20 , 
                 hidden_size = 8, tau=1.0, src_info = 'LR_TPE'):
        super().__init__()
        self.base_model = base_model
        self.num_states = num_states
        self.horizon = horizon
        self.tau = tau
        self.src_info = src_info
        self.probs_record = None

        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size + base_dim, num_states)
        self.predictors = nn.Linear(base_dim, num_states)
        self.global_steps = 0

    def forward(self, inputs):
        x , hist_loss = (inputs , None) if self.num_states == 1 else inputs
        hidden = self.base_model(x)
        # in-case multiple output of base_model
        if isinstance(hidden , (list , tuple)): hidden = hidden[0]
        # in-case time-series output of base_model , use the latest one
        if hidden.dim() == 3: hidden = hidden[:,-1]
        preds = self.predictors(hidden)

        if self.num_states == 1:
            final_pred = preds
            probs = None
        else:
            # information type
            router_out, _ = self.router(hist_loss[:,:-self.horizon])
            if "LR" in self.src_info:
                latent_representation = hidden
            else:
                latent_representation = torch.randn(hidden.shape).to(hidden)
            if "TPE" in self.src_info:
                temporal_pred_error = router_out[:, -1]
            else:
                temporal_pred_error = torch.randn(router_out[:, -1].shape).to(hidden)

            # print(hidden.shape , preds.shape , temporal_pred_error.shape , latent_representation.shape)
            out = self.fc(torch.cat([temporal_pred_error, latent_representation], dim=-1))
            probs = nn.functional.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)

            if self.training:
                final_pred = (preds * probs).sum(dim=-1 , keepdim = True)
            else:
                final_pred = preds[range(len(preds)), probs.argmax(dim=-1)].unsqueeze(-1)

        self.preds = preds
        self.probs = probs
        probs_sum  = probs.detach().sum(dim = 0 , keepdim = True)
        self.probs_record = probs_sum if self.probs_record is None else torch.concat([self.probs_record , probs_sum])
        return final_pred , preds
    
    def get_probs(self):
        return self.probs_record / self.probs_record.sum(dim=1 , keepdim = True)
    
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
        return metric

    def modifier_update(self , update , batch_data , ModelDate):
        if self.num_states > 1 and self.preds is not None:
            i = batch_data['i']
            v = self.preds.detach().to(ModelDate.buffer['hist_preds'])
            ModelDate.buffer['hist_preds'][i[:,0],i[:,1]] = v[:]
            ModelDate.buffer['hist_loss'][i[:,0],i[:,1]] = (v - ModelDate.buffer['hist_labels'][i[:,0],i[:,1]]).square()
            del self.preds , self.probs

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
