import torch

from torch import nn , Tensor

class tra_module:
    '''
    Decorator to grant a module dynamic data assign, and output probs (tra feature)
    '''
    def __call__(self, original_class):
        class new_tra_class(original_class):
            def __init__(self , *args , **kwargs):
                super().__init__(*args , **kwargs)
                self.dynamic_data_assigned = False
            def __call__(self , *args , **kwargs):
                assert self.dynamic_data_assigned , f'Run dynamic_data_assign first'
                self.dynamic_data_assigned = False
                return super().__call__(*args , **kwargs)
            def dynamic_data_assign(self , batch_data , model_data , **kwargs):
                if not hasattr(self, 'dynamic_data'): self.dynamic_data = {}
                self.dynamic_data.update({'model_data':model_data,'batch_data':batch_data,**kwargs})
                self.dynamic_data_assigned = True
                return 
            def dynamic_data_access(self , *args , **kwargs):
                if not hasattr(self, 'dynamic_data'): self.dynamic_data = {}
                return self.dynamic_data
            def penalty_data_access(self , *args , **kwargs):
                self.global_steps += 1
                return {'probs':self.probs,'num_states':self.num_states,'global_steps':self.global_steps,}
            def dynamic_data_unlink(self , *args , **kwargs):
                if hasattr(self, 'dynamic_data'): del self.dynamic_data
                return self
            def get_probs(self):
                if self.probs_record is not None: return self.probs_record / self.probs_record.sum(dim=1,keepdim=True)   
        return new_tra_class

class tra_component:
    '''
    Decorator to identify a component of a module as tra components, apply pipeline dynamic data assign, and output probs (tra feature)
    '''
    def __init__(self, *args):
        self.tra_component_list = args
    def __call__(self, original_class):
        tra_component_list = self.tra_component_list
        class new_tra_class(original_class):
            def dynamic_data_assign(self , *args , **kwargs):
                [getattr(self , comp).dynamic_data_assign(*args , **kwargs) for comp in tra_component_list]
            def dynamic_data_access(self , *args , **kwargs):
                dynamic_data = [getattr(self , comp).dynamic_data_access(*args , **kwargs) for comp in tra_component_list]
                return dynamic_data if len(dynamic_data) > 1 else dynamic_data[0]
            def dynamic_data_unlink(self , *args , **kwargs):
                for comp in tra_component_list: getattr(self , comp).dynamic_data_unlink(*args , **kwargs)
                return self
            def penalty_data_access(self , *args , **kwargs):
                penalty_data = [getattr(self , comp).penalty_data_access(*args , **kwargs) for comp in tra_component_list]
                return penalty_data if len(penalty_data) > 1 else penalty_data[0]
            def get_probs(self , *args , **kwargs):
                probs = [getattr(self,comp).get_probs(*args , **kwargs) for comp in tra_component_list]
                return probs if len(probs) > 1 else probs[0]
        return new_tra_class
    
class mod_tra(nn.Module):
    '''
    Temporal Routing Adaptor (TRA)

    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
        rho (float): calculate Optimal Transport penalty
    '''

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

    def forward(self , inputs : Tensor | tuple[Tensor , Tensor]) -> tuple[Tensor , Tensor]:
        if self.num_states == 1:
            x , hist_loss = inputs , None
        else:
            x , hist_loss = inputs

        hidden = self.base_model(x)
        # in-case multiple output of base_model
        if isinstance(hidden , (list , tuple)): hidden = hidden[0]
        # in-case time-series output of base_model , use the latest one
        if hidden.dim() == 3: hidden = hidden[:,-1]
        preds = self.predictors(hidden)

        if self.num_states == 1 or hist_loss is None:
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
            probs_sum  = probs.detach().sum(dim = 0 , keepdim = True)
            self.probs_record = probs_sum if self.probs_record is None else torch.concat([self.probs_record , probs_sum])

        self.preds = preds
        self.probs = probs
        
        return final_pred , preds
    
    def get_probs(self):
        if self.probs_record is not None:
            return self.probs_record / self.probs_record.sum(dim=1 , keepdim = True)
    
    def modifier_inputs(self , x : Tensor , batch_data , model_data) -> Tensor | tuple[Tensor , Tensor]:
        if self.num_states > 1:
            x = batch_data.x
            i = batch_data.i
            d = model_data.buffer['hist_loss']
            rw = model_data.seqs['hist_loss']
            hist_loss = torch.stack([d[i[:,0],i[:,1]+j+1-rw] for j in range(rw)],dim=-2).nan_to_num(1)
            return x , hist_loss
        else:
            return x
    
    def modifier_metric(self , metric , batch_data , model_data):
        return metric

    def modifier_update(self , update , batch_data , model_data) -> None:
        if self.num_states > 1 and self.preds is not None:
            i = batch_data.i
            v = self.preds.detach().to(model_data.buffer['hist_preds'])
            model_data.buffer['hist_preds'][i[:,0],i[:,1]] = v[:]
            model_data.buffer['hist_loss'][i[:,0],i[:,1]] = (v - model_data.buffer['hist_labels'][i[:,0],i[:,1]]).square()
            del self.preds , self.probs

@tra_module()
class block_tra(nn.Module):
    """
    Temporal Routing Adaptor (TRA) mapping segment
    """

    def __init__(self, hidden_dim , tra_dim = 8 , num_states = 1, horizon = 20 , 
                 tau=1.0, src_info = 'LR_TPE'):
        super().__init__()
        self.num_states = num_states
        self.global_steps = -1
        self.horizon = horizon
        self.tau = tau
        self.src_info = src_info
        self.probs_record = None

        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=tra_dim,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim + tra_dim, num_states)
        self.predictors = nn.Linear(hidden_dim, num_states)

    def forward(self , x : Tensor) -> tuple[Tensor , Tensor]:
        if self.num_states > 1:
            dynamic_data : dict = getattr(self , 'dynamic_data')
            i0 , i1 = dynamic_data['batch_data'].i[:,0] , dynamic_data['batch_data'].i[:,1]
            d = dynamic_data['model_data'].buffer['hist_loss']
            rw = dynamic_data['model_data'].seqs['hist_loss']
            hist_loss = torch.stack([d[i0 , i1+j+1-rw] for j in range(rw)],dim=-2).nan_to_num(1)
            preds = self.predictors(x)

            # information type
            router_out, _ = self.router(hist_loss[:,:-self.horizon])
            if "LR" in self.src_info:
                latent_representation = x
            else:
                latent_representation = torch.randn(x.shape).to(x)
            if "TPE" in self.src_info:
                temporal_pred_error = router_out[:, -1]
            else:
                temporal_pred_error = torch.randn(router_out[:, -1].shape).to(x)

            # print(x.shape , preds.shape , latent_representation.shape) , temporal_pred_error.shape
            probs = self.fc(torch.cat([latent_representation , temporal_pred_error], dim=-1))
            probs = nn.functional.gumbel_softmax(probs, dim=-1, tau=self.tau, hard=False)

            # get final prediction in either train (weighted sum) or eval (max probability)
            if self.training:
                final_pred = (preds * probs).sum(dim=-1 , keepdim = True)
            else:
                final_pred = preds[range(len(preds)), probs.argmax(dim=-1)].unsqueeze(-1)

            # record training history probs
            probs_agg  = probs.detach().sum(dim = 0 , keepdim = True)

            self.probs = probs.detach()
            self.probs_record = probs_agg if self.probs_record is None else torch.concat([self.probs_record , probs_agg])

            # update dynamic buffer
            vp = preds.detach().to(dynamic_data['model_data'].buffer['hist_preds'])
            v0 = dynamic_data['model_data'].buffer['hist_labels'][i0,i1].nan_to_num(0)
            dynamic_data['model_data'].buffer['hist_preds'][i0,i1] = vp
            dynamic_data['model_data'].buffer['hist_loss'][i0,i1] = (vp - v0).square()
        else: 
            final_pred = preds = self.predictors(x)
            
        return final_pred , preds

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
