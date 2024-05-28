import torch

from torch import nn , Tensor

from .rnn import rnn_univariate
class tra_module:
    '''Decorator to grant a module dynamic data assign, and output probs (tra feature)'''
    def __call__(self, original_class):
        class new_tra_class(original_class):
            def __init__(self , *args , **kwargs):
                super().__init__(*args , **kwargs)
            def __call__(self , *args , **kwargs):
                assert self.dynamic_data_assigned , f'Run dynamic_data_assign first'
                return super().__call__(*args , **kwargs)
            def dynamic_data_assign(self , obj):
                if not self.dynamic_data_assigned: self._dynamic = {'data_module':obj.data_module,'batch_data':obj.batch_data}
                return self
            def dynamic_data_unlink(self):
                if self.dynamic_data_assigned: del self._dynamic
                return self
            @property
            def dynamic_data_assigned(self): return hasattr(self , '_dynamic_data')
            @property
            def dynamic_data(self):
                # if not hasattr(self, 'dynamic_data'): self.dynamic_data = {}
                assert self.dynamic_data_assigned , f'Run dynamic_data_assign first'
                return self.dynamic_data
            @property
            def penalty_data(self):
                self.global_steps += 1
                return {'probs':self.probs,'num_states':self.num_states,'global_steps':self.global_steps,}
            @property
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
            def dynamic_data_assign(self , obj):
                [getattr(self , comp).dynamic_data_assign(obj) for comp in tra_component_list]
            def dynamic_data_unlink(self):
                [getattr(self , comp).dynamic_data_unlink() for comp in tra_component_list]
                return self
            @property
            def dynamic_data(self):
                v = [getattr(self , comp).dynamic_data for comp in tra_component_list]
                return v if len(v) > 1 else v[0]
            @property
            def penalty_data(self):
                v = [getattr(self , comp).penalty_data for comp in tra_component_list]
                return v if len(v) > 1 else v[0]
            @property
            def get_probs(self):
                v = [getattr(self,comp).get_probs for comp in tra_component_list]
                return v if len(v) > 1 else v[0]
        return new_tra_class
    
@tra_module()
class block_tra(nn.Module):
    '''Temporal Routing Adaptor (TRA) mapping segment'''
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
            i0 , i1 = dynamic_data['data_module'].i[:,0] , dynamic_data['batch_data'].i[:,1]
            d  = dynamic_data['data_module'].buffer['hist_loss']
            rw = dynamic_data['data_module'].seqs['hist_loss']
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
            vp = preds.detach().to(dynamic_data['data_module'].buffer['hist_preds'])
            v0 = dynamic_data['data_module'].buffer['hist_labels'][i0,i1].nan_to_num(0)
            dynamic_data['data_module'].buffer['hist_preds'][i0,i1] = vp
            dynamic_data['data_module'].buffer['hist_loss'][i0,i1] = (vp - v0).square()
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

@tra_component('mapping')
class tra_lstm(rnn_univariate):
    def __init__(self , input_dim , hidden_dim , tra_num_states=1, tra_horizon = 20 ,rnn_type = 'lstm' , num_output = 1 , **kwargs):
        super().__init__(input_dim , hidden_dim , rnn_type = 'lstm' , num_output=1 , **kwargs)
        self.mapping = block_tra(hidden_dim , num_states = tra_num_states, horizon = tra_horizon)
        self.set_multiloss_params()

    def forward(self, inputs) -> tuple[Tensor , dict]:
        # inputs.shape : (bat_size, seq, input_dim)
        hidden = self.encoder(inputs) # hidden.shape : (bat_size, hidden_dim)
        hidden = self.decoder(hidden) # hidden.shape : tuple of (bat_size, hidden_dim) , len is num_output
        if isinstance(hidden , tuple): hidden = hidden[0]
        output , hidden = self.mapping(hidden) # output.shape : (bat_size, num_output)   
        return output , {'hidden' : hidden}