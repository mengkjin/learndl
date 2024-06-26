import torch
from torch import nn , Tensor
from typing import Optional

from .Recurrent import get_rnn_mod

class block_tra(nn.Module):
    '''Temporal Routing Adaptor (TRA) mapping segment'''
    def __init__(self, hidden_dim , tra_dim = 8 , num_states = 1, hist_loss_seq_len = 60 , horizon = 20 , 
                 tau=1.0, src_info = 'LR_TPE' , gamma = 0.01 , rho = 0.999 , **kwargs):
        super().__init__()
        self.num_states = num_states
        self.global_steps = -1
        self.hist_loss_seq_len = hist_loss_seq_len
        self.horizon = horizon
        self.tau = tau
        self.src_info = src_info
        self.probs_record = None
        self.gamma = gamma 
        self.rho = rho

        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=tra_dim,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim + tra_dim, num_states)
        self.predictors = nn.Linear(hidden_dim, num_states)
    
    def forward(self , x : Tensor , hist_loss : Optional[Tensor] = None , y : Optional[Tensor] = None) -> tuple[Tensor , dict]:
        if self.num_states > 1:
            assert hist_loss is not None and y is not None

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

            # print(x.shape , preds.shape , latent_representation.shape, temporal_pred_error.shape)
            probs = self.fc(torch.cat([latent_representation , temporal_pred_error], dim=-1))
            if probs.isnan().any():
                print(preds , x)
                print(probs)
                print(latent_representation , temporal_pred_error)
                from src import api
                setattr(api , 'net' , self)
                setattr(api , 'x' , x)
                setattr(api , 'hist_loss' , hist_loss)

                raise ValueError

            probs = nn.functional.gumbel_softmax(probs, dim=-1, tau=self.tau, hard=False)
            if probs.isnan().any():
                print(probs)
                from src import api
                setattr(api , 'net' , self)
                setattr(api , 'x' , x)
                setattr(api , 'hist_loss' , hist_loss)
                raise ValueError
            
            # get final prediction in either train (weighted sum) or eval (max probability)
            if self.training:
                final_pred = (preds * probs).sum(dim=-1 , keepdim = True)
            else:
                final_pred = preds[range(len(preds)), probs.argmax(dim=-1)].unsqueeze(-1)

            # record training history probs
            probs_agg  = probs.detach().sum(dim = 0 , keepdim = True)

            self.probs = probs
            self.probs_record = probs_agg if self.probs_record is None else torch.concat([self.probs_record , probs_agg])
        else: 
            self.probs = None
            final_pred = preds = self.predictors(x)
        if self.training and self.probs is not None and self.num_states > 1 and y is not None:
            loss_opt_transport = self.loss_opt_transport(preds , y)
        else:
            loss_opt_transport = torch.Tensor([0])
            
        return final_pred , {'loss_opt_transport' : loss_opt_transport , 'hidden': preds , 'preds': preds}
    
    def loss_opt_transport(self , preds : Tensor , label : Tensor) -> Tensor:
        '''special penalty for tra'''
        assert self.probs is not None
        self.global_steps += 1
        square_error = (preds - label).square()
        min_se = square_error.min(dim=-1, keepdim=True).values
        square_error = square_error - min_se + 1e-6  # normalize & ensure positive input
        P = sinkhorn(-square_error, epsilon=0.01)  # sample assignment matrix
        lamb = self.gamma * (self.rho ** self.global_steps)
        reg = (self.probs + 1e-4).log().mul(P).sum(dim=-1).mean()

        loss = - lamb * reg
        if loss.isnan().any():
            print(label.isnan().any())
            print(square_error)
            print(lamb , self.gamma , self.rho , self.global_steps)
            print(self.probs.isnan().any())
            print((self.probs + 1e-4).log().isnan().any())
            from src import api
            setattr(api , 'net' , self)
            raise ValueError
        return loss

    @property
    def get_probs(self):
        if self.probs_record is not None: return self.probs_record / self.probs_record.sum(dim=1,keepdim=True)  

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    valid = torch.isfinite(inp_tensor)

    if ~valid.all():
        m = torch.max(inp_tensor[valid])
        inp_tensor[~valid] = m

    return inp_tensor

def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for _ in range(n_iters):
            Q = (Q / Q.sum(dim=0, keepdim=True)).nan_to_num_(0)
            Q = (Q / Q.sum(dim=1, keepdim=True)).nan_to_num_(0)
    return Q

class tra(nn.Module):
    def __init__(self , input_dim , hidden_dim , rnn_type = 'lstm' , rnn_layers = 2 , 
                 num_states=1, hist_loss_seq_len = 60 , hist_loss_horizon = 20 , **kwargs):
        super().__init__()
        self.num_states = num_states
        self.hist_loss_seq_len = hist_loss_seq_len
        self.hist_loss_horizon = hist_loss_horizon
        self.rnn = get_rnn_mod(rnn_type)(input_dim , hidden_dim , num_layers = rnn_layers , dropout = 0)
        self.tra_mapping = block_tra(hidden_dim , num_states = num_states, horizon=hist_loss_horizon , **kwargs)

    def forward(self, x : Tensor , **kwargs) -> tuple[Tensor , dict]:
        x = self.rnn(x)[:,-1] # [bs x hidden_dim]

        o , h = self.tra_mapping(x , **kwargs) # output.shape : (bat_size, num_output)   
        return o , h