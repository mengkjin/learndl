"""TRA: Temporal Routing Adaptor for multi-state stock prediction.

Reference: Ye et al. (2021) "Temporal Routing Adaptor for Dynamic Scene
Understanding."
"""
import torch
from torch import nn , Tensor

from src.proj import Logger
from .RNN import get_rnn_mod

class block_tra(nn.Module):
    """Temporal Routing Adaptor (TRA) mapping head.

    Maintains ``num_states`` parallel predictors and an LSTM-based router that
    assigns each sample to a state using Gumbel-Softmax.  When ``num_states=1``
    it degenerates to a single predictor (no routing).

    Args:
        hidden_dim:       Input hidden representation dimension.
        tra_dim:          LSTM router hidden dimension (default ``8``).
        num_states:       Number of parallel predictor states (default ``1``).
        hist_loss_seq_len: Length of the historical loss sequence fed to the
                          router (default ``60``).
        horizon:          Number of future time steps to skip in the router
                          input (default ``20``).
        tau:              Gumbel-Softmax temperature (default ``1.0``).
        src_info:         Information sources for the router — combination of
                          ``'LR'`` (latent representation) and ``'TPE'``
                          (temporal prediction error).
        gamma:            Initial Sinkhorn regularization coefficient.
        rho:              Decay rate for ``gamma`` (``gamma * rho^step``).

    Forward args (beyond ``x``):
        hist_loss: Historical per-state loss sequence ``[bs, hist_loss_seq_len, num_states]``.
                   Required when ``num_states > 1``.
        y:         Current target ``[bs, 1]``.  Required during training when
                   ``num_states > 1``.

    Returns:
        ``(final_pred, {'loss_opt_transport': ..., 'hidden': preds, 'preds': preds})``
    """
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
    
    def forward(self , x : Tensor , hist_loss : Tensor | None = None , y : Tensor | None = None) -> tuple[Tensor , dict]:
        if self.num_states > 1:
            assert hist_loss is not None and y is not None , \
                f'{self.__class__.__name__} hist_loss or y are None'

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

            # Logger.stdout(x.shape , preds.shape , latent_representation.shape, temporal_pred_error.shape)
            probs = self.fc(torch.cat([latent_representation , temporal_pred_error], dim=-1))
            if probs.isnan().any():
                Logger.stdout(preds , x)
                Logger.stdout(probs)
                Logger.stdout(latent_representation , temporal_pred_error)
                from src import api
                setattr(api , 'net' , self)
                setattr(api , 'x' , x)
                setattr(api , 'hist_loss' , hist_loss)

                raise ValueError

            probs = nn.functional.gumbel_softmax(probs, dim=-1, tau=self.tau, hard=False)
            if probs.isnan().any():
                Logger.stdout(probs)
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
        """Sinkhorn optimal transport regularization loss for TRA.

        Penalizes the negative log-likelihood of the router assignment under
        the optimal transport plan.  The regularization coefficient decays as
        ``gamma * rho^global_steps``, allowing gradual reduction of the
        routing constraint over training.

        Args:
            preds:  Per-state predictions ``[bs, num_states]``.
            label:  Ground-truth returns ``[bs, 1]``.

        Returns:
            Scalar loss tensor (negative, to be *minimized*).
        """
        assert self.probs is not None , f'{self.__class__.__name__} probs are None'
        self.global_steps += 1
        square_error = (preds - label).square()
        min_se = square_error.min(dim=-1, keepdim=True).values
        square_error = square_error - min_se + 1e-6  # normalize & ensure positive input
        P = sinkhorn(-square_error, epsilon=0.01)  # sample assignment matrix
        lamb = self.gamma * (self.rho ** self.global_steps)
        reg = (self.probs + 1e-4).log().mul(P).sum(dim=-1).mean()

        loss = - lamb * reg
        if loss.isnan().any():
            Logger.stdout(int(label.isnan().any()))
            Logger.stdout(square_error)
            Logger.stdout(lamb , self.gamma , self.rho , self.global_steps)
            Logger.stdout(self.probs.isnan().any())
            Logger.stdout((self.probs + 1e-4).log().isnan().any())
            from src import api
            setattr(api , 'net' , self)
            raise ValueError
        return loss

    @property
    def get_probs(self):
        if self.probs_record is not None: 
            return self.probs_record / self.probs_record.sum(dim=1,keepdim=True)  

def shoot_infs(inp_tensor):
    """Replace ``inf`` values with the maximum finite value in the tensor.

    Used to stabilize the Sinkhorn normalization; prevents ``exp(inf/eps)``
    from producing NaN during the iterative row/column normalization.
    """
    valid = torch.isfinite(inp_tensor)

    if ~valid.all():
        m = torch.max(inp_tensor[valid])
        inp_tensor[~valid] = m

    return inp_tensor

def sinkhorn(Q, n_iters=3, epsilon=0.01):
    """Sinkhorn-Knopp iterative row-and-column normalization.

    Converts a log-score matrix into a doubly-stochastic assignment matrix
    used as the optimal transport plan in ``block_tra.loss_opt_transport``.

    Args:
        Q:       Score matrix (negative squared error), shape ``[bs, num_states]``.
        n_iters: Number of Sinkhorn iterations (default ``3``).
        epsilon: Temperature for ``exp(Q / epsilon)``; smaller → sharper
                 assignments.

    Returns:
        Doubly-stochastic matrix of the same shape as ``Q`` (values sum to 1
        across both axes after normalization).
    """
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for _ in range(n_iters):
            Q = (Q / Q.sum(dim=0, keepdim=True)).nan_to_num_(0)
            Q = (Q / Q.sum(dim=1, keepdim=True)).nan_to_num_(0)
    return Q

class tra(nn.Module):
    """TRA model: RNN backbone + ``block_tra`` routing head.

    ``_default_category = 'tra'`` signals to the training loop that this model
    requires extra forward arguments: ``hist_loss`` and ``y`` must be provided
    during training when ``num_states > 1``.

    Args:
        input_dim:          Input feature dimension.
        hidden_dim:         RNN and TRA hidden dimension.
        rnn_type:           RNN backbone type (default ``'lstm'``).
        rnn_layers:         Number of RNN layers (default ``2``).
        num_states:         Number of TRA predictor states (default ``1``).
        hist_loss_seq_len:  Length of historical loss window (default ``60``).
        hist_loss_horizon:  Future horizon trimmed from router input
                            (default ``20``).

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``
        Output: ``([bs, 1], {'loss_opt_transport': ..., 'hidden': ..., 'preds': ...})``
    """
    _default_category = 'tra'
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