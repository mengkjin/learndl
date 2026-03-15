import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.proj import Logger
from src.res.algo.nn import layer as Layer

class mod_gru(nn.Module):
    def __init__(self , input_dim , output_dim , dropout=0.0 , num_layers = 2):
        super().__init__()
        num_layers = min(3,num_layers)
        self.gru = nn.GRU(input_dim , output_dim , num_layers = num_layers , dropout = dropout , batch_first = True)
    def forward(self, x : Tensor) -> Tensor:
        return self.gru(x)[0]

class Astgnn(nn.Module):
    def __init__(self,input_dim,hidden_dim = 128,dropout = 0.1,rnn_layers = 2,enc_in=None,enc_in_dim=64,
                 act_type='leaky',dec_mlp_layers=2,dec_mlp_dim=128,
                 alpha_num = 60 , beta_num = 10 , loss_corr_lamb = 0.1,
                 **kwargs):
        super().__init__()
        self.fc_enc_in = nn.Sequential(nn.Linear(input_dim, enc_in_dim),nn.Tanh())

        rnn_kwargs = {'input_dim':enc_in_dim,'output_dim':hidden_dim,'num_layers':rnn_layers, 'dropout':dropout}
        self.fc_rnn = mod_gru(**rnn_kwargs)

        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim , alpha_num), 
            Layer.Act.get_activation_fn(act_type), 
            nn.Dropout(dropout)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(hidden_dim , beta_num), 
            Layer.Act.get_activation_fn(act_type), 
            nn.Dropout(dropout)
        )
        self.alpha_map_out = Layer.EwLinear()
        self.loss_corr_lamb = loss_corr_lamb

    def forward(self, input : Tensor | tuple[Tensor,...] | list[Tensor]):
        '''
        in: [bs x seq_len x input_dim]
        out:[bs x hidden_dim]
        '''
        x = input if isinstance(input , Tensor) else torch.concat(input , dim = -1) 
        x = self.fc_enc_in(x)
        x = self.fc_rnn(x)
        x , x_0 = x[:,-1] , x[: , 0]
        alphas = self.alpha_net(x)
        betas = self.beta_net(x)
        betas_peer = self.beta_net(x_0)

        pred = self.alpha_map_out(alphas) 
        if False:
            Logger.stdout(f'input shape: {x.shape}')
            Logger.stdout(f'enc_in shape: {x.shape}')
            Logger.stdout(f'rnn outpur shape: {x.shape}')
            Logger.stdout(f'last rnn output shape: {x.shape}')
            Logger.stdout(f'initial rnn output shape: {x_0.shape}')
            Logger.stdout(f'alphas shape: {alphas.shape}')
            Logger.stdout(f'betas shape: {betas.shape}')
            Logger.stdout(f'pred shape: {pred.shape}')
        return pred , {'alphas':alphas , 'betas':betas , 'betas_peer':betas_peer}

    def loss(self, pred : torch.Tensor , label : torch.Tensor , alphas : torch.Tensor , betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        assert label.shape[-1] == 2 , label.shape
        mse = F.mse_loss(pred.squeeze() , label[...,0].squeeze())
        rsquare = self.rsquare_loss(alphas , label[...,1])
        corr = self.corr_loss(betas)
        turnover = self.turnover_loss(betas , betas_peer)
        
        return mse + rsquare + self.loss_corr_lamb * corr + turnover

    def rsquare_loss(self, hiddens : torch.Tensor , label : torch.Tensor , **kwargs):
        assert hiddens.ndim == 2 , hiddens.shape
        y_norm = label.norm()
        pred = hiddens @ (hiddens.T @ hiddens).inverse() @ hiddens.T @ label
        res_norm = (label - pred).norm()
        return 1 - res_norm / y_norm

    def corr_loss(self, hiddens : torch.Tensor , **kwargs):
        h = (hiddens - hiddens.mean(dim=0,keepdim=True)) / (hiddens.std(dim=0,keepdim=True) + 1e-6)
        pen = h.T.cov().norm()
        return pen

    def turnover_loss(self, betas : torch.Tensor , betas_peer : torch.Tensor , **kwargs):
        return (betas - betas_peer).norm()