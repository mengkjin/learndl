import torch
from torch import nn,Tensor
from torch.distributions import Normal , kl_divergence

from ..layer.MLP import MLP

class FactorVAE(nn.Module):
    _default_category = 'vae'
    
    def __init__(
        self,
        input_dim : int = 6 ,
        hidden_dim : int = 32,
        factor_num : int = 64,
        gru_input_size : int = 16 ,
        portfolio_size : int = 100 ,
        encoder_h_size : int = 32 ,
        alpha_h_size : int = 64,
        predictor_h_size : int = 16 ,
        monte_carlo : int = 1000 ,
        gamma : float = 0.001 ,
        **kwargs ,
    ):
        super().__init__()
        self.factor_num = factor_num
        self.portfolio_size = portfolio_size
        self.monte_carlo = monte_carlo
        self.gamma = gamma
        self.feature_extractor = FeatureExtractor(input_dim , hidden_dim , gru_input_size)
        self.factor_encoder = FactorEncoder(hidden_dim , factor_num , portfolio_size , encoder_h_size)
        self.factor_decoder = FactorDecoder(hidden_dim , factor_num , alpha_h_size)
        self.factor_predictor = FactorPredictor(hidden_dim , factor_num , predictor_h_size)
        self.bn = nn.BatchNorm1d(1)
        self.sampling = VAESampling()

    def forward(self, x : Tensor, y : Tensor | None = None , 
                factor_noise : Tensor | None = None ,
                alpha_noise : Tensor | None = None):
        '''
        in  : 
            x : [bs x seq_len x input_dim]
            y : [bs x 1]
            factor_noise : [factor_num]
            alpha_noise : [bs]
        out :
            in training : y_hat , loss_KL
            in eval     : y_hat , 
        '''
        if self.training:
            assert y is not None , f'{self.__class__.__name__} y is None when training'
            assert factor_noise is None or factor_noise.numel() == self.factor_num , factor_noise
            assert alpha_noise is None or alpha_noise.numel() == y.shape[0] , (alpha_noise.numel() , y.shape[0])
            latent_features = self.feature_extractor(x)
            mu_post, sigma_post = self.factor_encoder(latent_features, y)
            factors_post = self.sampling(mu_post, sigma_post , factor_noise)
            y_hat, mu_alpha, sigma_alpha, beta = self.factor_decoder(latent_features , factors_post , alpha_noise)
            
            mu_prior, sigma_prior = self.factor_predictor(latent_features)
            #print(mu_post, sigma_post , mu_prior, sigma_prior)
            loss_KL = kl_divergence(Normal(mu_post, sigma_post) , Normal(mu_prior, sigma_prior)).sum()
            y_hat = self.bn(y_hat)
            assert ~loss_KL.isnan().any() and abs(loss_KL <= 1.0e20) , loss_KL
            #mu_dec, sigma_dec = self.get_decoder_distribution(mu_alpha, sigma_alpha, mu_post, sigma_post, beta)
            #loss_negloglike = Normal(mu_dec, sigma_dec).log_prob(y.unsqueeze(-1)).sum()
            #loss_negloglike = loss_negloglike * (-1 / (self.portfolio_size * latent_features.shape[0]))
            return y_hat , {'loss_KL' : self.gamma * loss_KL}
        else:
            latent_features = self.feature_extractor(x)
            mu_prior, sigma_prior = self.factor_predictor(latent_features)
            y_hats = [self.sampling(mu_prior, sigma_prior) for _ in range(self.monte_carlo)]
            y_hats = [self.factor_decoder(latent_features , y_hat)[0] for y_hat in y_hats]
            y_hats = [self.bn(y_hat) for y_hat in y_hats]
            y_hat = torch.cat(y_hats , dim = -1).mean(dim = -1 , keepdim = True)
            return y_hat   

    def get_decoder_distribution(self, mu_alpha : Tensor, sigma_alpha : Tensor, mu_factor : Tensor,
                                 sigma_factor : Tensor, beta : Tensor):
        # print(mu_alpha.shape, mu_factor.shape, sigma_factor.shape, beta.shape)
        mu_dec = mu_alpha + torch.mm(beta, mu_factor.reshape(beta.shape[-1] , -1))
        sigma_dec = (sigma_alpha.square() + torch.mm(beta.square(), sigma_factor.square().reshape(beta.shape[-1] , -1))).sqrt()
        return mu_dec, sigma_dec
    
class FeatureExtractor(nn.Module):
    '''
    Use GRU to extract hidden features
    in  : [bs x seq_len x nvars]
    out : [bs x hidden_dim]
    '''
    def __init__(self , input_dim : int = 6 , hidden_dim : int = 32, gru_input_size : int = 16):
        super().__init__()
        # self.proj = MLP(input_dim,gru_input_size)
        self.proj = nn.Sequential(nn.Linear(input_dim,gru_input_size) , nn.LeakyReLU())
        self.gru = nn.GRU(gru_input_size, hidden_dim , num_layers=2 ,  batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x : Tensor) -> Tensor:
        x = self.proj(x)
        x = self.gru(x)[0][:,-1]
        x = self.bn(x)
        return x
    
class PortfolioLayer(nn.Module):
    '''
    Use hidden features to create synthetic portfolios
    in  : [bs x hidden_dim]
    out : [bs x portfolio_size]
    '''
    def __init__(self, hidden_dim : int = 32 , portfolio_size : int = 100):
        super().__init__()
        self.net = nn.Linear(hidden_dim , portfolio_size)

    def forward(self, x : Tensor) -> Tensor:
        p = torch.softmax(self.net(x), dim=0)
        return p
    
class VAESampling(nn.Module):
    def forward(self , mu : Tensor , sigma : Tensor , noise : Tensor | None = None):
        if noise is None:  
            return self.reparameterize(mu , sigma)
        else: 
            return mu + sigma * noise.reshape(sigma.shape).to(sigma)

    def reparameterize(self, mu : Tensor , sigma : Tensor):
        eps = torch.randn_like(sigma)
        return mu + sigma + eps

class DistributionLayer(nn.Module):
    '''
    Use hidden features to predict mu and sigma
    in  : [1 x input_dim]
    out : ([1 x output_dim] , [1 x output_dim])
    '''
    def __init__(self, input_dim : int , output_dim : int , hidden_size : int | None = None):
        super().__init__()
        if hidden_size is None: 
            hidden_size = output_dim
        self.norm  = nn.LayerNorm(input_dim)
        self.mu    = MLP(input_dim,output_dim,hidden_size)
        self.sigma = MLP(input_dim,output_dim,hidden_size,out_activation='softplus')
    def forward(self, x : Tensor):
        x = self.norm(x)
        mu , sigma = self.mu(x) , self.sigma(x)
        return mu , sigma
    
class FactorEncoder(nn.Module):
    '''
    Factor Encoder : Return mu_post, sigma_post
    in  : ([bs x hidden_dim] , [bs x 1]) , hidden features and future returns
    out : ([1 x factor_num] , [1 x factor_num])
    '''
    def __init__(self, 
                 hidden_dim : int = 32,
                 factor_num : int = 64, 
                 portfolio_size : int = 100 ,
                 encoder_h_size : int = 32):
        super().__init__()
        self.portfolio_layer = PortfolioLayer(hidden_dim, portfolio_size)
        self.mapping_layer = DistributionLayer(portfolio_size, factor_num , encoder_h_size)

    def forward(self, x : Tensor, y : Tensor):
        '''
        x : [bs x hidden_dim] , hidden features 
        y : [bs x 1] , future returns
        '''
        p = self.portfolio_layer(x)         # [bs x portfolio_size]
        y = torch.mm(y.T , p)               # [1 x portfolio_size]
        mu_post, sigma_post = self.mapping_layer(y) # ([1 x factor_num] , [1 x factor_num])
        # m = Normal(mu_post, sigma_post)
        # z_post = m.sample()
        return mu_post, sigma_post
    
class BetaLayer(nn.Module):
    '''
    in  : [bs x hidden_dim] , hidden features 
    out : [bs x factor_num]
    '''
    def __init__(self, hidden_dim : int = 32 , factor_num : int = 64):
        super().__init__()
        self.beta_layer = nn.Linear(hidden_dim , factor_num)
    def forward(self, x : Tensor):
        return self.beta_layer(x)
    
class AlphaLayer(nn.Module):
    '''
    in  : [bs x hidden_dim] , hidden features
    out : [bs x 2] , mu and sigma for each sample
    '''
    def __init__(self, hidden_dim : int = 32 , hidden_size : int = 32):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(hidden_dim , hidden_size) , nn.LeakyReLU())
        self.alpha_net = DistributionLayer(hidden_size,1,hidden_size)

    def forward(self, x : Tensor):
        return self.alpha_net(self.mapping(x))
    
class FactorDecoder(nn.Module):
    '''
    Generate Stock return y_hat from hidden features and factor sample.
    in  : ([bs x hidden_dim] , [1 x factor_num]) , hidden features and factor sample
    out : ([bs x 1] , [bs x 1] , [bs x 1] , [bs x factor_num]) , y_hat , mu_alpha, sigma_alpha , beta
    '''
    def __init__(self, hidden_dim : int = 32 , factor_num : int = 64 , alpha_h_size : int = 64):
        super().__init__()
        self.factor_num = factor_num
        self.alpha_layer = AlphaLayer(hidden_dim , alpha_h_size)
        self.beta_layer = BetaLayer(hidden_dim , factor_num)
        self.alpha_sampling = VAESampling()

    def forward(self, x : Tensor , factors : Tensor , alpha_noise : Tensor | None = None) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        mu_alpha, sigma_alpha = self.alpha_layer(x)
        beta = self.beta_layer(x)
        alpha = self.alpha_sampling(mu_alpha, sigma_alpha , alpha_noise)
        exposed_factors = torch.mm(beta, factors.reshape(self.factor_num , -1))
        stock_returns = exposed_factors + alpha
        return stock_returns, mu_alpha, sigma_alpha, beta
    
class FactorPredictor(nn.Module):
    '''
    in  : [bs x hidden_dim]
    out : ([factor_num x 1] , [factor_num x 1])
    '''
    def __init__(self, hidden_dim : int = 32, factor_num : int = 64,  predictor_h_size : int = 32):
        super().__init__()
        self.factor_num = factor_num
        self.k_map = nn.Linear(hidden_dim , predictor_h_size)
        self.v_map = nn.Linear(hidden_dim , predictor_h_size)
        self.query = torch.nn.Parameter(torch.rand(predictor_h_size , factor_num))
        self.factor_net = DistributionLayer(predictor_h_size * factor_num , factor_num)

    def forward(self, x : Tensor):
        k , v = self.k_map(x) , self.v_map(x)
        q_norm = self.query.norm(dim=0,keepdim=True) + 1e-6
        k_norm = k.norm(dim=-1,keepdim=True) + 1e-6
        f = []
        for i in range(self.factor_num):
            a = torch.nn.ReLU()(self.query[:,i] * k / q_norm[:,i] / k_norm) + 1e-6
            a = a / a.sum(-1 , keepdim=True)
            f.append((a * v).sum(0))
        f = torch.cat(f)
        return self.factor_net(f)
    
if __name__ == '__main__':
    from src.res.model.util import BatchData

    batch_data = BatchData.random(batch_size=40)
    x = batch_data.x
    assert isinstance(x , Tensor) , f'{type(x)} is not a Tensor'
    f = batch_data.y

    extractor = FeatureExtractor()
    h : Tensor = extractor(x)
    print(f'Hidden Features shape : {h.shape}') 

    portfolio = PortfolioLayer()
    p = portfolio(h)
    print(f'Portoflio shape : {p.shape}') 

    encoder = FactorEncoder()
    mu_post, sigma_post = encoder(h , f)
    print(f'mu_post shape : {mu_post.shape} , sigma_post shape : {sigma_post.shape}') 

    factor_sampling = VAESampling()
    factors = factor_sampling(mu_post , sigma_post)
    print(f'A post factor sample : {factors} ') 

    beta = BetaLayer()
    print(f'Beta shape : {beta(h).shape}') 

    alpha = AlphaLayer()
    mu_alpha, sigma_alpha = alpha(h)
    print(f'mu_alpha shape : {mu_alpha.shape} , sigma_alpha shape : {sigma_alpha.shape}') 

    decorder = FactorDecoder()
    stock_returns, mu_alpha, sigma_alpha, beta = decorder(h , factors)
    print(f'stock_returns shape : {stock_returns.shape} , mu_alpha shape : {mu_alpha.shape} , sigma_alpha shape : {sigma_alpha.shape}') 

    predictor = FactorPredictor()
    mu_prior, sigma_prior = predictor(h)
    print(f'mu_prior shape : {mu_prior.shape} , sigma_prior shape : {sigma_prior.shape}') 

    factor_vae = FactorVAE()
    factor_vae.train()
    y_hat , loss_KL = factor_vae(x , f)
    print(f'y_hat for training shape : {y_hat.shape} , loss_KL is : {loss_KL}') 
    
    factor_vae.eval()
    y_hat = factor_vae(x , f)
    print(f'y_hat for eval shape : {y_hat.shape}') 
