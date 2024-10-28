import torch
import numpy as np
import pandas as pd

from torch import Tensor
from typing import Any

from ..util.basic import BasicBoosterModel , load_xingye_data
from ..util.io import BoosterInput

class AdaBoost(BasicBoosterModel):
    DEFAULT_TRAIN_PARAM = {
        'n_learner' : 30, 
        'n_bins' : 20 , 
        'max_nan_ratio' : 0.8 ,
    }
    DEFAULT_WEIGHT_PARAM = {
        'ts_type' : None ,
        'cs_type' : None ,
        'bm_type' : None , 
        'ts_lin_rate' : 0.5 ,
        'ts_half_life_rate' : 0.5 ,
        'cs_top_tau' : 0.75*np.log(0.5)/np.log(0.75) ,
        'cs_ones_rate' : 2. ,
        'bm_rate' : 2. ,
        'bm_secid' : None}

    def assert_param(self):
        super().assert_param()
        assert self.weight_param.get('cs_type') in ['ones' , None] , self.weight_param.get('cs_type')

    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silent = False):
        if train is None: train = self.data['train']
        if valid is None: valid = self.data['valid']

        device = torch.device('cuda:0' if torch.cuda.is_available() and self.cuda else 'cpu')
        train_data = BoosterInput.concat([train , valid])
        dset = train_data.Dataset().to(device)

        dset.x = self.input_transform(dset.x)
        dset.y = self.label_transform(dset.y)

        idx = (dset.y != 0).squeeze()

        self.model : StrongLearner = StrongLearner(**self.train_param)
        self.model.fit(dset.x[idx] , dset.y[idx] , dset.w[idx] , silent = silent , feature = train_data.feature)
        return self
    
    def predict(self , x : BoosterInput | str = 'test'):
        data = self.booster_input(x)
        X = self.input_transform(data.X())
        return data.output(self.model.predict(X))
    
    def to_dict(self):
        model_dict = super().to_dict()
        model_dict['model'] = self.model.to_dict()
        return model_dict
    
    def load_dict(self , model_dict : dict , cuda = False , seed = None):
        super().load_dict(model_dict , cuda , seed)
        self.model = StrongLearner.from_dict(model_dict['model'] , cuda = cuda)
        return self
    
    def input_transform(self , x : torch.Tensor):
        n_bins = self.train_param['n_bins']
        return torch.floor(self.tensor_rank(x , dim = 0 , pct = True) * n_bins).\
            clip(max = n_bins - 1).nan_to_num(-1).to(torch.int)
    
    def label_transform(self , y : torch.Tensor):
        return (torch.floor(self.tensor_rank(y , dim = 0 , pct = True) * 3) - 1).sign()

    @staticmethod
    def tensor_rank(x : torch.Tensor , dim : int = 0 , pct = False):
        rank = x.argsort(dim = dim).argsort(dim = dim).where(~x.isnan() , torch.nan)
        if pct: rank /= rank.nan_to_num().max(dim = dim , keepdim = True)[0] + 1e-6
        return rank
    
class StrongLearner:
    def __init__(self , n_learner = 30, n_bins = 20 , max_nan_ratio : float = 0.8 , **kwargs):   
        self.n_learner     = n_learner
        self.n_bins        = n_bins
        self.max_nan_ratio = max_nan_ratio
        self.weak_learners = [WeakLearner(n_bins , max_nan_ratio) for _ in range(n_learner)]

    def __repr__(self) -> str: 
        return f'{self.__class__.__name__}(n_learner={self.n_learner},n_bins={self.n_bins},max_nan_ratio={self.max_nan_ratio})'
    def __getitem__(self , i): return self.weak_learners[i]

    def update_weight(self , weight : Tensor , y : Tensor , y_pred : Tensor):
        weight = torch.exp(-y * y_pred.nan_to_num(0)) * weight
        return weight / weight.sum()

    def fit(self , x : Tensor , y : Tensor , w : Tensor , silent = False , feature = None):
        for i , learner in enumerate(self.weak_learners):  
            y_pred = learner.fit(x , y , w).predict(x)
            w = self.update_weight(w , y , y_pred)
            if not silent and i % 1 == 0: 
                txt = f'Round: {i+1}, Est-Err: {learner.min_feat_loss:.4f}'
                if feature is None: 
                    txt += f', F_idx: {learner.feat_idx}'
                else:
                    txt += f', F_name: {feature[learner.feat_idx]}'
                print(txt)
        return self
    
    def predict(self , x : np.ndarray | Tensor):
        pred = self.predictions(x).mean(dim = -1)
        pred = pred / pred.std()
        return pred.cpu().numpy()
    
    def predictions(self, x : np.ndarray | Tensor):
        if not isinstance(x , Tensor): x = torch.tensor(x)
        x = x.nan_to_num(-1).to(torch.int32)
        preds = torch.stack([learner.predict(x).nan_to_num(0) for learner in self.weak_learners] , dim = -1)
        return preds
    
    @property
    def feat_idx(self): return [learner.feat_idx for learner in self.weak_learners]
    @property
    def feat_err(self): return [learner.min_feat_loss for learner in self.weak_learners]
    
    def to_dict(self):
        return {'n_learner' : self.n_learner, 'n_bins' : self.n_bins , 'max_nan_ratio' : self.max_nan_ratio ,
                'weak_learners':[learner.to_dict() for learner in self.weak_learners]}
    
    @classmethod
    def from_dict(cls , model_dict : dict , cuda = False):
        obj = cls(**model_dict)
        [learner.load_dict(d , cuda) for learner , d in zip(obj.weak_learners , model_dict['weak_learners'])]
        return obj
    
class WeakLearner:
    EPS = 1e-6
    SLOTS = ['n_bins','max_nan_ratio','n_feat','feat_losses','feat_idx','bin_predictions']
    def __init__(self, n_bins : int , max_nan_ratio : float = 0.8): 
        self.n_bins = n_bins
        self.max_nan_ratio = max_nan_ratio

    def __repr__(self) -> str: 
        return  f'{self.__class__.__name__}(n_bins={self.n_bins},max_nan_ratio={self.max_nan_ratio})'

    def fit(self, X : Tensor , y : Tensor , weight : Tensor | None = None):
        if weight is None: weight = torch.ones_like(y) / len(y)
        assert isinstance(X , Tensor) and isinstance(y , Tensor) and isinstance(weight , Tensor) , (X , y , weight)
        assert torch.all(X < self.n_bins) , X.max()
        assert not torch.is_floating_point(X) , X
        self.n_feat = X.shape[-1]

        pos_wgt , neg_wgt = (weight * (y > 0))[:,None] , (weight * (y < 0))[:,None]
        pos_imp = torch.zeros(self.n_feat , self.n_bins).to(y)
        neg_imp = pos_imp * 0.

        for ibin in range(pos_imp.shape[-1]):
            where = X == ibin
            pos_imp[:,ibin] = (pos_wgt * where).sum(dim = 0)
            neg_imp[:,ibin] = (neg_wgt * where).sum(dim = 0)

        feat_imp = (pos_imp + neg_imp).sum(-1,keepdim=True)
        good_feat = 1 - feat_imp.squeeze() <= self.max_nan_ratio
        pos_imp , neg_imp = pos_imp / feat_imp , neg_imp / feat_imp

        self.feat_losses = (pos_imp * neg_imp).sqrt().sum(1).where(good_feat , torch.nan)
        self.feat_idx = self.feat_losses.nan_to_num(torch.inf).argmin()
        self.bin_predictions = 0.5 * torch.log((pos_imp[self.feat_idx] + self.EPS) / (neg_imp[self.feat_idx] + self.EPS))
        return self
    
    @property
    def min_feat_loss(self): return self.feat_losses[self.feat_idx]

    def predict(self, X : Tensor):
        assert X.shape[-1] == self.n_feat , (X.shape , self.n_feat)
        assert isinstance(X , Tensor) , X
        assert not torch.is_floating_point(X) , X
        group = X[:, self.feat_idx]
        return self.bin_predictions[group].where(group >= 0 ,torch.nan)
    
    def cpu(self):
        self.bin_predictions = self.bin_predictions.cpu()
        return self
    
    def to_dict(self):
        d = {}
        for k in self.SLOTS:
            v = getattr(self , k)
            if isinstance(v , Tensor): v = v.cpu()
            d[k] = v
        return d
    
    def load_dict(self , model_dict , cuda = False):
        for k in self.SLOTS:
            v = model_dict[k]
            if isinstance(v , Tensor) and cuda and torch.cuda.is_available(): v = v.cuda()
            setattr(self , k , v)
        return self

    @classmethod
    def from_dict(cls , model_dict , cuda = False):
        return cls(model_dict['n_bins'],model_dict['max_nan_ratio']).load_dict(model_dict , cuda)

if __name__ == '__main__':
    factor_data = load_xingye_data()
    MDTs = np.sort(factor_data['date'].unique())
    windows_len = 24 
    
    ic_dfs = []
    start_idx = 179
    for idx in range(max(windows_len , start_idx) , len(MDTs)):
        ada = AdaBoost()
        input_df = AdaBoost.df_input(factor_data , idx , windows_len)
        ada.fit(BoosterInput.from_dataframe(input_df['train']) , silent = True)
        ic_dfs.append(ada.calc_ic(input_df['test']))
    df = pd.concat(ic_dfs)
    print(df)