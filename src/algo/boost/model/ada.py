import numpy as np
import pandas as pd
import torch

from torch import Tensor
from typing import Any , Optional

from .basic import BasicBoosterModel
from ..util.io import BoosterInput
from ....basic import PATH

class AdaBoost(BasicBoosterModel):
    DEFAULT_TRAIN_PARAM = {
        'n_learner' : 30, 
        'n_bins' : 20 , 
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

    def new_weight_param(self , weight_param , cuda = True , **kwargs) -> dict[str,Any]:
        new_weight_param = super().new_weight_param(weight_param , cuda , **kwargs)
        assert new_weight_param.get('cs_type') in ['ones' , None] , new_weight_param.get('cs_type')
        return new_weight_param

    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silence = False):
        if train is None: train = self.data['train']
        if valid is None: valid = self.data['valid']

        device = torch.device('cuda:0' if torch.cuda.is_available() and self.cuda else 'cpu')
        train_data = BoosterInput.concat([train , valid])
        x , y , w = train_data.XYW(as_tensor=True , device=device)

        x = self.rescale(x , 0 , self.train_param['n_bins'] , -1 , torch.int32)

        self.model : StrongLearner = StrongLearner(**self.train_param)
        self.model.fit(x , y , w , silence = silence , feature = train_data.feature)
        return self
    
    def predict(self , test : BoosterInput | Any = None):
        if test is None: test = self.data['test']
        return test.output(np.array(self.model.predict(test.X())))
    
    def to_dict(self):
        model_dict = super().to_dict()
        model_dict['model'] = self.model.to_dict()
        return model_dict
    
    def load_dict(self , model_dict : dict , cuda = False):
        super().load_dict(model_dict , cuda)
        self.model = StrongLearner.from_dict(model_dict['model'])
        return self
    
    @staticmethod
    def rescale(x : Tensor , lb = 0 , ub = 20 , nan = None , dtype = None):
        min = x.nan_to_num(torch.inf).min(0,keepdim=True)[0]
        max = x.nan_to_num(-torch.inf).max(0,keepdim=True)[0] + 1e-6
        x = (x - min) / (max - min) * (ub - lb) + lb
        return x.nan_to_num(nan).to(dtype = dtype)
    
    @classmethod
    def df_input(cls , factor_data : Optional[pd.DataFrame] = None , idx : int = -1 , windows_len = 24) -> dict[str,Any]:
        if factor_data is None: factor_data = load_xingye_data()
        MDTs = np.sort(factor_data['date'].unique())

        idtEnd = MDTs[idx - 1]
        idtStart = MDTs[idx - windows_len]
        idtTest = MDTs[idx]

        train_data = factor_data.loc[(factor_data['date'] >= idtStart) & (factor_data['date'] <= idtEnd),:].set_index(['date', 'secid'])
        test_data = factor_data.loc[factor_data['date'] == idtTest,:].set_index(['date', 'secid'])
        train = cls.parse_input_df(train_data , True , 20)
        test = cls.parse_input_df(test_data , False , 20).loc[:,train.columns.values]
        return {'train':train , 'test':test}

    @staticmethod
    def parse_input_df(df : pd.DataFrame , training : bool , n_bins : int , label_column = 'y_label') -> pd.DataFrame:
        x = df.drop(columns=[label_column])
        y = df[label_column]

        if y.isnull().sum() / len(y) > 0.2: print('Error: label missing ratio > 0.2')
        
        df = (x * n_bins).clip(upper=n_bins - 1).join(y)
        if training: 
            #missing = df.isnull().sum() / len(df)
            #df = df.loc[:, missing[missing < 0.2].index.values] # .fillna(0.0)
            df = df.loc[:, df.isnull().sum() / len(df) < 0.2] # .fillna(0.0)
            df[label_column] = (df[label_column].groupby('date').rank(pct=True) * 3).astype(int) - 1
            df = df[df[label_column] != 0].dropna(subset=[label_column])
        return df
    
class StrongLearner:
    def __init__(self , n_learner = 30, n_bins = 20 , **kwargs):   
        self.n_learner , self.n_bins = n_learner , n_bins
        self.weak_learners = [WeakLearner(n_bins) for _ in range(n_learner)]

    def __repr__(self) -> str: return f'{self.__class__.__name__}(n_learner={self.n_learner},n_bins={self.n_bins})'
    def __getitem__(self , i): return self.weak_learners[i]

    def update_weight(self , weight : Tensor , y : Tensor , y_pred : Tensor):
        weight = torch.exp(-y * y_pred.nan_to_num(0)) * weight
        return weight / weight.sum()

    def fit(self , x : np.ndarray | Tensor , y : np.ndarray | Tensor , w : np.ndarray | Tensor , 
            silence = False , feature = None):
        if not isinstance(x , Tensor): x = torch.tensor(x)
        if not isinstance(y , Tensor): y = torch.tensor(y)
        if not isinstance(w , Tensor): w = torch.tensor(w)
        
        for i , learner in enumerate(self.weak_learners):  # 使用 tqdm 显示进度条
            y_pred = learner.fit(x , y , w).predict(x)
            w = self.update_weight(w , y , y_pred)
            if not silence and i % 1 == 0: 
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
        return {'n_learner' : self.n_learner, 'n_bins' : self.n_bins ,
                'weak_learners':[learner.to_dict() for learner in self.weak_learners]}
    
    @classmethod
    def from_dict(cls , model_dict : dict , cuda = False):
        obj = cls(n_learner = model_dict['n_learner'] , n_bins = model_dict['n_bins'])
        [learner.load_dict(d , cuda) for learner , d in zip(obj.weak_learners , model_dict['weak_learners'])]
        return obj
    
class WeakLearner:
    EPS = 1e-6
    def __init__(self, n_bins : int): 
        self.n_bins = n_bins

    def __repr__(self) -> str: 
        return  f'{self.__class__.__name__}(n_bins={self.n_bins})'

    def fit(self, X : Tensor , y : Tensor , weight : Tensor | None = None):
        if weight is None: weight = torch.ones_like(y) / len(y)
        assert isinstance(X , Tensor) and isinstance(y , Tensor) and isinstance(weight , Tensor) , (X , y , weight)
        assert torch.all(X < self.n_bins) , X.max()
        assert not torch.is_floating_point(X) , X
        self.n_feat = X.shape[-1]

        pos_wgt , neg_wgt = (weight * (y > 0))[:,None] , (weight * (y < 0))[:,None]
        pos_imp = torch.zeros(self.n_feat , self.n_bins).to(y)
        neg_imp = pos_imp * 0.

        for ibin in range(self.n_bins):
            where = X == ibin
            pos_imp[:,ibin] = (pos_wgt * where).sum(dim = 0)
            neg_imp[:,ibin] = (neg_wgt * where).sum(dim = 0)
        
        feat_imp = (pos_imp + neg_imp).sum(-1,keepdim=True)
        pos_imp , neg_imp = pos_imp / feat_imp , neg_imp / feat_imp
        
        self.feat_losses = torch.sqrt(pos_imp * neg_imp).sum(1)
        self.feat_idx = self.feat_losses.argmin()
        self.bin_predictions = 0.5 * torch.log((pos_imp[self.feat_idx] + self.EPS) / (neg_imp[self.feat_idx] + self.EPS))
        return self
    
    @property
    def min_feat_loss(self): return self.feat_losses[self.feat_idx]

    def predict(self, X : Tensor):
        assert X.shape[-1] == self.n_feat , (X.shape , self.n_feat)
        assert isinstance(X , Tensor) , X
        assert not torch.is_floating_point(X) , X.dtype
        group = X[:, self.feat_idx]
        return torch.where(group >= 0 , self.bin_predictions[group] , torch.nan)
    
    def cpu(self):
        self.bin_predictions = self.bin_predictions.cpu()
        return self
    
    def to_dict(self):
        return {getattr(self , k) for k in ['n_bins','n_feat','feat_losses','feat_idx','bin_predictions']}
    
    def load_dict(self , model_dict , cuda = False):
        for k in ['n_bins','n_feat','feat_losses','feat_idx','bin_predictions']:
            v = model_dict[k]
            if isinstance(v , Tensor) and cuda and torch.cuda.is_available(): v = v.cuda()
            setattr(self , k , v)
        return self

    @classmethod
    def from_dict(cls , model_dict , cuda = False):
        return cls(model_dict['n_bins']).load_dict(model_dict , cuda)
    
def load_xingye_data():
    factor_data = pd.read_feather(f'{PATH.data}/TreeData/CombStdByZXMkt_All_TrainLabel.feather') # 训练集，带Label
    factor_data['date'] = factor_data['date'].astype(str).str.replace('-','').astype(int)
    v = factor_data['StockID'].astype(str).str.slice(0, 6).replace({'T00018' : '600018'})
    v = v.where(v.str.isdigit() , '-1').astype(int)
    factor_data['secid'] = v

    index_list = ['secid','date']
    label_list = ['nextRtnM']

    factor_data = factor_data.drop(columns=['ZX','mktVal','mktValRank','StockID','nextRtnM_Label']).set_index(index_list)
    factor_rank = factor_data.drop(columns=label_list).groupby('date').rank(pct = True).\
        join(factor_data[label_list]).rename(columns={'nextRtnM':'y_label'})
    return factor_rank.reset_index().sort_index()

if __name__ == '__main__':
    factor_data = load_xingye_data()
    MDTs = np.sort(factor_data['date'].unique())
    windows_len = 24 
    
    ic_dfs = []
    start_idx = 179
    for idx in range(max(windows_len , start_idx) , len(MDTs)):
        ada = AdaBoost()
        input_df = AdaBoost.df_input(factor_data , idx , windows_len)
        ada.fit(BoosterInput.from_dataframe(input_df['train']) , silence = True)
        ic_dfs.append(ada.calc_ic(input_df['test']))
    df = pd.concat(ic_dfs)
    print(df)