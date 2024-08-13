import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from copy import deepcopy
from torch import Tensor
from typing import Any , Optional

from .basic import BasicBooster , BoosterData
from ...basic import PATH

DEFAULT_TRAIN_PARAM = {
    'n_booster' : 30, 'n_bins' : 20 , 
}

class WeakBooster:
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

        pos_wgt , neg_wgt = (weight * (y == 1))[:,None] , (weight * (y == -1))[:,None]
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
    
class StrongBooster:
    def __init__(self , n_booster = 30, n_bins = 20):   
        self.n_booster , self.n_bins = n_booster , n_bins
        self.boosters = [WeakBooster(n_bins) for _ in range(n_booster)]

    def __repr__(self) -> str: return f'{self.__class__.__name__}(n_booster={self.n_booster},n_bins={self.n_bins})'
    def __getitem__(self , i): return self.boosters[i]

    def update_weight(self , weight : Tensor , y : Tensor , y_pred : Tensor):
        weight = torch.exp(-y * y_pred.nan_to_num(0)) * weight
        return weight / weight.sum()

    def fit(self , x : np.ndarray | Tensor , y : np.ndarray | Tensor , w : np.ndarray | Tensor , 
            silence = False , feature = None):
        if not isinstance(x , Tensor): x = torch.tensor(x)
        if not isinstance(y , Tensor): y = torch.tensor(y)
        if not isinstance(w , Tensor): w = torch.tensor(w)
        x = x.nan_to_num(-1).to(torch.int32)
        for i , booster in enumerate(self.boosters):  # 使用 tqdm 显示进度条
            y_pred = booster.fit(x , y , w).predict(x)
            w = self.update_weight(w , y , y_pred)
            if not silence and i % 5 == 0: 
                txt = f'Round: {i+1}, Est-Err: {booster.min_feat_loss:.4f}'
                if feature is None: 
                    txt += f', F_idx: {booster.feat_idx}'
                else:
                    txt += f', F_name: {feature[booster.feat_idx]}'
                print(txt)
        return self
    
    def predict(self , x : np.ndarray | Tensor):
        pred = self.predictions(x).mean(dim = -1)
        pred = pred / pred.std()
        return pred.cpu().numpy()
    
    def predictions(self, x : np.ndarray | Tensor):
        if not isinstance(x , Tensor): x = torch.tensor(x)
        x = x.nan_to_num(-1).to(torch.int32)
        preds = torch.stack([booster.predict(x).nan_to_num(0) for booster in self.boosters] , dim = -1)
        return preds
    
    @property
    def feat_idx(self): return [booster.feat_idx for booster in self.boosters]
    @property
    def feat_err(self): return [booster.min_feat_loss for booster in self.boosters]
    
    def to_dict(self):
        return {'ada_boosters':[booster.to_dict() for booster in self.boosters] ,
                'n_booster' : self.n_booster, 'n_bins' : self.n_bins}
    
    @classmethod
    def from_dict(cls , model_dict : dict , cuda = False):
        obj = cls(n_booster = model_dict['n_booster'] , n_bins = model_dict['n_bins'])
        [booster.load_dict(d , cuda) for booster , d in zip(obj.boosters , model_dict['ada_boosters'])]
        return obj

class AdaBoost(BasicBooster):
    def __init__(self , 
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 feature = None , 
                 cs_weight = False , ts_weight = False , bm_weight = None , bm_secid = None , 
                 cuda = True , 
                 plot_path = None, # '../../figures' ,
                 **kwargs): 
        self.train_param = deepcopy(DEFAULT_TRAIN_PARAM)
        self.train_param.update(train_param)  
        self.model = StrongBooster(**self.train_param)

        self.data : dict[str , BoosterData] = {}
        self.feature = feature

        self.weight_param = weight_param
        if cs_weight: self.weight_param['cs_type'] = 'ones'
        if ts_weight: self.weight_param['ts_type'] = 'exp'
        if bm_weight: self.weight_param['bm_type'] = 'in'
        if bm_secid is not None: self.weight_param['bm_secid'] = bm_secid
        
        assert self.weight_param.get('cs_type') in ['ones' , None] , self.weight_param.get('cs_type')

        self.plot_path = plot_path
        self.cuda = cuda
        self.data_import(train = train , test = test)

    def __repr__(self) -> str: return f'{self.__class__.__name__}'

    def update_weight(self , weight : Tensor , y : Tensor , y_pred : Tensor):
        weight = torch.exp(-y * y_pred.nan_to_num(0)) * weight
        return weight / weight.sum()

    def fit(self , use_feature = None , train = None , valid = None , silence = False):
        self.data_import(train = train)
        device = torch.device('cuda:0' if torch.cuda.is_available() and self.cuda else 'cpu')
        x = torch.tensor(self.data['train'].X()).nan_to_num(-1).to(torch.int32).to(device)
        y = torch.tensor(self.data['train'].Y()).to(device)
        w = torch.tensor(self.data['train'].W()).to(device)

        self.model.fit(x , y , w , silence = silence , feature = self.data['train'].feature)
        return self
    
    def predict(self , x : Optional[BoosterData] = None , reshape = True , reform = True):
        if x is None: x = self.data['test']
        pred = np.array(self.model.predict(x.X()))
        if reshape: pred = x.reshape_pred(pred)
        if reform: pred = x.reform_pred(pred)
        return pred
    
    def to_dict(self):
        return self.model.to_dict()
    
    @classmethod
    def from_dict(cls , model_dict : dict , cuda = False):
        obj = cls(cuda = cuda)
        obj.model = StrongBooster.from_dict(model_dict = model_dict)
        return obj
    
    @classmethod
    def df_input(cls , factor_data : Optional[pd.DataFrame] = None , idx : int = -1 , windows_len = 24) -> dict[str,Any]:
        if factor_data is None: factor_data = load_xingye_data()
        MDTs = np.sort(factor_data['date'].unique())

        idtEnd = MDTs[idx - 1]
        idtStart = MDTs[idx - windows_len]
        idtTest = MDTs[idx]

        print(f'Train: {idtStart} - {idtEnd} , Test : {idtTest}')
        train_data = factor_data.loc[(factor_data['date'] >= idtStart) & (factor_data['date'] <= idtEnd),:].\
            drop(columns=['nextRtnM']).rename(columns={'nextRtnM_Label':'y_label'}).set_index(['date', 'secid'])
        test_data = factor_data.loc[factor_data['date'] == idtTest,:].\
            drop(columns=['nextRtnM_Label']).rename(columns={'nextRtnM':'y_label'}).set_index(['date', 'secid'])
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
            df = df[df[label_column] != 0].dropna(subset=[label_column])
        return df

def load_xingye_data():
    factor_data = pd.read_feather('./data/TreeData/CombStdByZXMkt_All_TrainLabel.feather') # 训练集，带Label
    factor_data['date'] = factor_data['date'].astype(str).str.replace('-','').astype(int)
    v = factor_data['StockID'].astype(str).str.slice(0, 6).replace({'T00018' : '600018'})
    v = v.where(v.str.isdigit() , '-1').astype(int)
    factor_data['secid'] = v

    index_list = ['secid','date']
    label_list = ['nextRtnM' , 'nextRtnM_Label']

    factor_data = factor_data.drop(columns=['ZX','mktVal','mktValRank','StockID']).set_index(index_list)
    factor_rank = factor_data.drop(columns=label_list).groupby('date').rank(pct = True).join(factor_data[label_list])
    return factor_rank.reset_index().sort_index()

if __name__ == '__main__':
    factor_data = load_xingye_data()
    MDTs = np.sort(factor_data['date'].unique())
    windows_len = 24 
    
    ic_dfs = []
    start_idx = 179
    for idx in range(max(windows_len , start_idx) , len(MDTs)):
        input_df = AdaBoost.df_input(factor_data , idx , windows_len)
        ada = AdaBoost(**input_df)
        ic_dfs.append(ada.fit(silence = True).calc_ic())
    df = pd.concat(ic_dfs)
    print(df)