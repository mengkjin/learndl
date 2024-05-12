import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os , torch

from copy import deepcopy
from dataclasses import dataclass
from src.environ import DIR
# from src.algo.boost.lgbt import LgbtPlot , LgbtWeight
from typing import Any , ClassVar , Literal , Optional

from ...classes import BoosterData
from ...func import match_values, np_nanic_2d , np_nanrankic_2d

plt.style.use('seaborn-v0_8') 
# %%
"""
file_path = '/root/autodl-tmp/rnn_fac/data/risk_model.h5'
rm_file = h5py.File(file_path , mode='r+')
# pd.DataFrame(rm_file.get('20000106')[:])
file_path = '/root/autodl-tmp/rnn_fac/data/day_ylabels_data.h5'
y_file = h5py.File(file_path , mode='r+')
y , yTradeDate , ySecID = tuple([y_file.get(arr)[:] for arr in ['Y10' , 'TradeDate' , 'SecID']])

TradeDate = sorted(np.intersect1d(np.array(list(rm_file.keys() - ['colnames'])).astype(int) , yTradeDate))
dict_date = {
    'test':TradeDate[-50:],
    'valid':TradeDate[-50:],
    'train':TradeDate[-200:-50],
}
dict_df = {}
feature_names = np.concatenate((rm_file.get('colnames')[6:].astype(str),['y']))
for set_name in ['test' , 'valid' , 'train']:
    df_set = None
    for d in dict_date[set_name]:
        tmp = pd.DataFrame(rm_file.get(str(d))[:])
        pos = np.intersect1d(tmp.SecID , ySecID , return_indices= True)
        df = pd.DataFrame({'TradeDate' : str(d) , **tmp.iloc[pos[1],:] , 'y':y[pos[2],np.where(yTradeDate == d)].flatten()})
        df = df.set_index(['TradeDate','SecID']).loc[:,feature_names]
        df_set = pd.concat((df_set , df))
    dict_df[set_name] = df_set
"""
# %%

class Lgbt():
    var_date = ['TradeDate','datetime'] 
    def __init__(self , 
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 feature = None , 
                 plot_path = None, # '../../figures' ,
                 cuda = False , **kwargs):   
        self.train_param = {
            'objective': 'regression', 
            'verbosity': -1 , 
            'linear_tree': True, 
            'learning_rate': 0.3, 
            'lambda_l2': 1e-05, 
            'alpha': 1e-07, 
            'num_leaves': 31,
            'max_depth': 6, 
            # 'min_data_in_leaf' : 1, 
            'min_sum_hessian_in_leaf': 1, 
            'feature_fraction': 0.6, 
            'bagging_fraction': 0.75, 
            'force_col_wise': True, 
            'monotone_constraints': 1 , 
            'early_stopping' : 50 , 
            'zero_as_missing' : False ,
            'device_type': 'gpu' if cuda else 'cpu', # 'cuda' 'cpu'
            'seed': 42,
        }
        self.plot_path = plot_path
        self.train_param.update(kwargs)
        self.data_import(train , valid , test , feature)

    def data_import(self , train , valid , test , feature = None):
        assert type(train) == type(valid) == type(test) , f'type of train/valid/test must be identical'
        self.data : dict[str , BoosterData] = {}
        if train is None: return
        if isinstance(train , str):
            train = pd.read_csv(train,index_col=[0,1])
            valid = pd.read_csv(valid,index_col=[0,1])
            test  = pd.read_csv(test ,index_col=[0,1])
        assert train.shape[1] == valid.shape[1] == test.shape[1] , (train.shape[1] , valid.shape[1] , test.shape[1])
        if isinstance(train , pd.DataFrame) and isinstance(valid , pd.DataFrame) and isinstance(test , pd.DataFrame):
            assert type(train) == type(valid) == type(test) , f'type of train/valid/test must be identical'
            self.data['train'] = BoosterData(train.iloc[:,:-1] , train.iloc[:,-1] , feature = feature)
            self.data['valid'] = BoosterData(valid.iloc[:,:-1] , valid.iloc[:,-1] , feature = feature)
            self.data['test']  = BoosterData(test.iloc[:,:-1]  , test.iloc[:,-1]  , feature = feature)
        elif not (isinstance(train , pd.DataFrame) or isinstance(valid , pd.DataFrame) or isinstance(test , pd.DataFrame)):
            self.data['train'] = BoosterData(train[...,:-1] , train[...,-1] , feature = feature)
            self.data['valid'] = BoosterData(valid[...,:-1] , valid[...,-1] , feature = feature)
            self.data['test']  = BoosterData(test[...,:-1]  , test[...,-1]  , feature = feature)

    def setup(self , use_feature = None , weight_param = None):
        mono_constr = self.train_param['monotone_constraints']
        nfeat = len(use_feature) if use_feature else self.data['train'].nfeat
        if isinstance(mono_constr , list):
            if len(mono_constr) == 0: mono_constr = None
            elif len(mono_constr) != nfeat: mono_constr = (mono_constr * nfeat)[:nfeat]
        else:
            mono_constr = [mono_constr for _ in range(nfeat)]
        self.train_param['monotone_constraints'] = mono_constr
        
        [d.update_feature(use_feature) for d in self.data.values()]
        self.train_dataset = self.lgbt_dataset(self.data['train'] , weight_param)
        self.valid_dataset = self.lgbt_dataset(self.data['valid'] , weight_param , reference = self.train_dataset)

    def lgbt_dataset(self , data : BoosterData , weight_param = None , reference = None):
        return lgb.Dataset(data.X() , data.Y() , weight = data.W(weight_param) , reference = reference)

    def fit(self , use_feature = None , weight_param = None):
        self.setup(use_feature , weight_param)
        self.evals_result = dict()
        self.model = lgb.train(
            self.train_param ,
            self.train_dataset, 
            valid_sets=[self.train_dataset, self.valid_dataset], 
            valid_names=['train', 'valid'] , 
            num_boost_round=1000 , 
            callbacks=[lgb.record_evaluation(self.evals_result)],
        )
        
    def predict(self , inputs : Optional[BoosterData] = None , reform = True):
        if inputs is None: inputs = self.data['test']
        pred = self.model.predict(inputs.X())
        if reform: pred = inputs.reform_pred(pred)
        return pred
    
    def test_result(self):
        pred = np.array(self.predict(self.data['test'], reform = False)).reshape(*self.data['test'].y)
        df = self.calc_ic(pred , self.data['test'].y)
        plt.figure()
        df.cumsum().plot(title='average IC/RankIC = {:.4f}/{:.4f}'.format(*df.mean().values))
        if self.plot_path is not None:
            os.makedirs(self.plot_path, exist_ok=True)
            plt.savefig('/'.join([self.plot_path,'test_prediction.png']),dpi=1200)
        return df
    
    def calc_ic(self , pred : np.ndarray , label : np.ndarray):
        if pred.ndim == 1: pred , label = pred.reshape(-1,1) , label.reshape(-1,1)
        ic = np_nanic_2d(pred , label , dim = 0)
        ric = np_nanrankic_2d(pred , label , dim = 0)
        return pd.DataFrame({'ic' : ic , 'rankic' : ric})
    
    @property
    def plot(self): return LgbtPlot(self)

    @property
    def initiated(self): return bool(self.data)

class LgbtPlot:
    def __init__(self , lgb : Any) -> None:
        self.__lgbm = lgb

    @property
    def plot_path(self) -> str: return self.__lgbm.plot_path
    @property
    def evals_result(self) -> dict: return self.__lgbm.evals_result
    @property
    def model(self) -> lgb.Booster: return self.__lgbm.model
    @property
    def data(self) -> dict[str,BoosterData]: return self.__lgbm.data
    @property
    def train_param(self) -> dict: return self.__lgbm.train_param

    def training(self , show_plot = True , xlim = None , ylim = None , yscale = None):
        plt.figure()
        ax = lgb.plot_metric(self.evals_result, metric='l2')
        plt.scatter(self.model.best_iteration,list(self.evals_result['valid'].values())[0][self.model.best_iteration],label='best iteration')
        plt.legend()
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if yscale is not None: plt.yscale(yscale)
        if show_plot: plt.show()
        if self.plot_path is not None:
            os.makedirs(self.plot_path, exist_ok=True)
            plt.savefig('/'.join([self.plot_path,'training_process.png']),dpi=1200)
        return ax
    
    def importance(self):
        lgb.plot_importance(self.model)
        if self.plot_path is not None:
            os.makedirs(self.plot_path, exist_ok=True)
            plt.savefig('/'.join([self.plot_path,'feature_importance.png']),dpi=1200)

    def histogram(self , feature_idx='all'):
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        os.makedirs(self.plot_path, exist_ok=True)
        if isinstance(feature_idx,str):
            assert feature_idx=='all'
            feature_idx = range(len(self.model.feature_name()))
        n_subplot = len(feature_idx)
        ncol = n_subplot if n_subplot < 5 else 5
        nrow = n_subplot // 5 + (1 if (n_subplot % 5 > 0) else 0)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))
        if isinstance(axes , np.ndarray): 
            axes = axes.flatten()
        else:
            axes = [axes]
        for i in feature_idx:
            feature_importance = self.model.feature_importance()[i]
            if feature_importance ==0:
                axes[i].set_title(f'feature {self.model.feature_name()[i]} has 0 importance')
            else:
                lgb.plot_split_value_histogram(self.model, ax = axes[i] , feature=self.model.feature_name()[i], bins='auto' ,title="feature @feature@")
        fig.suptitle('split value histogram for feature(s)' , fontsize = 'x-large')
        plt.tight_layout()
        plt.savefig('/'.join([self.plot_path,'feature_histogram.png']),dpi=1200)

    def tree(self , num_trees_list=[0]):   
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        os.makedirs(self.plot_path, exist_ok=True)
        for num_trees in num_trees_list:
            fig, ax = plt.subplots(figsize=(12,12))
            ax = lgb.plot_tree(self.model,tree_index=num_trees, ax=ax)
            if self.plot_path is not None and os.path.exists(self.plot_path):
                plt.savefig('/'.join([self.plot_path , f'explainer_tree_{num_trees}.png']),dpi=1200)

    def shap(self , group='train'):
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        import shap
        
        # 定义计算SHAP模型，这里使用TreeExplainer
        explainer = shap.TreeExplainer(self.model)
        X_df = deepcopy(self.data[group].X())
            
        # 计算全部因子SHAP
        shap_values = explainer.shap_values(X_df)
        os.makedirs('/'.join([self.plot_path , self.plot_path , 'explainer_shap']) ,exist_ok=True)
        for file in os.listdir('/'.join([self.plot_path , 'explainer_shap'])):
            os.remove('/'.join([self.plot_path , 'explainer_shap' , file]))
        
        # 全部特征SHAP柱状图
        shap.summary_plot(shap_values,X_df,plot_type='bar',title='|SHAP|',show=False)
        plt.savefig('/'.join([self.plot_path , 'explainer_shap' , 'explainer_shap_bar.png']) ,dpi=100,bbox_inches='tight')
        
        # 全部特征SHAP点图
        shap.summary_plot(shap_values,X_df,plot_type='dot',title='SHAP',show=False)
        plt.savefig('/'.join([self.plot_path , 'explainer_shap' , 'explainer_shap_dot.png']),dpi=100,bbox_inches='tight')
        
        # 单个特征SHAP点图
        
        for feature , imp in zip(self.model.feature_name() , self.model.feature_importance()):
            if imp == 0: continue
            shap.dependence_plot(feature,shap_values,X_df,interaction_index=None,title=f'SHAP of {feature}',show=False)
            plt.savefig('/'.join([self.plot_path , 'explainer_shap' , f'explainer_shap_dot_{feature}.png']),dpi=100,bbox_inches='tight')
    
    def sdt(self , group='train'):
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        x = self.data[group].X()
        pred = self.model.predict(x)
        dtrain = lgb.Dataset(x, label=pred)
        _params = deepcopy(self.train_param)
        del _params['early_stopping']
        SDT = lgb.train(_params, dtrain, num_boost_round=1)
        fig, ax = plt.subplots(figsize=(12,12))
        ax = lgb.plot_tree(SDT, tree_index=0, ax=ax)
        plt.savefig('/'.join([self.plot_path,'explainer_sdt.png']),dpi=1200)

    def pdp(self , group='train'):
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        os.makedirs('/'.join([self.plot_path , 'explainer_pdp']) , exist_ok=True)
        for file in os.listdir('/'.join([self.plot_path , 'explainer_pdp'])):
            os.remove('/'.join([self.plot_path , 'explainer_pdp' , file]))
        for feature , imp in zip(self.model.feature_name() , self.model.feature_importance()):
            if imp == 0: continue
            x = deepcopy(self.data[group].X())
            if isinstance(x , pd.DataFrame): x = x.values
            ifeat = match_values(feature , self.data[group].feature)
            # when calculating PDP，factor range is -5:0.2:5
            x_range = np.arange(np.floor(min(x[:,ifeat])), np.ceil(max(x[:,ifeat])), 0.2)
            
            # initialization and calculation
            pdp = np.zeros_like(x_range)
            for i, c in enumerate(x_range):
                x[:,ifeat] = c
                pdp[i] = np.array(self.model.predict(x)).mean()

            # plotPDP
            plt.figure()
            plt.plot(x_range,pdp)
            plt.title(f'PDP of {feature}')
            plt.xlabel(f'{feature}')
            plt.ylabel('y')
            plt.savefig('/'.join([self.plot_path , 'explainer_pdp' , f'explainer_pdp_{feature}.png']))
            plt.close()
    
    
def main():
    plt.style.use('seaborn-v0_8') 
    dict_df : dict[str,Any] = {
        'train' : pd.read_csv('../../data/tree_data/df_train.csv' , index_col=[0,1]) , 
        'valid' : pd.read_csv('../../data/tree_data/df_valid.csv' , index_col=[0,1]) , 
        'test'  : pd.read_csv('../../data/tree_data/df_test.csv' , index_col=[0,1]) , 
    }

    # %%
    a = Lgbt(**dict_df)
    a.fit()

    # %%
    seqn = np.arange(dict_df['train'].shape[1])
    use_features = dict_df['train'].iloc[:,[*seqn[-22:-1]]].columns.values
    print(use_features)
    a.fit()
    a.plot.training()


    a.plot.training(xlim=[0,a.model.best_iteration+10] , ylim=[0.8,4.])
    a.test_result()
    a.plot.importance()
    a.plot.histogram()

    # %%
    a.plot.tree()
    a.plot.sdt('train')
    a.plot.pdp('train')
    if a.train_param['linear_tree']==False:
        a.plot.shap('train') # Error now due to Numpy >= 1.24 and shap from pip not compatible
# %%
if __name__ == '__main__':
    main()
