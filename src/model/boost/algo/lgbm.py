import torch
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from pathlib import Path
from typing import Any

from .basic import BasicBoosterModel
from ..io import BoosterInput

from ....func import match_values

PLOT_PATH : Path | None = None

class Lgbm(BasicBoosterModel):
    DEFAULT_TRAIN_PARAM = {
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
        'monotone_constraints': 0 , 
        'early_stopping' : 50 , 
        'zero_as_missing' : False ,
        'device_type': 'cpu',
        'seed': 42,
    }

    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silence = False):
        if train is None: train = self.data['train']
        x , y , w = train.XYW()
        train_set = lgb.Dataset(x.cpu().numpy() , y.cpu().numpy() , weight = w.cpu().numpy())

        if valid is None: valid = self.data['valid']
        x , y , w = valid.XYW()
        valid_set = lgb.Dataset(x.cpu().numpy() , y.cpu().numpy() , weight = w.cpu().numpy() , reference = train_set)
        
        train_param = deepcopy(self.train_param)
        train_param = {k:v for k,v in train_param.items() if k in self.DEFAULT_TRAIN_PARAM}
        train_param.update({'seed':self.seed , 
                            'device_type': 'gpu' if self.cuda and torch.cuda.is_available() else 'cpu' ,
                            'monotone_constraints' : self.mono_constr(train_param['monotone_constraints'] , 
                                                                      len(train.use_feature)) ,
                            'verbosity': -1 if silence else 1}) 
        print(silence)

        self.evals_result = dict()
        self.model : lgb.Booster = lgb.train(
            train_param ,
            train_set = train_set, 
            valid_sets= [train_set, valid_set], 
            valid_names=['train', 'valid'] , 
            num_boost_round=1000 , 
            callbacks=[lgb.record_evaluation(self.evals_result)],
        )
        return self
        
    def predict(self , test : BoosterInput | Any = None):
        if test is None: test = self.data['test']
        x = test.X().cpu().numpy()
        return test.output(self.model.predict(x))
    
    def to_dict(self):
        model_dict = super().to_dict()
        model_dict['model'] = self.model.model_to_string()
        return model_dict
    
    def load_dict(self , model_dict : dict , cuda = False , seed = None):
        super().load_dict(model_dict , cuda , seed)
        self.model = lgb.Booster(model_str = model_dict['model'])
        return self
    
    @property
    def plot(self): return LgbmPlot(self)

    @staticmethod
    def mono_constr(raw_mono_constr , nfeat : int):
        if isinstance(raw_mono_constr , list):
            if len(raw_mono_constr) == 0: 
                mono_constr = None
            elif len(raw_mono_constr) == 1: 
                mono_constr = raw_mono_constr * nfeat
            else: 
                assert len(raw_mono_constr) == nfeat , (len(raw_mono_constr) , nfeat)
                mono_constr = raw_mono_constr
        else:
            mono_constr = [raw_mono_constr] * nfeat
        return mono_constr

class LgbmPlot:
    def __init__(self , lgb : 'Lgbm' , plot_path : Path | None = PLOT_PATH) -> None:
        self.lgb = lgb
        self.plot_path = plot_path
        if self.plot_path is not None: self.plot_path.mkdir(exist_ok= True)

    def training(self , show_plot = True , xlim = None , ylim = None , yscale = None):
        plt.figure()
        ax = lgb.plot_metric(self.lgb.evals_result, metric='l2')
        plt.scatter(self.lgb.model.best_iteration,list(self.lgb.evals_result['valid'].values())\
                    [0][self.lgb.model.best_iteration],label='best iteration')
        plt.legend()
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if yscale is not None: plt.yscale(yscale)
        if show_plot: plt.show()
        if self.plot_path:
            self.plot_path.joinpath('training_process.png')
            plt.savefig(self.plot_path.joinpath('training_process.png'),dpi=1200)
        return ax
    
    def importance(self):
        lgb.plot_importance(self.lgb.model)
        if self.plot_path:
            plt.savefig(self.plot_path.joinpath('feature_importance.png'),dpi=1200)

    def histogram(self , feature_idx='all'):
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        if isinstance(feature_idx,str):
            assert feature_idx=='all'
            feature_idx = range(len(self.lgb.model.feature_name()))
        n_subplot = len(feature_idx)
        ncol = n_subplot if n_subplot < 5 else 5
        nrow = n_subplot // 5 + (1 if (n_subplot % 5 > 0) else 0)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))
        if isinstance(axes , np.ndarray): 
            axes = axes.flatten()
        else:
            axes = [axes]
        for i in feature_idx:
            feature_importance = self.lgb.model.feature_importance()[i]
            if feature_importance ==0:
                axes[i].set_title(f'feature {self.lgb.model.feature_name()[i]} has 0 importance')
            else:
                lgb.plot_split_value_histogram(self.lgb.model, ax = axes[i] , 
                                               feature=self.lgb.model.feature_name()[i], 
                                               bins='auto' ,title="feature @feature@")
        fig.suptitle('split value histogram for feature(s)' , fontsize = 'x-large')
        plt.tight_layout()
        plt.savefig(self.plot_path.joinpath('feature_histogram.png'),dpi=1200)

    def tree(self , num_trees_list=[0]):   
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        for num_trees in num_trees_list:
            fig, ax = plt.subplots(figsize=(12,12))
            ax = lgb.plot_tree(self.lgb.model,tree_index=num_trees, ax=ax)
            if self.plot_path:
                plt.savefig(self.plot_path.joinpath(f'explainer_tree_{num_trees}.png'),dpi=1200)

    def shap(self , train : BoosterInput | Any = None):
        if train is None: train = self.lgb.data['train']
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        import shap
        
        # 定义计算SHAP模型，这里使用TreeExplainer
        explainer = shap.TreeExplainer(self.lgb.model)
        X_df = deepcopy(train.X())
            
        # 计算全部因子SHAP
        shap_values = explainer.shap_values(X_df)
        self.plot_path.joinpath('explainer_shap').mkdir(exist_ok=True)
        [file.unlink() for file in self.plot_path.joinpath('explainer_shap').iterdir()]
        
        # 全部特征SHAP柱状图
        shap.summary_plot(shap_values,X_df,plot_type='bar',title='|SHAP|',show=False)
        plt.savefig(self.plot_path.joinpath('explainer_shap' , 'explainer_shap_bar.png') ,dpi=100,bbox_inches='tight')
        
        # 全部特征SHAP点图
        shap.summary_plot(shap_values,X_df,plot_type='dot',title='SHAP',show=False)
        plt.savefig(self.plot_path.joinpath('explainer_shap' , 'explainer_shap_dot.png') ,dpi=100,bbox_inches='tight')
        
        # 单个特征SHAP点图
        
        for feature , imp in zip(self.lgb.model.feature_name() , self.lgb.model.feature_importance()):
            if imp == 0: continue
            shap.dependence_plot(feature,shap_values,X_df,interaction_index=None,title=f'SHAP of {feature}',show=False)
            plt.savefig(self.plot_path.joinpath('explainer_shap' , f'explainer_shap_dot_{feature}.png'),dpi=100,bbox_inches='tight')
    
    def sdt(self , train : BoosterInput | Any = None):
        if train is None: train = self.lgb.data['train']
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        x = train.X().cpu().numpy()
        pred = np.array(self.lgb.model.predict(x))
        dtrain = lgb.Dataset(x, label=pred)
        _params = deepcopy(self.lgb.train_param)
        del _params['early_stopping']
        SDT = lgb.train(_params, dtrain, num_boost_round=1)
        fig, ax = plt.subplots(figsize=(12,12))
        ax = lgb.plot_tree(SDT, tree_index=0, ax=ax)
        plt.savefig(self.plot_path.joinpath('explainer_sdt.png'),dpi=1200)

    def pdp(self , train : BoosterInput | Any = None):
        if train is None: train = self.lgb.data['train']
        if self.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        self.plot_path.joinpath('explainer_pdp').mkdir(exist_ok=True)
        [file.unlink() for file in self.plot_path.joinpath('explainer_pdp').iterdir()]

        for feature , imp in zip(self.lgb.model.feature_name() , self.lgb.model.feature_importance()):
            if imp == 0: continue
            x = deepcopy(train.X()).cpu().numpy()
            if isinstance(x , pd.DataFrame): x = x.values
            ifeat = match_values(feature , train.feature)
            # when calculating PDP，factor range is -5:0.2:5
            x_range = np.arange(np.floor(min(x[:,ifeat])), np.ceil(max(x[:,ifeat])), 0.2)
            
            # initialization and calculation
            pdp = np.zeros_like(x_range)
            for i, c in enumerate(x_range):
                x[:,ifeat] = c
                pdp[i] = np.array(self.lgb.model.predict(x)).mean()

            # plotPDP
            plt.figure()
            plt.plot(x_range,pdp)
            plt.title(f'PDP of {feature}')
            plt.xlabel(f'{feature}')
            plt.ylabel('y')
            plt.savefig(self.plot_path.joinpath('explainer_pdp' , f'explainer_pdp_{feature}.png'))
            plt.close()
    
def main():
    dict_df : dict[str,Any] = {
        'train' : pd.read_csv('../../data/tree_data/df_train.csv' , index_col=[0,1]) , 
        'valid' : pd.read_csv('../../data/tree_data/df_valid.csv' , index_col=[0,1]) , 
        'test'  : pd.read_csv('../../data/tree_data/df_test.csv' , index_col=[0,1]) , 
    }

    # %%
    a = Lgbm().import_data(**dict_df)
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
    a.plot.sdt()
    a.plot.pdp()
    if a.train_param['linear_tree']==False:
        a.plot.shap() # Error now due to Numpy >= 1.24 and shap from pip not compatible
# %%
if __name__ == '__main__':
    main()
