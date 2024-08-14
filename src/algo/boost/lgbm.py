import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os , torch

from copy import deepcopy
from typing import Any , Optional

from .basic import BasicBooster , BoosterData
from ...basic import PATH
from ...func import match_values

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
    'device_type': 'cpu', # 'cuda' 'cpu'
    'seed': 42,
}

class Lgbm(BasicBooster):
    def __init__(self , 
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 feature = None , 
                 plot_path = None, # '../../figures' ,
                 cuda = True , **kwargs):
        self.train_param = deepcopy(DEFAULT_TRAIN_PARAM)
        self.train_param.update(train_param)
        if cuda and torch.cuda.is_available(): self.train_param.update({'device_type': 'gpu'})
        self.plot_path = plot_path
        self.train_param.update({k:v for k,v in kwargs.items() if k in self.train_param.keys()})

        self.evals_result = dict()
        self.data : dict[str , BoosterData] = {}

        self.feature = feature
        self.data_import(train = train , valid = valid , test = test)

    def __repr__(self) -> str: return f'{self.__class__.__name__}'

    def fit(self , use_feature = None , train = None , valid = None , silence = False):
        self.data_import(train = train , valid = valid)
        
        mono_constr = self.train_param['monotone_constraints']
        nfeat = len(use_feature) if use_feature else self.data['train'].nfeat
        if isinstance(mono_constr , list):
            if len(mono_constr) == 0: mono_constr = None
            elif len(mono_constr) != nfeat: mono_constr = (mono_constr * nfeat)[:nfeat]
        else:
            mono_constr = [mono_constr for _ in range(nfeat)]
        self.train_param['monotone_constraints'] = mono_constr
        
        [d.update_feature(use_feature) for d in self.data.values()]

        x , y , w = self.data['train'].XYW()
        train_dataset = lgb.Dataset(x , y , weight = w)

        x , y , w = self.data['valid'].XYW()
        valid_dataset = lgb.Dataset(x , y , weight = w , reference = train_dataset)
        
        self.evals_result = dict()
        self.model = lgb.train(
            self.train_param ,
            train_dataset, 
            valid_sets=[train_dataset, valid_dataset], 
            valid_names=['train', 'valid'] , 
            num_boost_round=1000 , 
            callbacks=[lgb.record_evaluation(self.evals_result)],
        )
        return self
        
    def predict(self , x : Optional[BoosterData] = None , reshape = True , reform = True):
        if x is None: x = self.data['test']
        pred = np.array(self.model.predict(x.X()))
        if reshape: pred = x.reshape_pred(pred)
        if reform: pred = x.reform_pred(pred)
        return pred
    
    def to_dict(self):
        return {'model_str':self.model.model_to_string()}
    
    @classmethod
    def from_dict(cls , model_dict : dict , cuda = False):
        obj = cls(cuda = cuda)
        obj.model = lgb.Booster(model_str = model_dict['model_str'])
        return obj
    
    @property
    def plot(self): return LgbmPlot(self)
    
    @staticmethod
    def df_input() -> dict[str,Any]:
        train = pd.read_csv(f'{PATH.data}/tree_data/df_train.csv' , index_col=[0,1])
        valid = pd.read_csv(f'{PATH.data}/tree_data/df_valid.csv' , index_col=[0,1])
        test  = pd.read_csv(f'{PATH.data}/tree_data/df_test.csv' , index_col=[0,1])

        return {'train':train , 'valid':valid , 'test':test}

class LgbmPlot:
    def __init__(self , lgb : 'Lgbm') -> None:
        self.lgb = lgb

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
        if self.lgb.plot_path is not None:
            os.makedirs(self.lgb.plot_path, exist_ok=True)
            plt.savefig('/'.join([self.lgb.plot_path,'training_process.png']),dpi=1200)
        return ax
    
    def importance(self):
        lgb.plot_importance(self.lgb.model)
        if self.lgb.plot_path is not None:
            os.makedirs(self.lgb.plot_path, exist_ok=True)
            plt.savefig('/'.join([self.lgb.plot_path,'feature_importance.png']),dpi=1200)

    def histogram(self , feature_idx='all'):
        if self.lgb.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        os.makedirs(self.lgb.plot_path, exist_ok=True)
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
        plt.savefig('/'.join([self.lgb.plot_path,'feature_histogram.png']),dpi=1200)

    def tree(self , num_trees_list=[0]):   
        if self.lgb.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        os.makedirs(self.lgb.plot_path, exist_ok=True)
        for num_trees in num_trees_list:
            fig, ax = plt.subplots(figsize=(12,12))
            ax = lgb.plot_tree(self.lgb.model,tree_index=num_trees, ax=ax)
            if self.lgb.plot_path is not None and os.path.exists(self.lgb.plot_path):
                plt.savefig('/'.join([self.lgb.plot_path , f'explainer_tree_{num_trees}.png']),dpi=1200)

    def shap(self , group='train'):
        if self.lgb.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        import shap
        
        # 定义计算SHAP模型，这里使用TreeExplainer
        explainer = shap.TreeExplainer(self.lgb.model)
        X_df = deepcopy(self.lgb.data[group].X())
            
        # 计算全部因子SHAP
        shap_values = explainer.shap_values(X_df)
        os.makedirs('/'.join([self.lgb.plot_path , self.lgb.plot_path , 'explainer_shap']) ,exist_ok=True)
        for file in os.listdir('/'.join([self.lgb.plot_path , 'explainer_shap'])):
            os.remove('/'.join([self.lgb.plot_path , 'explainer_shap' , file]))
        
        # 全部特征SHAP柱状图
        shap.summary_plot(shap_values,X_df,plot_type='bar',title='|SHAP|',show=False)
        plt.savefig('/'.join([self.lgb.plot_path , 'explainer_shap' , 'explainer_shap_bar.png']) ,dpi=100,bbox_inches='tight')
        
        # 全部特征SHAP点图
        shap.summary_plot(shap_values,X_df,plot_type='dot',title='SHAP',show=False)
        plt.savefig('/'.join([self.lgb.plot_path , 'explainer_shap' , 'explainer_shap_dot.png']),dpi=100,bbox_inches='tight')
        
        # 单个特征SHAP点图
        
        for feature , imp in zip(self.lgb.model.feature_name() , self.lgb.model.feature_importance()):
            if imp == 0: continue
            shap.dependence_plot(feature,shap_values,X_df,interaction_index=None,title=f'SHAP of {feature}',show=False)
            plt.savefig('/'.join([self.lgb.plot_path , 'explainer_shap' , f'explainer_shap_dot_{feature}.png']),dpi=100,bbox_inches='tight')
    
    def sdt(self , group='train'):
        if self.lgb.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        x = self.lgb.data[group].X()
        pred = np.array(self.lgb.model.predict(x))
        dtrain = lgb.Dataset(x, label=pred)
        _params = deepcopy(self.lgb.train_param)
        del _params['early_stopping']
        SDT = lgb.train(_params, dtrain, num_boost_round=1)
        fig, ax = plt.subplots(figsize=(12,12))
        ax = lgb.plot_tree(SDT, tree_index=0, ax=ax)
        plt.savefig('/'.join([self.lgb.plot_path,'explainer_sdt.png']),dpi=1200)

    def pdp(self , group='train'):
        if self.lgb.plot_path is None:
            print(f'plot path not given, will not proceed')
            return
        os.makedirs('/'.join([self.lgb.plot_path , 'explainer_pdp']) , exist_ok=True)
        for file in os.listdir('/'.join([self.lgb.plot_path , 'explainer_pdp'])):
            os.remove('/'.join([self.lgb.plot_path , 'explainer_pdp' , file]))
        for feature , imp in zip(self.lgb.model.feature_name() , self.lgb.model.feature_importance()):
            if imp == 0: continue
            x = deepcopy(self.lgb.data[group].X())
            if isinstance(x , pd.DataFrame): x = x.values
            ifeat = match_values(feature , self.lgb.data[group].feature)
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
            plt.savefig('/'.join([self.lgb.plot_path , 'explainer_pdp' , f'explainer_pdp_{feature}.png']))
            plt.close()
    
    
def main():
    dict_df : dict[str,Any] = {
        'train' : pd.read_csv('../../data/tree_data/df_train.csv' , index_col=[0,1]) , 
        'valid' : pd.read_csv('../../data/tree_data/df_valid.csv' , index_col=[0,1]) , 
        'test'  : pd.read_csv('../../data/tree_data/df_test.csv' , index_col=[0,1]) , 
    }

    # %%
    a = Lgbm(**dict_df)
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
