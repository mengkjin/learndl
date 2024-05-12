import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Any , Literal , Optional

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
    var_date=['TradeDate','datetime'] 
    def __init__(self , 
                 train : Any = '../../data/tree_data/df_train.csv' , 
                 valid : Any = '../../data/tree_data/df_valid.csv' ,
                 test  : Any = '../../data/tree_data/df_test.csv' , 
                 use_features = None , 
                 plot_path = '../../figures' ,
                 cuda = False , **kwargs):   
        if 'weight_param' in kwargs.keys():
            self.weight_param = kwargs['weight_param']
        else:
            self.weight_param = {'tau':0.75*np.log(0.5)/np.log(0.75) , 'decay':'exp' , 'rate':0.5}  
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
        self.data_import(train , valid , test)
        self.use_features = use_features
        #self.data_prepare(use_features)

    def calc_ic(self , pred , label , dropna = False):
        df = pd.DataFrame({'pred': pred, 'label': label})
        var_date = [v for v in self.var_date if v in df.index.names][0]
        ic = df.groupby(var_date).apply(lambda df: df['pred'].corr(df['label']))
        ric = df.groupby(var_date).apply(lambda df: df['pred'].corr(df['label'], method='spearman'))
        return (ic.dropna(), ric.dropna()) if dropna else (ic, ric)
        
    def data_import(self , train , valid , test):
        if isinstance(train , str): train = pd.read_csv(train,index_col=[0,1])
        if isinstance(valid , str): valid = pd.read_csv(valid,index_col=[0,1])
        if isinstance(test , str):  test = pd.read_csv(test,index_col=[0,1])
        self.raw_dataset = {'train' : train.copy() , 'valid' : valid.copy() , 'test'  : test.copy()}
        assert len(self.raw_dataset['train'].columns) == len(self.raw_dataset['valid'].columns) == len(self.raw_dataset['test'].columns)

    def setup(self , use_features = None , weight_param = None):
        self.confirm_features(use_features)
        self.create_dataset(weight_param)

    def confirm_features(self , use_features = None):
        self.features = self.raw_dataset['train'].columns[:-1].values if use_features is None else use_features
        mono_constr = self.train_param['monotone_constraints']
        n_feat = len(self.features)
        if isinstance(mono_constr , list):
            if len(mono_constr) == 0: mono_constr = None
            elif len(mono_constr) != n_feat: mono_constr = (mono_constr*n_feat)[:n_feat]
        else:
            mono_constr = [mono_constr for _ in range(n_feat)]
        self.train_param['monotone_constraints'] = mono_constr

    def create_dataset(self , weight_param = None):
        weight_param = self.weight_param if weight_param is None else weight_param
        self.x_train = self.raw_dataset['train'].loc[:,self.features]
        self.x_valid = self.raw_dataset['valid'].loc[:,self.features] 
        self.y_train, self.y_valid = self.raw_dataset['train'].iloc[:,-1], self.raw_dataset['valid'].iloc[:,-1]
        assert self.y_train.values.ndim == 1 , "XGBoost doesn't support multi-label training"
        self.w_train = self.sample_weight(self.y_train , weight_param)
        self.w_valid = self.sample_weight(self.y_valid , weight_param)

        self.train_dataset = lgb.Dataset(self.x_train, self.y_train, weight=self.w_train)
        self.valid_dataset = lgb.Dataset(self.x_valid, self.y_valid, weight=self.w_valid , reference=self.train_dataset)

    def fit(self , use_features = None , weight_param = None):
        self.setup(use_features , weight_param)
        self.evals_result = dict()
        self.model = lgb.train(
            self.train_param ,
            self.train_dataset, 
            valid_sets=[self.train_dataset, self.valid_dataset], 
            valid_names=['train', 'valid'] , 
            num_boost_round=1000 , 
            callbacks=[lgb.record_evaluation(self.evals_result)],
        )
        
    def predict(self , inputs : Any = None):
        if inputs is None: inputs = self.raw_dataset['test'].loc[:,self.features]
        if isinstance(inputs , (pd.DataFrame | pd.Series)):
            return pd.Series(np.array(self.model.predict(inputs)) , index=inputs.index)
        else:
            return self.model.predict(inputs)
    
    def test_result(self):
        x_test = self.raw_dataset['test'].loc[:,self.features]
        label  = self.raw_dataset['test'].iloc[:,-1]
        pred = self.predict(x_test)
        ic , ric = self.calc_ic(pred, label , dropna=True)
        plt.figure()
        ric.cumsum().plot(title='average Rank IC = %.4f'%ric.mean())
        plt.savefig('/'.join([self.plot_path,'test_prediction.png']),dpi=1200)
        return {'ic':ic , 'ric':ric}
    
    @property
    def plot(self): return LgbtPlot(self)

    def sample_weight(self , dataset , weight_param : dict[str,Any]):
        return LgbtWeight.calculate_weight(dataset , weight_param)
    
class LgbtWeight:
    var_sec =['SecID','instrument']
    var_date=['TradeDate','datetime']

    def __init__(self , y : pd.Series , weight_param = {'tau':0.75*np.log(0.5)/np.log(0.75)}) -> None:
        self.y = y
        self.weight_param = weight_param

    @property
    def weight(self): return self.calculate_weight(self.y , self.weight_param)

    @classmethod
    def calculate_weight(cls , y : pd.Series , weight_param : Optional[dict] = {'tau':0.75*np.log(0.5)/np.log(0.75)}):
        weight_param = {} if weight_param is None else weight_param
        w = cls.raw_weight(y)
        w = cls.weight_top_return(y,weight_param.get('tau'),w)
        w = cls.weight_time_decay(y,weight_param.get('rate'),weight_param.get('decay'),w)
        return w

    @classmethod
    def pivot_factor(cls , factor_1d : pd.DataFrame | pd.Series):
        var_sec = [v for v in cls.var_sec if v in factor_1d.index.names][0]
        var_date = [v for v in cls.var_date if v in factor_1d.index.names][0]
        factor_1d = factor_1d.reset_index()
        factor_2d = factor_1d.pivot_table(index=var_sec,columns=var_date,values=factor_1d.columns[-1])
        return factor_2d

    @classmethod
    def melt_factor(cls , factor_2d : pd.DataFrame):
        var_sec = [v for v in cls.var_sec if v == factor_2d.index.name][0]
        var_date = [v for v in cls.var_date if v in factor_2d.columns.name][0]
        factor_2d[var_sec] = factor_2d.index
        factor_1d = factor_2d.melt(id_vars=var_sec, var_name=var_date)
        factor_1d = factor_1d.set_index([var_date,var_sec])
        return factor_1d
    
    @staticmethod
    def raw_weight(y : pd.Series): return y * 0 + 1

    @classmethod
    def weight_top_return(cls , y : pd.Series , tau : Optional[float] = 0.75*np.log(0.5)/np.log(0.75) , original_weight = None):
        weight = cls.raw_weight(y) if original_weight is None else original_weight
        if tau is None: return weight
        data = cls.pivot_factor(y)
        for dt in range(data.shape[1]):
            v = data.iloc[:,dt].to_numpy()
            v[~np.isnan(v)] = v[~np.isnan(v)].argsort()
            data.iloc[:,dt] = np.exp((1 - v / np.nanmax(v))*np.log(0.5) / tau)
        add_w = cls.melt_factor(data).dropna()
        add_w = add_w['value'] if isinstance(weight , pd.Series) else add_w.rename(columns={'value':weight.columns[0]})
        return weight * add_w

    @classmethod
    def weight_time_decay(cls , y : pd.Series , rate : Optional[float] = 0.5 , 
                          decay : Optional[Literal['lin' , 'exp']] = 'lin' , original_weight = None,):
        weight = cls.raw_weight(y) if original_weight is None else original_weight
        if rate is None or decay is None: return weight
        data = cls.pivot_factor(weight)
        l = data.shape[1]
        w = np.linspace(rate,1,l) if decay == 'lin' else np.power(2 , -np.arange(l)[::-1] / int(rate * l))
        data *= w.reshape(1,-1)
        add_w = cls.melt_factor(data).dropna()
        add_w = add_w['value'] if isinstance(weight , pd.Series) else add_w.rename(columns={'value':weight.columns[0]})
        return weight * add_w

class LgbtPlot:
    def __init__(self , lgb : Lgbt | Any) -> None:
        self.__lgbm = lgb

    @property
    def plot_path(self): return self.__lgbm.plot_path
    @property
    def evals_result(self): return self.__lgbm.evals_result
    @property
    def model(self): return self.__lgbm.model
    @property
    def raw_dataset(self): return self.__lgbm.raw_dataset
    @property
    def train_param(self): return self.__lgbm.train_param

    def training(self , show_plot = True , xlim = None , ylim = None , yscale = None):
        os.makedirs(self.plot_path, exist_ok=True)
        plt.figure()
        ax = lgb.plot_metric(self.evals_result, metric='l2')
        plt.scatter(self.model.best_iteration,list(self.evals_result['valid'].values())[0][self.model.best_iteration],label='best iteration')
        plt.legend()
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if yscale is not None: plt.yscale(yscale)
        if show_plot: plt.show()
        plt.savefig('/'.join([self.plot_path,'training_process.png']),dpi=1200)
        return ax
    
    def importance(self):
        os.makedirs(self.plot_path, exist_ok=True)
        lgb.plot_importance(self.model)
        plt.savefig('/'.join([self.plot_path,'feature_importance.png']),dpi=1200)

    def histogram(self , feature_idx='all'):
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
        os.makedirs(self.plot_path, exist_ok=True)
        for num_trees in num_trees_list:
            fig, ax = plt.subplots(figsize=(12,12))
            ax = lgb.plot_tree(self.model,tree_index=num_trees, ax=ax)
            plt.savefig('/'.join([self.plot_path , f'explainer_tree_{num_trees}.png']),dpi=1200)
    
    def sdt(self , group='train'):
        x = self.raw_dataset[group].iloc[:,:-1] 
        y_pred = pd.Series(self.model.predict(x.values), index=x.index)  # type: ignore
        dtrain = lgb.Dataset(x, label=y_pred)
        _params = deepcopy(self.train_param)
        del _params['early_stopping']
        SDT = lgb.train(_params, dtrain, num_boost_round=1)
        fig, ax = plt.subplots(figsize=(12,12))
        ax = lgb.plot_tree(SDT, tree_index=0, ax=ax)
        plt.savefig('/'.join([self.plot_path,'explainer_sdt.png']),dpi=1200)

    def pdp(self , group='train'):
        os.makedirs('/'.join([self.plot_path , 'explainer_pdp']) , exist_ok=True)
        for file in os.listdir('/'.join([self.plot_path , 'explainer_pdp'])):
            os.remove('/'.join([self.plot_path , 'explainer_pdp' , file]))
        for feature , imp in zip(self.model.feature_name() , self.model.feature_importance()):
            if imp == 0: continue
            x = deepcopy(self.raw_dataset[group].iloc[:,:-1])
            # when calculating PDP，factor range is -5:0.2:5
            x_range = np.arange(np.floor(min(x.loc[:,feature])),
                                np.ceil(max(x.loc[:,feature])),
                                0.2)
            
            # initialization and calculation
            pdp = np.zeros_like(x_range)
            for i, c in enumerate(x_range):
                x.loc[:,feature] = c
                pdp[i] = self.model.predict(x.values).mean() # type: ignore

            # plotPDP
            plt.figure()
            plt.plot(x_range,pdp)
            plt.title(f'PDP of {feature}')
            plt.xlabel(f'{feature}')
            plt.ylabel('y')
            plt.savefig('/'.join([self.plot_path , 'explainer_pdp' , f'explainer_pdp_{feature}.png']))
            plt.close()

    def shap(self , group='train'):
        import shap
        
        # 定义计算SHAP模型，这里使用TreeExplainer
        explainer = shap.TreeExplainer(self.model)
        X_df = deepcopy(self.raw_dataset[group].iloc[:,:-1].copy())
            
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
