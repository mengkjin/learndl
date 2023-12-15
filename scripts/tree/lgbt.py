import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy , h5py
import os

plt.style.use('seaborn-v0_8-dark-palette') 

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
class lgbm():
    def __init__(self , 
                 train = '../../data/tree_data/df_train.csv' , 
                 valid = '../../data/tree_data/df_valid.csv' ,
                 test  = '../../data/tree_data/df_test.csv' , 
                 use_features = None , 
                 plot_path = '../../figures' ,
                 cuda = False , **kwargs):   
        self.var_sec =['SecID','instrument']
        self.var_date=['TradeDate','datetime'] 
        self.weight_param = {'tau':0.75*np.log(0.5)/np.log(0.75)} if 'weight_param' not in kwargs.keys() else kwargs['weight_param']
        # {'tau':0.75*np.log(0.5)/np.log(0.75) , 'time':'exp' , 'rate':0.5}   
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
            'device_type': 'gpu' if cuda else 'cpu', # 'cuda' 'cpu'
            'seed': 42,
        }
        self.plot_path = plot_path
        self.train_param.update(kwargs)

        self.use_features = use_features
        self.dataset_import(train , valid , test)
        self.dataset_prepare()        
        
    def pivot_factor(self,factor_1d):
        var_sec = [v for v in self.var_sec if v in factor_1d.index.names][0]
        var_date = [v for v in self.var_date if v in factor_1d.index.names][0]
        factor_1d = factor_1d.reset_index()
        factor_2d = factor_1d.pivot_table(index=var_sec,columns=var_date,values=factor_1d.columns[-1])
        return factor_2d

    def melt_factor(self,factor_2d):
        var_sec = [v for v in self.var_sec if v == factor_2d.index.name][0]
        var_date = [v for v in self.var_date if v in factor_2d.columns.name][0]
        factor_2d[var_sec] = factor_2d.index
        factor_1d = factor_2d.melt(id_vars=var_sec, var_name=var_date)
        factor_1d = factor_1d.set_index([var_date,var_sec])
        return factor_1d

    def weight_top_return(self , y , tau=0.75*np.log(0.5)/np.log(0.75) , original_weight = None):
        weight = pd.DataFrame(1,index=y.index,columns=[y.name]) if original_weight is None else original_weight
        data = self.pivot_factor(y)
        for dt in range(data.shape[1]):
            rank = data.iloc[:,dt].argsort()
            rank[rank < 0] = np.nan
            rank = rank.argsort()
            data.iloc[:,dt] = np.exp((1 - rank / rank.max())*np.log(0.5)/tau)
            data.iloc[rank < 0 , dt] = 0
        weight[y.name] = weight[y.name] * self.melt_factor(data)['value']
        return weight

    def weight_time_decay(self , y , func,rate = 0.5 , original_weight = None,):
        if rate is None: rate = 0.5
        weight = pd.DataFrame(1,index=y.index,columns=[y.name]) if original_weight is None else original_weight
        var_date = [v for v in self.var_date if v in y.index.names][0]
        dates = y.index.get_level_values(level=var_date).drop_duplicates() 
        if func == 'linear': # 线性衰减
            w = np.linspace(rate,1,len(dates))
        elif func == 'exp': # 指数衰减
            w = np.linspace(1,len(dates),len(dates))
            w = np.power(2,(w-len(w))/int(rate*len(w)))
        w = pd.Series(w,index=dates.values) 
        # w.plot() # 绘图
        for date in dates: weight.loc[date,:] = weight.loc[date,:].values * w[date]
        return weight

    def calc_ic(self , pred , label , dropna = False):
        df = pd.DataFrame({"pred": pred, "label": label})
        var_date = [v for v in self.var_date if v in df.index.names][0]
        ic = df.groupby(var_date).apply(lambda df: df["pred"].corr(df["label"]))
        ric = df.groupby(var_date).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
        return (ic.dropna(), ric.dropna()) if dropna else (ic, ric)
        
    def dataset_import(self , train , valid , test):
        if isinstance(train , str): train = pd.read_csv(train,index_col=[0,1])
        if isinstance(valid , str): valid = pd.read_csv(valid,index_col=[0,1])
        if isinstance(test , str): test = pd.read_csv(test,index_col=[0,1])
        self.raw_dataset = {
            'train' : train.fillna(0).copy() ,
            'valid' : valid.fillna(0).copy() ,
            'test'  : test.fillna(0).copy()
        }
        assert len(self.raw_dataset['train'].columns) == len(self.raw_dataset['valid'].columns) == len(self.raw_dataset['test'].columns)
        if self.use_features is None:
            self.features = self.raw_dataset['train'].columns[:-1]
        else:
            self.features = self.use_features
        if isinstance(self.train_param['monotone_constraints'] , list):
            if len(self.train_param['monotone_constraints']) == 0:
                self.train_param['monotone_constraints'] = None
            elif len(self.train_param['monotone_constraints']) != len(self.features):
                self.train_param['monotone_constraints'] = (self.train_param['monotone_constraints']*len(self.features))[:len(self.features)]
        else:
            self.train_param['monotone_constraints'] = [self.train_param['monotone_constraints']]*len(self.features)
    
    def dataset_prepare(self , weight_param = None):
        weight_param = self.weight_param if weight_param is None else weight_param
        x_train, x_valid = self.raw_dataset['train'].loc[:,self.features], self.raw_dataset['valid'].loc[:,self.features]
        y_train, y_valid = self.raw_dataset['train'].iloc[:,-1], self.raw_dataset['valid'].iloc[:,-1]
        assert y_train.values.ndim == 1 , "XGBoost doesn't support multi-label training"
        w_train, w_valid = pd.DataFrame(1,index=y_train.index,columns=[y_train.name]) , pd.DataFrame(1,index=y_valid.index,columns=[y_valid.name])
        if 'tau' in weight_param.keys():
            w_train = self.weight_top_return(y_train,weight_param.get('tau'),w_train)
            w_valid = self.weight_top_return(y_valid,weight_param.get('tau'),w_valid)
        if 'time' in weight_param.keys():
            w_train = self.weight_time_decay(y_train,weight_param.get('time'),weight_param.get('rate'),w_train)
            w_valid = self.weight_time_decay(y_valid,weight_param.get('time'),weight_param.get('rate'),w_valid)

        self.train_dataset = lgb.Dataset(x_train, y_train, weight=w_train.iloc[:,0])
        self.valid_dataset = lgb.Dataset(x_valid, y_valid, weight=w_valid.iloc[:,0] , reference=self.train_dataset)

    def change_features(self , use_features):
        self.use_features = use_features
        if self.use_features is None:
            self.features = self.raw_dataset['train'].columns[:-1]
        else:
            self.features = self.use_features
        if isinstance(self.train_param['monotone_constraints'] , list):
            if len(self.train_param['monotone_constraints']) == 0:
                self.train_param['monotone_constraints'] = None
            elif len(self.train_param['monotone_constraints']) != len(self.features):
                self.train_param['monotone_constraints'] = (self.train_param['monotone_constraints']*len(self.features))[:len(self.features)]
        else:
            self.train_param['monotone_constraints'] = [self.train_param['monotone_constraints']]*len(self.features)
        self.dataset_prepare()        

    def train_model(self , show_plot = False):
        self.evals_result = dict()
        self.model = lgb.train(
            self.train_param ,
            self.train_dataset, 
            valid_sets=[self.train_dataset, self.valid_dataset], 
            valid_names=['train', 'valid'] , 
            num_boost_round=1000 , 
            callbacks=[lgb.record_evaluation(self.evals_result)],
        )
        if show_plot: self.plot_training()
        
    def test_prediction(self , show_plot = True):
        pred = pd.Series(self.model.predict(self.raw_dataset['test'].loc[:,self.features]), index=self.raw_dataset['test'].index)
        ic, ric = self.calc_ic(pred, self.raw_dataset['test'].iloc[:,-1], dropna=True)
        if show_plot: 
            plt.figure()
            ric.cumsum().plot(title='average Rank IC = %.4f'%ric.mean())
            plt.savefig('/'.join([self.plot_path,'test_prediction.png']),dpi=1200)
        return pred , (ic, ric)
    
    def plot_training(self , show_plot = True , xlim = None , ylim = None , yscale = None):
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
    
    def plot_importance(self):
        os.makedirs(self.plot_path, exist_ok=True)
        lgb.plot_importance(self.model)
        plt.savefig('/'.join([self.plot_path,'feature_importance.png']),dpi=1200)

    def plot_histogram(self , feature_idx='all'):
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

    def plot_tree(self , num_trees_list=[0]):   
        os.makedirs(self.plot_path, exist_ok=True)
        for num_trees in num_trees_list:
            fig, ax = plt.subplots(figsize=(12,12))
            ax = lgb.plot_tree(self.model,tree_index=num_trees, ax=ax)
            plt.savefig('/'.join([self.plot_path , f'explainer_tree_{num_trees}.png']),dpi=1200)
    
    def plot_sdt(self , group='train'):
        x = self.raw_dataset[group].iloc[:,:-1] 
        y_pred = pd.Series(self.model.predict(x.values), index=x.index) 
        dtrain = lgb.Dataset(x, label=y_pred)
        _params = copy.deepcopy(self.train_param)
        del _params['early_stopping']
        SDT = lgb.train(_params, dtrain, num_boost_round=1)
        fig, ax = plt.subplots(figsize=(12,12))
        ax = lgb.plot_tree(SDT, tree_index=0, ax=ax)
        plt.savefig('/'.join([self.plot_path,'explainer_sdt.png']),dpi=1200)

    def plot_pdp(self , group='train'):
        os.makedirs('/'.join([self.plot_path , 'explainer_pdp']) , exist_ok=True)
        for file in os.listdir('/'.join([self.plot_path , 'explainer_pdp'])):
            os.remove('/'.join([self.plot_path , 'explainer_pdp' , file]))
        for feature , imp in zip(self.model.feature_name() , self.model.feature_importance()):
            if imp == 0: continue
            x = copy.deepcopy(self.raw_dataset[group].iloc[:,:-1])
            # when calculating PDP，factor range is -5:0.2:5
            x_range = np.arange(np.floor(min(x.loc[:,feature])),
                                np.ceil(max(x.loc[:,feature])),
                                0.2)
            
            # initialization and calculation
            pdp = np.zeros_like(x_range)
            for i, c in enumerate(x_range):
                x.loc[:,feature] = c
                pdp[i] = self.model.predict(x.values).mean()

            # plotPDP
            plt.figure()
            plt.plot(x_range,pdp)
            plt.title(f'PDP of {feature}')
            plt.xlabel(f'{feature}')
            plt.ylabel('y')
            plt.savefig('/'.join([self.plot_path , 'explainer_pdp' , f'explainer_pdp_{feature}.png']))
            plt.close()

    def plot_shap(self , group='train'):
        import shap
        
        # 定义计算SHAP模型，这里使用TreeExplainer
        explainer = shap.TreeExplainer(self.model)
        X_df = copy.deepcopy(self.raw_dataset[group].iloc[:,:-1].copy())
            
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

# %%
if __name__ == '__main__':
    # %%
    from warnings import simplefilter
    simplefilter(action="ignore",category=FutureWarning)
    plt.style.use('seaborn-v0_8') 
    dict_df = {
        'train' : pd.read_csv('../../data/tree_data/df_train.csv' , index_col=[0,1]) , 
        'valid' : pd.read_csv('../../data/tree_data/df_valid.csv' , index_col=[0,1]) , 
        'test'  : pd.read_csv('../../data/tree_data/df_test.csv' , index_col=[0,1]) , 
    }


    # %%
    a = lgbm(**dict_df)
    a.train_model()

    # %%
    seqn = np.arange(dict_df['train'].shape[1])
    use_features = dict_df['train'].iloc[:,[*seqn[-22:-1]]].columns.values
    print(use_features)
    a.change_features(use_features)
    a.train_model()

    # %%
    a.plot_training()
    a.plot_training(xlim=[0,a.model.best_iteration+10] , ylim=[0.8,4.])
    pred = a.test_prediction()
    a.plot_importance()
    a.plot_histogram()

    # %%
    a.plot_tree()
    a.plot_sdt('train')
    a.plot_pdp('train')
    if a.train_param['linear_tree']==False:
        a.plot_shap('train') # Error now due to Numpy >= 1.24 and shap from pip not compatible


