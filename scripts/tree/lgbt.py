import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from learndl.scripts.data_utils.ModelData import pivot_factor , melt_factor
import h5py
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


class lgbm():
    def __init__(self , 
                 train = 'data/df_train.csv' , 
                 valid = 'data/df_valid.csv' ,
                 test  = 'data/df_test.csv' , 
                 cuda = False , **kwargs):    
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
        self.train_param.update(kwargs)
        self.dataset_import(train , valid , test)
        self.dataset_prepare()        
        
    def pivot_factor(self,factor_1d,var_sec=['SecID','instrument'],var_date=['TradeDate','datetime']):
        if isinstance(var_sec,(list,tuple)): var_sec = [v for v in var_sec if v in factor_1d.index.names][0]
        if isinstance(var_date,(list,tuple)): var_date = [v for v in var_date if v in factor_1d.index.names][0]
        factor_1d = factor_1d.reset_index()
        factor_2d = factor_1d.pivot_table(index=var_sec,columns=var_date,values=factor_1d.columns[-1])
        return factor_2d

    def melt_factor(self,factor_2d,var_sec=['SecID','instrument'],var_date=['TradeDate','datetime']):
        if isinstance(var_sec,(list,tuple)): var_sec = [v for v in var_sec if v == factor_2d.index.name][0]
        if isinstance(var_date,(list,tuple)): var_date = [v for v in var_date if v in factor_2d.columns.name][0]
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

    def weight_time_decay(self , y , func,rate = 0.5 , original_weight = None):
        if rate is None: rate = 0.5
        weight = pd.DataFrame(1,index=y.index,columns=[y.name]) if original_weight is None else original_weight
        dates = y.index.get_level_values(level='TradeDate').drop_duplicates() 
        if func == 'linear': # 线性衰减
            w = np.linspace(rate,1,len(dates))
        elif func == 'exp': # 指数衰减
            w = np.linspace(1,len(dates),len(dates))
            w = np.power(2,(w-len(w))/int(rate*len(w)))
        w = pd.Series(w,index=dates.values) 
        # w.plot() # 绘图
        for date in dates: weight.loc[date,:] = weight.loc[date,:].values * w[date]
        return weight

    def calc_ic(self , pred , label , date_col = "TradeDate", dropna = False):
        df = pd.DataFrame({"pred": pred, "label": label})
        ic = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"]))
        ric = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
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
        assert self.raw_dataset['train'].isna().sum().sum() == 0
        assert self.raw_dataset['valid'].isna().sum().sum() == 0
        assert self.raw_dataset['test'].isna().sum().sum() == 0
        assert len(self.raw_dataset['train'].columns) == len(self.raw_dataset['valid'].columns) == len(self.raw_dataset['test'].columns)
        self.n_features = len(self.raw_dataset['train'].columns)-1
        if isinstance(self.train_param['monotone_constraints'] , list):
            if len(self.train_param['monotone_constraints']) == 0:
                self.train_param['monotone_constraints'] = None
            elif len(self.train_param['monotone_constraints']) != self.n_features:
                self.train_param['monotone_constraints'] = (self.train_param['monotone_constraints']*self.n_features)[:self.n_features]
        else:
            self.train_param['monotone_constraints'] = [self.train_param['monotone_constraints']]*self.n_features
    
    def dataset_prepare(self , weight_param = None):
        weight_param = self.weight_param if weight_param is None else weight_param
        x_train, x_valid = self.raw_dataset['train'].iloc[:,:self.n_features], self.raw_dataset['valid'].iloc[:,:self.n_features]
        y_train, y_valid = self.raw_dataset['train'].iloc[:,self.n_features], self.raw_dataset['valid'].iloc[:,self.n_features]
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
        pred = pd.Series(self.model.predict(self.raw_dataset['test'].iloc[:,:-1]), index=self.raw_dataset['test'].index)
        ic, ric = self.calc_ic(pred, self.raw_dataset['test'].iloc[:,-1], dropna=True)
        if show_plot:
            ric.cumsum().plot(title='average Rank IC = %.4f'%ric.mean())
        return pred , (ic, ric)
    
    def plot_training(self , show_plot = True):
        plt.figure()
        plt.style.use('seaborn') 
        ax = lgb.plot_metric(self.evals_result, metric='l2')
        plt.scatter(self.model.best_iteration,list(self.evals_result['valid'].values())[0][self.model.best_iteration],label='best iteration')
        plt.legend()
        if show_plot: plt.show()
        return ax
    
    def plot_importance(self):
        lgb.plot_importance(self.model)

    def plot_histogram(self , feature_idx='all'):
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
        fig.suptitle('plit value histogram for feature(s)' , fontsize = 'x-large')
        plt.tight_layout()

a = lgbm(cuda = False , **dict_df)
a.train_model()

a.plot_training()
pred = a.test_prediction()
a.plot_importance()
a.plot_histogram()