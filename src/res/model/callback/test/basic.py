import warnings
import pandas as pd
import numpy as np
from typing import Any

from src.proj import Logger , DB
from src.proj.util import AsyncSaver

from src.res.model.util import BaseCallBack

class BasicTestResult(BaseCallBack):
    '''Basic Test of RankIC'''
    
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.snap_folder.mkdir(exist_ok=True , parents=True)

    @property
    def model_test_dates(self) -> np.ndarray: return self.trainer.data.model_test_dates
    @property
    def test_full_dates(self) -> np.ndarray: return self.trainer.data.test_full_dates
    @property
    def snap_folder(self): return self.config.base_path.snapshot('basic_test')
    @property
    def path_test_df(self): return self.snap_folder.joinpath('test_by_date.feather')
    @property
    def path_result(self): return self.config.base_path.rslt('basic_test.xlsx')

    def complete_test_df(self , vb_level : Any = 3) -> pd.DataFrame:
        df = DB.load_df(self.path_test_df).dropna() if self.config.is_resuming else pd.DataFrame(columns=['model_num' , 'model_date' , 'submodel' , 'date' , 'value'])

        target_dates = np.setdiff1d(self.test_full_dates , df['date'].unique())
        preds = self.record.get_preds(target_dates).dropna()

        grouped = preds.groupby(by=['model_num' , 'model_date' , 'submodel' , 'date'], as_index=True)
        def df_ic(subdf : pd.DataFrame , **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='An input array is constant; the correlation coefficient is not defined' , category=RuntimeWarning)
                warnings.filterwarnings('ignore', message='invalid value encountered in divide' , category=RuntimeWarning)
                return subdf[['pred']].corrwith(subdf['label'], method='spearman')
        new_df = grouped.apply(df_ic , include_groups = False).rename(columns = {'pred' : 'value'}).reset_index(drop=False)
        if df.empty:
            df = new_df
        elif not new_df.empty:
            df = pd.concat([df , new_df]).drop_duplicates(subset=['model_num' , 'model_date' , 'submodel' , 'date'] , keep='last').\
                sort_values(by=['model_num' , 'model_date' , 'submodel' , 'date']).reset_index(drop=True).dropna()
        
        AsyncSaver.df(df , self.path_test_df , copy_for_safety = False , overwrite = True , vb_level = 'never')
        Logger.footnote(f'Basic Test Result saved to {self.path_test_df}' , vb_level = vb_level) 

        return df

    def on_test_end(self): 
        """update test_summary"""
        df = self.complete_test_df()
        df_date = df.query('date in @self.test_full_dates')
        if df_date.empty:
            return

        with Logger.Paragraph('Test Summary' , 3): 
            if df_date['model_date'].nunique() == 1 or self.config.base_path.is_null_model:
                # only one model_date, calculate by year
                df_date['model_date'] = (df_date['date'].astype(int) // 10000).apply(lambda x:f'Y{x}')
            else:
                df_date['model_date'] = df_date['model_date'].astype(str)
                
            df_model = df_date.groupby(['model_num' , 'model_date' , 'submodel'])['value'].mean().reset_index().rename(columns={'model_date':'stat'})

            dfs : dict[str,pd.DataFrame|pd.Series|Any] = {}
            dfs['Avg'] = df_date.groupby(['model_num','submodel'])['value'].mean()
            dfs['Sum'] = df_date.groupby(['model_num','submodel'])['value'].sum()
            dfs['Std'] = df_date.groupby(['model_num','submodel'])['value'].std()

            dfs['T']   = ((dfs['Avg'] / dfs['Std']) * (len(df_date['date'].unique())**0.5))
            dfs['IR']  = ((dfs['Avg'] / dfs['Std']) * ((240 / 10)**0.5))

            stat_df = pd.concat([df.reset_index().assign(stat=k) for k,df in dfs.items()])

            # display summary
            df = pd.concat([df_model , stat_df])
            cat_stat = [md for md in df_model['stat'].unique()] + ['Avg' , 'Sum' , 'Std' , 'T' , 'IR']
            cat_subm = ['best' , 'swalast' , 'swabest']

            base_name = self.config.model_module
            if self.config.module_type == 'boost' and self.config.boost_optuna: 
                base_name += '.optuna'
            df['model_num'] = df['model_num'].map(lambda x: f'{base_name}.{x}')
            df['submodel']  = pd.Categorical(df['submodel'] , categories = cat_subm, ordered=True) 
            df['stat']      = pd.Categorical(df['stat']     , categories = cat_stat, ordered=True) 

            test_summary = df.rename(columns={'model_num':'model'}).pivot_table('value' , 'stat' , ['model' , 'submodel'] , observed=False).round(4)
            self.container.dataframes['test_summary'] = test_summary

            # more than 100 rows of test_df_model means the cycle is month / day
            df_display = test_summary
            if len(df_display) > 100: 
                df_display = df_display.loc[['Avg' , 'Sum' , 'Std' , 'T' , 'IR']]          
            criterion_accuracy = list(self.config.criterion_accuracy.keys())[0]
            Logger.display(df_display , caption = f'Table: Test Summary ({criterion_accuracy}) for Models:' , vb_level = 2)
            
            # export excel
            rslt = {'test_summary' : test_summary , 'test_by_model' : df_model}
            for model_num in self.config.model_num_list:
                df : pd.DataFrame = df_date[df_date['model_num'] == model_num].pivot_table(
                    'value' , 'date' , 'submodel' , observed=False)
                df_cum = df.cumsum().rename(columns = {submodel:f'{submodel}_cum' for submodel in df.columns})
                df = df.merge(df_cum , on = 'date').rename_axis(None , axis = 'columns')
                rslt[f'{model_num}'] = df
            [AsyncSaver.df(df , self.snap_folder.joinpath(f'{key}.feather') , copy_for_safety = False , overwrite = True , vb_level = 'never') for key,df in rslt.items()]
            AsyncSaver.dfs(rslt , self.path_result, print_prefix = 'Test Summary')