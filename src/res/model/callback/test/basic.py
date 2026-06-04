from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
from typing import Any

from src.proj import DB
from src.proj.util.io.async_save import AsyncSaver

from src.res.model.util import BaseCallBack

class BasicTestResult(BaseCallBack):
    '''Basic Test of RankIC'''

    _stat_cols : tuple[str, ...] = ('rankic' , 'top5pct' , 'mid20pct' , 'bot5pct')
    
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.snap_folder.mkdir(exist_ok=True , parents=True)

    @property
    def model_test_dates(self) -> np.ndarray: 
        return self.trainer.data.model_test_dates
    @property
    def test_full_dates(self) -> np.ndarray: 
        return self.trainer.data.test_full_dates
    @property
    def snap_folder(self): 
        return self.config.base_path.snapshot('basic_test')
    @property
    def path_test_df(self): 
        return self.snap_folder.joinpath('test_by_date.feather')
    @property
    def path_result(self): 
        return self.config.base_path.rslt('basic_test.xlsx')

    @property
    def stat_cols(self) -> list[str]:
        return list(self._stat_cols)

    @property
    def test_df_cols(self) -> list[str]:
        return ['model_num' , 'model_date' , 'submodel' , 'date' , *self.stat_cols]

    def complete_test_df(self) -> pd.DataFrame:
        df = DB.load_df(self.path_test_df).dropna() if self.config.is_resuming else pd.DataFrame()
        if df.empty or not all(col in df.columns for col in self.stat_cols):
            df = pd.DataFrame(columns=self.test_df_cols)
        else:
            df = df[['model_num' , 'model_date' , 'submodel' , 'date' , *self.stat_cols]]

        target_dates = np.setdiff1d(self.test_full_dates , df['date'].unique())
        preds = self.record.get_preds(target_dates).dropna()

        grouped = preds.groupby(by=['model_num' , 'model_date' , 'submodel' , 'date'], as_index=True)
        def df_rankic(subdf : pd.DataFrame , **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='An input array is constant; the correlation coefficient is not defined' , category=RuntimeWarning)
                warnings.filterwarnings('ignore', message='invalid value encountered in divide' , category=RuntimeWarning)
                return subdf[['pred']].corrwith(subdf['label'], method='spearman')
        def df_top5pct(subdf : pd.DataFrame , **kwargs):
            std_label = (subdf[['label']] - subdf['label'].mean()) / (subdf['label'].std() + 1e-6)
            pred_rank = subdf['pred'].rank(pct=True)
            return std_label.loc[pred_rank >= 0.95].mean()
        def mid_20pct(subdf : pd.DataFrame , **kwargs):
            std_label = (subdf[['label']] - subdf['label'].mean()) / (subdf['label'].std() + 1e-6)
            pred_rank = subdf['pred'].rank(pct=True)
            return std_label.loc[(pred_rank >= 0.4) & (pred_rank < 0.6)].mean()
        def df_bot5pct(subdf : pd.DataFrame , **kwargs):
            std_label = (subdf[['label']] - subdf['label'].mean()) / (subdf['label'].std() + 1e-6)
            pred_rank = subdf['pred'].rank(pct=True)
            return std_label.loc[pred_rank < 0.05].mean()
        rankic_df = grouped.apply(df_rankic , include_groups = False).rename(columns={'pred':'rankic'})[['rankic']]
        top5pct_df = grouped.apply(df_top5pct , include_groups = False).rename(columns={'label':'top5pct'})[['top5pct']]
        mid20pct_df = grouped.apply(mid_20pct , include_groups = False).rename(columns={'label':'mid20pct'})[['mid20pct']]
        bot5pct_df = grouped.apply(df_bot5pct , include_groups = False).rename(columns={'label':'bot5pct'})[['bot5pct']]
        new_df = rankic_df.join(top5pct_df).join(mid20pct_df).join(bot5pct_df).reset_index(drop=False)
        if df.empty:
            df = new_df
        elif not new_df.empty:
            df = pd.concat([df , new_df]).drop_duplicates(subset=['model_num' , 'model_date' , 'submodel' , 'date'] , keep='last').\
                sort_values(by=['model_num' , 'model_date' , 'submodel' , 'date']).reset_index(drop=True).dropna()
        
        AsyncSaver.df(
            df , self.path_test_df , copy_for_safety = False , overwrite = True , 
            prefix = f'Basic Test Result' , vb_level = self.vb_level + 1 , indent = self.indent + 1)

        return df

    def on_test_end(self): 
        """update test_summary"""
        df = self.complete_test_df()
        df_date = df.query('date in @self.test_full_dates')
        if df_date.empty:
            return

        with self.logger.paragraph('Test Summary' , 3): 
            if df_date['model_date'].nunique() == 1 or self.config.base_path.is_null_model:
                # only one model_date, calculate by year
                df_date['model_date'] = (df_date['date'].astype(int) // 10000).apply(lambda x:f'Y{x}')
            else:
                df_date['model_date'] = df_date['model_date'].astype(str)

            df_date = df_date.melt(
                id_vars=['model_num' , 'model_date' , 'submodel' , 'date'] , var_name='stat' , value_name='value').\
                reset_index(drop=False)
            df_model = df_date.groupby(['model_num' , 'model_date' , 'submodel' , 'stat'])[['value']].\
                mean().reset_index().rename(columns={'model_date':'entry'})

            col_cats = {
                'submodel' : ['best' , 'swalast' , 'swabest'],
                'stat' : self.stat_cols,
                'entry' : [md for md in df_model['entry'].unique()] + ['Avg' , 'Sum' , 'Std' , 'T' , 'IR']
            }

            for col, cats in col_cats.items():
                if col in df_date.columns:
                    df_date[col] = pd.Categorical(df_date[col] , categories = cats, ordered=True) 
                if col in df_model.columns:
                    df_model[col] = pd.Categorical(df_model[col] , categories = cats, ordered=True) 

            df_date = df_date.sort_values(by=['model_num' , 'submodel' , 'model_date' , 'stat' , 'date']).reset_index(drop=True)
            df_model = df_model.sort_values(by=['model_num' , 'submodel' , 'entry' , 'stat']).reset_index(drop=True)

            dfs : dict[str,pd.DataFrame|pd.Series|Any] = {}
            dfs['Avg'] = df_date.groupby(['model_num','submodel','stat'],observed=True)[['value']].mean()
            dfs['Sum'] = df_date.groupby(['model_num','submodel','stat'],observed=True)[['value']].sum()
            dfs['Std'] = df_date.groupby(['model_num','submodel','stat'],observed=True)[['value']].std()

            dfs['T']   = ((dfs['Avg'] / dfs['Std']) * (len(df_date['date'].unique())**0.5))
            dfs['IR']  = ((dfs['Avg'] / dfs['Std']) * ((240 / 10)**0.5))

            stat_df = pd.concat([df.reset_index().assign(entry=k) for k,df in dfs.items()])

            # display summary
            df = pd.concat([df_model , stat_df])

            base_name = self.config.model_module
            if self.config.module_type == 'boost' and self.config.boost_optuna: 
                base_name += '.optuna'
            df['model_num'] = df['model_num'].map(lambda x: f'{base_name}.{x}')
            
            test_summary = df.rename(columns={'model_num':'model'}).\
                pivot_table('value' , 'entry' , ['model' , 'submodel' , 'stat'] , observed=True).round(4)
            self.container.dataframes['test_summary'] = test_summary

            # more than 100 rows of test_df_model means the cycle is month / day
            df_display = test_summary
            if len(df_display) > 100: 
                df_display = df_display.loc[['Avg' , 'Sum' , 'Std' , 'T' , 'IR']]          
            criterion_accuracy = list(self.config.criterion_accuracy.keys())[0]
            caption = f'Table: Test Summary for Models (rankic={criterion_accuracy},pct=normailized_position):'
            self.logger.display(df_display.round(3) , caption = caption)
            
            # export excel
            test_summary = self._revert_col_names(test_summary)
            rslt = {'test_summary' : test_summary , 'test_by_model' : df_model}
            for model_num in self.config.model_num_list:
                df = df_date[df_date['model_num'] == model_num]
                df_day = df.pivot_table('value' , 'date' , ['submodel' , 'stat'] , observed=True)
                df_cum = df_day.cumsum()
                df_cum.columns = pd.MultiIndex.from_arrays(
                    [df_cum.columns.get_level_values(0) , [f'{col}_cum' for col in df_cum.columns.get_level_values(1)]],
                    names=['submodel' , 'stat']
                )
                df = self._revert_col_names(df_day).merge(df_cum , on = 'date')
                rslt[f'{model_num}'] = df
            [AsyncSaver.df(df , self.snap_folder.joinpath(f'{key}.feather') , copy_for_safety = False , overwrite = True , vb_level = 'never') for key,df in rslt.items()]
            AsyncSaver.dfs(rslt , self.path_result, prefix = 'Test Summary' , indent = self.indent + 1 , vb_level = self.vb_level + 1)

    @staticmethod
    def _revert_col_names(df : pd.DataFrame) -> pd.DataFrame:
        df.columns = pd.MultiIndex.from_arrays(
            [df.columns.get_level_values(i).astype(str) for i in range(df.columns.nlevels)],
            names=df.columns.names
        )
        return df