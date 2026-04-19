import pandas as pd
from pathlib import Path
from src.res.factor.util.agency.portfolio_accountant import PortfolioAccount
from src.res.model.model_module.application.trainer import ModelTrainer
from src.res.factor.util.stats.aggregate import eval_period_ic_multi
from src.proj import Logger , PATH , CONST , DB

from .util import wrap_update

def display_account_summary(accounts : dict[str , dict[str , Path]] , account_type : str , by_max_columns : int = 12):
    dfs = {model : PortfolioAccount.EvalPeriodRet(paths) for model , paths in accounts.items()}
    dfs = concat_dfs_split(dfs , by_max_columns = by_max_columns)
    for i , df in enumerate(dfs):
        caption = f'Summary of {account_type.title()} Account Period Return (Total {len(dfs)} Tables):' if i == 0 else None
        Logger.display(df , caption = caption)
    return dfs

def model_account_summary(by_max_columns : int = 12):
    acc_paths : dict[str , dict[str , Path]] = {}
    fmp_types = {'t50' : 't50' , 'scr' : 'screen' , 'rein' : 'reinforce'}
    model_paths = ModelTrainer.all_resumable_models()
    for model_path in model_paths:
        acc_paths[model_path.model_name] = {}
        for col , fmp in fmp_types.items():
            available_paths = list(model_path.snapshot('detailed_alpha' , f'{fmp}_fmp_test' , 'account').glob('*best*.tar'))
            if available_paths:
                acc_paths[model_path.model_name][col] = available_paths[0]
    return display_account_summary(acc_paths , 'Model FMP' , by_max_columns = by_max_columns)

def tracking_port_account_summary(by_max_columns : int = 12):
    acc_paths : dict[str , dict[str , Path]] = {}
    for tport in CONST.Conf.TradingPort.tracking_ports:
        available_paths = list(PATH.rslt_trade.joinpath('tracking', tport).glob('account.tar'))
        if available_paths:
            acc_paths[tport] = {'port':available_paths[0]} 
    return display_account_summary(acc_paths , 'Tracking Portfolio' , by_max_columns = by_max_columns)

def backtest_port_account_summary(by_max_columns : int = 12):
    acc_paths : dict[str , dict[str , Path]] = {}
    for tport in CONST.Conf.TradingPort.backtest_ports:
        available_paths = list(PATH.rslt_trade.joinpath('backtest', tport).glob('account.tar'))
        if available_paths:
            acc_paths[tport] = {'port':available_paths[0]} 
    return display_account_summary(acc_paths , 'Backtest Portfolio' , by_max_columns = by_max_columns)

def model_ic_summary():
    paths : dict[str , Path] = {}
    
    model_paths = ModelTrainer.all_resumable_models()
    for model_path in model_paths:
        if (path := model_path.snapshot('basic_test' , f'test_by_date.feather')).exists():
            paths[model_path.model_name] = path
    ic_tables = {}
    for name , path in paths.items():
        df = DB.load_df(path)
        ic_tables[name] = df.astype({'date' : 'int'}).query('submodel == "best"').groupby('date')['value'].mean().\
            reset_index(drop = False).dropna().rename(columns = {'value' : 'ic'})
    df = eval_period_ic_multi(ic_tables)
    return df

def ic_summaries(by_max_columns : int = 12):
    df = model_ic_summary()
    for i in range(0, len(df.columns), by_max_columns):
        Logger.display(df.iloc[:,i:i+by_max_columns].dropna(how = 'all'))

def account_summaries(by_max_columns : int = 12):
    model_account_summary(by_max_columns)
    tracking_port_account_summary(by_max_columns)
    backtest_port_account_summary(by_max_columns)

def concat_dfs_split(dfs : dict[str,pd.DataFrame] , by_max_columns : int = 12) -> list[pd.DataFrame]:
    if not dfs:
        return []
    out_dfs : list[pd.DataFrame] = []
    current_batch : dict[str,pd.DataFrame] = {}
    for i , (name , df) in enumerate(dfs.items()):
        if current_batch and sum([len(df.columns) for df in current_batch.values()]) + len(df.columns) > by_max_columns:
            out_dfs.append(concat_dfs(current_batch))
            current_batch.clear()
        current_batch[name] = df
    out_dfs.append(concat_dfs(current_batch))
    out_dfs = [df for df in out_dfs if not df.empty]
    return out_dfs

def concat_dfs(accounts : dict[str,pd.DataFrame]) -> pd.DataFrame:
    longest_index = max(accounts.values(), key=len).index
    return pd.concat(accounts , axis = 1).reindex(index = longest_index).rename_axis(index = '')

class SummaryAPI:
    @classmethod
    def update(cls):
        wrap_update(cls.process , 'Summary')

    @classmethod
    def process(cls):
        ic_summaries()
        account_summaries()