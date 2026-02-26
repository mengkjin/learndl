import pandas as pd
from pathlib import Path
from src.res.factor.util.agency.portfolio_accountant import PortfolioAccount
from src.res.model.model_module.application.trainer import ModelTrainer
from src.proj import Logger , Proj , PATH

from .util import wrap_update

def summary_account_period_ret():
    acc_paths : dict[str , dict[str , Path]] = {}
    fmp_types = {'t50' : 't50' , 'scr' : 'screen' , 'rein' : 'reinforce'}
    model_paths = ModelTrainer.all_resumable_models()
    for model_path in model_paths:
        acc_paths[model_path.model_name] = {}
        for col , fmp in fmp_types.items():
            available_paths = list(model_path.snapshot('detailed_alpha' , f'{fmp}_fmp_test' , 'account').glob('*best*.tar'))
            if available_paths:
                acc_paths[model_path.model_name][col] = available_paths[0]
    
    for tport in Proj.Conf.TradingPort.focused_ports:
        acc_paths[tport] = {}
        available_paths = list(PATH.rslt_trade.joinpath(tport).glob('account.tar'))
        if available_paths:
            acc_paths[tport]['port'] = available_paths[0]
    
    df = pd.concat({model : PortfolioAccount.EvalPeriodRet(paths) for model , paths in acc_paths.items()} , axis = 1)
    Logger.display(df , caption = 'Summary of Account Period Return:')

class SummaryAPI:
    @classmethod
    def update(cls):
        wrap_update(cls.process , 'Summary')

    @classmethod
    def process(cls):
        summary_account_period_ret()