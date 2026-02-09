import pandas as pd
from pathlib import Path
from src.res.factor.util.agency.portfolio_accountant import PortfolioAccount
from src.res.model.model_module.application.trainer import ModelTrainer
from src.proj import Logger , Proj , PATH

from .util import wrap_update

def summary_account_period_ret():
    acc_paths : dict[str , dict[str , Path]] = {}
    fmp_types = {'t50' : 't50' , 'scr' : 'screen' , 'rein' : 'reinforce'}
    for model in ModelTrainer.resumable_models():
        acc_paths[model] = {}
        for col , fmp in fmp_types.items():
            available_paths = list(Path(f'models/{model}/snapshot/detailed_alpha/{fmp}_fmp_test/account').glob('*best*.tar'))
            if available_paths:
                acc_paths[model][col] = available_paths[0]
    
    for factor in ModelTrainer.resumable_factors():
        acc_paths[factor] = {}
        for col , fmp in fmp_types.items():
            available_paths = list(Path(f'results/null_models/factor@{factor}/snapshot/detailed_alpha/{fmp}_fmp_test/account').glob('*best*.tar'))
            if available_paths:
                acc_paths[factor][col] = available_paths[0]
    
    for trading_port in Proj.Conf.TradingPort.focused_ports:
        acc_paths[trading_port] = {}
        available_paths = list(PATH.rslt_trade.joinpath(trading_port).glob('account.tar'))
        if available_paths:
            acc_paths[trading_port]['port'] = available_paths[0]
    
    df = pd.concat({model : PortfolioAccount.EvalPeriodRet(paths) for model , paths in acc_paths.items()} , axis = 1)
    Logger.display(df , caption = 'Summary of Account Period Return:')

class SummaryAPI:
    @classmethod
    def update(cls):
        wrap_update(cls.process , 'Summary')

    @classmethod
    def process(cls):
        summary_account_period_ret()