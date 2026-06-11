from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any , Literal
from matplotlib.figure import Figure

from src.proj import Proj , Const , Save

from src.proj.bases import TestType
from src.res.factor.util import StockFactor 
from src.res.factor.api import FactorTestAPI
from src.res.model.util import BaseCallBack

class DetailedAlphaAnalysis(BaseCallBack):
    """Detailed Factor and Portfolio Level Analysis"""
    CB_ORDER : int = 50
    CB_KEY_PARAMS = ['tasks']
    TABLE_VB_LEVELS = {'factor@frontface':'max'}
    FIGURE_VB_LEVELS = {
        'factor@ic_curve@best.market': 2 , 
        'factor@group_return@best':'max' ,
        't50@perf_curve@best.univ':2 ,
        # 'screen@perf_curve@best.univ':2 ,
        'reinforce@perf_curve@best.univ':2 ,
    }

    def __init__(self , 
        trainer , 
        tasks : list[TestType | str] = ['factor' , 't50' , 'reinforce'] , **kwargs
    ) -> None:
        super().__init__(trainer , **kwargs)
        assert all(task in TestType for task in tasks) , \
            f'TASKS must be a list of valid tasks: {TestType} , but got {tasks}'

        self.tasks = ','.join(tasks)
        self.factor_tasks = [TestType(task) for task in tasks if task in ['factor']]
        self.fmp_tasks = [TestType(task) for task in tasks if task not in ['factor']]

        self.test_results : dict[str , pd.DataFrame] = {}
        self.test_figures : dict[str , Figure] = {}
        self.snap_folder.mkdir(exist_ok=True , parents=True)

    def __bool__(self):
        return bool(self.tasks)
        
    @property
    def snap_folder(self): return self.config.base_path.snapshot('detailed_alpha')
    @property
    def path_result_data(self): return self.config.base_path.rslt('detailed_alpha_data.xlsx')
    @property
    def path_result_plot(self): return self.config.base_path.rslt('detailed_alpha_plot.pdf')
    @property
    def table_vb_levels(self) -> dict[str,Any]: return {k:v for k,v in self.TABLE_VB_LEVELS.items() if k in self.test_results}
    @property
    def figure_vb_levels(self) -> dict[str,Any]: return {k:v for k,v in self.FIGURE_VB_LEVELS.items() if k in self.test_figures}
    @property
    def factor_names(self) -> list[str] | Any: 
        return self.trainer.model_submodels
    @property
    def test_dates(self) -> np.ndarray: return self.trainer.data.test_full_dates

    def get_factor(self , pred_dates : np.ndarray , which : Literal['first' , 'avg'] = 'avg') -> StockFactor:
        if which == 'first':
            df = self.record.get_preds(pred_dates = pred_dates , model_num = 0 , closest = True)
        elif which == 'avg':
            df = self.record.get_avg_preds(pred_dates = pred_dates , closest = True)
        else:
            raise ValueError(f'Invalid which: {which}')
        df = df.rename(columns={'submodel':'factor_name'}).pivot_table('pred',['secid','date'],'factor_name').reset_index()
        factor = StockFactor(df , factor_names = self.factor_names)
        return factor

    def get_factor_for_factor_test(self):
        test_dates = self.test_dates[::5]
        if Const.Model.resume_factor_perf is False:
            return self.get_factor(test_dates)
        else:
            saved_dates = FactorTestAPI.factor_stats_saved_dates(self.snap_folder)
            target_dates = np.setdiff1d(test_dates , saved_dates)
            if len(target_dates) == 0:
                target_dates = test_dates[-1:]
            factor = self.get_factor(target_dates).set_pseudo_date(test_dates)
            return factor

    def get_factor_for_fmp_test(self):
        if Const.Model.resume_fmp is False:
            return self.get_factor(self.test_dates)
        elif Const.Model.resume_fmp.startswith('trailing_'):
            trailing_days = int(Const.Model.resume_fmp.removeprefix('trailing_'))
            assert trailing_days > 0 , f'trailing_days must be greater than 0 , but got {trailing_days}'
            pred_last_date = self.config.resumed_max_pred_date
            port_last_date = FactorTestAPI.last_portfolio_date(self.snap_folder , self.fmp_tasks)
            last_date = min(pred_last_date , port_last_date)
            test_date_num = sum(self.test_dates > last_date) + trailing_days
            return self.get_factor(self.test_dates[-test_date_num:])
        else:
            raise ValueError(f'Invalid resuming test fmp option: {Const.Model.resume_fmp}')

    def factor_test(self):
        if not self.factor_tasks:
            return
        self.logger.note('Factor Perf Test')
        with self.logger.subprocess(idt = 1):
            with self.logger.timer(f'FactorPerfTest.get_factor'):
                factor = self.get_factor_for_factor_test()
            with self.logger.timer(f'FactorPerfTest.load_day_rets'):
                factor.day_returns()
            with self.logger.timer(f'FactorPerfTest.within_benchmarks'):
                factor.within_benchmarks()

            for task in self.factor_tasks:
                self.logger.divider(vb = 2)
                with self.logger.timer(f'FactorPerfTest.{task}' , enter_vb=1):
                    results = FactorTestAPI.run_test(
                        task , factor , test_path = self.snap_folder , 
                        resume = self.config.is_resuming , save_resumable = True , 
                        start = self.trainer.config.beg_date , end = self.trainer.config.end_date ,
                        indent = self.indent + 1 , vb_level = self.vb_level + 1,
                        title_prefix=self.config.model_name)

                    self.test_results.update({f'{task}@{k}':v for k,v in results.get_rslts().items()})
                    self.test_figures.update({f'{task}@{k}':v for k,v in results.get_figs().items()})

    def fmp_test(self):
        if not self.fmp_tasks:
            return
        self.logger.note('Factor FMP Test')
        with self.logger.subprocess(idt = 1):
            with self.logger.timer(f'FactorFMPTest.get_factor'):
                factor = self.get_factor_for_fmp_test()
            with self.logger.timer(f'FactorFMPTest.load_alpha_models'):
                factor.alpha_models()
            with self.logger.timer(f'FactorFMPTest.load_risk_models'):
                factor.risk_model()
            with self.logger.timer(f'FactorFMPTest.load_universe'):
                factor.universe(load = True)
            with self.logger.timer(f'FactorFMPTest.load_day_quotes' ):
                factor.day_quotes()

            for task in self.fmp_tasks:
                self.logger.divider(vb = 2)
                with self.logger.timer(f'FactorFMPTest.{task}' , enter_vb=1):
                    results = FactorTestAPI.run_test(
                        task , factor , test_path = self.snap_folder , 
                        resume = self.config.is_resuming , save_resumable = True , 
                        start = self.trainer.config.beg_date , end = self.trainer.config.end_date,
                        indent = self.indent + 1 , vb_level = self.vb_level + 1,
                        title_prefix=self.config.model_name)

                    self.test_results.update({f'{task}@{k}':v for k,v in results.get_rslts().items()})
                    self.test_figures.update({f'{task}@{k}':v for k,v in results.get_figs().items()})

    def display_export(self):
        self.logger.note('Display Analytic Results')
        for name , vb_level in self.table_vb_levels.items():
            if not Proj.verbose(vb_level):
                continue
            df = self.test_results[name].copy()
            df = df.reset_index(drop=isinstance(df.index , pd.RangeIndex))
            for col in df.columns:
                if col in ['pf','bm','excess','annualized','mdd','te','ret']: 
                    df[col] = df[col].map(lambda x:f'{x:.2%}')
                elif col in ['ir','calmar','turnover','IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , '|IC|_avg']: 
                    df[col] = df[col].map(lambda x:f'{x:.3f}')
                elif df.columns.name in ['group'] and (isinstance(col , int) or str(col).isdigit()):
                    df[col] = df[col].map(lambda x:f'{x:.3%}')
            self.logger.display(df , title = f'Table: {name.title()}:')

        for name , vb_level in self.figure_vb_levels.items():
            if not Proj.verbose(vb_level):
                continue
            self.logger.display(self.test_figures[name] , title = f'Figure: {name.title()}:')

        Save.dfs(
            self.test_results , self.path_result_data , async_save = True ,
            prefix='Detailed Alpha Analysis Datas' , indent = self.indent + 1 , vb_level = self.vb_level + 1)
        Save.figs(
            self.test_figures , self.path_result_plot , async_save = True ,
            prefix='Detailed Alpha Analysis Plots' , indent = self.indent + 1 , vb_level = self.vb_level + 1)
        Proj.exit_files.extend(self.path_result_data , self.path_result_plot)

    def on_test_end(self):
        if not self.tasks:
            return
        assert self.factor_names , 'factor_names is empty'
        with self.logger.paragraph('Detailed Alpha Analysis' , 3):
            self.factor_test()
            self.fmp_test()
            self.display_export()