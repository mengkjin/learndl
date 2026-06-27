"""Optuna-driven hyper-parameter search on top of :class:`GeneralBoostModel`.

Classes:
    OptunaSilent    — context manager that suppresses Optuna log output.
    OptunaBoostModel — sub-class of :class:`GeneralBoostModel` that wraps
                       the full ``fit()`` call in an Optuna study; the best
                       trial params are applied before the final fit.
"""
from __future__ import annotations
import optuna

from pathlib import Path
from typing import Any , Literal , TypeAlias

from src.proj import PATH , MACHINE , Proj 
from src.proj.bases import BoostModuleType
from .general import GeneralBoostModel

__all__ = ['OptunaBoostModel']

OptunaPlotType : TypeAlias = Literal['slice' , 'optimization_history' , 'param_importances' , 'contour']

class OptunaSilent:
    """Context manager that sets Optuna log level to ERROR on entry and restores it on exit."""
    def __init__(self , silent : bool = True):
        self.silent = silent
    def __enter__(self):
        if self.silent:
            self.old_level = optuna.logging.get_verbosity()
            optuna.logging.set_verbosity(optuna.logging.ERROR)
    def __exit__(self , *args , **kwargs):
        if self.silent:
            optuna.logging.set_verbosity(self.old_level)

class OptunaBoostModel(GeneralBoostModel):
    """Optuna-powered hyper-parameter search over :class:`GeneralBoostModel`.

    Fit workflow:
        1. :meth:`study_create` — create a named Optuna study (persisted to
           SQLite when ``DEFAULT_SAVE_STUDIES`` is ``True``).
        2. :meth:`study_optimize` — run ``n_trials`` trials where each trial
           calls :meth:`study_objective`, which fits the model and returns mean
           RankIC on the validation set.
        3. Apply ``best_params`` and do a final fit.

    Per-booster search spaces are defined in :meth:`trial_suggest_params`.
    """
    DEFAULT_N_TRIALS = 50 if MACHINE.platform_server else 20
    OPTUNA_SAVE_STUDIES = True
    
    @property
    def best_params(self):
        return self.study.best_trial.params

    @property
    def force_objective(self) -> str | None:
        param = (self.override_boost.get('param') or {})
        return param.get('objective')
    
    def update_param(self , params : dict[str, Any] | None = None , **kwargs):
        super().update_param(params , **kwargs)
        self.n_trials = kwargs.get('n_trials' , self.DEFAULT_N_TRIALS)
        return self

    def trial_suggest_params(self, trial : optuna.Trial):
        if self.boost_type == BoostModuleType.LGMB:
            params = {
                'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth':        trial.suggest_int('max_depth', 3, 12),
                'num_leaves':       trial.suggest_int('num_leaves', 20, 100, step=10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50, step=10),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-7, 100, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 100, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.1),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0, step=0.1),
            }
            if not self.force_objective:
                params['objective'] = trial.suggest_categorical('objective', ['mse', 'mae', 'rank'])
        elif self.boost_type == BoostModuleType.XGBOOST:
            params = {
                'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth':        trial.suggest_int('max_depth', 3, 12),
                'subsample':        trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-7, 100, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 100, log=True),
            }
            if not self.force_objective:
                params['objective'] = trial.suggest_categorical('objective', ['mse', 'mae', 'rank'])
        elif self.boost_type == BoostModuleType.CATBOOST:
            params = {
                'learning_rate':            trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth':                trial.suggest_int('max_depth', 3, 12),
                'l2_leaf_reg':              trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'bagging_temperature':      trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength':          trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'od_type':                  trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'min_data_in_leaf':         trial.suggest_int('min_data_in_leaf', 1, 100),
            }
            if not self.force_objective:
                params['objective'] = trial.suggest_categorical('objective', ['mse', 'mae', 'rank'])
        elif self.boost_type == BoostModuleType.ADA:
            params = {
                'n_learner' :       trial.suggest_int('n_learner', 10 , 50 , step = 5), 
                'n_bins' :          trial.suggest_int('n_bins', 10, 30, step = 5) , 
                'max_nan_ratio' :   trial.suggest_float('max_nan_ratio', 0.5, 0.9, step = 0.1) ,
            }
        else: 
            raise ValueError(f'Invalid boost type: {self.boost_type}')
        return params
    
    def fit(self , train = None , valid = None , use_feature = None , silent = False):
        self.import_data(train = train , valid = valid)
        self.boost.import_data(train=self.data['train'] , valid = self.data['valid'])
        self.boost.update_feature(use_feature)

        self.study_create()
        self.study_optimize(self.n_trials)
        
        self.update_param(self.study.best_trial.params).boost.fit(silent=True)
        return self

    @property
    def study_db_path(self) -> Path:
        return PATH.optuna / self.boost_type / f'{self.given_name}.sqlite3'

    @property
    def study_storage(self) -> str:
        return f'sqlite:///{self.study_db_path.resolve().as_posix()}'

    def study_create(self , direction='maximize'):
        with OptunaSilent(silent = not Proj.vb.is_max_level):
            if self.OPTUNA_SAVE_STUDIES:
                self.study_db_path.parent.mkdir(parents=True, exist_ok=True)
                storage = self.study_storage
            else:
                storage = None
            self.study = optuna.create_study(storage=storage,  direction = direction , study_name=self.sub_name)
        return self
    
    def study_objective(self , trial : optuna.Trial):
        params = self.trial_suggest_params(trial)
        boost = self.update_param(params).boost
        output = boost.fit(silent=True).predict('valid')
        if boost.valid_metric is not None:
            return boost.valid_metric.score_output(output)
        result = output.top5pct() if self.force_objective == 'rank' else output.rankic()
        return result.nanmean().item()

    def study_optimize(self , n_trials : int = DEFAULT_N_TRIALS):
        if self.boost_type in [BoostModuleType.LGMB , BoostModuleType.XGBOOST , BoostModuleType.CATBOOST]:
            max_trials = 100
        elif self.boost_type == BoostModuleType.ADA:
            max_trials = 20
        else: 
            raise ValueError(f'Invalid boost type: {self.boost_type}')
        n_trials = min(max_trials , n_trials)

        with OptunaSilent(silent = not Proj.vb.is_max_level):
            self.study.optimize(self.study_objective, n_trials = n_trials , show_progress_bar = Proj.vb.is_max_level)
            
        return self

    def study_plot(
        self , plot_type : OptunaPlotType , params : list[str] | None = None , **kwargs
    ):
        if plot_type == 'slice':
            return optuna.visualization.plot_slice(self.study , params = params , **kwargs)
        elif plot_type == 'optimization_history':
            return optuna.visualization.plot_optimization_history(self.study , **kwargs)
        elif plot_type == 'param_importances':
            return optuna.visualization.plot_param_importances(self.study , **kwargs)
        elif plot_type == 'contour':
            return optuna.visualization.plot_contour(self.study , params = params , **kwargs)
        
    def param_importances(self):
        return optuna.importance.get_param_importances(self.study)