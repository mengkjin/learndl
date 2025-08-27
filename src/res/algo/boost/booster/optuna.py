import optuna , random , string , time

from contextlib import nullcontext
from typing import Any , Literal

from src.basic import path as PATH , MACHINE
from .general import GeneralBooster

class OptunaSilent:
    def __enter__(self):
        self.old_level = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    def __exit__(self , exc_type , exc_value , traceback):
        optuna.logging.set_verbosity(self.old_level)

class OptunaBooster(GeneralBooster):
    DEFAULT_SILENT_CREATION = True
    DEFAULT_SILENT_STUDY = True
    DEFAULT_N_TRIALS = 50 if MACHINE.server else 20
    DEFAULT_SAVE_STUDIES = True
    DEFAULT_STORAGE = f'sqlite:///{PATH.optuna.relative_to(PATH.main)}/booster_{time.strftime("%Y%m") }.sqlite3'

    @property
    def best_params(self):
        return self.study.best_trial.params
    
    def update_param(self , params : dict[str,Any] , **kwargs):
        super().update_param(params , **kwargs)
        self.n_trials = kwargs.get('n_trials' , self.DEFAULT_N_TRIALS)
        return self

    def trial_suggest_params(self, trial : optuna.Trial):
        if self.booster_type == 'lgbm':
            params = {
                'objective':        trial.suggest_categorical('objective', ['mse', 'mae']),  # 'mse', 'mae', 'softmax'
                'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth':        trial.suggest_int('max_depth', 3, 12),
                'num_leaves':       trial.suggest_int('num_leaves', 20, 100, step=10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50, step=10),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-7, 100, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 100, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.1),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0, step=0.1),
            }
        elif self.booster_type == 'xgboost':
            params = {
                'objective':        trial.suggest_categorical('objective', ['reg:squarederror', 'reg:absoluteerror']), # 'reg:squarederror', 'reg:absoluteerror' , multi:softmax
                'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth':        trial.suggest_int('max_depth', 3, 12),
                'subsample':        trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-7, 100, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 100, log=True),
            }
        elif self.booster_type == 'catboost':
            params = {
                'objective':                trial.suggest_categorical('objective', ['RMSE', 'MAE']),
                'learning_rate':            trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth':                trial.suggest_int('max_depth', 3, 12),
                'l2_leaf_reg':              trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'bagging_temperature':      trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength':          trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'od_type':                  trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'min_data_in_leaf':         trial.suggest_int('min_data_in_leaf', 1, 100),
            }
        elif self.booster_type == 'ada':
            params = {
                'n_learner' :       trial.suggest_int('n_learner', 10 , 50 , step = 5), 
                'n_bins' :          trial.suggest_int('n_bins', 10, 30, step = 5) , 
                'max_nan_ratio' :   trial.suggest_float('max_nan_ratio', 0.5, 0.9, step = 0.1) ,
            }
        else: raise ValueError(f'Invalid booster type: {self.booster_type}')
        return params
    
    def fit(self , train = None , valid = None , use_feature = None , silent = False):
        self.import_data(train = train , valid = valid)
        self.booster.import_data(train=self.data['train'] , valid = self.data['valid'])
        self.booster.update_feature(use_feature)

        self.study_create(silent = silent or self.DEFAULT_SILENT_CREATION)
        self.study_optimize(self.n_trials , silent = silent or self.DEFAULT_SILENT_STUDY)
        
        self.update_param(self.study.best_trial.params).booster.fit(silent=True)
        return self
    
    def study_create(self , direction='maximize' , silent = False):
        name_str = self.given_name if self.given_name else self.booster.__class__.__name__
        time_str = time.strftime('%Y%m%d-%H%M%S') 
        rand_str =''.join(random.choices(string.ascii_letters + string.digits, k=10))
    
        with OptunaSilent() if silent else nullcontext():
            self.study = optuna.create_study(storage=self.DEFAULT_STORAGE if self.DEFAULT_SAVE_STUDIES else None, 
                                             direction = direction , study_name=f'{name_str}_{time_str}_{rand_str}')
        return self
    
    def study_objective(self , trial : optuna.Trial):
        params = self.trial_suggest_params(trial)
        booster = self.update_param(params).booster
        return booster.fit(silent=True).predict('valid').rankic().mean().item()

    def study_optimize(self , n_trials : int = DEFAULT_N_TRIALS , silent = False):
        if self.booster_type in ['lgbm' , 'xgboost' , 'catboost']:
            max_trials = 100
        elif self.booster_type == 'ada':
            max_trials = 20
        else: 
            raise ValueError(f'Invalid booster type: {self.booster_type}')
        n_trials = min(max_trials , n_trials)

        with OptunaSilent() if silent else nullcontext():
            self.study.optimize(self.study_objective, n_trials = n_trials)
            
        return self

    def study_plot(self , plot_type : Literal['slice' , 'optimization_history' , 'param_importances' , 'contour'] , 
                   params : list[str] | None = None , **kwargs):
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