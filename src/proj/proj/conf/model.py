# basic variables in factor package
from typing import Literal
from src.proj.abc import stderr

__all__ = ['ModelConfig' , 'Model']

def notify_change(title : str , value , prev):
    if value != prev:
        stderr(f'Project {title.title()} Changed from {prev} to {value}' , color = 'lightred' , bold = True)

class ModelTrainConfig:
    def __init__(self):
        self._resume_test : Literal[False , 'last_model_date' , 'last_pred_date'] = 'last_pred_date'
        self._resume_test_fmp : Literal[False] | str = 'trailing_5'
        self._resume_test_fmp_account : bool = True
        self._resume_test_factor_perf : bool = True

    @property
    def resume_test(self) -> Literal[False , 'last_model_date' , 'last_pred_date']:
        """resume option of model train: if resume testing"""
        return self._resume_test

    @resume_test.setter
    def resume_test(self , value):
        if value not in [False , 'False' , 'last_model_date' , 'last_pred_date']:
            raise ValueError(f'Invalid resuming test option: {value}')
        if value in ['False']:
            value = False
        notify_change('resume test' , value , self._resume_test)
        self._resume_test = value

    @property
    def resume_test_fmp(self) -> Literal[False] | str:
        """resume option of model train: if resume test fmp"""
        return self._resume_test_fmp

    @resume_test_fmp.setter
    def resume_test_fmp(self , value):
        if value is not False and not (isinstance(value , str) and value.startswith('trailing_')):
            raise ValueError(f'Invalid resuming test fmp option: {value}')
        notify_change('resume test fmp' , value , self._resume_test_fmp)
        self._resume_test_fmp = value

    @property
    def resume_test_fmp_account(self) -> bool:
        return self._resume_test_fmp_account

    @resume_test_fmp_account.setter
    def resume_test_fmp_account(self , value):
        if not isinstance(value , bool):
            raise ValueError(f'Invalid resuming test fmp account option: {value}')
        notify_change('resume test fmp account' , value , self._resume_test_fmp_account)
        self._resume_test_fmp_account = value

    @property
    def resume_test_factor_perf(self) -> bool:
        return self._resume_test_factor_perf

    @resume_test_factor_perf.setter
    def resume_test_factor_perf(self , value):
        if not isinstance(value , bool):
            raise ValueError(f'Invalid resuming test factor option: {value}')
        notify_change('resume test factor perf' , value , self._resume_test_factor_perf)
        self._resume_test_factor_perf = value

class ModelConfig:
    def __init__(self):
        self._train = ModelTrainConfig()

    @property
    def TRAIN(self):
        """config of model train , include resume options"""
        return self._train

Model = ModelConfig()