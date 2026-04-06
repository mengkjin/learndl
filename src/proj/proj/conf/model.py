"""Mutable training resume options and static model settings from configs."""

from __future__ import annotations
from typing import Literal , Any
from src.proj.core import stderr , singleton
from src.proj.env import MACHINE

__all__ = ['ModelSettingConfig']

def notify_change(title : str , value , prev):
    """Stderr banner when a train resume field changes."""
    if value != prev:
        stderr(f'Project {title.title()} Changed from {prev} to {value}' , color = 'lightred' , bold = True)

@singleton
class ModelTrainSettings:
    """
    User-tunable resume flags for training and downstream evaluations:
    - resume_test [Literal['False' , 'last_model_date' , 'last_pred_date']]: resume option of model train: if resume testing , and if so whether to resume from last model date or last pred date
    - resume_test_fmp [Literal[False] | str]: resume option of model train: if resume test fmp , and if so whether to resume from last pred date or trailing days
    - resume_test_fmp_account [bool]: resume option of model train: if resume test fmp account
    - resume_test_factor_perf [bool]: resume option of model train: if resume test factor perf
    """

    _resume_test : Literal[False , 'last_model_date' , 'last_pred_date'] = 'last_pred_date'
    _resume_test_fmp : Literal[False] | str = 'trailing_5'
    _resume_test_fmp_account : bool = True
    _resume_test_factor_perf : bool = True

    @classmethod
    def set_class_attrs(cls , name : str , value : Any):
        assert hasattr(cls , f'_{name}') , f'Attribute {name} not found in {cls}'
        notify_change(name.replace('_', ' ').title() , value , getattr(cls , f'_{name}'))
        setattr(cls , f'_{name}' , value)

    @property
    def resume_test(self) -> Literal[False , 'last_model_date' , 'last_pred_date']:
        """resume option of model train: if resume testing"""
        return self._resume_test

    @resume_test.setter
    def resume_test(self , value):
        assert isinstance(value , str) and value in [False , 'False' , 'last_model_date' , 'last_pred_date'] , f'Invalid value: {value}'
        self.set_class_attrs('resume_test' , value)

    @property
    def resume_test_fmp(self) -> Literal[False] | str:
        """resume option of model train: if resume test fmp"""
        return self._resume_test_fmp

    @resume_test_fmp.setter
    def resume_test_fmp(self , value):
        assert value is False or (isinstance(value , str) and value.startswith('trailing_')) , f'Invalid value: {value}'
        self.set_class_attrs('resume_test_fmp' , value)

    @property
    def resume_test_fmp_account(self) -> bool:
        """Whether to resume FMP account backtests when resuming tests."""
        return self._resume_test_fmp_account

    @resume_test_fmp_account.setter
    def resume_test_fmp_account(self , value):
        assert isinstance(value , bool) , f'Invalid value: {value}'
        self.set_class_attrs('resume_test_fmp_account' , value)

    @property
    def resume_test_factor_perf(self) -> bool:
        """Whether to resume factor performance evaluation when resuming tests."""
        return self._resume_test_factor_perf

    @resume_test_factor_perf.setter
    def resume_test_factor_perf(self , value):
        assert isinstance(value , bool) , f'Invalid value: {value}'
        self.set_class_attrs('resume_test_factor_perf' , value)

@singleton
class ModelSettingConfig:
    """
    Expose ``train`` toggles and ``settings`` from ``setting/model`` settings:
    - TRAIN: config of model train , include resume options
    - SETTINGS: settings of model , including prediction models / hidden extraction models
    """

    _train = ModelTrainSettings()
    _settings = MACHINE.configs('setting' , 'model')

    @property
    def TRAIN(self):
        """config of model train , include resume options"""
        return self._train

    @property
    def SETTINGS(self):
        """settings of model , including prediction models / hidden extraction models"""
        return self._settings