"""Mutable training resume options and static model settings from configs."""

from __future__ import annotations

from src.proj.core import SingletonMeta
from src.proj.env.machine import MACHINE

__all__ = ['ModelConstants']

class ModelConstants(metaclass=SingletonMeta):
    """
    User-tunable resume flags for training and downstream evaluations:
    - resume_test [False | 'last_model_date' | 'last_pred_date']: resume option of model train: if resume testing , and if so whether to resume from last model date or last pred date
    - resume_test_fmp [False | str]: resume option of model train: if resume test fmp , and if so whether to resume from last pred date or trailing days
    - resume_test_fmp_account [bool]: resume option of model train: if resume test fmp account
    - resume_test_factor_perf [bool]: resume option of model train: if resume test factor perf
    - SETTINGS: settings of model , including prediction models / hidden extraction models
    """

    @property
    def resume_test(self) -> bool:
        """resume option of model train: if resume testing"""
        value = MACHINE.preference('model' , 'resuming_options/testing')
        assert isinstance(value , bool) , f'Invalid value: {value}'
        return value

    @property
    def resume_test_start(self):
        """resume option of model train: if resume testing"""
        value = MACHINE.preference('model' , 'resuming_options/testing_start')
        assert value == 'last_model_date' or value == 'last_pred_date' , f'Invalid value: {value}'
        return value

    @property
    def resume_fmp(self):
        """resume option of model train: if resume test fmp"""
        value = MACHINE.preference('model' , 'resuming_options/fmp')
        assert value is False or (isinstance(value , str) and value.startswith('trailing_')) , f'Invalid value: {value}'
        return value

    @property
    def resume_fmp_account(self) -> bool:
        """Whether to resume FMP account backtests when resuming tests."""
        value = MACHINE.preference('model' , 'resuming_options/fmp_account')
        assert isinstance(value , bool) , f'Invalid value: {value}'
        return value

    @property
    def resume_factor_perf(self) -> bool:
        """Whether to resume factor performance evaluation when resuming tests."""
        value = MACHINE.preference('model' , 'resuming_options/factor_perf')
        assert isinstance(value , bool) , f'Invalid value: {value}'
        return value


    @property
    def strategies(self):
        """settings of model , including prediction models / hidden extraction models"""
        return MACHINE.config.get('strategy/model')