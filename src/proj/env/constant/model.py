"""Mutable training resume options and static model settings from configs."""

from __future__ import annotations
from typing import Literal
from src.proj.core import singleton
from src.proj.env import MACHINE

__all__ = ['ModelConstants']

@singleton
class ModelConstants:
    """
    User-tunable resume flags for training and downstream evaluations:
    - resume_test [Literal['False' , 'last_model_date' , 'last_pred_date']]: resume option of model train: if resume testing , and if so whether to resume from last model date or last pred date
    - resume_test_fmp [Literal[False] | str]: resume option of model train: if resume test fmp , and if so whether to resume from last pred date or trailing days
    - resume_test_fmp_account [bool]: resume option of model train: if resume test fmp account
    - resume_test_factor_perf [bool]: resume option of model train: if resume test factor perf
    - SETTINGS: settings of model , including prediction models / hidden extraction models
    """

    @property
    def resume_test(self) -> Literal[False , 'last_model_date' , 'last_pred_date']:
        """resume option of model train: if resume testing"""
        if not hasattr(self , '_resume_test'):
            self._resume_test = MACHINE.config.get('constant/default/model' , 'resuming_options/testing')
            assert self._resume_test in [False , 'last_model_date' , 'last_pred_date'] , f'Invalid value: {self._resume_test}'
        return self._resume_test

    @property
    def resume_fmp(self) -> Literal[False] | str:
        """resume option of model train: if resume test fmp"""
        if not hasattr(self , '_resume_test_fmp'):
            self._resume_test_fmp = MACHINE.config.get('constant/default/model' , 'resuming_options/fmp')
            assert self._resume_test_fmp is False or (isinstance(self._resume_test_fmp , str) and self._resume_test_fmp.startswith('trailing_')) , f'Invalid value: {self._resume_test_fmp}'
        return self._resume_test_fmp

    @property
    def resume_fmp_account(self) -> bool:
        """Whether to resume FMP account backtests when resuming tests."""
        if not hasattr(self , '_resume_test_fmp_account'):
            self._resume_test_fmp_account = MACHINE.config.get('constant/default/model' , 'resuming_options/fmp_account')
            assert isinstance(self._resume_test_fmp_account , bool) , f'Invalid value: {self._resume_test_fmp_account}'
        return self._resume_test_fmp_account

    @property
    def resume_factor_perf(self) -> bool:
        """Whether to resume factor performance evaluation when resuming tests."""
        if not hasattr(self , '_resume_test_factor_perf'):
            self._resume_test_factor_perf = MACHINE.config.get('constant/default/model' , 'resuming_options/factor_perf')
            assert isinstance(self._resume_test_factor_perf , bool) , f'Invalid value: {self._resume_test_factor_perf}'
        return self._resume_test_factor_perf


    @property
    def strategies(self):
        """settings of model , including prediction models / hidden extraction models"""
        return MACHINE.config.get('strategy/model')