"""Central data update config , specify the enforced start and end date for data update, on different machines."""
from __future__ import annotations

from functools import cached_property
from typing import Any

from src.proj.env import MACHINE
from src.proj.core import SingletonMeta , lit

__all__ = ['DataConstants']

class DataUpdateConfig(metaclass=SingletonMeta):
    """
    Default calendar window and step for factor refresh jobs:
    - start: start date of factor update
    - end: end date of factor update
    - step: step of factor update
    - init_date: uniform init date of factor update
    - target_dates: target dates of factor update
    """
    @cached_property
    def _use_schedule(self):
        """use schedule"""
        schedule = MACHINE.preference('project' , 'machine_schedules' , default = {}).get(MACHINE.name , 'testing')
        assert schedule == 'production' or schedule == 'testing' , f'Invalid schedule: {schedule}'
        return schedule
    @cached_property
    def _production(self) -> dict[str , Any]:
        """production schedule"""
        return MACHINE.preference('update_schedule' , 'production')
    @cached_property
    def _testing(self) -> dict[str , Any]:
        """testing schedule"""
        return MACHINE.preference('update_schedule' , 'testing')
    @property
    def schedule(self) -> dict[str , Any]:
        """schedule"""
        if self._use_schedule == 'production':
            return self._production
        elif self._use_schedule == 'testing':
            return self._testing
        else:
            raise ValueError(f'Invalid schedule: {self._use_schedule}')
    
    @property
    def update_from(self) -> int:
        """update from date"""
        return self.schedule.get('update_from' , 19000101)
    @property
    def update_to(self) -> int:
        """update to date"""
        return self.schedule.get('update_to' , 99991231)

    def start(self , key: lit.DataUpdateKey | None = None) -> int:
        if key is None:
            return self.update_from
        else:
            starts = self.schedule.get('start') or {}
            return starts.get(key , self.update_from)

    def end(self , key: lit.DataUpdateKey | None = None) -> int:
        if key is None:
            return self.update_to
        else:
            ends = self.schedule.get('end') or {}
            return ends.get(key , self.update_to)


class DataConstants(metaclass=SingletonMeta):
    """Aggregate accessor for data-related sub-configs (UPDATE, RISK, BENCH, ...):
    - UPDATE: factor update config
    - RISK: risk model config
    - BENCH: benchmarks config
    - TRADE: trade cost config
    - ROUNDING: rounding config
    - OPTIM: portfolio optimization config
    - STOCK: stock factor definition config
    - FMP: FMP config
    """

    @property
    def UPDATE(self):
        """config of factor update , include start , end , step , init_date"""
        return DataUpdateConfig()
