"""
Weekly-rank loser stock screener updater.

Identifies stocks that have consistently underperformed: in the trailing 50 weeks
they never appeared in the top 5% by weekly return but appeared in the bottom 5%
at least twice (> 2% of weeks).

Stores a boolean ``loser`` flag plus supporting statistics in ``exposure/week_rank_loser``.

Note: loads the full return history since 2007 on every call (see TODO_data.md item E8
for the proposed windowed optimisation).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import pad
from typing import Literal

from src.proj import CALENDAR , DB , Base
from src.func.tensor import rank_pct
from src.data.loader import DATAVENDOR
from src.data.update.custom.basic import BasicCustomUpdater

class WeekRankLoserUpdater(BasicCustomUpdater):
    """Registered updater for the weekly-rank loser stock screener."""
    START_DATE = max(20100101 , DB.min_date('trade_ts' , 'day' , use_alt=True))
    DB_SRC = 'exposure'
    DB_KEY = 'week_rank_loser'

    def update_all(self , update_type : Literal['recalc' , 'update' , 'rollback']):
        """Update loser flags for all missing dates."""
        if update_type == 'recalc':
            self.logger.warning(f'Recalculate all custom index is supported , but beware of the performance for {self.__class__.__name__}!')
            stored_dates = np.array([])
        elif update_type == 'update':
            stored_dates = DB.dates(self.DB_SRC , self.DB_KEY)
        elif update_type == 'rollback':
            rollback_date = CALENDAR.td(self._rollback_date)
            stored_dates = CALENDAR.slice(DB.dates(self.DB_SRC , self.DB_KEY) , 0 , rollback_date - 1)
        else:
            raise ValueError(f'Invalid update type: {update_type}')
            
        end = CALENDAR.updated()
        update_dates = CALENDAR.diffs(self.START_DATE , end , stored_dates)
        if len(update_dates) == 0:
            self.logger.skipping(f'{self.DB_SRC}/{self.DB_KEY} is up to date' , idt = 1 , vb = 1)
            return

        for date in update_dates:
            self.update_one(date)

        self.logger.success(f'Update {self.DB_SRC}/{self.DB_KEY} at {Base.Dates(update_dates)}' , idt = 1 , vb = 1)

    def update_one(self , date : int):
        """Compute and save loser flags for a single ``date``."""
        DB.save(calc_week_rank_loser(date) , self.DB_SRC , self.DB_KEY , date , indent = self.indent + 2 , vb_level = self.vb_level + 2)

def calc_week_rank_loser(date : int) -> pd.DataFrame:
    """
    Loser stocks (in the last 50 weeks, never top 5% but at least twice bottom 5%)
    """
    ret_block = DATAVENDOR.get_returns_block(20070101 , date)
    ret = ret_block.loc(feature = 'close').squeeze()
    logrtn = torch.log(ret + 1)
    week_rtn = pad(logrtn.unfold(1 , 5 , 1).sum(-1) , (4 , 0) , value = torch.nan)
    week_rank = torch.floor(rank_pct(week_rtn) / 0.05).clip(0,19) 

    i_date = sum(ret_block.date < date) 
    top_ratio = (week_rank[: , i_date-245:i_date+1:5].squeeze() >= 19).to(torch.float32).mean(-1)
    bottom_ratio = (week_rank[: , i_date-245:i_date+1:5].squeeze() <= 0).to(torch.float32).mean(-1)
    valid_weeks = week_rank[: , i_date-495:i_date+1:5].squeeze().isfinite().sum(-1) >= 50
    loser = (top_ratio == 0) & (bottom_ratio > 0.02) & (valid_weeks)
    df = pd.DataFrame({'secid' : ret_block.secid , 'top_ratio' : top_ratio.cpu().numpy() , 'bottom_ratio' : bottom_ratio.cpu().numpy() , 'valid_weeks' : valid_weeks.cpu().numpy() , 'loser' : loser.cpu().numpy()})
    return df