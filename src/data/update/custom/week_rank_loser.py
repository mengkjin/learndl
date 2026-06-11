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

from src.proj import CALENDAR , DB , Base
from src.func.tensor import rank_pct
from src.data.loader import DATAVENDOR
from src.data.update.custom.basic import BasicCustomUpdater

class WeekRankLoserUpdater(BasicCustomUpdater):
    """Registered updater for the weekly-rank loser stock screener."""
    START_DATE = max(20100101 , DB.min_date('trade_ts' , 'day' , use_alt=True))
    DB_SRC = 'exposure'
    DB_KEY = 'week_rank_loser'

    @classmethod
    def proceed_update(cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs) -> Base.UpdateFlag:
        """Update loser flags for all missing dates."""

        start = max(start or cls.START_DATE , cls.START_DATE)
        end = end or CALENDAR.updated()
        stored_dates = np.array([]) if overwrite else DB.dates(cls.DB_SRC , cls.DB_KEY)
        target_dates = CALENDAR.diffs(start , end , stored_dates)
        if len(target_dates) == 0:
            cls.logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        for date in target_dates:
            cls.update_one(date)
        
        return Base.UpdateFlag.SUCCESS

    @classmethod
    def update_one(cls , date : int):
        """Compute and save loser flags for a single ``date``."""
        DB.save(calc_week_rank_loser(date) , cls.DB_SRC , cls.DB_KEY , date , indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2)

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