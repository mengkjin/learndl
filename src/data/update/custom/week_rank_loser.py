import pandas as pd
import numpy as np
import torch
from torch.nn.functional import pad
from src.func import tensor as T

from typing import Any , Literal
from src.proj import Logger , CALENDAR , DB , Dates , Proj
from src.data.loader import DATAVENDOR
from src.data.update.custom.basic import BasicCustomUpdater

class WeekRankLoserUpdater(BasicCustomUpdater):
    START_DATE = max(20100101 , DB.min_date('trade_ts' , 'day' , use_alt=True))
    DB_SRC = 'exposure'
    DB_KEY = 'week_rank_loser'

    @classmethod
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback'] , indent : int = 1 , vb_level : Any = 1):
        vb_level = Proj.vb.level(vb_level)
        if update_type == 'recalc':
            Logger.warning(f'Recalculate all custom index is supported , but beware of the performance for {cls.__name__}!')
            stored_dates = np.array([])
        elif update_type == 'update':
            stored_dates = DB.dates(cls.DB_SRC , cls.DB_KEY)
        elif update_type == 'rollback':
            rollback_date = CALENDAR.td(cls._rollback_date)
            stored_dates = CALENDAR.slice(DB.dates(cls.DB_SRC , cls.DB_KEY) , 0 , rollback_date - 1)
        else:
            raise ValueError(f'Invalid update type: {update_type}')
            
        end_date = CALENDAR.updated()
        update_dates = CALENDAR.diffs(cls.START_DATE , end_date , stored_dates)
        if len(update_dates) == 0:
            Logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date' , indent = indent , vb_level = vb_level)
            return

        for date in update_dates:
            cls.update_one(date , indent = indent + 1 , vb_level = vb_level + 2)

        Logger.success(f'Update {cls.DB_SRC}/{cls.DB_KEY} at {Dates(update_dates)}' , indent = indent , vb_level = vb_level)

    @classmethod
    def update_one(cls , date : int , indent : int = 2 , vb_level : Any = 2):
        DB.save(calc_week_rank_loser(date) , cls.DB_SRC , cls.DB_KEY , date , indent = indent , vb_level = vb_level)

def calc_week_rank_loser(date : int) -> pd.DataFrame:
    """
    Loser stocks (in the last 50 weeks, never top 5% but at least twice bottom 5%)
    """
    ret_block = DATAVENDOR.get_returns_block(20070101 , date)
    ret = ret_block.loc(feature = 'close').squeeze()
    logrtn = torch.log(ret + 1)
    week_rtn = pad(logrtn.unfold(1 , 5 , 1).sum(-1) , (4 , 0) , value = torch.nan)
    week_rank = torch.floor(T.rank_pct(week_rtn) / 0.05).clip(0,19) 

    i_date = sum(ret_block.date < date) 
    top_ratio = (week_rank[: , i_date-245:i_date+1:5].squeeze() >= 19).to(torch.float32).mean(-1)
    bottom_ratio = (week_rank[: , i_date-245:i_date+1:5].squeeze() <= 0).to(torch.float32).mean(-1)
    valid_weeks = week_rank[: , i_date-495:i_date+1:5].squeeze().isfinite().sum(-1) >= 50
    loser = (top_ratio == 0) & (bottom_ratio > 0.02) & (valid_weeks)
    df = pd.DataFrame({'secid' : ret_block.secid , 'top_ratio' : top_ratio.cpu().numpy() , 'bottom_ratio' : bottom_ratio.cpu().numpy() , 'valid_weeks' : valid_weeks.cpu().numpy() , 'loser' : loser.cpu().numpy()})
    return df