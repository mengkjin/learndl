import numpy as np
import pandas as pd

from zoneinfo import ZoneInfo

from src.proj.env import MACHINE
import src.proj.db as DB

BJTZ = ZoneInfo("Asia/Shanghai")

class BasicCalendar:
    def __init__(self):
        calendar = DB.load('information_ts' , 'calendar' , raise_if_not_exist = True).loc[:,['calendar' , 'trade']]
        reserved = pd.DataFrame(MACHINE.configs('util' , 'calendar' , 'calendar'))
        if not reserved.empty:
            calendar = pd.concat([calendar , reserved.loc[:,['calendar' , 'trade']]]).drop_duplicates(subset='calendar', keep='first').sort_values('calendar')

        trd = calendar.query('trade == 1').reset_index(drop=True)
        trd['td'] = trd['calendar']
        trd['pre'] = trd['calendar'].shift(1, fill_value=-1)
        calendar = calendar.merge(trd.drop(columns='trade') , on = 'calendar' , how = 'left').ffill()
        calendar['cd_index'] = np.arange(len(calendar))
        calendar['td_index'] = calendar['trade'].cumsum() - 1
        calendar['td_forward_index'] = calendar['td_index'] + 1 - calendar['trade']
        calendar['td_forward'] = trd.iloc[calendar['td_forward_index'].clip(None , len(trd) - 1).to_numpy(int)]['calendar'].values
        calendar = calendar.astype(int).set_index('calendar')
        cal_cal = calendar.reset_index().set_index('cd_index')
        cal_trd = calendar[calendar['trade'] == 1].reset_index().set_index('td_index')

        self.full = calendar
        self.cal = cal_cal
        self.trd = cal_trd

        self.min_date = calendar.index.min()
        self.max_date = calendar.index.max()

        self.max_td_index : int = int(cal_trd.index.max())
    
BC = BasicCalendar()