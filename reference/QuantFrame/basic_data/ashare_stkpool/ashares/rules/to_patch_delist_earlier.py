from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
import os


def get_early_delist(start_date, end_date):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'no_st_delist_info.csv'))
    df['early_ann_date'] = CALENDAR_UTIL.get_last_dates(df['delist_date'].tolist(), True, 60)
    df['CalcDate'] = df.apply(lambda x: CALENDAR_UTIL.get_ranged_dates(x['early_ann_date'], x['delist_date']), axis=1)
    rtn = df[['Code', 'CalcDate']].explode('CalcDate')
    rtn = rtn.loc[rtn['CalcDate'].between(start_date, end_date), ['CalcDate', 'Code']].sort_values(['CalcDate', 'Code'])
    rtn['is_earlier_kicked'] = 1
    return rtn