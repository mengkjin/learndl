import numpy as np
import pandas as pd

from typing import Any

from src.data.download.tushare.basic import RollingFetcher , pro , code_to_secid

class AnalystReport(RollingFetcher):
    '''analyst report'''
    DB_SRC = 'analyst_ts'
    DB_KEY = 'report'
    START_DATE = 20100101
    ROLLING_DATE_COL = 'report_date'
    ROLLING_SEP_DAYS = 30
    ROLLING_BACK_DAYS = 30
    def get_data(self , start_date , end_date):
        assert start_date is not None and end_date is not None , 'start_date and end_date must be provided'
        df = self.iterate_fetch(pro.report_rc , limit = 2000 , max_fetch_times = 200 , start_date = int(start_date) , end_date = int(end_date))
        df = code_to_secid(df)
        return df