"""Tushare fetchers for sell-side analyst reports."""
# do not use relative import in this file because it will be running in top-level directory
from src.data.download.tushare.basic import RollingFetcher , TS

class AnalystReport(RollingFetcher):
    """analyst report infomation"""
    DB_SRC = 'analyst_ts'
    DB_KEY = 'report'
    START_DATE = 20100101
    ROLLING_DATE_COL = 'report_date'
    ROLLING_SEP_DAYS = 30
    ROLLING_BACK_DAYS = 30
    def get_data(self , start , end):
        assert start is not None and end is not None , 'start and end must be provided'
        df = self.iterate_fetch(self.api.report_rc , limit = 2000 , max_fetch_times = 200 , start_date = int(start) , end_date = int(end))
        df = TS.code_to_secid(df)
        return df