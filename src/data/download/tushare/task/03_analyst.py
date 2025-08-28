# do not use relative import in this file because it will be running in top-level directory
from src.data.download.tushare.basic import RollingFetcher , pro , ts_code_to_secid

class AnalystReport(RollingFetcher):
    '''analyst report'''
    DB_SRC = 'analyst_ts'
    DB_KEY = 'report'
    START_DATE = 20100101
    ROLLING_DATE_COL = 'report_date'
    ROLLING_SEP_DAYS = 30
    ROLLING_BACK_DAYS = 30
    def get_data(self , start_dt , end_dt):
        assert start_dt is not None and end_dt is not None , 'start_dt and end_dt must be provided'
        df = self.iterate_fetch(pro.report_rc , limit = 2000 , max_fetch_times = 200 , start_date = int(start_dt) , end_date = int(end_dt))
        df = ts_code_to_secid(df)
        return df