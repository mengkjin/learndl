# do not use relative import in this file because it will be running in top-level directory
from src.proj import MACHINE
from src.data.download.tushare.basic import InfoFetcher , DateFetcher

class ConvertibleBasic(InfoFetcher):
    """convertible basic infomation"""
    DB_KEY = 'cb_basic'
    UPDATE_FREQ = 'w'

    def get_data(self , date):
        df = self.iterate_fetch(self.pro.cb_basic , limit = 5000)
        return df

class ConvertibleDailyQuote(DateFetcher):
    """convertible daily quote"""
    START_DATE = 20180101 if MACHINE.server else 20241215
    DB_KEY = 'cb_day'
    def get_data(self , date : int):
        date_str = str(date)
        
        quote = self.iterate_fetch(self.pro.cb_daily , limit = 2000 , trade_date=date_str)
        if quote.empty: 
            return quote
        quote = quote.rename(columns={'pre_close':'preclose','vol':'volume' , 'pre_settle':'presettle'})
        quote['amount'] = quote['amount'] * 10000
        
        return quote