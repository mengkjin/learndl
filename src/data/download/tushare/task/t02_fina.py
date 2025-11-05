# do not use relative import in this file because it will be running in top-level directory
from src.data.download.tushare.basic import FinaFetcher , ts_code_to_secid

class FinaIndicator(FinaFetcher):
    """financial indicators"""
    DB_KEY = 'indicator'  
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = self.pro.fina_indicator_vip(period = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df
    
class IncomeStatement(FinaFetcher):
    """income statement"""
    DB_KEY = 'income'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = self.pro.income_vip(period = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df
    
class BalanceSheet(FinaFetcher):
    """balance sheet"""
    DB_KEY = 'balance'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = self.pro.balancesheet_vip(period = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df
    
class CashFlow(FinaFetcher):
    """cash flow"""
    DB_KEY = 'cashflow'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = self.pro.cashflow_vip(period = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df

class Dividend(FinaFetcher):
    """dividend infomation"""
    DB_KEY = 'dividend'
    def get_data(self , date : int):
        df = self.pro.dividend(end_date = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df

class Forecast(FinaFetcher):
    """forecast financial statement"""
    DB_KEY = 'forecast'
    CONSIDER_FUTURE = True
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = self.pro.forecast_vip(period = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df
    
class Express(FinaFetcher):
    """express financial statement"""
    DB_KEY = 'express'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = self.pro.express_vip(period = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df
    
class MainBusiness(FinaFetcher):
    """main business infomation"""
    DB_KEY = 'mainbz'
    DATA_FREQ = 'y'
    def get_data(self , date : int):
        df = self.pro.fina_mainbz_vip(period = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df
    
class DisclosureDate(FinaFetcher):
    """financial statement disclosure date"""
    DB_KEY = 'disclosure'
    CONSIDER_FUTURE = True
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = self.pro.disclosure_date(end_date = str(date))
        df = ts_code_to_secid(df , 'ts_code' , drop_old=False)
        return df
