from .abstract_fetcher import FinaFetcher
from ..basic import pro , code_to_secid

class FinaIndicator(FinaFetcher):
    '''financial indicators'''
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'indicator'    
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.fina_indicator_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
    
class IncomeStatement(FinaFetcher):
    '''income statement'''
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'income'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.income_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
    
class BalanceSheet(FinaFetcher):
    '''balance sheet'''
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'balance'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.balancesheet_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
    
class CashFlow(FinaFetcher):
    '''cash flow'''
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'cashflow'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.cashflow_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df

class Dividend(FinaFetcher):
    '''dividend'''
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'dividend'
    def get_data(self , date : int):
        df = pro.dividend(end_date = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df

class Forecast(FinaFetcher):
    '''forecast'''
    CONSIDER_FUTURE = True
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'forecast'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.forecast_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
    
class Express(FinaFetcher):
    '''express'''
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'express'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.express_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
    
class MainBusiness(FinaFetcher):
    '''main business'''
    DATA_FREQ = 'y'
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'mainbz'
    def get_data(self , date : int):
        df = pro.fina_mainbz_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
    
class DisclosureDate(FinaFetcher):
    '''disclosure date'''
    CONSIDER_FUTURE = True
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'disclosure'
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.disclosure_date(end_date = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
