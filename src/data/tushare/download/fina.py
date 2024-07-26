from ..basic import pro , code_to_secid , FinaFetecher

class FinaIndicator(FinaFetecher):
    '''Tonghuashun Concept'''
    def db_src(self): return 'financial_ts'
    def db_key(self): return 'indicator'    
    def get_data(self , date : int):
        assert date % 10000 in [331,630,930,1231] , date
        df = pro.fina_indicator_vip(period = str(date))
        df = code_to_secid(df , 'ts_code' , retain=True)
        return df
    
    