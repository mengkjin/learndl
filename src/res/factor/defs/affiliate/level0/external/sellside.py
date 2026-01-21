from src.res.factor.calculator.factor_calc import SellsideFactor

class df_scores_v0(SellsideFactor):
    init_date = 20180101
    description = '东方金工,scores_v0'
    load_db_key = 'dongfang.scores_v0'
    load_db_col = 'avg'
    alternative_dbs = [
        {'src' : 'sellside' , 'key' : 'huayuan.scores_v0' , 'col' : 'scores_v0'}
    ]

    def calc_factor(self , date : int):
        return self.load_factor(date)

class ht_master_combined(SellsideFactor):
    init_date = 20180101
    description = '华泰金工,master_combined'
    load_db_key = 'huatai.master_combined'
    load_db_col = 'master_combined'

    def calc_factor(self , date : int):
        return self.load_factor(date)