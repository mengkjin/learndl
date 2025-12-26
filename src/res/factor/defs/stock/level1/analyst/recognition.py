import pandas as pd

from src.math.transform import winsorize , whiten

from src.res.factor.calculator import CoverageFactor
from src.res.factor.defs.stock.level0.analyst.coverage import (
    cov_inst_3m , cov_inst_12m , cov_report_3m , cov_report_12m
)

class analyst_recognition(CoverageFactor):
    init_date = 20110101
    description = '分析师覆盖4因子复合'
    
    def calc_factor(self, date: int):
        inst_3m = cov_inst_3m.EvalSeries(date)
        inst_12m = cov_inst_12m.EvalSeries(date)
        report_3m = cov_report_3m.EvalSeries(date)
        report_12m = cov_report_12m.EvalSeries(date)
        covs = [inst_3m , inst_12m , report_3m , report_12m]
        v = pd.concat([whiten(winsorize(cov)) for cov in covs] , axis = 1).mean(axis = 1)
        return whiten(winsorize(v))