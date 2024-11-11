from src.factor.classes import StockFactorHierarchy
from src.factor.perf.api import PerfManager
from src.factor.fmp.api import FmpManager

def perf_test():
    pm = PerfManager.random_test(nfactor=1)
    return pm

def fmp_test():
    pm = FmpManager.random_test(nfactor=1 , verbosity=2)
    return pm

def factor_hierarchy():
    return StockFactorHierarchy()

