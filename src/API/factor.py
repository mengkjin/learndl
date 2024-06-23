from ..factor.perf.api import PerfManager
from ..factor.fmp.api import FmpManager

def perf_test(save = True):
    pm = PerfManager.random_test(nfactor=1)
    if save : pm.save('factor_result/random')
    return pm

def fmp_test(save = True):
    pm = FmpManager.random_test(nfactor=1 , verbosity=2)
    if save : pm.save('factor_result/random')
    return pm
