from ..factor.perf.api import PerfManager
from ..factor.fmp.api import FmpManager
from ..basic import PATH

def perf_test():
    pm = PerfManager.random_test(nfactor=1)
    return pm

def fmp_test():
    pm = FmpManager.random_test(nfactor=1 , verbosity=2)
    return pm
