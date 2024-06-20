from .basic import DATAVENDOR , Benchmark
from .perf.api import PerfManager

def perf_test():
    factor_val = DATAVENDOR.random_factor(20230701 , 20240331 , nfactor=1).to_dataframe()
    benchmark  = None # Benchmark('csi500')

    pm = PerfManager(all=True)
    pm.calc(factor_val , benchmark).plot(show=True)
