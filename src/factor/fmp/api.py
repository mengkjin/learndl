from ..basic import DATAVENDOR , AlphaModel , AVAIL_BENCHMARK

from .stat import group_accounting
from .builder import group_optimize

from ..loader import factor as factor_loader

def main_test(nfactor = 1 , nbench = 2 , nlags = 2 , verbosity = 2):
    config_path  = 'custom_opt_config.yaml'

    factor_val = factor_loader.random(20230701 , 20230930 , nfactor=nfactor)
    alpha_models = [AlphaModel.from_dataframe(factor_val[[factor_name]]) for factor_name in factor_val.columns]

    port_optim_tuples = group_optimize(alpha_models , AVAIL_BENCHMARK[:nbench] , [0,1,2][:nlags] ,
                                       config_path = config_path , verbosity = verbosity)
    port_optim_tuples = group_accounting(port_optim_tuples , verbosity=verbosity)

    return port_optim_tuples

if __name__ == '__main__':
    
    main_test()