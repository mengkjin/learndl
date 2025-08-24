import os


START_CALC_DATE = '2003-01-01'


def get_data_path(root_path_, pool_type_):
    path = os.path.sep.join([root_path_, 'stock_pool', pool_type_])
    return path
