import os


START_CALC_DATE = '2003-01-01'


def get_path(root_path, category):
    path = os.path.sep.join([root_path, 'common_data', 'industry', category])
    return path
