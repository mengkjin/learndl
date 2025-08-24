import os


START_CALC_DATE = '2003-01-01'


def get_data_path(root_path, barra_type):
    path = os.sep.join([root_path, "barra_data", "factors", barra_type])
    return path