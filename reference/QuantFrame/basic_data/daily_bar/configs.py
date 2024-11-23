import os


def get_data_path(root_path_, category_):
    path = os.path.sep.join([root_path_, 'common_data', 'daily_bar', category_])
    return path