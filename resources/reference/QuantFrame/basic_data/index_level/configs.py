import os


def get_data_path(root_path, category):
    path = os.path.sep.join([root_path, 'common_data', 'index_level', category])
    return path