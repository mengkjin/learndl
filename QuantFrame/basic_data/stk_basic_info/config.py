import os


def get_file_path(root_path, category):
    rtn = os.path.sep.join([root_path, "common_data", "stk_basic_info", "{0}.csv".format(category)])
    return rtn