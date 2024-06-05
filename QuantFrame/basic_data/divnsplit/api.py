import pandas as pd
import os


def load_divnsplit_data(root_path, sed, eed):
    path = os.path.sep.join([root_path, 'common_data', 'divnsplit'])
    file_path = os.path.join(path, 'divnsplit.ftr')
    rtn = pd.read_feather(file_path)
    rtn = rtn[rtn["ExDate"].between(sed, eed)].copy()
    return rtn