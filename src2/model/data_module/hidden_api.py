import numpy as np
import pandas as pd

from ...basic import PATH


def hidden_model_dates(hidden_key : str):
    model_name , model_num , model_type = hidden_key.split('.')
    prefix = f'hidden.{model_num}.{model_type}.'
    suffix = f'.feather'
    model_dates = []
    for p in PATH.hidden.joinpath(model_name).iterdir():
        if p.name.startswith(prefix) and p.name.endswith(suffix):
            model_dates.append(int(p.name.removeprefix(prefix).removesuffix(suffix)))
    return np.sort(model_dates)

def get_hidden_df(hidden_key : str , model_date : int):
    possible_hmd = hidden_model_dates(hidden_key)
    hmd = possible_hmd[possible_hmd <= model_date].max()
    model_name , model_num , model_type = hidden_key.split('.')
    hidden_path = PATH.hidden.joinpath(model_name , f'hidden.{model_num}.{model_type}.{hmd}.feather')
    hidden_df = pd.read_feather(hidden_path)
    return hidden_df