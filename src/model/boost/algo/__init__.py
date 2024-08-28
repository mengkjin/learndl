from .basic import BasicBoosterModel

from .ada import AdaBoost
from .lgbm import Lgbm
from .xgboost import XgBoost

def choose_booster_model(booster_type : str):
    if booster_type == 'lgbm':
        return Lgbm
    elif booster_type == 'ada':
        return AdaBoost
    elif booster_type == 'xgb':
        return XgBoost
    else:
        raise KeyError(booster_type)