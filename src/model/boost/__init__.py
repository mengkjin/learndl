from .io import BoosterInput
from .algo import Lgbm , AdaBoost , BasicBoosterModel
from .booster import GeneralBooster , choose_booster_model

VALID_BOOSTERS = ['lgbm' , 'ada' , 'xgboost' , 'catboost']

def choose_booster_model(booster_type):
    if booster_type == 'lgbm':
        return Lgbm
    elif booster_type == 'ada':
        return AdaBoost
    else:
        raise KeyError(booster_type)
    