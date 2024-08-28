from .basic import BasicBoosterModel

from .ada import AdaBoost
from .catboost import CatBoost
from .lgbm import Lgbm
from .xgboost import XgBoost

VALID_BOOSTERS = ['lgbm' , 'ada' , 'xgboost' , 'catboost']
_booster_dict = {
    'lgbm' : Lgbm,
    'ada' : AdaBoost,
    'xgboost' : XgBoost,
    'catboost' : CatBoost,
}

def choose_booster_model(booster_type : str):
    return _booster_dict[booster_type]