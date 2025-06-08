from .util import BoosterInput , BoosterOutput
from .booster import GeneralBooster , OptunaBooster , AVAILABLE_BOOSTERS

def valid_booster(booster_type : str):
    return booster_type in AVAILABLE_BOOSTERS