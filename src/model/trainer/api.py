from .basic import TrainerModule
from .net import NetTrainer
from .booster import BoosterTrainer
from .aggregator import AggregatorTrainer

from ..classes import TrainConfig
from ...basic import REG_MODELS , THIS_IS_SERVER

class Trainer:
    @staticmethod
    def initialize(stage = -1 , resume = -1 , checkname = -1 , config_path = None , **kwargs) -> TrainerModule:
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no] , [1,yes]
        checkname: [-1,choose] , [0,default] , [1,yes]
        '''
        module_name = TrainConfig.guess_module(config_path)
        module_type = TrainConfig.get_module_type(module_name)

        use_trainer = {
            'nn' : NetTrainer ,
            'booster' : BoosterTrainer ,
            'aggregator' : AggregatorTrainer ,
        }[module_type]
        app = use_trainer(stage = stage , resume = resume , checkname = checkname , **kwargs)
        return app

    @classmethod
    def update_models(cls):
        if not THIS_IS_SERVER:
            print('This is not server! Will not update models!')
            return
        for model in REG_MODELS:
            config_path = TrainConfig.get_config_path(model.name)
            cls.initialize(stage = 0 , resume = 1 , checkname = 0 , config_path = config_path).go()