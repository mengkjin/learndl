# from .basic import TrainerModule
from ..model_module.trainer import TrainerModule
# from .boost import BoosterTrainer


from ..util import TrainConfig
from ...basic import REG_MODELS , THIS_IS_SERVER

class Trainer:
    @staticmethod
    def initialize(stage = -1 , resume = -1 , checkname = -1 , base_path = None , **kwargs):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no] , [1,yes]
        checkname: [-1,choose] , [0,default] , [1,yes]
        '''
        
        '''
        module_name = TrainConfig.guess_module(base_path)
        module_type = TrainConfig.get_module_type(module_name)

        use_trainer = {
            'nn' : NNTrainer ,
            # 'boost' : BoosterTrainer ,
        }[module_type]
        '''
        app = TrainerModule(stage = stage , resume = resume , checkname = checkname , **kwargs)
        return app

    @classmethod
    def update_models(cls):
        if not THIS_IS_SERVER:
            print('This is not server! Will not update models!')
            return
        for model in REG_MODELS:
            cls.initialize(stage = 0 , resume = 1 , checkname = 0 , base_path = model.model_path).go()