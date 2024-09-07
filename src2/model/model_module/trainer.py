from .module_selector import module_selector
from ..callback import CallBackManager
from ..data_module import DataModule
from ..util.classes import BaseTrainer

class TrainerModule(BaseTrainer):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data     = DataModule(self.config)
    def init_model(self , **kwargs):
        self.model    = module_selector(self.config.model_module).create_from_trainer(self , **kwargs)
    def init_callbacks(self , **kwargs) -> None: 
        self.callback = CallBackManager.setup(self)