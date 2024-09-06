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

    def load_model(self , training : bool , model_type = 'best' , lr_multiplier = 1.):
        self.model.load_model(training , model_type , lr_multiplier)

    def stack_model(self):
        self.on_before_save_model()
        for model_type in self.model_types:
            model_dict = self.model.collect(model_type)
            self.deposition.stack_model(model_dict , self.model_date , self.model_num , model_type) 

    def save_model(self):
        if self.metrics.better_attempt(self.status.best_attempt_metric): self.stack_model()
        [self.deposition.dump_model(self.model_date , self.model_num , model_type) for model_type in self.model_types]