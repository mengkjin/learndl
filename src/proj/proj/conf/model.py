# basic variables in factor package
from typing import Literal

__all__ = ['ModelConfig' , 'Model']

class ModelTrainConfig:
    @property
    def resume_option(self) -> Literal['last_model_date' , 'last_pred_date']:
        """resume option of model train"""
        return 'last_pred_date'


class ModelConfig:
    def __init__(self):
        self._train = ModelTrainConfig()


    @property
    def TRAIN(self):
        """config of model train , include resume_option"""
        return self._train

Model = ModelConfig()