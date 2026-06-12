"""
Future utils for trainer
"""
from __future__ import annotations

__all__ = ['FutureUtils']

class FutureUtils:
    @classmethod
    def model(cls , trainer_or_config):
        from src.res.model.util.trainer import PredictorModel
        return PredictorModel.initialize(trainer_or_config)
    @classmethod
    def callback(cls , trainer_or_config):
        from src.res.model.callback import ConsolidateCallBack
        return ConsolidateCallBack.initialize(trainer_or_config)
    @classmethod
    def data(cls , trainer_or_config):  
        from src.res.model.util.data import DataModule
        return DataModule.initialize(trainer_or_config)
    @classmethod
    def status(cls , trainer_or_config):
        from src.res.model.util.trainer import TrainerStatus
        return TrainerStatus.initialize(trainer_or_config)
    @classmethod
    def record(cls , trainer_or_config):
        from src.res.model.util.trainer import PredRecorder
        return PredRecorder(trainer_or_config)
    @classmethod
    def texts(cls , trainer_or_config):
        from src.res.model.util.trainer import TrainerTexts
        return TrainerTexts(trainer_or_config)
    @classmethod
    def container(cls , trainer_or_config):
        from src.res.model.util.storage import TypedContainer
        return TypedContainer()
    @classmethod
    def metrics(cls , trainer_or_config):
        from src.res.model.util.metric import TrainerMetrics
        return TrainerMetrics(trainer_or_config)
    @classmethod
    def checkpoint(cls , trainer_or_config):
        from src.res.model.util.storage import Checkpoint
        return Checkpoint()
    @classmethod
    def deposition(cls , trainer_or_config):
        from src.res.model.util.storage import Deposition
        return Deposition(trainer_or_config)
    @classmethod
    def get_util(cls , name : str , trainer_or_config):
        return getattr(cls , name)(trainer_or_config)