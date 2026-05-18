from __future__ import annotations

class FutureUtils:
    @classmethod
    def assert_trainer(cls , config_or_trainer):
        from src.res.model.util.trainer import BaseTrainer
        assert isinstance(config_or_trainer , BaseTrainer) , f'config_or_trainer must be BaseTrainer, but got {type(config_or_trainer)}'
    @classmethod
    def model(cls , config_or_trainer):
        from src.res.model.util.trainer import PredictorModel
        return PredictorModel.initialize(config_or_trainer)
    @classmethod
    def callback(cls , config_or_trainer):
        from src.res.model.callback import ConsolidateCallBack
        return ConsolidateCallBack.initialize(config_or_trainer)
    @classmethod
    def data(cls , config_or_trainer):  
        from src.res.model.util.data import DataModule
        return DataModule.initialize(config_or_trainer)
    @classmethod
    def status(cls , config_or_trainer):
        from src.res.model.util.trainer import TrainerStatus
        return TrainerStatus()
    @classmethod
    def record(cls , config_or_trainer):
        cls.assert_trainer(config_or_trainer)
        from src.res.model.util.trainer import PredRecorder
        return PredRecorder(config_or_trainer)
    @classmethod
    def texts(cls , config_or_trainer):
        cls.assert_trainer(config_or_trainer)
        from src.res.model.util.trainer import TrainerTexts
        return TrainerTexts(config_or_trainer)
    @classmethod
    def container(cls , config_or_trainer):
        cls.assert_trainer(config_or_trainer)
        from src.res.model.util.storage import TypedContainer
        return TypedContainer()
    @classmethod
    def metrics(cls , config_or_trainer):
        from src.res.model.util.metric import TrainerMetrics
        return TrainerMetrics(config_or_trainer)
    @classmethod
    def checkpoint(cls , config_or_trainer):
        cls.assert_trainer(config_or_trainer)
        from src.res.model.util.storage import Checkpoint
        return Checkpoint()
    @classmethod
    def deposition(cls , config_or_trainer):
        from src.res.model.util.storage import Deposition
        return Deposition(config_or_trainer)
    @classmethod
    def get_util(cls , name : str , config_or_trainer):
        return getattr(cls , name)(config_or_trainer)