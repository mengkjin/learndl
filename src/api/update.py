from src.proj import Logger , MACHINE , CALENDAR
from src.proj.util import Options
from src.proj.func import print_disk_space_info
from src.res.model.util import PredictionModel

from .data import DataAPI
from .factor import FactorAPI
from .model import ModelAPI

from .trading import TradingAPI
from .notification import NotificationAPI

class UpdateAPI:
    @classmethod
    def daily(cls):
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        DataAPI.update()
        if not DataAPI.is_updated():
            Logger.conclude('Data is not updated to the latest date, skip model update' , level = 'error')
            return
        FactorAPI.Market.update()
        FactorAPI.Stock.update(timeout = 3)
        FactorAPI.Affiliate.update()
        ModelAPI.update()
        FactorAPI.Pooling.update(timeout = 3)
        FactorAPI.Stats.update()
        FactorAPI.Hierarchy.update()
        TradingAPI.update()
        NotificationAPI.update()
        print_disk_space_info()

    @classmethod
    def rollback(cls , rollback_date : int):
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        CALENDAR.check_rollback_date(rollback_date)
        DataAPI.rollback(rollback_date)
        FactorAPI.Market.rollback(rollback_date)
        FactorAPI.Stock.rollback(rollback_date , timeout = 10)
        FactorAPI.Affiliate.rollback(rollback_date)
        FactorAPI.Pooling.rollback(rollback_date , timeout = 10)
        FactorAPI.Stats.rollback(rollback_date)
        FactorAPI.Hierarchy.rollback(rollback_date)

    @classmethod
    def weekly(cls):
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        ModelAPI.update_models()

    @classmethod
    def resume_testing(cls):
        models = PredictionModel.SelectModels()
        factors = Options.available_factors()
        Logger.remark(f'Resume Testing {len(models) + len(factors)} models and factors')
        [Logger.remark(f'Resume Testing Model {model.model_path.name}' , indent = 1) for model in models]
        [Logger.remark(f'Resume Testing Factor {factor}' , indent = 1) for factor in factors]
        for model in PredictionModel.SelectModels():
            if model.model_path.name not in ModelAPI.Trainer.available_models():
                Logger.warning(f'Model {model.model_path.name} is not available at {MACHINE.name}, skip testing')
                continue
            with Logger.ParagraphI(f'Resume Testing Model {model.model_path}'):
                ModelAPI.test_model(model.model_path.name , resume = 1)
        for factor in Options.available_factors():
            with Logger.ParagraphI(f'Resume Testing Factor {factor}'):
                ModelAPI.test_factor(factor , resume = 1)
        
    