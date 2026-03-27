class _Trainer:
    """custom class to record trainer instance"""
    def __get__(self , instance, owner):
        from src.res.model.util.classes import BaseTrainer
        return BaseTrainer._instance

class _Account:
    """custom class to record account instance"""
    def __get__(self , instance, owner):
        from src.res.factor.util import PortfolioAccount
        return PortfolioAccount._account
class _Factor:
    """custom class to record factor instance"""
    def __get__(self , instance, owner):
        from src.res.factor.util import StockFactor
        return StockFactor._factor

class InstanceCollection:
    """
    custom class to record anything
    example:
        Proj.States.trainer = trainer # for src.res.model.util.classes.BaseTrainer
        Proj.States.account = account # for src.res.factor.util.agency.portfolio_accountant.PortfolioAccount
        Proj.States.factor = factor   # for src.res.factor.util.classes.StockFactor
    """
    trainer = _Trainer()
    account = _Account()
    factor = _Factor()

    def __repr__(self):
        return f'InstanceCollection(trainer={self.trainer} , account={self.account} , factor={self.factor})'

    def info(self) -> list[str]:
        """return the machine info list"""
        return [
            f'Proj States    : {list(self.status().keys())}', 
        ]

    def status(self) -> dict:
        """return the machine status dict"""
        status = {}
        for name in ['trainer' , 'account' , 'factor']:
            obj = getattr(self , name)
            if obj is not None:
                status[name] = obj
        return status

INSTANCES = InstanceCollection()