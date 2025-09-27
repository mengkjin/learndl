import pandas as pd

class InstanceRecord:
    """
    singleton class to record instances

    example:
        INSTANCE_RECORD.update_trainer(trainer) # for src.res.model.util.classes.BaseTrainer
        INSTANCE_RECORD.update_account(account) # for pandas.DataFrame portfolio account
        INSTANCE_RECORD.update_factor(factor) # for src.res.factor.util.classes.StockFactor
    """
    _instance = None
    _slots = ['trainer' , 'account' , 'factor']

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        attrs = {name:type(getattr(self , name)) for name in self._slots if hasattr(self , name)}
        return f'{self.__class__.__name__}({attrs})'
    
    def update_trainer(self , trainer):
        """
        update the trainer instance
        """
        from src.res.model.util.classes import BaseTrainer
        assert isinstance(trainer , BaseTrainer) , f'trainer is not a BaseTrainer instance: {trainer}'
        self.trainer = trainer

    def update_account(self , account):
        """
        update the account instance
        """
        assert isinstance(account , pd.DataFrame) , f'account is not a pd.DataFrame instance: {account}'
        self.account = account

    def update_factor(self , factor):
        """
        update the factor instance
        """
        from src.res.factor.util import StockFactor
        assert isinstance(factor , StockFactor) , f'factor is not a StockFactor instance: {factor}'
        self.factor = factor

INSTANCE_RECORD = InstanceRecord()
