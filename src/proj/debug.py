import pandas as pd

class InstanceRecord:
    """
    custom class to record instances

    example:
        InstanceRecord.update_trainer(trainer) # for src.res.model.util.classes.BaseTrainer
        InstanceRecord.update_account(account) # for pandas.DataFrame portfolio account
        InstanceRecord.update_factor(factor) # for src.res.factor.util.classes.StockFactor
    """
    _slots = ['trainer' , 'account' , 'factor']
    
    @classmethod
    def update_trainer(cls , trainer):
        """
        update the trainer instance
        """
        from src.res.model.util.classes import BaseTrainer
        assert isinstance(trainer , BaseTrainer) , f'trainer is not a BaseTrainer instance: {trainer}'
        cls.trainer = trainer

    @classmethod
    def update_account(cls , account):
        """
        update the account instance
        """
        assert isinstance(account , pd.DataFrame) , f'account is not a pd.DataFrame instance: {account}'
        cls.account = account

    @classmethod
    def update_factor(cls , factor):
        """
        update the factor instance
        """
        from src.res.factor.util import StockFactor
        assert isinstance(factor , StockFactor) , f'factor is not a StockFactor instance: {factor}'
        cls.factor = factor

