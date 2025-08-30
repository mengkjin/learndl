import pandas as pd
class InstanceRecord:
    '''singleton class to record instances'''
    _instance = None
    _slots = ['trainer' , 'account' , 'factor']

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        print(f'src.basic.INSTANCE_RECORD can be accessed to check {self._slots}')
    
    def __repr__(self):
        attrs = {name:type(getattr(self , name)) for name in self._slots if hasattr(self , name)}
        return f'{self.__class__.__name__}({attrs})'
    
    def update_trainer(self , trainer):
        from src.res.model.util.classes import BaseTrainer
        assert isinstance(trainer , BaseTrainer) , f'trainer is not a BaseTrainer instance: {trainer}'
        self.trainer = trainer

    def update_account(self , account):
        assert isinstance(account , pd.DataFrame) , f'account is not a pd.DataFrame instance: {account}'
        self.account = account

    def update_factor(self , factor):
        from src.res.factor.util import StockFactor
        assert isinstance(factor , StockFactor) , f'factor is not a StockFactor instance: {factor}'
        self.factor = factor

INSTANCE_RECORD = InstanceRecord()
