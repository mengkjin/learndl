"""Lazy accessors for optional global objects (trainer, portfolio account, factor engine)."""

from __future__ import annotations

from src.proj.core import singleton

class _Trainer:
    """Descriptor returning ``BaseTrainer._instance``."""
    def __get__(self , instance, owner):
        from src.res.model.util.classes import BaseTrainer
        return BaseTrainer._trainer

class _Account:
    """Descriptor returning ``PortfolioAccount._account``."""
    def __get__(self , instance, owner):
        from src.res.factor.util import PortfolioAccount
        return PortfolioAccount._account
class _Factor:
    """Descriptor returning ``StockFactor._factor``."""
    def __get__(self , instance, owner):
        from src.res.factor.util import StockFactor
        return StockFactor._factor

@singleton
class InstanceCollection:
    """
    Holds optional instances used across research code.
        Proj.instances.trainer  # BaseTrainer instance
        Proj.instances.account  # PortfolioAccount instance
        Proj.instances.factor   # StockFactor instance
    """
    trainer = _Trainer()
    account = _Account()
    factor = _Factor()

    def __repr__(self):
        return f'InstanceCollection(trainer={self.trainer} , account={self.account} , factor={self.factor})'

    def info(self) -> list[str]:
        """One-line summary of which named slots are populated."""
        return [
            f'Proj States    : {list(self.status().keys())}', 
        ]

    def status(self) -> dict:
        """Map slot name to object for non-empty trainer / account / factor."""
        status = {}
        for name in ['trainer' , 'account' , 'factor']:
            obj = getattr(self , name)
            if obj is not None:
                status[name] = obj
        return status