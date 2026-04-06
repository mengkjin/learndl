"""Typed accessors for domain config trees under ``configs/`` (factor, model, trading, etc.)."""

from .factor import FactorConfig
from .model import ModelSettingConfig
from .interactive import InteractiveConfig
from .trading import TradingPortConfig

__all__ = ['Conf']

class Conf:
    """Aggregate accessor for domain config trees under ``configs/`` (factor, model, trading, interactive, etc.)."""
    Factor = FactorConfig()
    Model = ModelSettingConfig()
    Interactive = InteractiveConfig()
    TradingPort = TradingPortConfig()