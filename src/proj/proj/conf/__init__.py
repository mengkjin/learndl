"""Typed accessors for domain config trees under ``configs/`` (factor, model, trading, etc.)."""

from .factor import FactorConfig
from .model import ModelConfig
from .interactive import InteractiveConfig
from .trading import TradingPortConfig

__all__ = ['Conf']

class Conf:
    """Aggregate accessor for domain config trees under ``configs/`` (factor, model, trading, interactive, etc.)."""
    Factor = FactorConfig()
    Model = ModelConfig()
    Interactive = InteractiveConfig()
    TradingPort = TradingPortConfig()