from .preference import Preference
from .factor import FactorConstants
from .model import ModelConstants
from .trading import TradingPortConstants

class Const:
    """Aggregate accessor for domain config trees under ``configs/`` (factor, model, trading, interactive, preference, etc.)."""
    Pref = Preference()
    Factor = FactorConstants()
    Model = ModelConstants()
    TradingPort = TradingPortConstants()

__all__ = ['Const']