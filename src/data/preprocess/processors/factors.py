"""Factors preprocessors."""
from __future__ import annotations
from ..core import FactorPreProcessor

class PrePro_quality(FactorPreProcessor):
    """Quality factor preprocessor (key: ``'quality'``)."""

class PrePro_growth(FactorPreProcessor):
    """Growth factor preprocessor (key: ``'growth'``)."""

class PrePro_value(FactorPreProcessor):
    """Value factor preprocessor (key: ``'value'``)."""

class PrePro_earning(FactorPreProcessor):
    """Earnings factor preprocessor (key: ``'earning'``)."""

class PrePro_surprise(FactorPreProcessor):
    """Earnings surprise factor preprocessor (key: ``'surprise'``)."""

class PrePro_coverage(FactorPreProcessor):
    """Analyst coverage factor preprocessor (key: ``'coverage'``)."""

class PrePro_forecast(FactorPreProcessor):
    """Earnings forecast factor preprocessor (key: ``'forecast'``)."""

class PrePro_adjustment(FactorPreProcessor):
    """Analyst revision/adjustment factor preprocessor (key: ``'adjustment'``)."""

class PrePro_hf_momentum(FactorPreProcessor):
    """High-frequency momentum factor preprocessor (key: ``'hf_momentum'``)."""

class PrePro_hf_volatility(FactorPreProcessor):
    """High-frequency volatility factor preprocessor (key: ``'hf_volatility'``)."""

class PrePro_hf_correlation(FactorPreProcessor):
    """High-frequency correlation factor preprocessor (key: ``'hf_correlation'``)."""

class PrePro_hf_liquidity(FactorPreProcessor):
    """High-frequency liquidity factor preprocessor (key: ``'hf_liquidity'``)."""

class PrePro_momentum(FactorPreProcessor):
    """Price momentum factor preprocessor (key: ``'momentum'``)."""

class PrePro_volatility(FactorPreProcessor):
    """Price volatility factor preprocessor (key: ``'volatility'``)."""

class PrePro_correlation(FactorPreProcessor):
    """Return correlation factor preprocessor (key: ``'correlation'``)."""

class PrePro_liquidity(FactorPreProcessor):
    """Liquidity factor preprocessor (key: ``'liquidity'``)."""

class PrePro_holding(FactorPreProcessor):
    """Institutional holding factor preprocessor (key: ``'holding'``)."""

class PrePro_trading(FactorPreProcessor):
    """Trading behaviour factor preprocessor (key: ``'trading'``)."""

