"""Concrete processors, imported explicitly for automatic registration."""
from .registry import (
    PrePros as PrePros,
    DIRECT_FFILL_DATABLOCK_KEYS as DIRECT_FFILL_DATABLOCK_KEYS,
    TRADE_ALLOW_INACTIVE as TRADE_ALLOW_INACTIVE,
)
from .labels import PrePro_y as PrePro_y, PrePro_y_lag0 as PrePro_y_lag0
from .labels import PrePro_y_open as PrePro_y_open, PrePro_y_vwap as PrePro_y_vwap
from .trade import (
    PrePro_day as PrePro_day,
    PrePro_15m as PrePro_15m,
    PrePro_30m as PrePro_30m,
    PrePro_30mcont as PrePro_30mcont,
    PrePro_60m as PrePro_60m,
    PrePro_week as PrePro_week,
)
from .risk import PrePro_style as PrePro_style, PrePro_indus as PrePro_indus
from .factors import (
    PrePro_quality as PrePro_quality,
    PrePro_growth as PrePro_growth,
    PrePro_value as PrePro_value,
    PrePro_earning as PrePro_earning,
    PrePro_surprise as PrePro_surprise,
    PrePro_coverage as PrePro_coverage,
    PrePro_forecast as PrePro_forecast,
    PrePro_adjustment as PrePro_adjustment,
    PrePro_hf_momentum as PrePro_hf_momentum,
    PrePro_hf_volatility as PrePro_hf_volatility,
    PrePro_hf_correlation as PrePro_hf_correlation,
    PrePro_hf_liquidity as PrePro_hf_liquidity,
    PrePro_momentum as PrePro_momentum,
    PrePro_volatility as PrePro_volatility,
    PrePro_correlation as PrePro_correlation,
    PrePro_liquidity as PrePro_liquidity,
    PrePro_holding as PrePro_holding,
    PrePro_trading as PrePro_trading,
)
from .level2 import PrePro_dfl2 as PrePro_dfl2, PrePro_dfl2cs as PrePro_dfl2cs
from .minchars import PrePro_minc as PrePro_minc, PrePro_mincr as PrePro_mincr
from .market import PrePro_market as PrePro_market

from .original_15m import PrePro_15m_original as PrePro_15m_original

from ..factor_sequence.processors import (
    PrePro_factorseq, PrePro_factorperf, PrePro_factormarket,
)
