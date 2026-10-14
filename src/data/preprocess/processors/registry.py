"""Preprocessor registry facade and shared constants."""
from __future__ import annotations
from src.proj import Base
from src.data.util import DataBlock
from ..core import PreProcessor

TRADE_ALLOW_INACTIVE : bool = True

# DataBlock keys whose inday axis is a synthetic pack with overlapping calendar
# days (not a true chronological fold like 15m/30m). For these, ffill/autofill
# must use date-axis-only fill (stretch=False), not date×inday stretch.
DIRECT_FFILL_DATABLOCK_KEYS : tuple[str , ...] = ('week' ,)

class PrePros:
    """
    Thin registry facade over ``PreProcessorMeta.registry``.

    Provides key listing, processor instantiation, and iteration helpers
    used by ``ModuleData`` and ``PreProcessorTask``.
    """
    @classmethod
    def keys(cls) -> list[str]:
        """Return all registered preprocessor keys."""
        return [name for name in PreProcessor.registry.keys()]

    @classmethod
    def enabled_keys(cls) -> list[str]:
        """Return registered keys whose processor is ``ENABLED``."""
        return [name for name , kls in PreProcessor.registry.items() if kls.ENABLED]

    @classmethod
    def start_date(cls , frame : Base.lit.DataBlockTimeFrame = 'predict') -> int:
        """Return the start date used for the given ``frame`` across all processors."""
        return PreProcessor.start_date(frame)

    @classmethod
    def get_processor(
        cls , key : str , frame : Base.lit.DataBlockTimeFrame , 
        indent : int = 0 , vb_level : Base.lit.VerbosityLevel = 'max'
    ) -> PreProcessor:
        """Instantiate and return the preprocessor registered under ``key``."""
        return PreProcessor.registry[key](frame , indent = indent , vb_level = vb_level)

    @classmethod
    def allow_inactive(cls , key : str) -> bool:
        """Return whether the preprocessor is allowed to be inactive."""
        return PreProcessor.registry[key].AllowInactive

    @classmethod
    def use_direct_ffill(cls , key : str) -> bool:
        """
        Return True if ``key`` should ffill on date axis only (no date×inday stretch).

        Matches ``DIRECT_FFILL_DATABLOCK_KEYS`` after ``DataBlock.data_type_abbr``.
        """
        return DataBlock.data_type_abbr(key) in DIRECT_FFILL_DATABLOCK_KEYS

