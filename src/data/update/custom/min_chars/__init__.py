"""
Minute-reconstructed characteristics in three stages.

Each updater writes its own ``min_chars`` key (separate feathers; backfill independently).

1. ``MinCharsDailyUpdater``  → ``min_chars/min_chars``
2. ``MinCharsRollUpdater``   → ``min_chars/min_chars_roll``
3. ``MinCharsTaggedUpdater`` → ``min_chars/min_chars_tag``

Column formulas: ``FACTORS.md``.  One-row-per-column inventory: ``factors.csv``.
"""
from src.data.update.custom.min_chars.daily import (
    APPROX_FEATURES ,
    COMPUTABLE_FEATURES ,
    DROPPED_FEATURES ,
    MinCharsDailyUpdater ,
    OMITTED_OHLC ,
    OUTPUT_COLUMNS ,
    REDEFINED_FEATURES ,
    calc_min_chars ,
)
from src.data.update.custom.min_chars.rolling import (
    MinCharsRollUpdater ,
    ROLL_COLUMNS ,
    ROLL_WINDOW ,
    calc_min_chars_roll ,
)
from src.data.update.custom.min_chars.tagged import (
    MinCharsTaggedUpdater ,
    TAG_COLUMNS ,
    TAGS ,
    calc_min_chars_tag ,
)

# Backward-compatible alias (same class object; not a second registry entry).
MinCharsUpdater = MinCharsDailyUpdater

__all__ = [
    'MinCharsDailyUpdater' ,
    'MinCharsRollUpdater' ,
    'MinCharsTaggedUpdater' ,
    'MinCharsUpdater' ,
    'calc_min_chars' ,
    'calc_min_chars_roll' ,
    'calc_min_chars_tag' ,
    'OUTPUT_COLUMNS' ,
    'ROLL_COLUMNS' ,
    'TAG_COLUMNS' ,
    'TAGS' ,
    'ROLL_WINDOW' ,
    'COMPUTABLE_FEATURES' ,
    'REDEFINED_FEATURES' ,
    'APPROX_FEATURES' ,
    'DROPPED_FEATURES' ,
    'OMITTED_OHLC' ,
]
