"""Registered updater for the full listing-reference table."""
from src.proj import Base
from src.data.util.prices.listing import update_listing_prices
from .basic import BasicCustomUpdater

class ListingPriceUpdater(BasicCustomUpdater):
    START_DATE = 19970101
    UPDATE_ORDER = -90
    DEPENDS_ON = ('AdjPriceUpdater',)
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE, Base.UpdateType.ROLLBACK, Base.UpdateType.RECALC)

    @classmethod
    def proceed_update(cls, start=None, end=None, overwrite=False, **kwargs):
        changed = update_listing_prices(start=start, end=end, overwrite=overwrite)
        return Base.UpdateFlag.SUCCESS if changed else Base.UpdateFlag.SKIPPED
