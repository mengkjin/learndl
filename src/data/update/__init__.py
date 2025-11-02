from .multi_kline import MultiKlineUpdater
from .labels import ClassicLabelsUpdater

class AffiliatedDataUpdater:
    @classmethod
    def update(cls):
        MultiKlineUpdater.update()
        ClassicLabelsUpdater.update()

    @classmethod
    def update_rollback(cls , rollback_date : int):
        MultiKlineUpdater.update_rollback(rollback_date)
        ClassicLabelsUpdater.update_rollback(rollback_date)