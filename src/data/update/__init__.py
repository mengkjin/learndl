from .multi_kline import MultiKlineUpdater
from .labels import ClassicLabelsUpdater

class OtherDataUpdater:
    @classmethod
    def update(cls):
        MultiKlineUpdater.update()
        ClassicLabelsUpdater.update()