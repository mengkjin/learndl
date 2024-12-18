from .multi_kline import MultiKlineUpdater
from .labels import ClassicLabelsUpdater
from .pre_process import DataPreProcessor

class OtherDataUpdater:
    @classmethod
    def update(cls):
        MultiKlineUpdater.update()
        ClassicLabelsUpdater.update()