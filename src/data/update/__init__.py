from .ndays import NDaysUpdater
from .labels import ClassicLabelsUpdater

class DataUpdater:
    @classmethod
    def proceed(cls):
        NDaysUpdater.proceed()
        ClassicLabelsUpdater.proceed()