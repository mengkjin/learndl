from . import daily , fina , index , info
from ..basic import TushareFetecher

class TushareDownloader:
    DOWNLOAD_TASKS = [
        info.Calendar ,
        info.Description ,
        info.SWIndustry ,
        info.ChangeName , 
        index.THSConcept ,
        daily.DailyValuation ,
        daily.DailyQuote ,
        fina.FinaIndicator ,
    ]
    def __init__(self) -> None:
        pass

    @classmethod
    def proceed(cls):
        for task_down in cls.DOWNLOAD_TASKS:
            task = task_down()
            assert isinstance(task , TushareFetecher) , task
            task.update()
