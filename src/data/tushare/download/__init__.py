from . import daily , fina , index , info
from ..basic import TushareFetecher

TUSHARE_DOWNLOAD_TASK = [
        info.Calendar ,
        info.Description ,
        info.SWIndustry ,
        info.ChangeName , 
        index.THSConcept ,
        daily.DailyValuation ,
        daily.DailyQuote ,
        fina.FinaIndicator ,
    ]