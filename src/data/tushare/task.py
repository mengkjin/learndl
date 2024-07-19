from . import basic , info , index , daily

def main():
    tasks : list[basic.TushareFetecher] = [
        info.Calendar() ,
        info.Description() ,
        info.SWIndustry() ,
        info.ChangeName() , 
        index.THSConcept() ,
        daily.DailyValuation() ,
        daily.DailyQuote() ,
    ]
    for task in tasks:
        task.update()