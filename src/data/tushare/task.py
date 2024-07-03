from . import basic , info , index

def main():
    tasks : list[basic.TushareFetecher] = [
        info.Calendar() ,
        info.Description() ,
        info.SWIndustry() ,
        info.ChangeName() , 
        index.THSConcept() ,
    ]
    for task in tasks:
        task.update()