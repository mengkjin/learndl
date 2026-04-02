import pandas as pd

from src.data import DATAVENDOR
from src.proj import CALENDAR , DB

class Omission:
    def __init__(self , name: str = 'warnst'):
        self.name = name

    def __repr__(self):
        return f'Omission(name={self.name})'

    def __call__(self , date: int) -> list[int]:
        return self.get_secid(date)

    def get_secid(self , date: int) -> list[int]:
        secid = []
        if 'warnst' in self.name:
            secid.extend(self.warnst(date))
        return list(set(secid))

    @classmethod
    def collect_announcements(cls , start_date: int , end_date: int): # -> dict[int , list[str]]:
        dates = CALENDAR.range(start_date , end_date , 'cd')
        paths = [path for ex in ['bse','szse','sse'] for path in [DB.path('crawler' , f'announcement_{ex}' , date) for date in dates]]
        df = DB.load_df(paths)
        return df
        
    @classmethod
    def recent_announcements(cls , date: int , window: int = 90): # -> dict[int , list[str]]:
        start_date = CALENDAR.cd(date , -window + 1)
        df = cls.collect_announcements(start_date , date)
        df = df.loc[df['secid'].isin(DATAVENDOR.secid(date))]
        return df

    @classmethod
    def warnst(cls , date: int , window: int = 90) -> list[int]:
        anns = cls.recent_announcements(date , window)
        if anns.empty:
            return []
        pats = [r"可能.*(?:风险警示)",r"可能.*(?:退市)"]
        cond = pd.concat([anns["title"].str.contains(pat, regex=True, na=False) for pat in pats] , axis = 1).any(axis = 1)
        df = anns.loc[cond].loc[:,['sec_name','secid','title','date']].sort_values(by = 'secid')
        return df['secid'].unique().tolist()